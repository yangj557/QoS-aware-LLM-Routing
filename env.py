import os
import random
import numpy as np
from typing import Optional
from itertools import accumulate

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import torch
from tensordict import TensorDict

from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
from torchrl.envs import EnvBase, TransformedEnv, DoubleToFloat
from torchrl.envs.utils import check_env_specs

from vllm_expert import vLLM_Expert
from datasets import load_from_disk

def _set_seed(self, seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)
    self.rng = rng

def _step(self, tensordict):
    # model selection
    weights = tensordict["action"].squeeze(-1)
    expert_idx = torch.argmax(weights, dim=-1).unsqueeze(dim=0)
    
    # do action
    penalty = 0 
    if expert_idx < self.num_experts:
        future = self.experts[expert_idx].add.remote(self.times_idx)
        ray.get(future)

        # penalty
        data = self.dataset[self.times_idx]
        prompt = data["instruction"] + " " + data["input"]
        input_tokens = self.experts[expert_idx].get_length.remote(prompt)
        outputs = data["candidates"][self.expert_params[expert_idx][0]]["text"]
        output_tokens = self.experts[expert_idx].get_length.remote(outputs)
        penalty = self.experts[expert_idx].penalty.remote(input_tokens, output_tokens)
        penalty = ray.get(penalty)

    # reward
    futrues = [expert.get_and_reset_reward.remote() for expert in self.experts]
    reward = sum(ray.get(futrues)) - penalty

    # step time
    self.times_idx += 1
    self.current_time = self.workload[self.times_idx]
    futures = [expert.step_to_time.remote(self.current_time, self.times_idx >= 5000) for expert in self.experts]
    ray.get(futures)

    # done
    done = True if self.times_idx == 5000 else False

    # new observation
    data = self.dataset[self.times_idx % 5000]
    prompt = data["instruction"] + " " + data["input"]
    input_tokens = [expert.get_length.remote(prompt) for expert in self.experts]
    input_tokens = ray.get(input_tokens)
    pred_output_tokens = [data["pred_output_tokens"][self.expert_params[i][0]] for i in range(self.num_experts)]
    pred_scores = [data["pred_score"][self.expert_params[i][0]] for i in range(self.num_experts)]

    new_observation = []
    new_observation.extend(input_tokens)
    new_observation.extend(pred_scores)
    new_observation.extend(pred_output_tokens)
    futrues = [expert.get_features.remote() for expert in self.experts]
    features = ray.get(futrues)
    for feature in features:
        new_observation.extend(feature)

    # res
    out = TensorDict(
        {
            "observation": new_observation,
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
        device=self.device,
    )
    return out

def _reset(self, tensordict):
    # reset env
    self.times_idx = 0
    self.current_time = self.workload[self.times_idx]
    futrues = [expert.reset.remote() for expert in self.experts]
    ray.get(futrues)

    # observation
    data = self.dataset[self.times_idx]
    prompt = data["instruction"] + " " + data["input"]
    input_tokens = [expert.get_length.remote(prompt) for expert in self.experts]
    input_tokens = ray.get(input_tokens)
    pred_output_tokens = [data["pred_output_tokens"][self.expert_params[i][0]] for i in range(self.num_experts)]
    pred_scores = [data["pred_score"][self.expert_params[i][0]] for i in range(self.num_experts)]
    new_observation = []
    new_observation.extend(input_tokens)
    new_observation.extend(pred_scores)
    new_observation.extend(pred_output_tokens)
    futrues = [expert.get_features.remote() for expert in self.experts]
    features = ray.get(futrues)
    for feature in features:
        new_observation.extend(feature)

    out = TensorDict(
        {
            "observation": new_observation,
        },
        batch_size=[],
        device=self.device,
    )
    return out

def _make_spec(self):
    self.observation_spec = CompositeSpec(
        observation=UnboundedContinuousTensorSpec(shape=(126 * self.num_experts,), dtype=torch.float64, device=self.device),
        device=self.device,
    )

    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = BoundedTensorSpec(
        low=-1,
        high=1,
        shape=(self.num_experts + 1,),
        dtype=torch.float32,
        device=self.device
    )
    
    self.reward_spec = CompositeSpec(
        reward=UnboundedContinuousTensorSpec(shape=(1,), device=self.device, dtype=torch.float64),
        device=self.device,
    )

class MyEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, num_experts=3, lam=3.0, L=0.03, seed=None, device="cpu"):
        super().__init__(device=device, batch_size=[])
        self.num_experts = num_experts
        self.lam = lam
        self.L = L

        self.dataset = load_from_disk("/root/nfs/drl_scheduling/simulator2/datasets/llm-blender/mix-instruct/train")

        ray.init()
        pg = placement_group(
            name="llm_pg",
            bundles=[{"GPU": 1, "CPU": 1} for _ in range(num_experts)],
            strategy="STRICT_PACK"  # or "PACK" or "SPREAD" depending on your needs
        )
        ray.get(pg.ready())

        self.expert_params = [
            (2, "/root/nfs/models/chavinlo/alpaca-native", 7.2 * 1e-5, 2.5 * 1e-5, self.L),
            (9, "/root/nfs/models/THUDM/chatglm-6b", 6.8 * 1e-5, 2.5 * 1e-5, self.L),
            (11, "/root/nfs/models/mosaicml/mpt-7b-instruct", 7.2 * 1e-5, 2.5 * 1e-5, self.L),
            (1, "/root/nfs/models/TheBloke/koala-7B-HF", 7.2 * 1e-5, 2.5 * 1e-5, self.L),
            (3, "/root/nfs/models/mosesjun0h/llama-7b-hf-baize-lora-bf16", 7.2 * 1e-5, 2.5 * 1e-5, self.L),
            (10, "/root/nfs/models/mosaicml/mpt-7b", 7.2 * 1e-5, 2.5 * 1e-5, self.L),
        ]

        self.experts = [
            vLLM_Expert.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i
                )
            ).remote(*self.expert_params[i]) for i in range(self.num_experts)
        ]

        # init spec
        self._make_spec()

        # set seed
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

        # workload
        workload = np.random.exponential(1 / self.lam, 4999)
        workload = [0] + list(accumulate(workload)) + [np.inf]
        self.workload = [round(x, 2) for x in workload]
        self.times_idx = 0

        self.cumulated_reward = 0.0

    # Helpers: _make_spec, _process, _filter
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = _step
    _set_seed = _set_seed

if __name__ == "__main__":
    env = MyEnv(seed=2024)
    env = TransformedEnv(
        env,
        DoubleToFloat(),
    )
    check_env_specs(env)
