import os
import ray
import time
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_from_disk

@ray.remote(num_gpus=1, num_cpus=1)
class vLLM_Expert:
    def __init__(self, id, model_name, k1, k2, L=0.03):
        self.id = id # id in dataset
        self.model_name = model_name
        self.k1 = k1
        self.k2 = k2
        self.L = L

        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
        self.expert = LLM(model_name,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.9,
                        swap_space=0,
                        trust_remote_code=True)
        self.dataset = load_from_disk("/root/nfs/drl_scheduling/simulator2/datasets/llm-blender/mix-instruct/train")
        
        self.current_time = 0.0
        self.info = None
        self.reward = 0
        self.features = None
        self.rid2start_time = {} # request id to start time
        self.rid2idx = {} # request id to index in dataset

    def add(self, idx):
        data = self.dataset[idx]
        prompt = data["instruction"] + " " + data["input"]
        output = data["candidates"][self.id]["text"]
        output_tokens = len(self.expert.get_tokenizer()(output)["input_ids"])
        sampling_params = SamplingParams(max_tokens=output_tokens, ignore_eos=True)
        self.expert._add_request(prompt, sampling_params)

        rid = str(self.expert.request_counter.counter-1)
        self.rid2start_time[rid] = self.current_time
        self.rid2idx[rid] = idx

        return rid

    def step(self):
        if not self.expert.llm_engine.has_unfinished_requests():
            self.current_time += 0.01
            return

        start = time.time()
        self.info = self.expert.llm_engine.step()
        end = time.time()
        self.current_time += end - start

        for i in self.info:
            if i.finished:
                rid = i.request_id
                idx = self.rid2idx[rid]
                data = self.dataset[idx]
                score = data["candidates"][self.id]["scores"]["bertscore"]

                latency = (self.current_time - self.rid2start_time[rid]) / len(i.outputs[0].token_ids)
                self.reward += score if latency < self.L else 0.0
        return self.current_time
    
    def step_to_time(self, time, end=False):
        while self.current_time < time if not end else self.expert.llm_engine.has_unfinished_requests():
            self.step()
        
        return self.current_time

    def get_features(self):
        total = self.expert.llm_engine.cache_config.num_gpu_blocks
        used = 0
        num_running = len(self.expert.llm_engine.scheduler[0].running)
        num_waiting = len(self.expert.llm_engine.scheduler[0].waiting)
        for r in self.expert.llm_engine.scheduler[0].running:
            used += len(r.get_seqs()[0].get_token_ids())
        gpu_util = used / total
        features = [gpu_util, num_running, num_waiting]

        running_request_feature = []
        for i in range(10):
            if i < num_running:
                r = self.expert.llm_engine.scheduler[0].running[i]
                rid = r.request_id
                idx = self.rid2idx[rid]
                data = self.dataset[idx]

                input_tokens = r.get_seqs()[0].get_prompt_len()
                pred_score = data["pred_score"][self.id]
                pred_output_tokens = data["pred_output_tokens"][self.id]
                gpu_util = len(r.get_seqs()[0].get_token_ids()) / total
                current_output_tokens = r.get_seqs()[0].get_output_len()
                current_latency = (self.current_time - self.rid2start_time[rid]) / current_output_tokens if current_output_tokens > 0 else 0
                running_request_feature.extend([input_tokens, pred_score, pred_output_tokens, gpu_util, current_output_tokens, current_latency])
            else:
                running_request_feature.extend([0, 0, 0, 0, 0, 0])

        waiting_request_feature = []
        for i in range(10):
            if i < num_waiting:
                r = self.expert.llm_engine.scheduler[0].waiting[i]
                rid = r.request_id
                idx = self.rid2idx[rid]
                data = self.dataset[idx]

                input_tokens = r.get_seqs()[0].get_prompt_len()
                pred_score = data["pred_score"][self.id]
                pred_output_tokens = data["pred_output_tokens"][self.id]
                gpu_util = 0.0
                current_output_tokens = r.get_seqs()[0].get_output_len()
                current_latency = (self.current_time - self.rid2start_time[rid]) / current_output_tokens if current_output_tokens > 0 else 0
                waiting_request_feature.extend([input_tokens, pred_score, pred_output_tokens, gpu_util, current_output_tokens, current_latency])
            else:
                waiting_request_feature.extend([0, 0, 0, 0, 0, 0])

        features.extend(running_request_feature)
        features.extend(waiting_request_feature)
        return features
    
    def get_length(self, prompt):
        return len(self.expert.get_tokenizer()(prompt)["input_ids"])

    def get_and_reset_reward(self):
        reward = self.reward
        self.reward = 0
        return reward
    
    def penalty(self, pi, di):
        penalty = 0.0
        for r in self.expert.llm_engine.scheduler[0].running:
            rid = r.request_id
            idx = self.rid2idx[rid]
            data = self.dataset[idx]
            output = data["candidates"][self.id]["text"]
            score = data["candidates"][self.id]["scores"]["bertscore"]

            output_tokens = len(self.expert.get_tokenizer()(output)["input_ids"])
            current_output_tokens = r.get_seqs()[0].get_output_len()
            second = 0.0
            for k in range(1, min(output_tokens-current_output_tokens, di)):
                second += pi + k
            l_plus = (self.k1 * pi + self.k2 * second) / output_tokens
            current_l = (self.current_time - self.rid2start_time[rid]) / current_output_tokens if current_output_tokens > 0 else 0
            l_pred = current_l + l_plus

            if l_pred > self.L:
                penalty += score

        return penalty
    
    def reset(self):
        self.current_time = 0.0
        self.info = None
        self.reward = 0
        self.features = None
        self.rid2start_time = {} # request id to start time
        self.rid2idx = {} # request id to index in dataset

        self.expert._run_engine(use_tqdm=False)

if __name__ == "__main__":
    # set seed
    np.random.seed(42)

    expert = vLLM_Expert(2, "/root/nfs/models/chavinlo/alpaca-native", 7.2 * 1e-5, 2.5 * 1e-5)
    timestamp = [0.0, 10.0]

    result_path = "/root/nfs/drl_scheduling/simulator2/datasets/llm-blender/mix-instruct/train"
    dataset = load_from_disk(result_path)
    for i, t in enumerate(timestamp):
        while expert.current_time < t:
            expert.step()
        expert.add(i)
        print("reward: ", expert.get_and_reset_reward())
        print("features: ", expert.get_features())

        data = dataset[i]
        prompt = data["instruction"] + " " + data["input"]
        output = data["candidates"][expert.id]["text"]
        pi = len(expert.expert.get_tokenizer()(prompt)["input_ids"])
        di = len(expert.expert.get_tokenizer()(output)["input_ids"])
        print("penalty: ", expert.penalty(pi, di))
