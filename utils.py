import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torch_geometric.nn import HANConv
# ====================================================================
# Collector and replay buffer
# ---------------------------

def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----

class HAN(nn.Module):
    def __init__(self, num_experts, hidden_channels=64, heads=8):
        super().__init__()
        self.num_experts = num_experts
        in_channels = {
            "Arrived": 3 * num_experts,
            "Expert": 3,
            "Running": 6,
            "Waiting": 6,
        }
        metadata = (
            ["Arrived", "Expert", "Running", "Waiting"],
            [("Arrived", "to", "Expert"),
             ("Expert", "to", "Arrived"),
            ("Expert", "to", "Running"),
            ("Running", "to", "Expert"),
            ("Expert", "to", "Waiting"),
            ("Waiting", "to", "Expert")],
        )
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                metadata=metadata)
        self.han_conv2 = HANConv(hidden_channels, hidden_channels, heads=heads,
                                metadata=metadata)

    def forward(self, x):
        x = x.squeeze(0)
        a2e = [[], []]
        e2a = [[], []]
        for i in range(self.num_experts):
            a2e[0].append(0)
            a2e[1].append(i)
            e2a[0].append(i)
            e2a[1].append(0)

        e2r = [[], []]
        r2e = [[], []]
        e2w = [[], []]
        w2e = [[], []]
        expert_features = []
        running_features = []
        waiting_features = []
        start = 3 * self.num_experts
        for i in range(self.num_experts):
            expert_features.append(x[start:start+3])
            start += 3

            for _ in range(10):
                if x[start:start+6].sum() != 0:
                    e2r[0].append(i)
                    e2r[1].append(len(running_features))
                    r2e[0].append(len(running_features))
                    r2e[1].append(i)

                    running_features.append(x[start:start+6])
                start += 6
            
            for _ in range(10):
                if x[start:start+6].sum() != 0:
                    e2w[0].append(i)
                    e2w[1].append(len(waiting_features))
                    w2e[0].append(len(waiting_features))
                    w2e[1].append(i)

                    waiting_features.append(x[start:start+6])
                start += 6
        
        x_dict = {
            'Arrived': x[:3 * self.num_experts],
            'Expert': torch.stack(expert_features, dim=0),
            'Running': torch.stack(running_features, dim=0) if len(running_features) > 0 else torch.zeros(0, 6),
            'Waiting': torch.stack(waiting_features, dim=0) if len(waiting_features) > 0 else torch.zeros(0, 6),
        }
        
        edge_index_dict = {
            ('Arrived', 'to', 'Expert'): torch.LongTensor(a2e),
            ('Expert', 'to', 'Arrived'): torch.LongTensor(e2a),
            ('Expert', 'to', 'Running'): torch.LongTensor(e2r) if len(running_features) > 0 else torch.zeros(2, 0, dtype=torch.long),
            ('Running', 'to', 'Expert'): torch.LongTensor(r2e) if len(running_features) > 0 else torch.zeros(2, 0, dtype=torch.long),
            ('Expert', 'to', 'Waiting'): torch.LongTensor(e2w) if len(waiting_features) > 0 else torch.zeros(2, 0, dtype=torch.long),
            ('Waiting', 'to', 'Expert'): torch.LongTensor(w2e) if len(waiting_features) > 0 else torch.zeros(2, 0, dtype=torch.long),
        }

        out = self.han_conv(x_dict, edge_index_dict)
        out = self.han_conv2(out, edge_index_dict)
        return out['Arrived']
    
class Actor(nn.Module):
    def __init__(self, han, actor_net_kwargs):
        super().__init__()
        self.han = han
        self.mlp = MLP(**actor_net_kwargs)
    
    def forward(self, obs):
        obs = obs.view(-1, 126 * self.han.num_experts)
        obss = []
        for i in range(obs.shape[0]):
            obss.append(self.han(obs[i]))
        obs = torch.concat(obss, dim=0)
        obs = self.mlp(obs)
        return obs

class Critic(nn.Module):
    def __init__(self, han, qvalue_net_kwargs):
        super().__init__()
        self.han = han
        self.mlp = MLP(**qvalue_net_kwargs)
    
    def forward(self, action, obs):
        obs = obs.view(-1, 126 * self.han.num_experts)
        obss = []
        for i in range(obs.shape[0]):
            obss.append(self.han(obs[i]))
        obs = torch.concat(obss, dim=0)
        action = action.view(obs.shape[0], -1)
        obs = self.mlp(torch.cat([action, obs], dim=-1))
        return obs

def make_sac_agent(cfg, train_env, device):
    han = HAN(cfg.env.num_experts)

    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = Actor(han, actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = Critic(han, qvalue_net_kwargs)

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = train_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    return model, model[0]


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError