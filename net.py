import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)

def mlp_modify(sizes, activation, output_activation=nn.Identity, dropout=0.2):
    layers = []
    for j in range(len(sizes) - 1):
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers.append(layer)

        if j < len(sizes) - 2:
            norm_layer = nn.LayerNorm(sizes[j + 1])

            nn.init.constant_(norm_layer.weight, 1.0)
            nn.init.constant_(norm_layer.bias, 0.0)
            layers.append(norm_layer)

        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act())


    return nn.Sequential(*layers)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, use_std=True):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, nn.Tanh)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.use_std = use_std
        if self.use_std:
            self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(
            self,
            obs,
            deterministic=False,
            with_logprob=True,
            with_mean_std=False,
    ):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)

        if self.use_std:
            log_std = self.log_std_layer(net_out)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)

            # Pre-squash distribution and sample
            pi_distribution = Normal(mu, std)

            if deterministic:
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            if with_logprob:

                entropy = pi_distribution.entropy().sum(axis=-1)

                tanh_correction = torch.log(1 - torch.tanh(pi_action) ** 2 + 1e-6).sum(
                    axis=-1)
                entropy = entropy - tanh_correction

            else:
                entropy = None

            pi_action = torch.tanh(pi_action)

            if with_mean_std:
                return pi_action, entropy, mu, log_std

            return pi_action, entropy
        else:
            # Deterministic output
            return torch.tanh(mu), None


class EnsembleQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2, use_mlp_modify=False):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        if not use_mlp_modify:
            self.q_nets = nn.ModuleList(
                [
                    mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
                    for i in range(num_q)
                ]
            )
        else:
            self.q_nets = nn.ModuleList(
                [
                    mlp_modify([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
                    for i in range(num_q)
                ]
            )

    def forward(self, obs, act=None):
        data = obs if act is None else torch.cat([obs, act], dim=-1)
        return [torch.squeeze(q(data), -1) for q in self.q_nets]

    def predict(self, obs, act):
        q_list = self.forward(obs, act)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def loss(self, target, q_list=None):
        losses = [((q - target) ** 2).mean() for q in q_list]
        return sum(losses)

class EnsembleVCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2, use_mlp_modify=False):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        if not use_mlp_modify:
            self.q_nets = nn.ModuleList(
                [
                    mlp([obs_dim] + list(hidden_sizes) + [1], nn.ReLU)
                    for i in range(num_q)
                ]
            )
        else:
            self.q_nets = nn.ModuleList(
                [
                    mlp_modify([obs_dim] + list(hidden_sizes) + [1], nn.ReLU)
                    for i in range(num_q)
                ]
            )

    def forward(self, obs):
        data = obs
        return [torch.squeeze(q(data), -1) for q in self.q_nets]

    def predict(self, obs):
        q_list = self.forward(obs)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def loss(self, target, q_list=None):
        losses = [((q - target) ** 2).mean() for q in q_list]
        return sum(losses)