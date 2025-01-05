import random
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

# from o2o_safe_rl import  net
from net import SquashedGaussianMLPActor, EnsembleQCritic, VAE, EnsembleVCritic
from utils import Lagrange, PIDLagrangian

class SAC_Lag(object):
    def __init__(
            self,
            project,
            name,
            state_dim,
            action_dim,
            max_action,
            cost_limit,
            device,
            seed,
            actor_lr=5e-5,
            critic_lr=5e-5,
            cost_critic_lr=5e-5,
            lambda_lr=5e-5,
            # results_dir,
            discount=0.99,
            tau=0.005,
            alpha=1e-5,
            kl_coeff=0.1,
            use_reward_critic_norm=False,
            use_cost_critic_norm=False,
            if_pid=False,
            lagrangian_multiplier_init=0.0,
            args=None
    ):

        self.device = device

        # if args.env == "OfflineDroneCircle-v0":
        if args is not None:
            if args.env == "OfflineDroneCircle-v0" or args.env == "OfflineAntCircle-v0" or args.env == "OfflineDroneRun-v0" or args.env == "OfflineAntRun-v0":
                print("Using OfflineDroneCircle!!!")
                self.actor = SquashedGaussianMLPActor(state_dim, action_dim, [256, 256, 256], nn.ReLU).to(self.device)
            else:
                self.actor = SquashedGaussianMLPActor(state_dim, action_dim, [256, 256], nn.ReLU).to(self.device)

        else:
            self.actor = SquashedGaussianMLPActor(state_dim, action_dim, [256, 256], nn.ReLU).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.initial_actor = copy.deepcopy(self.actor)

        if not use_reward_critic_norm:
            self.critic = EnsembleQCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=2).to(self.device)
        else:
            self.critic = EnsembleQCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=2, use_mlp_modify=True).to(self.device)
        if args.env == "OfflineDroneCircle-v0" or args.env == "OfflineAntCircle-v0" or args.env == "OfflineDroneRun-v0" or args.env == "OfflineAntRun-v0":
            self.critic = EnsembleQCritic(state_dim, action_dim, [256, 256, 256], nn.ReLU, num_q=2).to(
            self.device)
        else:
            self.critic = EnsembleQCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=2).to(
                self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        if not use_cost_critic_norm:
            self.cost_critic = EnsembleQCritic(state_dim, action_dim, [256, 256, 256], nn.ReLU, num_q=1).to(self.device)
        else:
            self.cost_critic = EnsembleQCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=1, use_mlp_modify=True).to(
                self.device)

        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_critic_lr)

        self.judge_safe = EnsembleVCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=1, use_mlp_modify=True).to(self.device)
        self.judge_safe_optimizer = torch.optim.Adam(self.judge_safe.parameters(), lr=cost_critic_lr)


        self.lagrange = Lagrange(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer='Adam',
            lagrangian_upper_bound=None
        )

        self.if_pid =  if_pid
        if self.if_pid:
            self.pid_lagrange = PIDLagrangian(
                cost_limit=cost_limit,
                pid_kp=5e-6,
                pid_ki=5e-6,
                pid_kd=5e-6,
                pid_d_delay=10,
                lagrangian_multiplier_init=0.0
            )

        self.lagrange_refine = Lagrange(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer='Adam',
            lagrangian_upper_bound=None
        )
        
        self.vae = VAE(obs_dim=state_dim, act_dim=action_dim, hidden_size=512,
                  latent_dim=action_dim * 2, act_lim=max_action, device=device).to(device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)

        self.v_critic = EnsembleVCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=1).to(self.device)
        self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=critic_lr)
        self.cost_v_critic = EnsembleVCritic(state_dim, action_dim, [256, 256], nn.ReLU, num_q=1).to(self.device)
        self.cost_v_critic_optimizer = torch.optim.Adam(self.cost_v_critic.parameters(), lr=cost_critic_lr)

        self.max_action = max_action
        self.cost_limit = cost_limit
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.l2_reg_coeff = 0 #0.0001
        self.max_grad_norm = 40
        self.kl_coeff = kl_coeff

        self.total_it = 0

        self.logger = wandb.init(project=project, entity='keru_chen', name=f'seed{seed}', group=name,
                                   config={
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                            "max_action": max_action,
                            "cost_limit": cost_limit,
                            "device": str(device),
                            "actor_lr": actor_lr,
                            "critic_lr": critic_lr,
                            "cost_critic_lr": cost_critic_lr,
                            "lambda_lr": lambda_lr,
                            "discount": discount,
                            "tau": tau,
                            "alpha": alpha,
                            "kl_coeff": kl_coeff
                        })
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    def train(self, replay_buffer, offline_dataset, batch_size=256, episode_cost=0, online_ratio=1, if_kl=False,
              if_use_initial_actor=True, if_so2=False):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, cost = replay_buffer.sample(int(batch_size * online_ratio))
        dataset = TensorDataset(
            state.to(self.device),
            action.to(self.device),
            next_state.to(self.device),
            reward.to(self.device),
            cost.to(self.device),
            not_done.to(self.device)
        )

        if offline_dataset and not(online_ratio == 1):
            sampled_indices = np.random.choice(len(offline_dataset), int(batch_size * (1 - online_ratio)), replace=False)
            offline_dataset = Subset(offline_dataset, sampled_indices)
            dataset = ConcatDataset([offline_dataset, dataset])

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        Q, Qc = [], []
        kl = []

        for mini_batch in dataloader:
            mini_state, mini_action, mini_next_state, mini_reward, mini_cost, mini_not_done = mini_batch

            with torch.no_grad():
                next_action, next_log_pi = self.actor(mini_next_state)

                if if_so2:
                    std_dev = 0.1
                    noise = torch.normal(mean=0.0, std=std_dev, size=next_action.shape)
                    next_action = next_action + noise

                # Compute the entropy
                entropy = torch.mean(next_log_pi)

                # Compute the target Q value
                target_Q, _ = self.critic_target.predict(mini_next_state, next_action)
                target_Q = target_Q - self.alpha * entropy
                target_Q = mini_reward + mini_not_done * self.discount * target_Q

                # Compute the target cost value
                target_cost, _ = self.cost_critic_target.predict(mini_next_state, next_action)
                target_cost = mini_cost + mini_not_done * self.discount * target_cost

            # Get current Q estimates
            _, current_Q = self.critic.predict(mini_state, mini_action)
            # Compute critic loss
            critic_loss = self.critic.loss(target_Q, current_Q)

            l2_reg_critic = sum(param.pow(2).sum() for param in self.critic.parameters())
            critic_loss += self.l2_reg_coeff * l2_reg_critic

            Q.append(torch.mean(target_Q).item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            self.critic_optimizer.step()

            # Get current cost estimates
            _, current_cost = self.cost_critic.predict(mini_state, mini_action)
            # Compute cost critic loss
            cost_critic_loss = self.cost_critic.loss(target_cost, current_cost)

            l2_reg_cost_critic = sum(param.pow(2).sum() for param in self.cost_critic.parameters())
            cost_critic_loss += self.l2_reg_coeff * l2_reg_cost_critic

            Qc.append(torch.mean(target_cost).item())

            # Optimize the cost critic
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()

            self.cost_critic_optimizer.step()

            # Update the Lagrange multiplier
            self.lagrange.update_lagrange_multiplier(episode_cost)
            if self.if_pid:
                self.pid_lagrange.pid_update(episode_cost)

            lagrange_item = self.lagrange.get_lagrangian_multiplier
            if self.if_pid:
                lagrange_item = self.pid_lagrange.get_lagrangian_multiplier


            # Compute actor loss with entropy
            pi_action, logp_pi, mu, log_std = self.actor(mini_state, with_logprob=True, with_mean_std=True)
            std = log_std.exp()
            entropy = -torch.mean(logp_pi)

            q_pi, _ = self.critic_target.predict(mini_state, pi_action)
            qc_pi, _ = self.cost_critic_target.predict(mini_state, pi_action)

            if if_kl:
                initial_pi_action, initial_logp_pi, initial_mu, initial_log_std = self.initial_actor(mini_state, with_logprob=True, with_mean_std=True)
                initial_std = initial_log_std.exp()
                kl_divergence = 0.5 * torch.mean(torch.log(initial_std / std) + (std.pow(2) + (mu - initial_mu).pow(2)) / (2 * initial_std.pow(2)) - 0.5)
                kl.append(kl_divergence.item())
                actor_loss = (-q_pi + lagrange_item * qc_pi - self.alpha * entropy + self.kl_coeff * kl_divergence).mean()
            else:
                actor_loss = (-q_pi + lagrange_item * qc_pi - self.alpha * entropy).mean()

            if not if_use_initial_actor:
                self.initial_actor = copy.deepcopy(self.actor)


            l2_reg_actor = sum(param.pow(2).sum() for param in self.actor.parameters())
            actor_loss += self.l2_reg_coeff * l2_reg_actor

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.logger.log({
            "critic_loss": critic_loss.item(),
            "cost_critic_loss": cost_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_target_Q": np.mean(Q),
            "mean_target_Qc": np.mean(Qc),
            "entropy": entropy.item(),
            "lagrange":lagrange_item
        }, step=self.total_it)

        if if_kl:
            wandb.log({"mean_kl_divergence": np.mean(kl)}, step=self.total_it)

        if if_kl:
            return np.mean(Q), np.mean(Qc), np.mean(kl)
        else:
            return np.mean(Q), np.mean(Qc)


    def vpa(self, replay_buffer, if_critic=True, if_cost_critic=True, batch_size=256):
        state, action, next_state, reward, not_done, cost = replay_buffer.sample(batch_size)
        dataset = TensorDataset(
            state.to(self.device),
            action.to(self.device),
            next_state.to(self.device),
            reward.to(self.device),
            cost.to(self.device),
            not_done.to(self.device)
        )

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for mini_batch in dataloader:
            (
                observations,
                actions,
                next_observations,
                rewards,
                costs,
                not_dones,
            ) = mini_batch

            with torch.no_grad():

                next_action, next_log_pi = self.initial_actor(next_observations)

                # Compute the target Q value
                target_Q, _ = self.critic_target.predict(next_observations, next_action)
                target_Q = rewards + not_dones * self.discount * target_Q

                # Compute the target cost value
                target_cost, _ = self.cost_critic_target.predict(next_observations, next_action)
                target_cost = costs + not_dones * self.discount * target_cost

            # Get current Q estimates
            _, current_Q = self.critic.predict(observations, actions)
            # Compute critic loss
            critic_loss = self.critic.loss(target_Q, current_Q)

            if if_critic:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            # Get current cost estimates
            _, current_cost = self.cost_critic.predict(observations, actions)
            # Compute cost critic loss
            cost_critic_loss = self.cost_critic.loss(target_cost, current_cost)

            if if_cost_critic:
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            if if_critic:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if if_cost_critic:
                for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def vpa_entropy(self, replay_buffer, if_critic=True, if_cost_critic=True, batch_size=256):
        state, action, next_state, reward, not_done, cost = replay_buffer.sample(batch_size)
        dataset = TensorDataset(
            state.to(self.device),
            action.to(self.device),
            next_state.to(self.device),
            reward.to(self.device),
            cost.to(self.device),
            not_done.to(self.device)
        )

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for mini_batch in dataloader:
            (
                observations,
                actions,
                next_observations,
                rewards,
                costs,
                not_dones,
            ) = mini_batch

            with torch.no_grad():
                next_action, next_log_pi = self.initial_actor(next_observations)

                # Compute the entropy
                entropy = -torch.mean(next_log_pi) * 0.1

                # Compute the target Q value
                target_Q, _ = self.critic_target.predict(next_observations, next_action)
                target_Q = rewards + not_dones * self.discount * (target_Q - entropy)

                # Compute the target cost value
                target_cost, _ = self.cost_critic_target.predict(next_observations, next_action)
                target_cost = costs + not_dones * self.discount * (target_cost - entropy)


            # Get current Q estimates
            _, current_Q = self.critic.predict(observations, actions)
            # Compute critic loss
            critic_loss = self.critic.loss(target_Q, current_Q)

            if if_critic:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            # Get current cost estimates
            _, current_cost = self.cost_critic.predict(observations, actions)
            cost_critic_loss = self.cost_critic.loss(target_cost, current_cost)

            if if_cost_critic:
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            if if_critic:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if if_cost_critic:
                for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
