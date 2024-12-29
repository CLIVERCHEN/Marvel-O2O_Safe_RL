import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
import gymnasium as gym
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
import argparse
import os
from online.sac_lag import SAC_Lag
from numpy.random import PCG64, Generator
import bullet_safety_gym  # noqa
from tqdm.auto import trange  # noqa
import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from torch.distributions import Normal
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
import matplotlib.pyplot as plt
from tqdm.auto import trange  # noqa
import copy
from utils import *
import datetime
import pickle
from config.warmstart_config import warmstart_config
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JSRL:
    def __init__(
            self,
            agent,
            args
    ):
        self.agent = agent
        self.args = args

    def set_env(self):
        self.env = gym.make(self.args.env)
        self.env = wrap_env(
            env=self.env,
            reward_scale=0.1,
        )
        env = OfflineEnvWrapper(self.env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, device=DEVICE)

    def set_dataset(self):
        with open(f'../offline_dataset/{self.args.env}/dataset.pkl', 'rb') as file:
            loaded_dataset = pickle.load(file)
        for i in range(len(loaded_dataset['actions'])):
            obs = torch.tensor(loaded_dataset['observations'][i], dtype=torch.float32)
            next_obs = torch.tensor(loaded_dataset['next_observations'][i], dtype=torch.float32)
            action = torch.tensor(loaded_dataset['actions'][i], dtype=torch.float32)
            reward = torch.tensor(loaded_dataset['rewards'][i] * 0.1, dtype=torch.float64)
            cost = torch.tensor(loaded_dataset['costs'][i], dtype=torch.float64)
            done = torch.tensor(loaded_dataset['done'][i], dtype=torch.float32)
            self.buffer.add(obs, action, next_obs, reward, done, cost)

    def sample_action(self, obs, timestep):
        offline_ratio = 1 - (self.agent.total_it / self.args.max_timesteps)
        current_ratio = timestep / self.args.max_envsteps
        if current_ratio < offline_ratio:
            action_online, _ = self.agent.initial_actor(
                torch.tensor(obs[None, ...], dtype=torch.float32).to(DEVICE), with_logprob=True)
            action = np.squeeze(action_online.cpu().numpy(), axis=0)
        else:
            action_offline, _ = self.agent.actor(
                torch.tensor(obs[None, ...], dtype=torch.float32).to(DEVICE), with_logprob=True)
            action = np.squeeze(action_offline.cpu().numpy(), axis=0)

        chosen_action = np.squeeze(action.cpu().numpy(), axis=0)

        return chosen_action

    def rollout(self):
        episode_reward = 0
        episode_cost = 0
        # Rollout
        for _ in range(self.args.rollout_num * 5):
            with torch.no_grad():
                obs, _ = self.env.reset()
                done = False
                timestep = 0
                for _ in range(self.args.max_envsteps):
                    while not done:
                        action = self.sample_action(obs, timestep)
                        obs_next, reward, terminated, truncated, info = self.env.step(action)
                        # episode_reward += reward
                        cost = info["cost"]
                        # episode_cost += cost
                        done = 1 if terminated or truncated else 0
                        self.buffer.add(obs, action, obs_next, reward, done, cost)
                        obs = obs_next
                        timestep += 1

        with torch.no_grad():
            obs, _ = self.env.reset()
            done = False
            timestep = 0
            for _ in range(self.args.max_envsteps):
                while not done:
                    action = self.sample_action(obs, timestep)
                    obs_next, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                    cost = info["cost"]
                    episode_cost += cost
                    done = 1 if terminated or truncated else 0
                    self.buffer.add(obs, action, obs_next, reward, done, cost)
                    obs = obs_next
                    timestep += 1

        return episode_reward, episode_cost

    def train(self, seed):
        rollout_reward, rollout_cost = [], []
        Q, Qc = [], []
        evaluations_reward, evaluations_cost = [], []

        train_num = self.args.max_timesteps // self.args.rollout_num
        for t in trange(int(train_num), desc="Training"):

            # Evaluate episode
            if t % self.args.eval_freq == 0:
                avg_reward, avg_cost = eval_policy(policy=self.agent, env_name=self.args.env, device=DEVICE, seed=seed)
                evaluations_reward.append(avg_reward)
                evaluations_cost.append(avg_cost)

                self.agent.logger.log({
                    "eval_reward": avg_reward * 10,
                    "eval_cost": avg_cost
                }, step=self.agent.total_it)

            # Rollout
            episode_reward, episode_cost = self.rollout()

            episode_reward /= (self.args.rollout_num * 5)
            episode_cost /= (self.args.rollout_num * 5)

            for _ in range(self.args.rollout_num):
                q, qc = self.agent.train(replay_buffer=self.buffer, batch_size=self.args.batch_size, episode_cost=episode_cost,
                                     online_ratio=1, offline_dataset=None, if_kl=False)
                rollout_reward.append(episode_reward)
                rollout_cost.append(episode_cost)
                Q.append(q)
                Qc.append(qc)

                self.agent.logger.log({
                    "rollout_reward": episode_reward * 10,
                    "rollout_cost": episode_cost
                }, step=self.agent.total_it)

            print(
                f"Episode: {t + 1} Reward: {episode_reward * 10:.3f} Cost: {episode_cost}")
            # Reset environment
            state, _ = self.env.reset()

    def run(self, seed):  # every seed
        # for seed in self.args.seeds:
        self.set_env()
        # self.set_dataset()

        self.train(seed)
        self.agent.logger.finish()