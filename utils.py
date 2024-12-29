from __future__ import annotations
import numpy as np
import torch
import gymnasium as gym
# import gym
from gym.wrappers import RecordEpisodeStatistics, RecordVideo
from typing import Optional, Tuple, Union, List
import os
import random
from dsrl.offline_env import OfflineEnvWrapper
from tqdm.auto import trange
from tqdm import tqdm
import pybullet as pb
from collections import deque

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e7)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros(max_size)
        self.not_done = np.zeros(max_size)
        self.cost = np.zeros(max_size)

        self.device = device

    def add(self, state, action, next_state, reward, done, cost):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1 - done
        self.cost[self.ptr] = cost

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device)
        )

class Lagrange:
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        self.cost_limit: float = cost_limit
        self.lambda_lr: float = lambda_lr
        self.lagrangian_upper_bound: float | None = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 1e-5)
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

    @property
    def get_lagrangian_multiplier(self) -> float:
        return self.lagrangian_multiplier.item()

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )

class PIDLagrangian:
    def __init__(
        self,
        pid_kp=0.1,
        pid_ki=0.01,
        pid_kd=0.01,
        pid_d_delay=5,
        pid_delta_p_ema_alpha=0.9,
        pid_delta_d_ema_alpha=0.9,
        sum_norm=True,
        diff_norm=False,
        penalty_max=100,
        lagrangian_multiplier_init=0.001,
        cost_limit=20.0,
    ) -> None:
        self._pid_kp: float = pid_kp
        self._pid_ki: float = pid_ki
        self._pid_kd: float = pid_kd
        self._pid_kp_ori: float = pid_kp
        self._pid_ki_ori: float = pid_ki
        self._pid_kd_ori: float = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._pid_i: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._cost_penalty: float = 0.0

    @property
    def get_lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    def pid_update(self, ep_cost_avg: float) -> None:
        delta = float(ep_cost_avg - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)

        self.adapt_pid_parameters(ep_cost_avg)

    def adapt_pid_parameters(self, ep_cost_avg):
        # Calculate statistics
        costs = np.array(self._cost_ds)
        avg_cost = np.mean(costs + 1e-1)
        std_dev = np.std(costs)

        alpha = 0.05
        beta = 0.05
        gamma = 0.05

        # Adaptive PID parameters
        if avg_cost > self._cost_limit:
            self._pid_kp *= (1 + alpha * (avg_cost - self._cost_limit) / avg_cost)
            self._pid_ki *= (1 + beta * (avg_cost - self._cost_limit) / avg_cost)
        else:
            self._pid_kp *= (1 - alpha * (self._cost_limit - avg_cost) / avg_cost)
            self._pid_ki *= (1 - beta * (self._cost_limit - avg_cost) / avg_cost)

        # Adjust Kd based on the standard deviation of the costs
        self._pid_kd *= (1 + gamma * std_dev / avg_cost)

        # Ensure PID parameters stay within reasonable bounds
        self._pid_kp = min(max(self._pid_kp, self._pid_kp_ori / 10), self._pid_kp_ori * 10)
        self._pid_ki = min(max(self._pid_ki, self._pid_ki_ori / 10), self._pid_ki_ori * 10)
        self._pid_kd = min(max(self._pid_kd, self._pid_kd_ori / 10), self._pid_kd_ori * 10)


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (
                       state - state_mean
               ) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        #env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

import matplotlib.pyplot as plt

def plot_variable(variable, title, file_name):
    plt.figure(figsize=(25, 8))
    plt.plot(variable)
    plt.title(f"{title} over Time")
    plt.xlabel("Time Step")
    plt.ylabel(title)
    plt.savefig(f"{file_name}")
    plt.close()

@torch.no_grad()
def eval_policy(policy, env_name, device, seed, eval_episodes=10):
    avg_reward, avg_cost = 0.0, 0.0
    for s in range(eval_episodes):
        eval_env = gym.make(env_name)
        eval_env = wrap_env(
            env=eval_env,
            reward_scale=0.1,
        )
        eval_env = OfflineEnvWrapper(eval_env)
        set_seed(env=eval_env, seed=seed+5, deterministic_torch=True)
        done = False

        for _ in range(300):
            obs, _ = eval_env.reset()
            while not done:
                action, _ = policy.actor(torch.tensor(obs[None, ...], dtype=torch.float32).to(device), deterministic=True,
                                         with_logprob=False)
                action = np.squeeze(action.cpu().numpy(), axis=0)
                obs_next, reward, terminated, truncated, info = eval_env.step(action)
                avg_reward += reward
                avg_cost += info["cost"]
                done = terminated or truncated
                obs = obs_next

    avg_reward *= 10
    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Reward evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Cost evaluation over {eval_episodes} episodes: {avg_cost:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_cost

@torch.no_grad()
def eval_policy_explore(policy, env_name, device, seed, eval_episodes=10):
    avg_reward, avg_cost = 0.0, 0.0
    for s in range(eval_episodes):
        eval_env = gym.make(env_name)
        eval_env = wrap_env(
            env=eval_env,
            reward_scale=0.1,
        )
        eval_env = OfflineEnvWrapper(eval_env)
        set_seed(env=eval_env, seed=seed+5, deterministic_torch=True)
        done = False

        for _ in range(200):
            obs, _ = eval_env.reset()
            while not done:
                action, _ = policy.actor_explore(torch.tensor(obs[None, ...], dtype=torch.float32).to(device), deterministic=True,
                                         with_logprob=False)
                action = np.squeeze(action.cpu().numpy(), axis=0)
                obs_next, reward, terminated, truncated, info = eval_env.step(action)
                avg_reward += reward
                avg_cost += info["cost"]
                done = terminated or truncated
                obs = obs_next

    avg_reward *= 10
    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Reward evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Cost evaluation over {eval_episodes} episodes: {avg_cost:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_cost

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


@torch.no_grad()
def eval_critic(policy, env, device, num_eval=10):

    reward_bias_list, cost_bias_list = [], []

    for _ in range(num_eval):
        obs, _ = env.reset()
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(device)
        action, _ = policy.actor(obs, with_logprob=True)

        critic_q, _ = policy.critic_target.predict(obs, action)
        critic_qc, _ = policy.cost_critic_target.predict(obs, action)

        done = False

        mean_reward, mean_cost = [], []

        for _ in range(10):
            eval_reward, eval_cost = 0, 0
            for _ in range(300):
                if not done:
                    obs_next, reward, terminated, truncated, info = env.step(action.cpu().detach().numpy())
                    eval_reward += reward
                    cost = info["cost"]
                    eval_cost += cost
                    done = 1 if terminated or truncated else 0
                    obs = obs_next
                    action, _ = policy.actor(torch.tensor(obs[None, ...], dtype=torch.float32).to(device),
                                             deterministic=True, with_logprob=True)
                else:
                    break
            mean_reward.append(eval_reward)
            mean_cost.append(eval_cost)

        reward_bias = (critic_q - np.mean(mean_reward)) / abs(np.mean(mean_reward) + 1e-4)
        cost_bias = (critic_qc - np.mean(mean_cost)) / abs(np.mean(mean_cost) + 1e-4)

        reward_bias_list.append(reward_bias.cpu().detach().numpy())
        cost_bias_list.append(cost_bias.cpu().detach().numpy())

    return np.mean(reward_bias_list), np.std(reward_bias_list), np.mean(cost_bias_list), np.std(cost_bias_list)


@torch.no_grad()
def eval_critic_from_offline_data_BallCircle(policy, env, device, offline_dataset, num_eval=20):

    reward_bias_list, cost_bias_list = [], []
    reward_list, cost_list = [], []
    for data in tqdm(offline_dataset):
        original_obs, next_obs, action, reward, cost, done = data

        obs_, _ = env.reset()
        env = set_state_BallCircle(env=env, obs=original_obs)

        original_obs = torch.tensor(original_obs).squeeze().to(device)
        action = torch.tensor(action).squeeze().to(device)

        critic_q, _ = policy.critic_target.predict(original_obs, action)
        critic_qc, _ = policy.cost_critic_target.predict(original_obs, action)

        _reward_bias_list, _cost_bias_list = [], []
        _reward_list, _cost_list = [], []

        for _ in range(num_eval):

            done = False
            eval_reward, eval_cost = 0, 0

            for _ in range(300):
                while not done:
                    obs_next, reward, terminated, truncated, info = env.step(action.cpu().detach().numpy())
                    eval_reward += reward
                    cost = info["cost"]
                    eval_cost += cost
                    done = 1 if terminated or truncated else 0
                    obs = obs_next
                    action, _ = policy.actor(torch.tensor(obs[None, ...], dtype=torch.float32).to(device),
                                           deterministic=True, with_logprob=True)

            reward_bias = abs(critic_q - eval_reward) / abs(eval_reward)
            cost_bias = abs(critic_qc - eval_cost) / (abs(eval_cost) + 1e-4)

            _reward_bias_list.append(reward_bias.cpu().detach().numpy())
            _cost_bias_list.append(cost_bias.cpu().detach().numpy())
            _reward_list.append(eval_reward)
            _cost_list.append(eval_cost)

        reward_bias_list.append(np.mean(_reward_bias_list))
        cost_bias_list.append(np.mean(_cost_bias_list))
        reward_list.append(np.mean(_reward_list))
        cost_list.append(np.mean(_cost_list))

    return reward_bias_list, reward_list, cost_bias_list, cost_list

def set_state_BallCircle(env, obs):
    agent = env.env.agent
    bc = env.env.agent.bc
    agent_obs = np.array(obs[:-1])
    task_obs = obs[-1]

    position = np.concatenate([agent_obs[:2] / 0.1, [agent.init_xyz[2]]])
    linear_velocity = np.concatenate([agent_obs[2:4] / 0.2, [0]])

    bc.resetBasePositionAndOrientation(agent.body_id, position, agent.get_quaternion())
    bc.resetBaseVelocity(agent.body_id, linear_velocity, agent_obs[4:] / 0.1)
    bc.resetBasePositionAndOrientation(agent.body_id, position, agent.get_quaternion())
    bc.resetBaseVelocity(agent.body_id, linear_velocity, agent_obs[4:])

    dist_from_center = np.linalg.norm(agent.get_position()[:2])
    circle_radius = dist_from_center - (task_obs * env.env.task.world.env_dim)
    env.env.task.circle.radius = circle_radius

    return env

def set_state_CarRun(env, obs):
    agent = env.env.agent
    bc = agent.bc

    xy = obs[:2] / 0.1
    xy_dot = obs[2:4]
    sin_yaw = obs[4]
    cos_yaw = obs[5]
    yaw = np.arctan2(sin_yaw, cos_yaw)
    yaw_dot = obs[6] / 0.1

    position = np.concatenate([xy, [agent.init_xyz[2]]])
    velocity = np.concatenate([xy_dot, [0]])
    angular_velocity = [0, 0, yaw_dot]

    orientation_quat = pb.getQuaternionFromEuler([0, 0, yaw])
    bc.resetBasePositionAndOrientation(agent.body_id, position, orientation_quat)
    bc.resetBaseVelocity(agent.body_id, velocity, angular_velocity)

    return env


@torch.no_grad()
def eval_critic_from_offline_data_CarRun(policy, env, device, offline_dataset, num_eval=20):
    reward_bias_list, cost_bias_list = [], []
    reward_list, cost_list = [], []
    for data in tqdm(offline_dataset):
        original_obs, next_obs, action, reward, cost, done = data

        obs_, _ = env.reset()
        env = set_state_CarRun(env=env, obs=original_obs)

        original_obs = torch.tensor(original_obs).squeeze().to(device)
        action = torch.tensor(action).squeeze().to(device)

        critic_q, _ = policy.critic_target.predict(original_obs, action)
        critic_qc, _ = policy.cost_critic_target.predict(original_obs, action)

        _reward_bias_list, _cost_bias_list = [], []
        _reward_list, _cost_list = [], []

        for _ in range(num_eval):

            done = False
            eval_reward, eval_cost = 0, 0

            for _ in range(300):
                while not done:
                    obs_next, reward, terminated, truncated, info = env.step(action.cpu().detach().numpy())
                    eval_reward += reward
                    cost = info["cost"]
                    eval_cost += cost
                    done = 1 if terminated or truncated else 0
                    obs = obs_next
                    action, _ = policy.actor(torch.tensor(obs[None, ...], dtype=torch.float32).to(device),
                                             deterministic=True, with_logprob=True)

            reward_bias = abs(critic_q - eval_reward) / abs(eval_reward)
            cost_bias = abs(critic_qc - eval_cost) / (abs(eval_cost) + 1e-4)

            _reward_bias_list.append(reward_bias.cpu().detach().numpy())
            _cost_bias_list.append(cost_bias.cpu().detach().numpy())
            _reward_list.append(eval_reward)
            _cost_list.append(eval_cost)

        reward_bias_list.append(np.mean(_reward_bias_list))
        cost_bias_list.append(np.mean(_cost_bias_list))
        reward_list.append(np.mean(_reward_list))
        cost_list.append(np.mean(_cost_list))

    return reward_bias_list, reward_list, cost_bias_list, cost_list

def load_filtered_state_dict(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    model_dict.update(filtered_dict)

    model.load_state_dict(model_dict, strict=False)


def process_and_save_metrics(all_reward, all_cost, all_q, all_qc, all_kl_div,
                             all_reward_bias_mean_list, all_reward_bias_std_list,
                             all_cost_bias_mean_list, all_cost_bias_std_list,
                             results_dir):

    all_reward = np.mean(all_reward, axis=0)
    all_cost = np.mean(all_cost, axis=0)
    all_q = np.mean(all_q, axis=0)
    all_qc = np.mean(all_qc, axis=0)
    all_kl_div = np.mean(all_kl_div, axis=0)

    all_reward_bias_mean_list = np.mean(all_reward_bias_mean_list, axis=0)
    all_reward_bias_std_list = np.mean(all_reward_bias_std_list, axis=0)
    all_cost_bias_mean_list = np.mean(all_cost_bias_mean_list, axis=0)
    all_cost_bias_std_list = np.mean(all_cost_bias_std_list, axis=0)

    all_reward_bias_mean_list = np.clip(all_reward_bias_mean_list, a_min=0, a_max=3)
    all_reward_bias_std_list = np.clip(all_reward_bias_std_list, a_min=0, a_max=3)
    all_cost_bias_mean_list = np.clip(all_cost_bias_mean_list, a_min=0, a_max=3)
    all_cost_bias_std_list = np.clip(all_cost_bias_std_list, a_min=0, a_max=3)

    np.save(f"{results_dir}/rollout_reward", all_reward)
    np.save(f"{results_dir}/rollout_cost", all_cost)
    np.save(f"{results_dir}/Q", all_q)
    np.save(f"{results_dir}/Qc", all_qc)
    np.save(f"{results_dir}/kl_div", all_kl_div)
    np.save(f"{results_dir}/reward_bias_mean", all_reward_bias_mean_list)
    np.save(f"{results_dir}/reward_bias_std", all_reward_bias_std_list)
    np.save(f"{results_dir}/cost_bias_mean", all_cost_bias_mean_list)
    np.save(f"{results_dir}/cost_bias_std", all_cost_bias_std_list)

    plot_variable(all_reward, "Rollout Reward", f"{results_dir}/rollout_reward")
    plot_variable(all_cost, "Rollout Cost", f"{results_dir}/rollout_cost")
    plot_variable(all_q, "Q Value", f"{results_dir}/Q")
    plot_variable(all_qc, "Qc Value", f"{results_dir}/Qc")
    plot_variable(all_kl_div, "KL Divergence", f"{results_dir}/kl_div")
    plot_variable(all_reward_bias_mean_list, "Reward Bias Mean", f"{results_dir}/reward_bias_mean")
    plot_variable(all_reward_bias_std_list, "Reward Bias Std", f"{results_dir}/reward_bias_std")
    plot_variable(all_cost_bias_mean_list, "Cost Bias Mean", f"{results_dir}/cost_bias_mean")
    plot_variable(all_cost_bias_std_list, "Cost Bias Std", f"{results_dir}/cost_bias_std")


def visual(policy, args, index, device):
    env = gym.make(args.env)
    env = RecordVideo(env,
                      video_folder=f'video_output/{args.env}_index{index}',
                      name_prefix="eval",
                      episode_trigger=lambda x: True)
    print("Start recording!!!")
    try:
        with torch.no_grad():
            obs, _ = env.reset()
            done = False
            for _ in range(args.max_envsteps):
                while not done:
                    action, _ = policy.actor(torch.tensor(obs[None, ...], dtype=torch.float32).to(device),
                                             with_logprob=True)
                    action = np.squeeze(action.cpu().numpy(), axis=0)
                    obs_next, reward, terminated, truncated, info = env.step(action)
                    done = 1 if terminated or truncated else 0
                    obs = obs_next
    finally:
        env.close()
    print("Finish recording!!!")