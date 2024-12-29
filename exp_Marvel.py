import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
import argparse
from online.sac_lag import SAC_Lag
import bullet_safety_gym
from tqdm.auto import trange
import dsrl
import gymnasium as gym
import numpy as np
import torch
import copy
from utils import *
import datetime
import pickle
from config.warmstart_config import warmstart_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="o2o_safe_rl")
    parser.add_argument("--policy", default="SAC")
    parser.add_argument("--env", default="OfflineHalfCheetahVelocityGymnasium-v1")
    parser.add_argument("--seeds", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--max_timesteps", default=500, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--rollout_num", default=5)
    parser.add_argument("--smooth_window", default=10, type=int)
    parser.add_argument("--vpa_steps", default=5000, type=int)
    parser.add_argument("--max_envsteps", default=200, type=int)

    # setting
    parser.add_argument("--cost_limit", default=20)
    parser.add_argument("--actor_lr", default=1e-5, type=float)
    parser.add_argument("--critic_lr", default=5e-4, type=float)
    parser.add_argument("--cost_critic_lr", default=8e-4, type=float)
    parser.add_argument("--lambda_lr", default=1e-5, type=float)
    parser.add_argument("--alpha", default=1e-4, type=float)
    parser.add_argument("--kl_coeff", default=0, type=float)
    parser.add_argument("--if_VPA", default=True, type=bool)
    parser.add_argument("--if_aPID", default=True, type=bool)

    parser.add_argument("--offline_algo", default="bearl")

    parser.add_argument("--if_use_actor_optim", default=True)
    parser.add_argument("--if_use_critic_optim", default=False)
    parser.add_argument("--if_use_cost_critic_optim", default=False)

    args = parser.parse_args()

    args = warmstart_config(args)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"{args.policy}_{args.env}_{start_time}"

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}")
    print("---------------------------------------")

    if not os.path.exists("./results_finetune"):
        os.makedirs("./results_finetune")

    if args.save_model and not os.path.exists("./models_finetune"):
        os.makedirs("./models_finetune")

    results_dir = f"./results_finetune/{file_name}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    hyperparameters = {
              "cost_limit": args.cost_limit,
              "actor_lr": args.actor_lr,
              "critic_lr": args.critic_lr,
              "cost_critic_lr": args.cost_critic_lr,
              "lambda_lr": args.lambda_lr,
              "alpha": args.alpha,
              "if_use_actor_optim": args.if_use_actor_optim,
              "if_use_critic_optim": args.if_use_critic_optim,
              "if_use_cost_critic_optim": args.if_use_cost_critic_optim,
              "kl_coeff": args.kl_coeff,
              "vpa_steps": args.vpa_steps
    }

    with open(f'{results_dir}/hyperparameters.txt', 'w') as file:
        for param, value in hyperparameters.items():
            file.write(f'{param}: {value}\n')

    with open(f'../offline_dataset/{args.env}/dataset.pkl', 'rb') as file:
        loaded_dataset = pickle.load(file)

    env = gym.make(args.env)
    env = wrap_env(
        env=env,
        reward_scale=0.1,
    )
    env = OfflineEnvWrapper(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    offline_replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, device=DEVICE)

    for i in range(len(loaded_dataset['actions'])):
        obs = torch.tensor(loaded_dataset['observations'][i], dtype=torch.float32)
        next_obs = torch.tensor(loaded_dataset['next_observations'][i], dtype=torch.float32)
        action = torch.tensor(loaded_dataset['actions'][i], dtype=torch.float32)
        reward = torch.tensor(loaded_dataset['rewards'][i]*0.1, dtype=torch.float64)
        cost = torch.tensor(loaded_dataset['costs'][i], dtype=torch.float64)
        done = torch.tensor(loaded_dataset['done'][i], dtype=torch.float32)
        offline_replay_buffer.add(obs, action, next_obs, reward, done, cost)

    all_reward, all_cost, all_q, all_qc = [], [], [], []
    all_kl_div = []
    all_reward_bias_mean_list, all_reward_bias_std_list, all_cost_bias_mean_list, all_cost_bias_std_list = [], [], [], []

    for seed in args.seeds:
        print(f'Training with seed {seed}')

        env = gym.make(args.env)
        env = wrap_env(
            env=env,
            reward_scale=0.1,
        )
        env = OfflineEnvWrapper(env)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        set_seed(seed=seed, env=env, deterministic_torch=True)

        kwargs = {
                "project": args.project,
                "name": f'{args.policy}-{args.env}-{args.offline_algo}-exp_WS-{start_time}',
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": args.discount,
                "tau": args.tau,
                "device": DEVICE,
                "cost_limit": args.cost_limit,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "cost_critic_lr": args.cost_critic_lr,
                "lambda_lr": args.lambda_lr,
                "alpha": args.alpha,
                'kl_coeff': args.kl_coeff,
                "seed":seed,
                "use_reward_critic_norm":False,
                "use_cost_critic_norm":False,
                "lagrangian_multiplier_init": 0.0,
                "if_pid":args.if_aPID
        }

        policy = SAC_Lag(**kwargs)

        policy.actor.load_state_dict(torch.load(f'../offline_model/{args.env}_{args.offline_algo}_actor.pth'))
        policy.initial_actor.load_state_dict(
            torch.load(f'../offline_model/{args.env}_{args.offline_algo}_actor.pth'))

        load_filtered_state_dict(policy.critic,f'../offline_model/{args.env}_{args.offline_algo}_critic.pth')
        load_filtered_state_dict(policy.cost_critic, f'../offline_model/{args.env}_{args.offline_algo}_cost_critic.pth')

        policy.critic_target = copy.deepcopy(policy.critic)
        policy.cost_critic_target = copy.deepcopy(policy.cost_critic)

        # optimizer
        if args.if_use_actor_optim:
            policy.actor_optimizer.load_state_dict(
                torch.load(f'../offline_model/{args.env}_{args.offline_algo}_actor_optim.pth'))
        if args.if_use_critic_optim:
            policy.critic_optimizer.load_state_dict(
                torch.load(f'../offline_model/{args.env}_{args.offline_algo}_critic_optim.pth'))
        if args.if_use_cost_critic_optim:
            policy.cost_critic_optimizer.load_state_dict(
                torch.load(f'../offline_model/{args.env}_{args.offline_algo}_cost_critic_optim.pth'))

        replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, device=DEVICE)

        # Evaluate untrained policy
        evaluations_reward, evaluations_cost = [], []

        state, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_timesteps = 0

        train_num = args.max_timesteps // args.rollout_num
        
        if args.if_VPA:
            for i in trange(int(args.vpa_steps), desc="vpa"):
                policy.vpa(replay_buffer=offline_replay_buffer, batch_size=256)

        rollout_reward, rollout_cost = [], []
        Q, Qc = [], []
        kl_div = []
        reward_bias_mean_list, reward_bias_std_list, cost_bias_mean_list, cost_bias_std_list = [], [], [], []

        for t in trange(int(train_num), desc="Training"):

            episode_timesteps += 1

            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            avg_reward, avg_cost = 0, 0

            # Evaluate episode
            if t % args.eval_freq == 0:
                avg_reward, avg_cost = eval_policy(policy=policy, env_name=args.env, device=DEVICE, seed=seed)
                evaluations_reward.append(avg_reward)
                evaluations_cost.append(avg_cost)

                np.save(f"{results_dir}/eval_reward", evaluations_reward)
                np.save(f"{results_dir}/eval_cost", evaluations_cost)

                policy.logger.log({
                    "eval_reward": avg_reward * 10,
                    "eval_cost": avg_cost
                }, step=policy.total_it)

            # Rollout
            for _ in range(args.rollout_num):
                with torch.no_grad():
                    obs, _ = env.reset()
                    done = False
                    for _ in range(args.max_envsteps):
                        while not done:
                            action, _ = policy.actor(torch.tensor(obs[None, ...], dtype=torch.float32).to(DEVICE),with_logprob=True)
                            action = np.squeeze(action.cpu().numpy(), axis=0)
                            obs_next, reward, terminated, truncated, info = env.step(action)
                            episode_reward += reward
                            cost = info["cost"]
                            episode_cost += cost
                            done = 1 if terminated or truncated else 0
                            replay_buffer.add(obs, action, obs_next, reward, done, cost)
                            obs = obs_next

            episode_reward /= args.rollout_num
            episode_cost /= args.rollout_num

            for _ in range(args.rollout_num):
                q, qc = policy.train(replay_buffer=replay_buffer, batch_size=args.batch_size, episode_cost=episode_cost,
                                     online_ratio=1, offline_dataset=None, if_kl=False)
                rollout_reward.append(episode_reward * 10)
                rollout_cost.append(episode_cost)
                Q.append(q)
                Qc.append(qc)
                kl_div.append(0)

                policy.logger.log({
                    "rollout_reward": episode_reward * 10,
                    "rollout_cost": episode_cost
                }, step=policy.total_it)

            print(
                f"Episode: {t + 1} Reward: {episode_reward * 10:.3f} Cost: {episode_cost}")
            # Reset environment
            state, _ = env.reset()


        rollout_reward = moving_average(rollout_reward, args.smooth_window)
        rollout_cost = moving_average(rollout_cost, args.smooth_window)
        Q = moving_average(Q, args.smooth_window)
        Qc = moving_average(Qc, args.smooth_window)
        kl_div = moving_average(kl_div, args.smooth_window)

        all_reward.append(rollout_reward)
        all_cost.append(rollout_cost)
        all_q.append(Q)
        all_qc.append(Qc)
        all_kl_div.append(kl_div)

        all_reward_bias_mean_list.append(reward_bias_mean_list)
        all_reward_bias_std_list.append(reward_bias_std_list)
        all_cost_bias_mean_list.append(cost_bias_mean_list)
        all_cost_bias_std_list.append(cost_bias_std_list)

        policy.logger.finish()

    process_and_save_metrics(all_reward, all_cost, all_q, all_qc, all_kl_div,
                             all_reward_bias_mean_list, all_reward_bias_std_list,
                             all_cost_bias_mean_list, all_cost_bias_std_list,
                             results_dir)

