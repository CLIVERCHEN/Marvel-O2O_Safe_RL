import argparse

def warmstart_config(args):
    if args.env == "OfflineBallCircle-v0":
        args.actor_lr=5e-5
        args.critic_lr=5e-3
        args.cost_critic_lr=5e-3
        args.lambda_lr=1e-4
        args.alpha=1e-3
        args.max_timesteps = 500
        args.max_envsteps = 200
    if args.env == "OfflineBallRun-v0":
        args.actor_lr=1e-5
        args.critic_lr=5e-3
        args.cost_critic_lr=5e-3
        args.lambda_lr=1e-4
        args.alpha=1e-3
        args.max_timesteps = 500
        args.max_envsteps = 100
    elif args.env == "OfflineCarCircle-v0":
        args.actor_lr = 5e-4
        args.critic_lr = 5e-3
        args.cost_critic_lr = 8e-3
        args.lambda_lr = 2e-3
        args.alpha = 1e-3
        args.max_timesteps = 500
        args.max_envsteps = 200
    elif args.env == "OfflineCarRun-v0":
        args.actor_lr = 5e-5
        args.critic_lr = 5e-3
        args.cost_critic_lr = 8e-3
        args.lambda_lr = 2e-3
        args.alpha = 5e-3
        args.max_timesteps = 500
        args.max_envsteps = 200
    elif args.env == "OfflineDroneCircle-v0":
        args.actor_lr = 5e-6
        args.critic_lr = 8e-4
        args.cost_critic_lr = 2e-3
        args.lambda_lr = 1e-4
        args.alpha = 5e-3
        args.max_timesteps = 1000
        args.max_envsteps = 300
    elif args.env == "OfflineDroneRun-v0":
        args.actor_lr = 2e-6
        args.critic_lr = 8e-3
        args.cost_critic_lr = 5e-3
        args.lambda_lr = 5e-3
        args.alpha = 1e-3

        args.max_timesteps = 5000
        args.max_envsteps = 200
    elif args.env == "OfflineAntCircle-v0":
        args.actor_lr = 5e-5
        args.critic_lr = 5e-4
        args.cost_critic_lr = 8e-4
        args.lambda_lr = 1e-5
        args.alpha = 5e-5
        args.max_timesteps = 5000
        args.max_envsteps = 500
    elif args.env == "OfflineAntRun-v0":
        args.actor_lr = 5e-5
        args.critic_lr = 2e-3
        args.cost_critic_lr = 5e-3
        args.lambda_lr = 1e-5
        args.alpha = 1e-4
        args.max_timesteps = 5000
        args.max_envsteps = 200
    elif args.env == "OfflineSwimmerVelocityGymnasium-v1":
        args.actor_lr = 1e-5
        args.critic_lr = 5e-4
        args.cost_critic_lr = 8e-4
        args.lambda_lr = 1e-4
        args.alpha = 1e-4
        args.max_timesteps = 2000
        args.max_envsteps = 1000
    elif args.env == "OfflineHalfCheetahVelocityGymnasium-v1":
        args.actor_lr = 1e-3
        args.critic_lr = 1e-2
        args.cost_critic_lr = 1e-2
        args.lambda_lr = 1e-4
        args.alpha = 5e-3
        args.max_timesteps = 5000
        args.max_envsteps = 1000
    elif args.env == "OfflineHopperVelocityGymnasium-v1":
        args.actor_lr = 1e-3
        args.critic_lr = 1e-2
        args.cost_critic_lr = 1e-2
        args.lambda_lr = 1e-4
        args.alpha = 5e-3
        args.max_timesteps = 5000
        args.max_envsteps = 1000

    return args