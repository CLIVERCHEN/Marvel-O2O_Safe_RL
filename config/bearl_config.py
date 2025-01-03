from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BEARLTrainConfig:
    # wandb params
    project: str = "warm_start_safe_rl_experiments"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BEARL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
    threads: int = 4
    reward_scale: float = 1  # 0.1
    cost_scale: float = 1
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    vae_lr: float = 0.001
    cost_limit: int = 20  # 10, 20, 40
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 50_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    lmbda: float = 0.75
    mmd_sigma: float = 50
    target_mmd_thresh: float = 0.05
    num_samples_mmd_match: int = 10
    start_update_policy_step: int = 0
    kernel: str = "gaussian"  # or "laplacian"
    num_q: int = 2
    num_qc: int = 1
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BEARLCarCircleConfig(BEARLTrainConfig):
    pass


@dataclass
class BEARLAntRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BEARLCarRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BEARLAntCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class BEARLBallRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class BEARLBallCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class BEARLHalfCheetahVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLHopperVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BEARLSwimmerVelocityConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


BEARL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BEARLCarCircleConfig,
    "OfflineAntRun-v0": BEARLAntRunConfig,
    "OfflineDroneRun-v0": BEARLDroneRunConfig,
    "OfflineDroneCircle-v0": BEARLDroneCircleConfig,
    "OfflineCarRun-v0": BEARLCarRunConfig,
    "OfflineAntCircle-v0": BEARLAntCircleConfig,
    "OfflineBallCircle-v0": BEARLBallCircleConfig,
    "OfflineBallRun-v0": BEARLBallRunConfig,

    "OfflineHalfCheetahVelocityGymnasium-v1": BEARLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BEARLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BEARLSwimmerVelocityConfig,
}