from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field

@dataclass
class CPQTrainConfig:
    # wandb params
    project: str = "warm_start_safe_rl_experiments"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CPQ"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = 0.0
    noise_scale: float = None # 0.01
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineAntCircle-v0"
    dataset: str = None
    seed: int = 0  # 5
    device: str = "cuda:0"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    alpha_lr: float = 0.0001
    vae_lr: float = 0.001
    cost_limit: int = 40  # 10, 20, 40, 80
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 500_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    num_q: int = 2
    num_qc: int = 1
    qc_scalar: float = 1.5
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class CPQCarCircleConfig(CPQTrainConfig):
    pass


@dataclass
class CPQAntRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class CPQDroneCircleConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class CPQCarRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class CPQAntCircleConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class CPQBallRunConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class CPQBallCircleConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200

@dataclass
class CPQHalfCheetahVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQHopperVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CPQSwimmerVelocityConfig(CPQTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


CPQ_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": CPQCarCircleConfig,
    "OfflineAntRun-v0": CPQAntRunConfig,
    "OfflineDroneRun-v0": CPQDroneRunConfig,
    "OfflineDroneCircle-v0": CPQDroneCircleConfig,
    "OfflineCarRun-v0": CPQCarRunConfig,
    "OfflineAntCircle-v0": CPQAntCircleConfig,
    "OfflineBallCircle-v0": CPQBallCircleConfig,
    "OfflineBallRun-v0": CPQBallRunConfig,

    "OfflineHalfCheetahVelocityGymnasium-v1": CPQHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": CPQHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": CPQSwimmerVelocityConfig,
}