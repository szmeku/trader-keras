"""Configuration — Hydra structured configs with type validation."""
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class PipelineConfig:
    name: str = "predict"
    steps: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)


@dataclass
class BackboneConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seed: int = 42


@dataclass
class TrainConfig:
    horizons: list[int] = field(default_factory=lambda: [10])
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    train_ratio: float = 0.8
    loss: str = "gaussian_nll"


@dataclass
class DataConfig:
    pattern: str = "icmarkets_*.parquet"
    data_dir: str = "~/projects/data"
    load_limit: int = 0
    lookback: int = 60
    stride: int = 1


@dataclass
class WandbConfig:
    tags: list[str] = field(default_factory=list)
    project: str = "trader-keras"


@dataclass
class RLConfig:
    n_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_grad_norm: float = 1.0


@dataclass
class EnvConfig:
    symbol: str = "EURUSD"
    date_start: str = "2024-01-01"
    date_end: str = "2024-12-31"
    lookback: int = 30
    balance: float = 10_000.0
    leverage: float = 20.0


@dataclass
class Config:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    train: Optional[TrainConfig] = None
    rl: Optional[RLConfig] = None
    env: Optional[EnvConfig] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
