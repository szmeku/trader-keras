"""Configuration — plain dataclasses for OmegaConf/Hydra."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage1Config:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lookback: int = 60
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 30, 60])
    bar_seconds: int = 60
    stride: int = 1
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    train_ratio: float = 0.8
    probabilistic: bool = True
    magnitude_alpha: float = 0.0
    seed: int = -1  # -1 = no seed (OmegaConf needs concrete types)


@dataclass
class DataConfig:
    pattern: str = "icmarkets_*.parquet"
    data_dir: str = "~/projects/data"
    load_limit: int = 0  # 0 = load everything


@dataclass
class LoggingConfig:
    provider: list[str] = field(default_factory=lambda: ["console"])
    tags: list[str] = field(default_factory=list)
    project: str = "trader-keras"


@dataclass
class Config:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
