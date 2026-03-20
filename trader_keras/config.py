"""Configuration — pydantic models + YAML loading."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, BeforeValidator


def _wrap_scalar(v: object) -> object:
    return [v] if isinstance(v, (int, float)) else v


class Stage1Config(BaseModel):
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lookback: int = 60
    horizons: Annotated[list[int], BeforeValidator(_wrap_scalar)] = [1, 5, 10, 30, 60]
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
    seed: int | None = None


class DataConfig(BaseModel):
    pattern: str = "icmarkets_*.parquet"
    data_dir: str = "~/projects/data"
    load_limit: int | None = None


class LoggingConfig(BaseModel):
    provider: str | list[str] = "console"
    tags: list[str] = []
    project: str = "trader-keras"


class Config(BaseModel):
    stage1: Stage1Config = Stage1Config()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(path: str | Path) -> Config:
    raw: dict = {}
    if path and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    return Config(**raw)
