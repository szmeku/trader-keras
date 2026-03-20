"""Configuration dataclasses and YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    pattern: str = "icmarkets_*.parquet"
    data_dir: str = "~/projects/data"
    load_limit: int | None = None


@dataclass
class Stage1Config:
    # Architecture
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    # Sequence
    lookback: int = 60
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 30, 60])
    bar_seconds: int = 1
    stride: int = 1
    # Training
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    train_ratio: float = 0.8
    # Loss
    probabilistic: bool = True   # Gaussian NLL with (mu, sigma)
    magnitude_alpha: float = 0.0  # weight exponent for large-move emphasis
    # Features
    features: list[str] | None = None  # None → default FEATURE_COLS
    seed: int | None = None


@dataclass
class LoggingConfig:
    provider: str | list[str] = "console"  # "wandb", "console", or both
    tags: list[str] = field(default_factory=list)
    project: str = "trader-keras"


@dataclass
class Config:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _coerce(value: Any, target: Any) -> Any:
    """Coerce value to match the type of target (best-effort, for scalars)."""
    if value is None:
        return value
    if isinstance(target, list) and not isinstance(value, list):
        return [value]
    if isinstance(target, float) and isinstance(value, str):
        return float(value)
    if isinstance(target, int) and isinstance(value, str):
        return int(value)
    return value


def _apply_dict(dc: Any, d: dict) -> None:
    """Apply dict values onto a dataclass instance, coercing types."""
    for k, v in d.items():
        if hasattr(dc, k):
            attr = getattr(dc, k)
            if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                _apply_dict(attr, v)
            else:
                setattr(dc, k, _coerce(v, attr))


def load_config(path: str | Path) -> Config:
    raw: dict = {}
    if path and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()
    _apply_dict(cfg, raw)
    return cfg
