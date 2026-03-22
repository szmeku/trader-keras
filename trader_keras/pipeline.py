"""Composable pipeline: Ctx → Ctx step functions, pipe runner."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypedDict

import numpy as np
import pandas as pd
from omegaconf import DictConfig

if TYPE_CHECKING:
    import keras
    from icmarkets_env import TradingEnv


class Ctx(TypedDict, total=False):
    """Pipeline context — steps read/write these keys."""
    cfg: DictConfig
    bars: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    feature_cols: list[str]
    model: keras.Model
    model_path: Path
    env: TradingEnv


StepFn = Callable[[Ctx], Ctx]
STEPS: dict[str, StepFn] = {}


def step(fn: StepFn) -> StepFn:
    """Register a step function by name."""
    STEPS[fn.__name__] = fn
    return fn


def pipe(ctx: Ctx, *steps: StepFn) -> Ctx:
    """Run steps sequentially, threading ctx through."""
    for s in steps:
        ctx = s(ctx)
    return ctx
