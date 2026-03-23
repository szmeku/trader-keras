#!/usr/bin/env python
"""Pipeline entry point.

Usage:
    python run.py                              # default: predict pipeline
    python run.py pipeline=rl                  # RL pipeline
    python run.py train.lr=0.001               # override params
    python run.py data.load_limit=50000
"""
import logging
import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import keras
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import trader_keras.config  # noqa: F401 — registers structured config
import trader_keras.steps  # noqa: F401 — registers pipeline steps
from trader_keras.pipeline import STEPS, pipe


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    keras.utils.set_random_seed(cfg.backbone.seed)
    step_names = list(cfg.pipeline.steps)
    skip = set(cfg.pipeline.get("skip", []))
    steps = [STEPS[name] for name in step_names if name not in skip]
    logger.info("Pipeline: %s  steps: %s  skip: %s",
                cfg.pipeline.name, [s.__name__ for s in steps], skip or "(none)")
    ctx = pipe({"cfg": cfg}, *steps)
    print(f"Done. Model: {ctx.get('model_path')}")


if __name__ == "__main__":
    main()
