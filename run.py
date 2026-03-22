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

from trader_keras.pipeline import pipe
from trader_keras.steps.data import load, featurize, window
from trader_keras.steps.model import model, checkpoint
from trader_keras.steps.train import fit_supervised, save
from trader_keras.steps.env import env
from trader_keras.steps.rl import fit_rl

PIPELINES = {
    "predict": [load, featurize, window, model, checkpoint, fit_supervised, save],
    "rl":      [load, featurize, env, fit_rl, save],
}


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pipeline_name = cfg.pipeline.name
    keras.utils.set_random_seed(cfg.backbone.seed)
    pipeline = PIPELINES[pipeline_name]
    skip = set(cfg.pipeline.get("skip", []))
    steps = [s for s in pipeline if s.__name__ not in skip]
    logger.info("Pipeline: %s  skip: %s", pipeline_name, skip or "(none)")
    ctx = pipe({"cfg": cfg}, *steps)
    print(f"Done. Model: {ctx.get('model_path')}")


if __name__ == "__main__":
    main()
