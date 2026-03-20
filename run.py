#!/usr/bin/env python
"""Train Stage 1 GRU predictor.

Usage:
    python run.py                          # default config.yml
    python run.py --config-name=bench      # use bench.yml
    python run.py stage1.lr=0.001          # override any param
    python run.py data.load_limit=50000 stage1.epochs=5
"""
import logging
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

# Work around cuBLAS autotuning failures on older GPUs (e.g. GTX 1050 Ti)
xla_flags = os.environ.get("XLA_FLAGS", "")
for flag in ("--xla_gpu_autotune_level=0", "--xla_gpu_enable_cublaslt=false"):
    if flag not in xla_flags:
        xla_flags = f"{xla_flags} {flag}".strip()
os.environ["XLA_FLAGS"] = xla_flags

import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from trader_keras.trainer import train

    model_path = train(cfg)
    print(f"Done. Model: {model_path}")


if __name__ == "__main__":
    main()
