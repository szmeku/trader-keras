#!/usr/bin/env python
"""Unified CLI runner.

Usage:
    uv run python run.py train [CONFIG]          # default: config.yml
    uv run python run.py train my_config.yml
"""
import logging
import os
import sys
from pathlib import Path

# Set JAX backend BEFORE any keras import
os.environ.setdefault("KERAS_BACKEND", "jax")

import typer

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.callback()
def main() -> None:
    """trader-keras CLI"""


@app.command("train")
def train(config: str = typer.Argument("config.yml", help="Path to YAML config")) -> None:
    """Train Stage 1 GRU predictor."""
    from trader_keras.config import load_config
    from trader_keras.trainer import train as do_train

    cfg = load_config(config)
    model_path = do_train(cfg)
    typer.echo(f"Done. Model: {model_path}")


if __name__ == "__main__":
    app(prog_name="run.py")
