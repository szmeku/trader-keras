"""Create logger from config."""
from __future__ import annotations

from typing import Any

from ..config import LoggingConfig
from .base import BaseLogger, ConsoleLogger, MultiLogger


def create_logger(cfg: LoggingConfig, run_config: dict[str, Any] | None = None) -> BaseLogger:
    providers = cfg.provider if isinstance(cfg.provider, list) else [cfg.provider]
    loggers: list[BaseLogger] = []
    for p in providers:
        if p == "console":
            loggers.append(ConsoleLogger())
        elif p == "wandb":
            from .wandb_logger import WandbLogger
            loggers.append(
                WandbLogger(
                    project=cfg.project,
                    tags=cfg.tags,
                    config=run_config,
                )
            )
        else:
            raise ValueError(f"Unknown logging provider: {p!r}")
    return MultiLogger(loggers) if len(loggers) > 1 else loggers[0]
