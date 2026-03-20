"""Logging — console + optional W&B."""
from __future__ import annotations

import os
from typing import Any, Protocol

from ..config import LoggingConfig


class Logger(Protocol):
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def finish(self) -> None: ...


class ConsoleLogger:
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        prefix = f"[step {step}] " if step is not None else ""
        print(prefix + "  ".join(parts))

    def finish(self) -> None:
        pass


class WandbLogger:
    def __init__(self, project: str, tags: list[str], config: dict[str, Any]) -> None:
        import wandb
        api_key = os.environ.get("WANDB_API_TOKEN") or os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        self._run = wandb.init(project=project, tags=tags, config=config, reinit=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import wandb
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        import wandb
        wandb.finish()


class MultiLogger:
    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = loggers

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for lg in self._loggers:
            lg.log(metrics, step)

    def finish(self) -> None:
        for lg in self._loggers:
            lg.finish()


def create_logger(cfg: LoggingConfig, run_config: dict[str, Any] | None = None) -> Logger:
    providers = cfg.provider if isinstance(cfg.provider, list) else [cfg.provider]
    loggers: list[Logger] = []
    for p in providers:
        if p == "console":
            loggers.append(ConsoleLogger())
        elif p == "wandb":
            loggers.append(WandbLogger(project=cfg.project, tags=cfg.tags, config=run_config or {}))
        else:
            raise ValueError(f"Unknown logging provider: {p!r}")
    return MultiLogger(loggers) if len(loggers) > 1 else loggers[0]
