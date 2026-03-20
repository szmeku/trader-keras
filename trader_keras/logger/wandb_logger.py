"""W&B logger — thin adapter over wandb Python SDK."""
from __future__ import annotations

import os
from typing import Any

from .base import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb  # lazy import so console-only runs don't need wandb

        api_key = os.environ.get("WANDB_API_TOKEN") or os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)

        self._run = wandb.init(
            project=project,
            tags=tags or [],
            config=config or {},
            reinit=True,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import wandb
        wandb.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(config, allow_val_change=True)

    def finish(self) -> None:
        import wandb
        wandb.finish()
