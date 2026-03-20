"""Logger abstraction — same interface for console and W&B."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

    def log_config(self, config: dict[str, Any]) -> None:
        pass  # optional; W&B logs config at init


class ConsoleLogger(BaseLogger):
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        prefix = f"[step {step}] " if step is not None else ""
        print(prefix + "  ".join(parts))

    def finish(self) -> None:
        pass


class MultiLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self._loggers = loggers

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for lg in self._loggers:
            lg.log(metrics, step)

    def log_config(self, config: dict[str, Any]) -> None:
        for lg in self._loggers:
            lg.log_config(config)

    def finish(self) -> None:
        for lg in self._loggers:
            lg.finish()
