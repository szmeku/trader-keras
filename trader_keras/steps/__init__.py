# Import step modules to trigger @step registration.
import logging as _logging

from . import data, model, train  # noqa: F401

try:
    from . import env, rl  # noqa: F401
except ImportError:
    _logging.getLogger(__name__).warning(
        "icmarkets_env not installed — RL pipeline steps (env, fit_rl) unavailable. "
        "Install with: uv pip install icmarkets-env"
    )
