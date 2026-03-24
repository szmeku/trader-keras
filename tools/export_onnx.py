"""Re-export ONNX from a training run dir.

Normally the save step exports automatically. Use this to re-export
from an existing run dir (e.g., after fixing the export code).

Usage:
    uv run python tools/export_onnx.py [run_dir]

    run_dir: Hydra output dir containing policy.keras and .hydra/config.yaml
             Default: latest timestamped dir under outputs/

Outputs (in run_dir):
    policy.onnx — ONNX model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import keras

from trader_keras.models.policy import HiddenSlice  # noqa: F401 — register before load
from icmarkets_env.onnx.export import export_policy_onnx


def _find_latest_run() -> Path:
    """Find the most recent Hydra output dir containing policy.keras."""
    outputs = Path("outputs")
    candidates = sorted(outputs.glob("*/*/policy.keras"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No policy.keras found under outputs/")
    return candidates[-1].parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Keras policy → ONNX")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Hydra output dir (default: latest run)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run()
    model_path = run_dir / "policy.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No policy.keras in {run_dir}")

    model = keras.saving.load_model(str(model_path), safe_mode=False)
    obs_dim = model.input_shape[0][-1]
    hidden_shape = model.input_shape[1][1:]

    onnx_path = export_policy_onnx(model, run_dir / "policy.onnx", obs_dim, hidden_shape)
    print(f"Run dir: {run_dir}")
    print(f"ONNX:    {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")
    print(f"  obs_dim={obs_dim}, hidden={hidden_shape}")


if __name__ == "__main__":
    main()
