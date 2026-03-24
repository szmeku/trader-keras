"""Unit tests for training pipeline steps."""
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import pytest
from omegaconf import OmegaConf

from trader_keras.pipeline import Ctx, pipe


def _test_cfg() -> OmegaConf:
    return OmegaConf.create({
        "backbone": {
            "hidden_size": 16, "num_layers": 1, "dropout": 0.0,
        },
        "data": {"lookback": 10},
        "train": {
            "horizons": [1, 5], "epochs": 5, "patience": 3, "batch_size": 8,
            "lr": 1e-3, "weight_decay": 0.0, "clip_grad_norm": 1.0,
            "loss": "gaussian_nll",
        },
        "wandb": {"project": "test", "tags": []},
    })


def _build_compiled_model(cfg):
    import keras

    from trader_keras.models.gru import LOSSES, build_gru_model

    bb = cfg.backbone
    tr = cfg.train
    model = build_gru_model(
        n_features=4, n_horizons=len(tr.horizons), cfg=bb,
        loss_name=tr.loss, lookback=cfg.data.lookback,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=tr.lr),
        loss=LOSSES[tr.loss],
    )
    return model


def _make_ctx(cfg=None) -> Ctx:
    cfg = cfg or _test_cfg()
    rng = np.random.default_rng(42)
    n_train, n_val = 64, 16
    lookback, n_features = cfg.data.lookback, 4

    model = _build_compiled_model(cfg)

    return Ctx(
        cfg=cfg,
        model=model,
        x_train=rng.normal(size=(n_train, lookback, n_features)).astype("float32"),
        y_train=rng.normal(size=(n_train, len(cfg.train.horizons))).astype("float32"),
        x_val=rng.normal(size=(n_val, lookback, n_features)).astype("float32"),
        y_val=rng.normal(size=(n_val, len(cfg.train.horizons))).astype("float32"),
    )


class TestFitSupervised:
    def test_returns_ctx_with_model_path(self):
        from trader_keras.steps.train import fit_supervised

        ctx = _make_ctx()
        ctx = fit_supervised(ctx)
        assert "model_path" in ctx
        assert ctx["model_path"].exists()

    def test_loss_is_finite(self):
        from trader_keras.steps.train import fit_supervised

        ctx = _make_ctx()
        ctx = fit_supervised(ctx)
        pred = ctx["model"](ctx["x_val"][:2], training=False)
        assert np.all(np.isfinite(np.array(pred)))

    def test_works_without_validation(self):
        from trader_keras.steps.train import fit_supervised

        ctx = _make_ctx()
        ctx["x_val"] = np.empty((0, 10, 4), dtype="float32")
        ctx["y_val"] = np.empty((0, 2), dtype="float32")
        ctx = fit_supervised(ctx)
        assert "model_path" in ctx

    def test_model_is_trained(self):
        """Loss should be finite after training (sanity check)."""
        import keras

        from trader_keras.steps.train import fit_supervised

        ctx = _make_ctx()
        loss_before = float(keras.ops.convert_to_numpy(
            ctx["model"].evaluate(ctx["x_train"], ctx["y_train"], verbose=0)
        ))
        ctx = fit_supervised(ctx)
        loss_after = float(keras.ops.convert_to_numpy(
            ctx["model"].evaluate(ctx["x_train"], ctx["y_train"], verbose=0)
        ))
        assert np.isfinite(loss_after)
        assert loss_after <= loss_before


class TestSave:
    def test_save_writes_model_to_run_dir(self, tmp_path, monkeypatch):
        import keras
        from unittest.mock import MagicMock
        from trader_keras.steps.train import save

        # Mock Hydra runtime to use tmp_path as output dir
        mock_hydra = MagicMock()
        mock_hydra.runtime.output_dir = str(tmp_path)
        monkeypatch.setattr("hydra.core.hydra_config.HydraConfig.get", lambda: mock_hydra)

        model = keras.Sequential([keras.layers.Dense(1, input_shape=(2,))])
        cfg = _test_cfg()
        cfg = OmegaConf.merge(cfg, {"save": {"export_onnx": False}})
        ctx = Ctx(cfg=cfg, model=model)
        ctx = save(ctx)

        assert ctx["model_path"] == tmp_path / "policy.keras"
        assert ctx["model_path"].exists()


class TestPipeIntegration:
    def test_fit_then_save(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from trader_keras.steps.train import fit_supervised, save

        mock_hydra = MagicMock()
        mock_hydra.runtime.output_dir = str(tmp_path)
        monkeypatch.setattr("hydra.core.hydra_config.HydraConfig.get", lambda: mock_hydra)

        cfg = _test_cfg()
        cfg = OmegaConf.merge(cfg, {"save": {"export_onnx": False}})
        ctx = _make_ctx(cfg)
        ctx = pipe(ctx, fit_supervised, save)
        assert ctx["model_path"].exists()
