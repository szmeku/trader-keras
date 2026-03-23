"""Tests for Hydra config loading with pipeline groups."""
from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import trader_keras.config  # noqa: F401 — registers structured config


@pytest.fixture()
def config_dir() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "conf")


def _compose(config_dir: str, overrides: list[str] | None = None) -> OmegaConf:
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])


class TestPredictPipeline:
    def test_default_pipeline_is_predict(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert cfg.pipeline.name == "predict"

    def test_pipeline_has_steps(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert "load" in cfg.pipeline.steps
        assert "fit_supervised" in cfg.pipeline.steps

    def test_backbone_defaults(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert cfg.backbone.hidden_size == 64
        assert cfg.backbone.seed == 42

    def test_data_defaults(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert cfg.data.lookback == 60
        assert cfg.data.stride == 1
        assert cfg.data.load_limit == 0
        assert cfg.data.data_dir == "~/projects/data"

    def test_train_defaults(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert cfg.train.lr == pytest.approx(3e-4)
        assert cfg.train.loss == "gaussian_nll"
        assert cfg.train.epochs == 100

    def test_wandb_defaults(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert cfg.wandb.project == "trader-keras"


class TestRLPipeline:
    def test_rl_pipeline_name(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["pipeline=rl"])
        assert cfg.pipeline.name == "rl"

    def test_rl_pipeline_steps(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["pipeline=rl"])
        assert list(cfg.pipeline.steps) == ["env", "fit_rl", "save"]

    def test_rl_has_env_config(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["pipeline=rl"])
        assert cfg.env.symbol == "EURUSD"
        assert cfg.env.balance == pytest.approx(10_000.0)

    def test_rl_has_rl_config(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["pipeline=rl"])
        assert cfg.rl.n_epochs == 4
        assert cfg.rl.gamma == pytest.approx(0.99)


class TestOverrides:
    def test_override_train_lr(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["train.lr=0.001"])
        assert cfg.train.lr == pytest.approx(0.001)

    def test_override_data_load_limit(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["data.load_limit=5000"])
        assert cfg.data.load_limit == 5000

    def test_override_pipeline_selection(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["pipeline=rl"])
        assert cfg.pipeline.name == "rl"


class TestSkip:
    def test_skip_empty_by_default(self, config_dir: str) -> None:
        cfg = _compose(config_dir)
        assert list(cfg.pipeline.skip) == []

    def test_skip_override(self, config_dir: str) -> None:
        cfg = _compose(config_dir, ["pipeline.skip=[load,featurize]"])
        assert "load" in cfg.pipeline.skip
        assert "featurize" in cfg.pipeline.skip
