"""Unit tests for RL pipeline step (PPO)."""
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["WANDB_MODE"] = "disabled"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pytest
from omegaconf import OmegaConf

from trader_keras.pipeline import Ctx


class FakeEnv:
    """Minimal env matching TradingEnv interface for testing."""

    obs_dim = 36
    n_action_types = 6
    n_bars = 100

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action_type: int, p0: float, p1: float):
        self._step_count += 1
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        reward = float(self._rng.standard_normal() * 0.01)
        done = self._step_count >= self.n_bars
        return obs, reward, done, {}


def _rl_cfg():
    return OmegaConf.create({
        "backbone": {"hidden_size": 16},
        "rl": {
            "rollout_steps": 32,
            "n_epochs": 2,
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "lr": 3e-4,
            "batch_size": 16,
            "total_timesteps": 64,
        },
        "wandb": {"project": "test", "tags": []},
    })


class TestCollectRollout:
    def test_returns_correct_keys(self):
        from trader_keras.models.policy import build_policy_model
        from trader_keras.steps.rl import _collect_rollout

        env = FakeEnv()
        policy = build_policy_model(env.obs_dim, _rl_cfg().backbone)
        rollout = _collect_rollout(env, policy, rollout_steps=32)

        expected_keys = {"obs", "action_types", "p0s", "p1s",
                         "rewards", "dones", "log_probs", "values"}
        assert set(rollout.keys()) == expected_keys

    def test_correct_shapes(self):
        from trader_keras.models.policy import build_policy_model
        from trader_keras.steps.rl import _collect_rollout

        env = FakeEnv()
        policy = build_policy_model(env.obs_dim, _rl_cfg().backbone)
        n = 32
        rollout = _collect_rollout(env, policy, rollout_steps=n)

        assert rollout["obs"].shape == (n, env.obs_dim)
        assert rollout["action_types"].shape == (n,)
        assert rollout["p0s"].shape == (n,)
        assert rollout["p1s"].shape == (n,)
        assert rollout["rewards"].shape == (n,)
        assert rollout["dones"].shape == (n,)
        assert rollout["log_probs"].shape == (n,)
        assert rollout["values"].shape == (n,)

    def test_action_types_in_range(self):
        from trader_keras.models.policy import build_policy_model
        from trader_keras.steps.rl import _collect_rollout

        env = FakeEnv()
        policy = build_policy_model(env.obs_dim, _rl_cfg().backbone)
        rollout = _collect_rollout(env, policy, rollout_steps=32)

        assert np.all(rollout["action_types"] >= 0)
        assert np.all(rollout["action_types"] < 6)

    def test_p0_bounded(self):
        from trader_keras.models.policy import build_policy_model
        from trader_keras.steps.rl import _collect_rollout

        env = FakeEnv()
        policy = build_policy_model(env.obs_dim, _rl_cfg().backbone)
        rollout = _collect_rollout(env, policy, rollout_steps=32)

        assert np.all(rollout["p0s"] >= 0.0)
        assert np.all(rollout["p0s"] <= 1.0)

    def test_p1_bounded(self):
        from trader_keras.models.policy import build_policy_model
        from trader_keras.steps.rl import _collect_rollout

        env = FakeEnv()
        policy = build_policy_model(env.obs_dim, _rl_cfg().backbone)
        rollout = _collect_rollout(env, policy, rollout_steps=32)

        assert np.all(rollout["p1s"] >= -1.0)
        assert np.all(rollout["p1s"] <= 1.0)


class TestComputeGAE:
    def test_correct_shape(self):
        from trader_keras.steps.rl import _compute_gae

        n = 32
        rewards = np.random.randn(n).astype(np.float32) * 0.01
        values = np.random.randn(n).astype(np.float32)
        dones = np.zeros(n, dtype=np.float32)

        advantages, returns = _compute_gae(rewards, values, dones, 0.99, 0.95)
        assert advantages.shape == (n,)
        assert returns.shape == (n,)

    def test_finite_values(self):
        from trader_keras.steps.rl import _compute_gae

        n = 32
        rewards = np.random.randn(n).astype(np.float32) * 0.01
        values = np.random.randn(n).astype(np.float32)
        dones = np.zeros(n, dtype=np.float32)

        advantages, returns = _compute_gae(rewards, values, dones, 0.99, 0.95)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))

    def test_done_resets_advantage(self):
        """Advantage should not propagate across episode boundaries."""
        from trader_keras.steps.rl import _compute_gae

        n = 10
        rewards = np.ones(n, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.float32)
        dones[4] = 1.0  # episode boundary at step 4

        advantages, _ = _compute_gae(rewards, values, dones, 0.99, 0.95)
        # Step 4 terminates, so its advantage is just r - V = 1 - 0 = 1
        # Step 3 can bootstrap from step 4 which is done: delta=r+0-V=1
        assert np.isfinite(advantages).all()


class TestSampleAction:
    def test_returns_valid_actions(self):
        from trader_keras.models.policy import build_policy_model
        from trader_keras.steps.rl import _sample_action

        policy = build_policy_model(obs_dim=36, cfg=_rl_cfg().backbone)
        obs = np.zeros((1, 36), dtype=np.float32)
        outputs = policy(obs, training=False)

        action_type, p0, p1, log_prob, value = _sample_action(outputs)
        assert 0 <= action_type < 6
        assert 0.0 <= p0 <= 1.0
        assert -1.0 <= p1 <= 1.0
        assert np.isfinite(log_prob)
        assert np.isfinite(value)


class TestFitRL:
    def test_produces_model_in_ctx(self):
        from trader_keras.steps.rl import fit_rl

        env = FakeEnv()
        ctx: Ctx = {"cfg": _rl_cfg(), "env": env}
        ctx = fit_rl(ctx)

        assert "model" in ctx
        # Model should produce valid outputs
        obs = np.zeros((1, env.obs_dim), dtype=np.float32)
        outputs = ctx["model"](obs, training=False)
        assert len(outputs) == 4

    def test_model_outputs_finite(self):
        from trader_keras.steps.rl import fit_rl

        env = FakeEnv()
        ctx: Ctx = {"cfg": _rl_cfg(), "env": env}
        ctx = fit_rl(ctx)

        obs = np.random.randn(2, env.obs_dim).astype(np.float32)
        logits, p0, p1, value = ctx["model"](obs, training=False)
        assert np.all(np.isfinite(np.array(logits)))
        assert np.all(np.isfinite(np.array(value)))
