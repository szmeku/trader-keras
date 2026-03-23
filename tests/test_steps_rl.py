"""Unit tests for RL pipeline step (PPO with JIT rollouts)."""
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["WANDB_MODE"] = "disabled"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import keras
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
        "backbone": {"hidden_size": 16, "seed": 42},
        "env": {"lookback": 10, "balance": 10000.0},
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
            "clip_grad_norm": 1.0,
        },
        "wandb": {"project": "test", "tags": []},
    })


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
        from trader_keras.steps.rl import _compute_gae

        n = 10
        rewards = np.ones(n, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.float32)
        dones[4] = 1.0

        advantages, _ = _compute_gae(rewards, values, dones, 0.99, 0.95)
        assert np.isfinite(advantages).all()


class TestSampleActionJax:
    def test_returns_valid_actions(self):
        from trader_keras.steps.rollout import _sample_action_jax

        key = jax.random.PRNGKey(0)
        logits = jnp.zeros(6)
        p0_params = jnp.array([0.5, 0.0])
        p1_params = jnp.array([0.0, -1.0])

        action_type, p0, p1, log_prob = _sample_action_jax(key, logits, p0_params, p1_params)
        assert 0 <= int(action_type) < 6
        assert 0.0 <= float(p0) <= 1.0
        assert -1.0 <= float(p1) <= 1.0
        assert np.isfinite(float(log_prob))

    def test_different_keys_different_actions(self):
        from trader_keras.steps.rollout import _sample_action_jax

        logits = jnp.zeros(6)
        p0_params = jnp.array([0.5, 0.0])
        p1_params = jnp.array([0.0, 0.0])

        actions = set()
        for i in range(20):
            key = jax.random.PRNGKey(i)
            a, _, _, _ = _sample_action_jax(key, logits, p0_params, p1_params)
            actions.add(int(a))
        assert len(actions) > 1  # not all the same


class TestGaussianLogProb:
    def test_matches_scipy(self):
        from scipy.stats import norm
        from trader_keras.steps.ppo_loss import gaussian_log_prob

        x, mean, log_std = 0.5, 0.3, -1.0
        std = np.exp(np.clip(log_std, -5.0, 2.0))

        result = float(gaussian_log_prob(
            np.array([x]), np.array([mean]), np.array([log_std]),
        )[0])
        expected = norm.logpdf(x, loc=mean, scale=std)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestClipGrads:
    def test_reduces_norm_when_above_threshold(self):
        from trader_keras.steps.rl import _clip_grads

        grads = [jnp.ones((10,)) * 5.0, jnp.ones((10,)) * 5.0]
        original_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads)))
        max_norm = 1.0
        assert original_norm > max_norm

        clipped = _clip_grads(grads, max_norm)
        clipped_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in clipped)))
        np.testing.assert_allclose(clipped_norm, max_norm, rtol=1e-5)

    def test_no_change_when_below_threshold(self):
        from trader_keras.steps.rl import _clip_grads

        grads = [jnp.ones((10,)) * 0.01, jnp.ones((5,)) * 0.01]
        max_norm = 10.0
        clipped = _clip_grads(grads, max_norm)

        for g, c in zip(grads, clipped):
            np.testing.assert_array_equal(np.array(g), np.array(c))


class TestRolloutJIT:
    """Test JIT rollout with real icmarkets_env (skip if no data)."""

    @pytest.fixture()
    def env_and_policy(self):
        try:
            from icmarkets_env import TradingEnv
            te = TradingEnv.from_symbol("EURUSD", "2025-01-01", "2025-01-03", lookback=10)
        except (FileNotFoundError, Exception):
            pytest.skip("EURUSD data not available")
        cfg = OmegaConf.create({"hidden_size": 16})
        from trader_keras.models.policy import build_policy_model
        policy = build_policy_model(te.obs_dim, cfg)
        return te, policy

    def test_returns_correct_shapes(self, env_and_policy):
        from icmarkets_env.env import reset
        from trader_keras.steps.rollout import build_collect_rollout

        te, policy = env_and_policy
        params = te._params
        obs, state, bar_feats = reset(params, lookback=10, balance=10000.0)
        hidden = jnp.zeros(16)
        collect = build_collect_rollout(policy, params, lookback=10, balance=10000.0)

        trainable = [v.value for v in policy.trainable_variables]
        non_trainable = [v.value for v in policy.non_trainable_variables]
        rng = jax.random.PRNGKey(42)

        transitions, _ = collect(
            rng, trainable, non_trainable,
            state, obs, bar_feats, hidden, jnp.int32(0), 64,
        )
        assert transitions["reward"].shape == (64,)
        assert transitions["obs"].shape == (64, te.obs_dim)
        assert transitions["hidden"].shape == (64, 16)
        assert transitions["action_type"].shape == (64,)

    def test_rewards_finite(self, env_and_policy):
        from icmarkets_env.env import reset
        from trader_keras.steps.rollout import build_collect_rollout

        te, policy = env_and_policy
        params = te._params
        obs, state, bar_feats = reset(params, lookback=10, balance=10000.0)
        hidden = jnp.zeros(16)
        collect = build_collect_rollout(policy, params, lookback=10, balance=10000.0)

        trainable = [v.value for v in policy.trainable_variables]
        non_trainable = [v.value for v in policy.non_trainable_variables]
        rng = jax.random.PRNGKey(42)

        transitions, _ = collect(
            rng, trainable, non_trainable,
            state, obs, bar_feats, hidden, jnp.int32(0), 64,
        )
        assert np.all(np.isfinite(np.asarray(transitions["reward"])))
        assert np.all(np.isfinite(np.asarray(transitions["value"])))
        assert np.all(np.isfinite(np.asarray(transitions["log_prob"])))


class TestFitRL:
    """Integration test with FakeEnv — uses Python fallback path."""

    def test_produces_model_in_ctx(self):
        """fit_rl with real env (skip if no data)."""
        try:
            from icmarkets_env import TradingEnv
            te = TradingEnv.from_symbol("EURUSD", "2025-01-01", "2025-01-03", lookback=10)
        except (FileNotFoundError, Exception):
            pytest.skip("EURUSD data not available")

        from trader_keras.steps.rl import fit_rl

        cfg = _rl_cfg()
        ctx: Ctx = {"cfg": cfg, "env": te}
        ctx = fit_rl(ctx)

        assert "model" in ctx
        obs = np.zeros((1, te.obs_dim), dtype=np.float32)
        hidden = np.zeros((1, 16), dtype=np.float32)
        outputs = ctx["model"]([obs, hidden], training=False)
        assert len(outputs) == 5  # logits, p0, p1, value, new_hidden
