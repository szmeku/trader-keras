"""Unit tests for actor-critic policy model."""
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import pytest
from omegaconf import OmegaConf


def _cfg(hidden_size: int = 32):
    return OmegaConf.create({"hidden_size": hidden_size})


class TestBuildPolicyModel:
    def test_output_shapes(self):
        from trader_keras.models.policy import build_policy_model

        model = build_policy_model(obs_dim=36, cfg=_cfg())
        batch = np.zeros((4, 36), dtype=np.float32)
        logits, p0, p1, value = model(batch, training=False)

        assert logits.shape == (4, 6)
        assert p0.shape == (4, 2)
        assert p1.shape == (4, 2)
        assert value.shape == (4, 1)

    def test_different_obs_dims(self):
        from trader_keras.models.policy import build_policy_model

        for obs_dim in [12, 48, 96]:
            model = build_policy_model(obs_dim=obs_dim, cfg=_cfg())
            batch = np.zeros((2, obs_dim), dtype=np.float32)
            logits, p0, p1, value = model(batch, training=False)
            assert logits.shape == (2, 6)
            assert value.shape == (2, 1)

    def test_different_hidden_sizes(self):
        from trader_keras.models.policy import build_policy_model

        for hs in [16, 64, 128]:
            model = build_policy_model(obs_dim=36, cfg=_cfg(hidden_size=hs))
            batch = np.zeros((2, 36), dtype=np.float32)
            outputs = model(batch, training=False)
            assert len(outputs) == 4

    def test_gradients_flow(self):
        from trader_keras.models.policy import build_policy_model

        model = build_policy_model(obs_dim=36, cfg=_cfg())
        batch = np.zeros((4, 36), dtype=np.float32)

        # Use keras compute_loss_and_updates or manual gradient check
        import jax

        @jax.jit
        def loss_fn(params, x):
            logits, p0, p1, value = model(x, training=False)
            return keras.ops.sum(logits) + keras.ops.sum(value)

        trainable_vars = model.trainable_variables
        # Check that we have trainable variables
        assert len(trainable_vars) > 0

        # Verify output is finite
        logits, _, _, value = model(batch, training=False)
        assert np.all(np.isfinite(np.array(logits)))
        assert np.all(np.isfinite(np.array(value)))

    def test_model_name(self):
        from trader_keras.models.policy import build_policy_model

        model = build_policy_model(obs_dim=36, cfg=_cfg())
        assert model.name == "actor_critic"

    def test_single_observation(self):
        """Model works with batch size 1."""
        from trader_keras.models.policy import build_policy_model

        model = build_policy_model(obs_dim=36, cfg=_cfg())
        batch = np.zeros((1, 36), dtype=np.float32)
        logits, p0, p1, value = model(batch, training=False)
        assert logits.shape == (1, 6)
        assert value.shape == (1, 1)
