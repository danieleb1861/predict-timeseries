import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp  # noqa: E402

from load_forecasting.models.itransformer_flax import build_itransformer_model  # noqa: E402


@pytest.mark.parametrize("lookback,horizon", [(20, 5), (50, 25)])
def test_itransformer_output_shape(lookback, horizon):
    num_features, output_dim, batch = 17, 1, 4
    model = build_itransformer_model(
        (lookback, num_features), (horizon, output_dim), target_indices=(0,)
    )
    x = jnp.asarray(np.random.default_rng(0).normal(size=(batch, lookback, num_features)), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, deterministic=True
    )
    y = model.apply(variables, x, deterministic=True)
    assert y.shape == (batch, horizon, output_dim)


def test_itransformer_target_indices_mismatch_raises():
    model = build_itransformer_model((20, 17), (5, 1), target_indices=(0, 1))
    x = jnp.ones((2, 20, 17), dtype=jnp.float32)
    with pytest.raises(ValueError, match="target_indices"):
        model.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, deterministic=True
        )


def test_itransformer_head_selects_target_channel():
    # Output dim must follow target_indices length for multi-target settings.
    model = build_itransformer_model((20, 5), (5, 2), target_indices=(0, 3))
    x = jnp.ones((2, 20, 5), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, deterministic=True
    )
    y = model.apply(variables, x, deterministic=True)
    assert y.shape == (2, 5, 2)
