import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp  # noqa: E402

from load_forecasting.models.dlinear_flax import build_dlinear_model  # noqa: E402


@pytest.mark.parametrize("lookback,horizon", [(20, 5), (50, 25)])
def test_dlinear_output_shape(lookback, horizon):
    num_features, output_dim, batch = 17, 1, 4
    model = build_dlinear_model((lookback, num_features), (horizon, output_dim))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(batch, lookback, num_features)), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    y = model.apply(variables, x, deterministic=True)
    assert y.shape == (batch, horizon, output_dim)


def test_dlinear_kernel_clamped_to_lookback():
    # moving_avg_kernel=25 > L=20 must not fail.
    model = build_dlinear_model((20, 3), (5, 1), moving_avg_kernel=25)
    x = jnp.ones((2, 20, 3), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    y = model.apply(variables, x, deterministic=True)
    assert y.shape == (2, 5, 1)


def test_dlinear_is_affine():
    # For an affine map f(x) = Ax + b: f(x1 + x2) + f(0) == f(x1) + f(x2).
    model = build_dlinear_model((20, 3), (5, 1))
    rng = np.random.default_rng(1)
    x1 = jnp.asarray(rng.normal(size=(2, 20, 3)), dtype=jnp.float32)
    x2 = jnp.asarray(rng.normal(size=(2, 20, 3)), dtype=jnp.float32)
    zero = jnp.zeros_like(x1)
    variables = model.init(jax.random.PRNGKey(0), x1, deterministic=True)

    def f(x):
        return model.apply(variables, x, deterministic=True)

    lhs = f(x1 + x2) + f(zero)
    rhs = f(x1) + f(x2)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-4)
