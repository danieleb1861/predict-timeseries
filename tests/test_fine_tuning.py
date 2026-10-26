import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from load_forecasting.fine_tuning import Scaler, restandardize, trainable_mask  # noqa: E402
from load_forecasting.training_deep import DeepTrainConfig  # noqa: E402
from load_forecasting.training_flax import create_train_state, fit_flax_model  # noqa: E402


class TwoLayer(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(4, name="body")(x.reshape((x.shape[0], -1)))
        return nn.Dense(3, name="head")(x)[..., None]


def _data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5, 2)).astype(np.float32)
    return X, (X.sum(axis=(1, 2))[:, None, None] * np.ones((1, 3, 1))).astype(np.float32)


def test_create_train_state_starts_from_init_params():
    module, cfg = TwoLayer(), DeepTrainConfig(seed=0)
    random_state = create_train_state(module, (5, 2), cfg)
    given = jax.tree_util.tree_map(lambda p: p + 1.0, random_state.params)
    state = create_train_state(module, (5, 2), cfg, init_params=given)
    for a, b in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(given)):
        np.testing.assert_array_equal(a, b)


def test_frozen_leaves_do_not_move_and_trainable_ones_do():
    X, Y = _data()
    module, cfg = TwoLayer(), DeepTrainConfig(batch_size=16, epochs=3, learning_rate=1e-2, loss="huber", patience=5)
    start = create_train_state(module, (5, 2), cfg).params
    mask = trainable_mask(start, frozen=[r"^body/"])
    state, _, _ = fit_flax_model(module, X, Y, X, Y, cfg, init_params=start, trainable_mask=mask)
    np.testing.assert_array_equal(state.params["body"]["kernel"], start["body"]["kernel"])
    assert not np.allclose(state.params["head"]["kernel"], start["head"]["kernel"])


def test_trainable_mask_patterns():
    params = create_train_state(TwoLayer(), (5, 2), DeepTrainConfig()).params
    assert trainable_mask(params) is None
    only_head = trainable_mask(params, trainable=[r"^head/"])
    assert not only_head["body"]["kernel"] and only_head["head"]["kernel"]
    with pytest.raises(ValueError):
        trainable_mask(params, frozen=[r"."])


def test_restandardize_is_an_exact_change_of_scaler():
    rng = np.random.default_rng(1)
    src = Scaler(np.array([1.0, -2.0], np.float32), np.array([2.0, 0.5], np.float32))
    dst = Scaler(np.array([0.5, 3.0], np.float32), np.array([4.0, 1.5], np.float32))
    z = rng.normal(size=(7, 5, 2)).astype(np.float32)
    physical = z * src.scale + src.mean
    np.testing.assert_allclose(restandardize(z, src, dst) * dst.scale + dst.mean, physical, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(restandardize(restandardize(z, src, dst), dst, src), z, rtol=1e-5, atol=1e-5)
