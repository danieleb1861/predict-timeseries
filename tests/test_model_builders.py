import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from load_forecasting.models.registry_flax import FLAX_MODEL_BUILDERS
from scripts.tune_hyperparams import build_module


SPECS = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "model_specs.json").read_text(
        encoding="utf-8"
    )
)
MODEL_SPECS = {**SPECS["deep_models"], **SPECS["hybrid_models"]}
# The 17-signal feature layout is read from the only spec that declares groups,
# so this suite keeps no second copy of the feature order.
FEATURE_NAMES = tuple(
    name
    for group in SPECS["deep_models"]["dual_stream_film"]["feature_groups"].values()
    for name in group
)
TARGET_NAMES = ("act_pow_pu",)


def test_every_registry_builder_is_covered_by_a_spec():
    assert set(MODEL_SPECS) == set(FLAX_MODEL_BUILDERS)


@pytest.mark.parametrize("model_key", sorted(MODEL_SPECS))
def test_registry_builder_emits_requested_horizon_shape(model_key):
    """
    Verify that every registered builder honours the (H, output_dim) contract.

    Per-model behaviour (dilation schedules, feature routing, head modes) is
    owned by the dedicated model tests; this case only pins the shared shape
    contract across the whole registry.
    """
    lookback, horizon, batch = 20, 5, 2
    input_shape = (lookback, len(FEATURE_NAMES))
    output_shape = (horizon, len(TARGET_NAMES))

    module = build_module(
        FLAX_MODEL_BUILDERS,
        MODEL_SPECS[model_key],
        input_shape,
        output_shape,
        {},
        FEATURE_NAMES,
        TARGET_NAMES,
        (FEATURE_NAMES.index(TARGET_NAMES[0]),),
    )
    x = jnp.asarray(
        np.random.default_rng(0).normal(size=(batch, *input_shape)), dtype=jnp.float32
    )
    variables = module.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        x,
        deterministic=True,
    )
    prediction = module.apply(variables, x, deterministic=True)

    assert prediction.shape == (batch, horizon, len(TARGET_NAMES))
    assert prediction.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(prediction)))
