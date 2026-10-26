import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from load_forecasting.deployment_flax import (
    load_flax_artifact,
    save_flax_deployment_artifact,
)
from load_forecasting.models.deep_flax import receptive_field
from load_forecasting.models.dual_stream_flax import (
    BoundedFiLM,
    DualStreamFiLMForecaster,
    PreNormCausalResidualBlock,
    build_dual_stream_film_model,
    resolve_feature_group_indices,
)
from load_forecasting.models.registry_flax import FLAX_DEEP_BUILDERS, FLAX_MODEL_BUILDERS
from scripts.tune_hyperparams import build_module


FEATURE_NAMES = (
    "act_pow_pu",
    "react_pow_pu",
    "air_temp",
    "sea_temp",
    "wind_intensity",
    "sea_state",
    "lever_diff",
    "pfactor",
    "spd_port",
    "spd_stbd",
    "dP",
    "moving_avg_5",
    "moving_avg_10",
    "moving_avg_20",
    "moving_std_5",
    "moving_std_10",
    "moving_std_20",
)
FEATURE_GROUPS = {
    "endogenous": (
        "act_pow_pu",
        "react_pow_pu",
        "pfactor",
        "dP",
        "moving_avg_5",
        "moving_avg_10",
        "moving_avg_20",
        "moving_std_5",
        "moving_std_10",
        "moving_std_20",
    ),
    "propulsion": ("lever_diff", "spd_port", "spd_stbd"),
    "environment": ("air_temp", "sea_temp", "wind_intensity", "sea_state"),
}
GROUP_INDICES = {
    "endogenous_indices": (0, 1, 7, 10, 11, 12, 13, 14, 15, 16),
    "propulsion_indices": (6, 8, 9),
    "environment_indices": (2, 3, 4, 5),
    "target_indices": (0,),
}


def _small_builder_args(**overrides):
    args = {
        **GROUP_INDICES,
        "dynamics_width": 8,
        "command_hidden": 4,
        "environment_hidden": 4,
        "context_width": 8,
        "dropout_rate": 0.0,
        "horizon_embed_dim": 4,
        "decoder_width": 8,
    }
    args.update(overrides)
    return args


def test_feature_groups_resolve_by_name_after_permutation():
    permuted = tuple(reversed(FEATURE_NAMES))
    resolved = resolve_feature_group_indices(
        permuted,
        ("act_pow_pu",),
        FEATURE_GROUPS,
    )

    for group_name in ("endogenous", "propulsion", "environment"):
        indices = resolved[f"{group_name}_indices"]
        assert tuple(permuted[index] for index in indices) == FEATURE_GROUPS[group_name]
    assert tuple(permuted[index] for index in resolved["target_indices"]) == (
        "act_pow_pu",
    )


@pytest.mark.parametrize(
    ("overrides", "output_shape", "match"),
    [
        ({"propulsion_indices": ()}, (5, 1), "must not be empty"),
        (
            {"propulsion_indices": (0, 6, 8, 9)},
            (5, 1),
            "must be disjoint",
        ),
        ({"environment_indices": (2, 3, 4)}, (5, 1), "cover every input"),
        ({"environment_indices": (2, 3, 4, 17)}, (5, 1), "between 0 and 16"),
        ({"target_indices": (0,)}, (5, 2), "expected output dimension 2"),
        ({"target_indices": (6,)}, (5, 1), "must belong to the endogenous"),
    ],
)
def test_builder_rejects_invalid_feature_routing(overrides, output_shape, match):
    with pytest.raises(ValueError, match=match):
        build_dual_stream_film_model(
            (20, 17),
            output_shape,
            **_small_builder_args(**overrides),
        )


@pytest.mark.parametrize(
    ("lookback", "horizon", "expected_dilations"),
    [
        (20, 5, (1, 2, 4, 8)),
        (50, 25, (1, 2, 4, 8, 16)),
    ],
)
def test_output_shape_and_adaptive_receptive_field(
    lookback,
    horizon,
    expected_dilations,
):
    module = build_dual_stream_film_model(
        (lookback, 17),
        (horizon, 1),
        **_small_builder_args(),
    )
    x = jnp.ones((2, lookback, 17), dtype=jnp.float32)
    variables = module.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        x,
        deterministic=True,
    )
    prediction = module.apply(variables, x, deterministic=True)

    assert module.dilations == expected_dilations
    assert receptive_field([3] * len(module.dilations), module.dilations) >= lookback
    assert receptive_field([3] * (len(module.dilations) - 1), module.dilations[:-1]) < lookback
    assert prediction.shape == (2, horizon, 1)
    assert prediction.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(prediction)))


def test_causal_residual_block_does_not_see_future():
    module = PreNormCausalResidualBlock(
        features=3,
        kernel_size=3,
        dilation_rate=2,
        dropout_rate=0.0,
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 10, 3))
    changed = x.at[:, 5:, :].add(100.0)
    variables = module.init(jax.random.PRNGKey(3), x, deterministic=True)

    original_output = module.apply(variables, x, deterministic=True)
    changed_output = module.apply(variables, changed, deterministic=True)
    np.testing.assert_allclose(original_output[:, :5, :], changed_output[:, :5, :])


def test_zero_initialized_film_is_identity_and_ignores_context():
    module = BoundedFiLM(features=4, scale=0.5)
    dynamics = jax.random.normal(jax.random.PRNGKey(4), (2, 4))
    context_a = jnp.zeros((2, 3), dtype=jnp.float32)
    context_b = jnp.full((2, 3), 100.0, dtype=jnp.float32)
    variables = module.init(jax.random.PRNGKey(5), dynamics, context_a)

    output_a = module.apply(variables, dynamics, context_a)
    output_b = module.apply(variables, dynamics, context_b)
    np.testing.assert_allclose(output_a, dynamics)
    np.testing.assert_allclose(output_b, dynamics)
    np.testing.assert_array_equal(variables["params"]["projection"]["kernel"], 0.0)
    np.testing.assert_array_equal(variables["params"]["projection"]["bias"], 0.0)


def test_horizon_residual_head_initializes_to_persistence():
    module = build_dual_stream_film_model(
        (20, 17),
        (5, 1),
        **_small_builder_args(head_mode="horizon_residual"),
    )
    x = jax.random.normal(jax.random.PRNGKey(6), (2, 20, 17))
    variables = module.init(
        {"params": jax.random.PRNGKey(7), "dropout": jax.random.PRNGKey(8)},
        x,
        deterministic=True,
    )
    prediction = module.apply(variables, x, deterministic=True)
    expected = jnp.broadcast_to(x[:, -1:, 0:1], prediction.shape)

    np.testing.assert_allclose(prediction, expected, rtol=0.0, atol=0.0)


def test_registry_contains_dual_stream_film():
    assert FLAX_DEEP_BUILDERS["dual_stream_film"] is build_dual_stream_film_model
    assert FLAX_MODEL_BUILDERS["dual_stream_film"] is build_dual_stream_film_model


def test_default_spec_builds_through_tuner_path():
    specs_path = Path(__file__).resolve().parents[1] / "config" / "model_specs.json"
    spec = json.loads(specs_path.read_text(encoding="utf-8"))["deep_models"][
        "dual_stream_film"
    ]

    module = build_module(
        FLAX_MODEL_BUILDERS,
        spec,
        (20, 17),
        (5, 1),
        {},
        FEATURE_NAMES,
        ("act_pow_pu",),
        (0,),
    )

    assert isinstance(module, DualStreamFiLMForecaster)
    assert module.endogenous_indices == GROUP_INDICES["endogenous_indices"]
    assert module.propulsion_indices == GROUP_INDICES["propulsion_indices"]
    assert module.environment_indices == GROUP_INDICES["environment_indices"]
    assert module.dilations == (1, 2, 4, 8)
    assert module.head_mode == "flat_direct"


def test_dual_stream_orbax_round_trip(tmp_path):
    input_shape = (8, 17)
    output_shape = (3, 1)
    builder_args = _small_builder_args(head_mode="flat_direct")
    module = build_dual_stream_film_model(
        input_shape,
        output_shape,
        **builder_args,
    )
    inputs = np.linspace(-1.0, 1.0, 2 * 8 * 17, dtype=np.float32).reshape(2, 8, 17)
    params = module.init(
        {"params": jax.random.PRNGKey(9), "dropout": jax.random.PRNGKey(10)},
        jnp.asarray(inputs),
        deterministic=True,
    )["params"]
    expected = np.asarray(
        module.apply({"params": params}, jnp.asarray(inputs), deterministic=True)
    )

    artifact_dir = tmp_path / "dual_stream_artifact"
    save_flax_deployment_artifact(
        module,
        params,
        artifact_dir,
        builder_key="dual_stream_film",
        builder_args=builder_args,
        input_shape=input_shape,
        output_shape=output_shape,
        feature_names=FEATURE_NAMES,
        target_names=("act_pow_pu",),
        feature_mean=np.zeros(17, dtype=np.float32),
        feature_scale=np.ones(17, dtype=np.float32),
        target_mean=np.zeros(1, dtype=np.float32),
        target_scale=np.ones(1, dtype=np.float32),
        sampling_rate="1s",
        golden_input_standardized=inputs[:1],
    )

    loaded = load_flax_artifact(artifact_dir)
    actual = loaded.predict_standardized(inputs, feature_names=FEATURE_NAMES)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    for expected_leaf, actual_leaf in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(loaded.params),
    ):
        np.testing.assert_array_equal(expected_leaf, actual_leaf)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    saved_args = manifest["model"]["builder_args"]
    assert saved_args["dilations"] == [1, 2, 4]
    assert saved_args["endogenous_indices"] == list(GROUP_INDICES["endogenous_indices"])
    assert loaded.verify_golden(rtol=1e-6, atol=1e-6)
