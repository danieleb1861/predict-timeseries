import json
from types import SimpleNamespace

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from load_forecasting.deployment_flax import (
    load_flax_artifact,
    load_flax_inference,
    load_flax_run_model,
    save_flax_deployment_artifact,
)
from load_forecasting.models.registry_flax import FLAX_MODEL_BUILDERS


def test_orbax_artifact_golden_prediction_round_trip(tmp_path):
    builder_key = "cnn"
    builder_args = {"kernel_size": 3}
    input_shape = (8, 3)
    output_shape = (2, 1)
    feature_names = ("active_power", "reactive_power", "command")
    target_names = ("active_power",)

    module = FLAX_MODEL_BUILDERS[builder_key](
        input_shape=input_shape,
        output_shape=output_shape,
        **builder_args,
    )
    inputs_z = np.linspace(-1.5, 1.5, 2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    params_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(17))
    params = module.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.asarray(inputs_z),
        deterministic=True,
    )["params"]
    expected_z = np.asarray(
        module.apply({"params": params}, jnp.asarray(inputs_z), deterministic=True)
    )

    feature_mean = np.asarray([10.0, -4.0, 0.5], dtype=np.float32)
    feature_scale = np.asarray([2.0, 0.25, 3.0], dtype=np.float32)
    target_mean = np.asarray([100.0], dtype=np.float32)
    target_scale = np.asarray([12.0], dtype=np.float32)
    inputs_physical = inputs_z * feature_scale.reshape((1, 1, -1)) + feature_mean.reshape((1, 1, -1))
    expected_physical = (
        expected_z * target_scale.reshape((1, 1, -1))
        + target_mean.reshape((1, 1, -1))
    )

    artifact_dir = tmp_path / "cnn_artifact"
    save_flax_deployment_artifact(
        module,
        params,
        artifact_dir,
        builder_key=builder_key,
        builder_args=builder_args,
        input_shape=input_shape,
        output_shape=output_shape,
        feature_names=feature_names,
        target_names=target_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        sampling_rate="1s",
        golden_input_standardized=inputs_z[:1],
    )

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["builder_args"]["dilations"] == [1, 2, 4]
    assert (artifact_dir / "params").is_dir()
    assert (artifact_dir / "preprocess.npz").is_file()
    assert (artifact_dir / "golden_io.npz").is_file()

    loaded = load_flax_artifact(artifact_dir)
    np.testing.assert_allclose(
        loaded.predict_standardized(inputs_z, feature_names=feature_names),
        expected_z,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        loaded.predict(inputs_physical, feature_names=feature_names),
        expected_physical,
        rtol=1e-6,
        atol=1e-6,
    )
    assert loaded.verify_golden(rtol=1e-6, atol=1e-6)

    with pytest.raises(ValueError, match="feature order mismatch"):
        loaded.predict(inputs_physical, feature_names=tuple(reversed(feature_names)))

    # The common loader preserves standardized inputs, batching, and golden checks.
    predictor = load_flax_inference(
        artifact_dir, input_shape=input_shape, output_shape=output_shape,
        feature_names=feature_names, target_names=target_names,
        feature_mean=feature_mean, feature_scale=feature_scale,
        target_mean=target_mean, target_scale=target_scale,
    )
    np.testing.assert_allclose(predictor.predict_standardized(inputs_z, batch_size=1), expected_z, rtol=1e-6, atol=1e-6)
    assert predictor.predict_standardized(inputs_z[:0]).shape == (0, 2, 1)


@pytest.fixture
def legacy_run(tmp_path):
    """A small real MsgPack model with archive metadata and two input splits."""
    artifact = tmp_path / "model_cnn_flax"
    artifact.mkdir()
    module = FLAX_MODEL_BUILDERS["cnn"](input_shape=(8, 3), output_shape=(2, 1), kernel_size=3)
    inputs = np.arange(3 * 8 * 3, dtype=np.float32).reshape(3, 8, 3) / 100
    params = module.init({"params": jax.random.PRNGKey(7), "dropout": jax.random.PRNGKey(8)},
                         jnp.asarray(inputs[:1]), deterministic=True)["params"]
    (artifact / "params.msgpack").write_bytes(serialization.to_bytes(params))
    (artifact / "metadata.json").write_text(json.dumps({"model_key": "cnn", "flops_per_window": 123}))
    spec = {"builder": "cnn", "label": "CNN", "builder_args": {"kernel_size": 3}}
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps({"deep_models": {"cnn": spec}}))
    names = ["power", "reactive", "command"]
    meta = {"featureNames": names, "targetNames": ["power"], "Lookback window L": 8,
            "Horizon forecasting window H": 2, "MU": pd.Series([10., 20., 30.], index=names),
            "SIG": pd.Series([2., 3., 4.], index=names)}
    expected = np.asarray(module.apply({"params": params}, jnp.asarray(inputs), deterministic=True))
    return tmp_path, artifact, spec, specs_file, meta, inputs, expected


def test_legacy_loader_and_latency_wrapper_preserve_predictions(legacy_run):
    from measure_inference_latency import load_flax_model

    _, artifact, _, specs_file, meta, inputs, expected = legacy_run
    loaded = load_flax_run_model(artifact, meta, specs_file)
    np.testing.assert_allclose(loaded.predict_standardized(inputs, batch_size=2), expected, rtol=1e-6, atol=1e-6)
    assert loaded.predict_standardized(inputs[:0]).shape == (0, 2, 1)
    module, params, metadata = load_flax_model(artifact, meta, specs_file)
    np.testing.assert_allclose(module.apply({"params": params}, jnp.asarray(inputs), deterministic=True), expected)
    assert metadata["flops_per_window"] == 123
    with pytest.raises(ValueError, match="positive integer"):
        loaded.predict_standardized(inputs, batch_size=0)
    with pytest.raises(ValueError, match="inputs must have shape"):
        loaded.predict_standardized(inputs[:, :, :2])


def test_legacy_loader_rejects_parameter_shape_mismatch(legacy_run):
    _, artifact, spec, _, meta, _, _ = legacy_run
    # Same tree keys but different convolution shapes must fail during restoration.
    spec = {**spec, "builder_args": {"kernel_size": 5}}
    with pytest.raises(ValueError, match="--model-specs matches the archive"):
        load_flax_run_model(artifact, meta, spec=spec)


@pytest.mark.parametrize("validation_size", [0, 1])
def test_peak_regeneration_restores_once_for_both_splits(legacy_run, monkeypatch, validation_size):
    from load_forecasting import deployment_flax
    from scripts.compute_peak_metrics import load_or_regenerate_predictions

    root, _, spec, specs_file, meta, inputs, expected = legacy_run
    original_loader = deployment_flax.load_flax_run_model
    loads = []

    def counted_loader(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        loads.append(loaded)
        return loaded

    monkeypatch.setattr(deployment_flax, "load_flax_run_model", counted_loader)
    result = load_or_regenerate_predictions(root, "cnn", spec, meta,
                                            {"Xte": inputs, "Xva": inputs[:validation_size]}, specs_file)
    assert len(loads) == 1
    np.testing.assert_allclose(result, expected * 2 + 10, rtol=1e-6, atol=1e-6)
    with np.load(root / "predictions" / "cnn.npz") as saved:
        assert saved["yhat_val_phys"].shape == (validation_size, 2, 1)
        np.testing.assert_allclose(saved["yhat_val_phys"], (expected * 2 + 10)[:validation_size], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("field, value, message", [
    ("input_shape", (7, 3), "input shape"),
    ("output_shape", (3, 1), "output shape"),
    ("feature_names", ("reactive", "power", "command"), "feature order"),
    ("target_names", ("reactive",), "target order"),
    ("feature_mean", [11., 20., 30.], "feature mean"),
    ("feature_scale", [2., 3., 5.], "feature scale"),
    ("target_mean", [11.], "target mean"),
    ("target_scale", [3.], "target scale"),
    ("target_mean", [10., 10.], "must have shape"),
])
def test_common_loader_rejects_archive_contract_mismatch(tmp_path, monkeypatch, field, value, message):
    from load_forecasting import deployment_flax

    contract = dict(input_shape=(8, 3), output_shape=(2, 1), feature_names=("power", "reactive", "command"),
                    target_names=("power",), feature_mean=[10., 20., 30.], feature_scale=[2., 3., 4.],
                    target_mean=[10.], target_scale=[2.])
    fake = SimpleNamespace(**contract, module=None, params=None, manifest={"model": {}})
    verified = []

    def restore(path, *, verify_golden):
        verified.append(verify_golden)
        return fake

    monkeypatch.setattr(deployment_flax, "load_flax_artifact", restore)
    with pytest.raises(ValueError, match=message):
        load_flax_inference(tmp_path, **{**contract, field: value})
    assert verified == [True]


def test_versioned_artifact_does_not_fall_back_after_golden_failure(legacy_run, monkeypatch):
    from load_forecasting import deployment_flax

    _, artifact, _, specs_file, meta, _, _ = legacy_run
    (artifact / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    (artifact / "metadata.json").unlink()  # Stray legacy files must not trigger spec loading.

    def invalid_golden(path, *, verify_golden):
        assert verify_golden
        raise ValueError("golden prediction mismatch")

    monkeypatch.setattr(deployment_flax, "load_flax_artifact", invalid_golden)
    with pytest.raises(ValueError, match="golden prediction mismatch"):
        load_flax_run_model(artifact, meta, specs_file)
