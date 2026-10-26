import subprocess
import sys
import pickle
from types import SimpleNamespace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from flax import serialization

from load_forecasting.models.registry_flax import FLAX_MODEL_BUILDERS
from scripts.gen_figures_4papers import (
    FigureModel,
    discover_archives,
    effective_archive_tag,
    load_metrics,
    parse_args,
    predict_flax,
    predict_flax_msgpack,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Probed in a subprocess so the assertion cannot be weakened by a TensorFlow
# import performed earlier in the session by another test.
CLI_IMPORT_PROBE = """
import sys

from scripts.gen_figures_4papers import parse_args

parse_args([])
print("tensorflow" in sys.modules)
"""


def _flax_archive(root, name):
    archive = root / name
    artifact = archive / "model_cnn_flax"
    artifact.mkdir(parents=True)
    (artifact / "params.msgpack").write_bytes(b"placeholder")
    return archive


def test_flax_archive_discovery_requires_requested_campaign_tag(tmp_path):
    wanted = _flax_archive(
        tmp_path, "20260720_ForecastModels_Navigation_L20_H5_flax7"
    )
    _flax_archive(
        tmp_path, "20260721_ForecastModels_Navigation_L20_H5_experimental"
    )
    keras = tmp_path / "20260722_7Models_Navigation_L20_H5"
    keras.mkdir()
    (keras / "model_cnn.keras").write_bytes(b"placeholder")

    resolved = discover_archives(
        tmp_path,
        backend="flax",
        modes=("navigation",),
        archive_tag="flax7",
        grid=((20, 5),),
    )

    assert resolved == {"navigation": {(20, 5): wanted}}


def test_archive_discovery_reports_every_missing_cell(tmp_path):
    _flax_archive(
        tmp_path, "20260720_ForecastModels_Manoeuvre_L20_H5_flax7"
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_archives(
            tmp_path,
            backend="flax",
            modes=("manoeuvre",),
            archive_tag="flax7",
            grid=((20, 5), (45, 25), (60, 25)),
        )

    message = str(exc_info.value)
    assert "manoeuvre:L45/H25" in message
    assert "manoeuvre:L60/H25" in message


def test_legacy_msgpack_prediction_rebuilds_model_from_registry(tmp_path):
    input_shape = (8, 3)
    output_shape = (2, 1)
    feature_names = ("active_power", "reactive_power", "command")
    target_names = ("active_power",)
    inputs = np.linspace(-1.0, 1.0, 3 * 8 * 3, dtype=np.float32).reshape(
        3, *input_shape
    )
    module = FLAX_MODEL_BUILDERS["cnn"](
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size=3,
    )
    params_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(13))
    params = module.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.asarray(inputs[:1]),
        deterministic=True,
    )["params"]
    expected = np.asarray(
        module.apply({"params": params}, jnp.asarray(inputs), deterministic=True)
    )

    artifact = tmp_path / "model_cnn_flax"
    artifact.mkdir()
    (artifact / "params.msgpack").write_bytes(serialization.to_bytes(params))
    figure_model = FigureModel(
        key="cnn",
        label="CNN",
        builder="cnn",
        artifact_name="cnn",
        spec={"label": "CNN", "builder": "cnn", "builder_args": {"kernel_size": 3}},
    )

    actual = predict_flax_msgpack(
        artifact,
        figure_model,
        inputs,
        output_shape=output_shape,
        feature_names=feature_names,
        target_names=target_names,
        batch_size=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_legacy_msgpack_rejects_mismatched_model_spec(tmp_path):
    artifact = tmp_path / "model_cnn_flax"
    artifact.mkdir()
    (artifact / "params.msgpack").write_bytes(b"not-a-compatible-tree")
    model = FigureModel(
        key="cnn",
        label="CNN",
        builder="cnn",
        artifact_name="cnn",
        spec={"label": "CNN", "builder": "cnn", "builder_args": {"kernel_size": 5}},
    )

    with pytest.raises(ValueError, match="--model-specs matches the archive"):
        predict_flax_msgpack(
            artifact,
            model,
            np.ones((1, 8, 3), dtype=np.float32),
            output_shape=(2, 1),
            feature_names=("active_power", "reactive_power", "command"),
            target_names=("active_power",),
            batch_size=1,
        )


def test_current_flax_artifact_receives_standardized_windows(monkeypatch, tmp_path):
    class FakeArtifact:
        module = None
        params = None
        manifest = {"model": {}}
        input_shape = (8, 3)
        output_shape = (2, 1)
        feature_names = ("active_power", "reactive_power", "command")
        target_names = ("active_power",)
        feature_mean = np.asarray([10.0, -4.0, 0.5], dtype=np.float32)
        feature_scale = np.asarray([2.0, 0.25, 3.0], dtype=np.float32)
        target_mean = np.asarray([100.0], dtype=np.float32)
        target_scale = np.asarray([12.0], dtype=np.float32)

        def __init__(self):
            self.received = []

        def predict_standardized(self, values, *, feature_names):
            assert tuple(feature_names) == self.feature_names
            self.received.append(np.asarray(values).copy())
            return np.zeros((len(values), *self.output_shape), dtype=np.float32)

    fake = FakeArtifact()
    from load_forecasting import deployment_flax

    monkeypatch.setattr(
        deployment_flax,
        "load_flax_artifact",
        lambda artifact_dir, verify_golden: fake,
    )
    archive = tmp_path / "archive"
    (archive / "model_cnn_flax").mkdir(parents=True)
    inputs = np.linspace(-2.0, 2.0, 3 * 8 * 3, dtype=np.float32).reshape(3, 8, 3)
    model = FigureModel("cnn", "CNN", "cnn", "cnn", {"builder": "cnn"})

    prediction = predict_flax(
        archive,
        model,
        inputs,
        output_shape=(2, 1),
        feature_names=fake.feature_names,
        target_names=fake.target_names,
        feature_mean=fake.feature_mean,
        feature_scale=fake.feature_scale,
        target_mean=fake.target_mean,
        target_scale=fake.target_scale,
        batch_size=2,
        verify_golden=True,
    )

    np.testing.assert_array_equal(np.concatenate(fake.received), inputs)
    assert prediction.shape == (3, 2, 1)


def test_metric_loader_supports_comma_and_space_vectors(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    pd.DataFrame(
        {
            "RMSE_horizon": ["[0.1 0.2]"],
            "MAE_horizon": ["[0.08 0.15]"],
            "MAPE_horizon": ["[1.0, 2.0]"],
            "MAPE_targetax": ["[unused, very, large]"],
            "RMSE_all": [0.16],
            "MAE_all": [0.12],
            "MAPE_all": [1.5],
        },
        index=["CNN"],
    ).to_csv(archive / "results_metrics.csv")
    model = FigureModel("cnn", "CNN", "cnn", "cnn", {"builder": "cnn"})

    metrics = load_metrics(archive, (model,))

    np.testing.assert_allclose(metrics["CNN"]["RMSE_h"], [0.1, 0.2])
    np.testing.assert_allclose(metrics["CNN"]["MAPE_h"], [1.0, 2.0])


def test_cli_defaults_to_flax_backend_and_campaign_tag():
    args = parse_args([])

    assert args.backend == "flax"
    assert effective_archive_tag(args.backend, args.archive_tag) == "flax8"


def test_figure_metrics_prefer_parquet_and_validate_requested_horizon(tmp_path):
    from load_forecasting.metrics import save_metrics_outputs

    save_metrics_outputs({"CNN": {"MAPE_horizon": [1.0, 2.0], "MAPE_all": 1.5}}, tmp_path)
    # The figure adapter must use the shared reader, including its format preference.
    (tmp_path / "results_metrics.csv").write_text("stale CSV", encoding="utf-8")
    model = FigureModel("cnn", "CNN", "cnn", "cnn", {"builder": "cnn"})
    metrics = load_metrics(tmp_path, (model,), expected_horizon=2)
    np.testing.assert_allclose(metrics["CNN"]["MAPE_h"], [1.0, 2.0])
    with pytest.raises(ValueError, match="Metric horizon 2 != expected 3"):
        load_metrics(tmp_path, (model,), expected_horizon=3)


def test_cli_parsing_does_not_import_tensorflow():
    completed = subprocess.run(
        [sys.executable, "-c", CLI_IMPORT_PROBE],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


@pytest.fixture
def figure_prediction_run(tmp_path, monkeypatch):
    """Small archive with observable inference calls and captured plotted curves."""
    from scripts import gen_figures_4papers as figures

    archive = tmp_path / "archive"
    archive.mkdir()
    artifact = archive / "model_cnn_flax"
    artifact.mkdir()
    (artifact / "params.msgpack").write_bytes(b"weights")
    (archive / "model_cnn.keras").write_bytes(b"weights")
    inputs = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 10
    targets = np.arange(16, dtype=np.float32).reshape(8, 2, 1) / 10
    np.savez_compressed(archive / "windows.npz", Xte=inputs, Yte=targets)
    metadata = {"featureNames": ["power", "command"], "targetNames": ["power"],
                "MU": pd.Series([10., 1.], index=["power", "command"]),
                "SIG": pd.Series([2., 3.], index=["power", "command"]), "WindowSlidingCoeff": 1}
    with (archive / "scalers_and_meta.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    # Source invalidation is tested on temporary files, never repository code.
    source = tmp_path / "runtime/load_forecasting"
    (source / "models").mkdir(parents=True)
    (source / "models/cnn.py").write_text("initial model source", encoding="utf-8")
    (source / "deployment_flax.py").write_text("initial loader source", encoding="utf-8")
    monkeypatch.setattr(figures, "REPO_ROOT", tmp_path / "runtime")
    model = FigureModel("cnn", "CNN", "cnn", "cnn", {"builder": "cnn"})
    calls, curves, logs = [], [], []
    native_flax = figures.predict_flax

    def predict(archive, model, values, **kwargs):
        calls.append((len(values), kwargs))
        return np.repeat(values[:, :1, :1], 2, axis=1)

    def save(figure, *args, **kwargs):
        curves.append([(line.get_xdata().copy(), line.get_ydata().copy())
                       for line in figure.axes[0].lines])
        figures.plt.close(figure)

    monkeypatch.setattr(figures, "predict_flax", predict)
    monkeypatch.setattr(figures, "predict_keras", predict)
    monkeypatch.setattr(figures, "save_figure", save)
    kwargs = dict(segment_length=3, batch_size=4, verify_golden=True,
                  output_dir=tmp_path / "figures", formats=("png",), dpi=30,
                  logger=SimpleNamespace(info=lambda fmt, *args: logs.append(fmt % args)))
    return SimpleNamespace(module=figures, archive=archive, model=model, kwargs=kwargs,
                           inputs=inputs, targets=targets, metadata=metadata,
                           source=source, calls=calls, curves=curves, logs=logs, native_flax=native_flax)


@pytest.mark.parametrize("backend", ["flax", "keras"])
@pytest.mark.parametrize("verify", [False, True])
def test_figure_cache_reuses_full_test_predictions(figure_prediction_run, backend, verify):
    """Hits preserve all plotted values, full-test MAPE, and Flax verification flags."""
    run = figure_prediction_run
    run.kwargs["verify_golden"] = verify
    # Unverified training predictions must not be trusted or overwritten.
    legacy = run.archive / "predictions/cnn.npz"
    legacy.parent.mkdir()
    np.savez_compressed(legacy, model_label="CNN", yhat_test_phys=np.full((8, 2, 1), -999.))
    legacy_bytes = legacy.read_bytes()
    for _ in range(2):
        run.module.timeseries_comparison(run.archive, "man", backend, [run.model], **run.kwargs)
    assert [n for n, _ in run.calls] == ([8, 0] if backend == "flax" else [8])
    if backend == "flax":
        assert all(options["verify_golden"] is verify for _, options in run.calls)
    for before, after in zip(run.curves[0], run.curves[1]):
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
    metrics = [line for line in run.logs if "First-horizon MAPE=" in line]
    assert len(metrics) == 2 and metrics[0] == metrics[1]
    assert legacy.read_bytes() == legacy_bytes


def test_figure_cache_with_real_flax_artifact(figure_prediction_run, monkeypatch):
    """Verify that a restored model accepts the empty-batch validation path."""
    run = figure_prediction_run
    module = FLAX_MODEL_BUILDERS["cnn"](input_shape=(3, 2), output_shape=(2, 1))
    params = module.init(jax.random.PRNGKey(7), jnp.asarray(run.inputs[:1]), deterministic=True)["params"]
    (run.archive / "model_cnn_flax/params.msgpack").write_bytes(serialization.to_bytes(params))
    calls = []

    def predict(archive, model, values, **kwargs):
        calls.append(len(values))
        return run.native_flax(archive, model, values, **kwargs)

    monkeypatch.setattr(run.module, "predict_flax", predict)
    for _ in range(2):
        run.module.timeseries_comparison(run.archive, "man", "flax", [run.model], **run.kwargs)
    assert calls == [8, 0]
    for before, after in zip(run.curves[0], run.curves[1]):
        np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("backend", ["flax", "keras"])
@pytest.mark.parametrize("change", ["weights", "windows", "scalers", "spec", "source", "batch",
                                    "shape", "nonfinite", "corrupt", "legacy", "runtime"])
def test_figure_cache_falls_back_when_provenance_or_contents_change(figure_prediction_run, monkeypatch, backend, change):
    """Stale or damaged cached results must trigger full inference again."""
    run = figure_prediction_run
    run.module.timeseries_comparison(run.archive, "man", backend, [run.model], **run.kwargs)
    cache = run.kwargs["output_dir"] / "prediction_cache/archive" / backend / "cnn.npz"
    if change == "weights":
        path = run.archive / ("model_cnn_flax/params.msgpack" if backend == "flax" else "model_cnn.keras")
        path.write_bytes(b"updated weights")
    elif change == "windows":
        np.savez_compressed(run.archive / "windows.npz", Xte=run.inputs + 1, Yte=run.targets)
    elif change == "scalers":
        run.metadata["MU"].iloc[0] += 1
        with (run.archive / "scalers_and_meta.pkl").open("wb") as handle:
            pickle.dump(run.metadata, handle)
    elif change == "spec":
        run.model.spec["builder_args"] = {"kernel_size": 5}
    elif change == "source":
        (run.source / "models/cnn.py").write_text("new model source", encoding="utf-8")
    elif change == "batch":
        run.kwargs["batch_size"] = 2
    elif change == "runtime":
        monkeypatch.setattr(run.module.importlib.metadata, "version", lambda package: "changed-version")
    elif change == "corrupt":
        cache.write_bytes(b"invalid npz")
    else:
        with np.load(cache) as saved:
            fingerprint, values = saved["fingerprint"], saved["physical_first_horizon"].copy()
        if change == "shape":
            values = values[:-1]
        elif change == "nonfinite":
            values[0] = np.nan
        fields = {} if change == "legacy" else {"fingerprint": fingerprint}
        np.savez_compressed(cache, physical_first_horizon=values, **fields)
    run.module.timeseries_comparison(run.archive, "man", backend, [run.model], **run.kwargs)
    assert [n for n, _ in run.calls] == [8, 8]


def test_cached_figure_cannot_bypass_golden_failure(figure_prediction_run, monkeypatch):
    """A cached curve must never suppress a requested artifact verification failure."""
    run = figure_prediction_run
    run.module.timeseries_comparison(run.archive, "man", "flax", [run.model], **run.kwargs)

    def reject(*args, **kwargs):
        assert kwargs["verify_golden"] is True
        raise ValueError("golden prediction mismatch")

    monkeypatch.setattr(run.module, "predict_flax", reject)
    with pytest.raises(ValueError, match="golden prediction mismatch"):
        run.module.timeseries_comparison(run.archive, "man", "flax", [run.model], **run.kwargs)
    assert len(run.curves) == 1
