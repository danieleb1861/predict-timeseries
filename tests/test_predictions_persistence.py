import numpy as np
import pytest

from load_forecasting.training_deep import save_predictions


def test_save_predictions_roundtrip(tmp_path):
    yhat_test = np.random.default_rng(0).normal(size=(10, 5, 1))
    yhat_val = np.random.default_rng(1).normal(size=(7, 5, 1))
    save_predictions(tmp_path, "tcn", "TCN", yhat_test, yhat_val)

    saved = np.load(tmp_path / "predictions" / "tcn.npz")
    assert saved["yhat_test_phys"].shape == (10, 5, 1)
    assert saved["yhat_val_phys"].shape == (7, 5, 1)
    assert saved["yhat_test_phys"].dtype == np.float32
    assert str(saved["model_label"]) == "TCN"


def test_save_predictions_val_optional(tmp_path):
    save_predictions(tmp_path, "ridge", "Ridge", np.zeros((4, 5, 1)))
    saved = np.load(tmp_path / "predictions" / "ridge.npz")
    assert "yhat_test_phys" in saved.files
    assert "yhat_val_phys" not in saved.files


@pytest.mark.parametrize("sample_key", ["Xte", "Yte"])
@pytest.mark.parametrize("problem", ["shape", "nonfinite", "label"])
def test_peak_reader_rejects_invalid_legacy_cache(tmp_path, problem, sample_key):
    from scripts.compute_peak_metrics import load_or_regenerate_predictions

    # Shape and contents can be validated even though old caches lack provenance.
    prediction = np.zeros((3, 2, 1))
    label = "CNN"
    if problem == "shape":
        prediction = prediction[:1]
    elif problem == "nonfinite":
        prediction[0, 0, 0] = np.nan
    else:
        label = "Other model"
    save_predictions(tmp_path, "cnn", label, prediction)
    with pytest.raises(ValueError, match="cnn.npz"):
        load_or_regenerate_predictions(
            tmp_path, "cnn", {"label": "CNN"},
            {"Horizon forecasting window H": 2, "targetNames": ["power"]},
            {sample_key: np.zeros((3, 8, 3))}, None,
        )


def test_peak_metrics_preserves_tail_energy_and_ramp_definitions():
    """Check every metric against hand-calculated values, including zero masking."""
    from scripts.compute_peak_metrics import peak_metrics

    truth = np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6], [4, 8, 12]], dtype=float)
    result = peak_metrics(truth, truth + 1, np.array([0, 2, 4, 6]), 1200, 0.5, [0.5, 0.9])
    assert result == pytest.approx({
        "tail_threshold_pu": 3.0, "tail_frac_%": 100 * 5 / 12,
        "tail_rmse_pu": 1.0, "tail_mae_pu": 1.0,
        "energy_err_mean_kwh": 1.0, "energy_err_p95_kwh": 1.0,
        "mape_ramp_low_%": 100 * (1 + 1 / 2 + 1 / 3) / 3,
        "mape_ramp_mid_%": 100 * (1 / 2 + 1 / 4 + 1 / 6) / 3,
        "mape_ramp_steep_%": 100 * (1 / 4 + 1 / 8 + 1 / 12) / 3,
        "n_windows_low": 2, "n_windows_mid": 1, "n_windows_steep": 1,
    })


@pytest.mark.parametrize("tail_from", ["train", "test"])
def test_peak_cli_reuses_targets_and_truth_selections(tmp_path, monkeypatch, tail_from):
    """Two cached models must share target reads and never decompress features."""
    import pickle
    import pandas as pd
    from scripts import compute_peak_metrics as peak

    truth = np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6], [4, 8, 12]], dtype=np.float32)
    targets = np.stack([truth * 2, truth], axis=-1)
    training = np.array([0, 2, 4, 6], dtype=np.float32).reshape(4, 1, 1)
    training = np.broadcast_to(training, targets.shape)
    windows_path = tmp_path / "windows.npz"
    np.savez_compressed(windows_path, Yte=targets, Ytr=training,
                        Xte=np.zeros((4, 8, 2)), Xva=np.zeros((2, 8, 2)))
    meta = {"Horizon forecasting window H": 3, "targetNames": ["P", "Q"],
            "MU": pd.Series([0, 0], index=["P", "Q"]),
            "SIG": pd.Series([1, 1], index=["P", "Q"])}
    with (tmp_path / "scalers_and_meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)
    specs = {"cnn": {"label": "CNN"}, "tcn": {"label": "TCN"}}
    expected = {}
    reference = training[:, :, 1] if tail_from == "train" else truth
    for offset, (key, spec) in enumerate(specs.items(), start=1):
        save_predictions(tmp_path, key, spec["label"], targets + offset)
        expected[spec["label"]] = peak.peak_metrics(
            truth, truth + offset, reference, 1200, 0.5, [0.5, 0.9])

    # Observe actual NPZ member reads, including repeated lazy decompression.
    with np.load(windows_path) as probe:
        archive_type = type(probe)
    original_getitem = archive_type.__getitem__
    reads = []

    def track_reads(self, key):
        if str(self.zip.filename) == str(windows_path):
            assert key not in {"Xte", "Xva"}, "Cached predictions must not read features"
            reads.append(key)
        return original_getitem(self, key)

    original_quantile = np.quantile
    quantiles = []

    def track_quantile(values, q, *args, **kwargs):
        quantiles.append(np.asarray(q).tolist())
        return original_quantile(values, q, *args, **kwargs)

    monkeypatch.setattr(archive_type, "__getitem__", track_reads)
    monkeypatch.setattr(np, "quantile", track_quantile)
    monkeypatch.setattr(peak, "load_model_specs", lambda path: (specs, {}))
    monkeypatch.setattr("sys.argv", ["compute_peak_metrics.py", "--archive", str(tmp_path),
                                   "--model-specs", "unused.json", "--rating-kw", "1200",
                                   "--tail-q", "0.5", "--tail-from", tail_from, "--d-plot", "1"])
    peak.main()

    suffix = "_testtail" if tail_from == "test" else ""
    result = pd.read_csv(tmp_path / f"peak_metrics{suffix}.csv", index_col=0)
    for label, metrics in expected.items():
        assert result.loc[label].to_dict() == pytest.approx(metrics)
    assert reads == ["Yte", "Ytr"]
    assert quantiles == [0.5, [0.5, 0.9], 0.95, 0.95]


def test_flat_model_saves_val_and_test_predictions(tmp_path):
    pytest.importorskip("sklearn")
    from load_forecasting.models.baselines import build_ridge_baseline
    from load_forecasting.training import train_eval_flat_model

    rng = np.random.default_rng(0)
    L, H, F = 8, 3, 2
    Xtr, Ytr = rng.normal(size=(40, L, F)), rng.normal(size=(40, H, 1))
    Xva = rng.normal(size=(12, L, F))
    Xte, Yte = rng.normal(size=(15, L, F)), rng.normal(size=(15, H, 1))

    yhat_phys, metrics = train_eval_flat_model(
        "ridge", "Ridge", build_ridge_baseline(),
        Xtr, Ytr, Xte, Yte, (H, 1),
        run_dir=str(tmp_path),
        inv_scale=lambda z: z,
        y_true_phys=Yte,
        d_plot=0,
        log_time=lambda *args: None,
        Xva=Xva,
    )

    saved = np.load(tmp_path / "predictions" / "ridge.npz")
    assert saved["yhat_test_phys"].shape == (15, H, 1)
    assert saved["yhat_val_phys"].shape == (12, H, 1)
    np.testing.assert_allclose(saved["yhat_test_phys"], yhat_phys.astype(np.float32), rtol=1e-5)


def test_inference_timer_excludes_validation_saving_and_scoring(tmp_path, monkeypatch, capsys):
    from load_forecasting import training
    from load_forecasting.metrics import compute_metrics

    clock = [0.0]

    def spend(seconds):
        clock[0] += seconds

    class Model:
        def fit(self, X, Y):
            spend(31)

        def predict(self, X):
            spend(2 if len(X) == 2 else 7)  # Test and validation have distinct costs.
            return np.ones((len(X), 2))

    def inverse(values):
        spend(3)
        return values

    def score(*args, **kwargs):
        assert kwargs["include_confidence"] is True
        spend(17)
        return compute_metrics(*args, **kwargs)

    monkeypatch.setattr(training, "tmer", lambda: clock[0])
    monkeypatch.setattr(training.joblib, "dump", lambda *args: spend(5))
    monkeypatch.setattr(training, "save_predictions", lambda *args: spend(11))
    monkeypatch.setattr(training, "compute_metrics", score)
    training.train_eval_flat_model(
        "fake", "Fake", Model(), np.ones((4, 3, 1)), np.ones((4, 2, 1)),
        np.ones((2, 3, 1)), np.ones((2, 2, 1)), (2, 1),
        run_dir=tmp_path, inv_scale=inverse, y_true_phys=np.ones((2, 2, 1)),
        d_plot=0, log_time=lambda *args: None, Xva=np.ones((3, 3, 1)),
    )
    output = capsys.readouterr().out
    assert "Fake test prediction time (s): 2.0" in output
    assert "Fake scoring time (s): 17.0" in output


@pytest.mark.parametrize("variant", ["standard", "aliases", "hybrid", "custom_loss", "string_loss"])
def test_keras_wrappers_preserve_training_contracts(tmp_path, monkeypatch, capsys, variant):
    """Exercise both wrappers while tracking options, outputs, and timing boundaries."""
    import sys
    from types import SimpleNamespace
    from load_forecasting import training

    # Lightweight backend objects let this contract test run without TensorFlow.
    keras = SimpleNamespace(
        optimizers=SimpleNamespace(Adam=lambda **kwargs: SimpleNamespace(**kwargs)),
        losses=SimpleNamespace(Huber=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(keras=keras))
    clock, calls = [0.0], {}
    history = SimpleNamespace(history={"loss": [1.0], "val_loss": [2.0]})
    xtr, ytr, xva, yva, xte = [np.full((3, 2, 1), value) for value in range(1, 6)]
    truth = np.full((3, 2, 1), 9.0)
    callbacks = [object()]
    metrics = {"RMSE_horizon": np.ones(2), "RMSE_all": 1.0, "MAE_all": 1.0, "MAPE_all": 1.0}

    class Model:
        def compile(self, **kwargs):
            calls["compile"] = kwargs

        def summary(self):
            calls["summary"] = True

        def fit(self, X, Y, **kwargs):
            assert X is xtr and Y is ytr
            calls["fit"] = kwargs
            clock[0] += 31
            return history

        def save(self, path):
            calls["save"] = path
            clock[0] += 11

        def predict(self, X, **kwargs):
            assert X is xte
            calls["predict"] = kwargs
            clock[0] += 2
            return np.full_like(truth, 4.0)

    def inverse(values):
        clock[0] += 3
        return values * 2

    def score(y_true, y_hat, d_plot, **kwargs):
        assert y_true is truth and d_plot == 0
        assert kwargs == {"include_confidence": True}
        np.testing.assert_array_equal(y_hat, np.full_like(truth, 8.0))
        clock[0] += 17
        return metrics

    def save_plot(directory, key):
        calls["plot"] = (directory, key)
        training.plt.close()

    monkeypatch.setattr(training, "tmer", lambda: clock[0])
    monkeypatch.setattr(training, "compute_metrics", score)
    kwargs = dict(
        model=Model(), Xtr=xtr, Ytr=ytr, Xva=xva, Yva=yva, Xte=xte, Yte=None,
        run_dir=tmp_path, inv_scale=inverse, y_true_phys=truth, d_plot=0,
        callbacks=callbacks, predict_batch_size=7, imgs_dir=tmp_path,
        save_plot=save_plot, log_time=lambda key, start: calls.update(log=(key, start)),
    )
    hybrid = variant in {"hybrid", "custom_loss", "string_loss"}
    if variant == "aliases":
        kwargs.update(artifact_key="weights", plot_key="curve", log_key="timing")
    elif variant in {"custom_loss", "string_loss"}:
        kwargs.update(loss=object() if variant == "custom_loss" else "mae",
                      lr=0.02, batch_size=4, epochs=3, shuffle=True)
    if hybrid:
        returned_history, prediction, returned_metrics = training.train_eval_hybrid(model_name="Hybrid", **kwargs)
        assert returned_history is history
    else:
        prediction, returned_metrics = training.train_eval_keras_model(model_key="mlp", model_label="MLP", **kwargs)

    assert returned_metrics is metrics
    np.testing.assert_array_equal(prediction, np.full_like(truth, 8.0))
    assert calls["summary"] is True
    assert calls["fit"] == dict(epochs=kwargs.get("epochs", 40), batch_size=kwargs.get("batch_size", 128),
                                shuffle=kwargs.get("shuffle", False), validation_data=(xva, yva),
                                callbacks=callbacks, verbose=1)
    assert calls["predict"] == {"batch_size": 7}
    assert calls["compile"]["optimizer"].learning_rate == kwargs.get("lr", 7e-4 if hybrid else 1e-3)
    if variant == "hybrid":
        assert calls["compile"]["loss"].delta == 1.0
    elif hybrid:
        assert calls["compile"]["loss"] is kwargs["loss"]
    else:
        assert calls["compile"]["loss"] == "mse"
    artifact = "Hybrid" if hybrid else ("weights" if variant == "aliases" else "mlp")
    plot = "hybrid" if hybrid else ("curve" if variant == "aliases" else "mlp")
    log = "Hybrid" if hybrid else ("timing" if variant == "aliases" else "mlp")
    assert calls["save"] == str(tmp_path / f"model_{artifact}.keras")
    assert calls["plot"] == (tmp_path, plot)
    assert calls["log"] == (f"model_{log}_train_eval", 0.0)
    label = "Hybrid" if hybrid else "MLP"
    output = capsys.readouterr().out
    assert f"{label} test prediction time (s): 2.0" in output
    assert f"{label} scoring time (s): 17.0" in output
