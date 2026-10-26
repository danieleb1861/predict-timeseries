"""Foundation-run planning checks without downloading pretrained checkpoints."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_foundation_models as runner


@pytest.mark.parametrize("key", ["chronos2", "timesfm", "tirex2", "tirex2_univariate"])
@pytest.mark.parametrize("state", ["complete", "partial", "missing", "other_label", "disabled"])
def test_main_loads_only_models_with_pending_archives(tmp_path, monkeypatch, key, state):
    """Skip by label and selected CSV, while preserving adapter options and order."""
    archives = [tmp_path / f"20260919_ForecastModels_Manoeuvre_L{l}_H{h}_flax8"
                for l, h in [(20, 5), (96, 129)]]
    label = key.upper()
    for i, archive in enumerate(archives):
        archive.mkdir()
        # Discovery needs the path; completed runs must never open the windows.
        (archive / "windows.npz").write_bytes(b"must not be read")
        if state != "missing" and not (state == "partial" and i == 0):
            pd.DataFrame({"MAPE_all": [1.0]}, index=["OTHER" if state == "other_label" else label]).to_csv(
                archive / "custom_metrics.csv")
    specs = tmp_path / "specs.json"
    specs.write_text(json.dumps({"deep_models": {key: {"label": label}}}), encoding="utf-8")
    loaded, runs = [], []
    adapter = object()

    def build(**kwargs):
        loaded.append(kwargs)
        return adapter

    def run(actual_adapter, archive, actual_label, **kwargs):
        assert actual_adapter is adapter and actual_label == label
        assert kwargs == {"batch_size": 3, "out_csv_name": "custom_metrics.csv",
                          "skip_existing": state != "disabled"}
        runs.append(archive)

    monkeypatch.setitem(runner.ADAPTER_BUILDERS, key, build)
    monkeypatch.setattr(runner, "run_model", run)
    args = ["--results-dir", str(tmp_path), "--model-specs", str(specs), "--models", key,
            "--out-csv-name", "custom_metrics.csv", "--device", "cpu", "--batch-size", "3",
            "--timesfm-internal-batch-size", "11", "--timesfm-force-flip-invariance",
            "--tirex2-internal-batch-size", "7", "--tirex2-force-sign-flip-invariance",
            "--tirex2-differencing"]
    if state != "disabled":
        args.append("--skip-existing")
    assert runner.main(args) == 0
    expected = [] if state == "complete" else (archives[:1] if state == "partial" else archives)
    assert runs == expected
    assert len(loaded) == int(bool(expected))
    if loaded:
        options = {"device": "cpu"}
        if key == "timesfm":
            # Keep compile dimensions based on all selected archives, including completed ones.
            options.update(per_core_batch_size=11, max_context=96, max_horizon=256,
                           force_flip_invariance=True)
        elif key in {"tirex2", "tirex2_univariate"}:
            options.update(internal_batch_size=7, tta_sign_flip=True, tta_diff=True)
        assert loaded == [options]


def test_run_model_still_skips_completed_direct_calls(tmp_path):
    """Direct callers retain their completion guard without touching the adapter."""
    pd.DataFrame({"MAPE_all": [1.0]}, index=["Chronos"]).to_csv(tmp_path / "metrics.csv")
    runner.run_model(object(), tmp_path, "Chronos", batch_size=3,
                     out_csv_name="metrics.csv", skip_existing=True)


def test_run_model_can_overwrite_completed_results(tmp_path, monkeypatch):
    """Disabling the flag still performs inference and preserves other CSV rows."""
    pd.DataFrame({"MAPE_all": [99.0, 17.0]}, index=["Chronos", "Other"]).to_csv(tmp_path / "metrics.csv")
    history = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    expected = history[:, -1:] + np.arange(2)
    batches = []

    def predict(values, horizon, covariates):
        assert horizon == 2 and covariates is None
        batches.append(len(values))
        return values[:, -1:] + np.arange(horizon)

    monkeypatch.setattr(runner, "build_inputs", lambda *args: (history, None, expected[:, :, None], 2))
    adapter = SimpleNamespace(key="chronos2", covariate_aware=False, predict_batch=predict)
    runner.run_model(adapter, tmp_path, "Chronos", batch_size=2,
                     out_csv_name="metrics.csv", skip_existing=False)
    assert batches == [2, 1]
    with np.load(tmp_path / "predictions/chronos2.npz") as saved:
        np.testing.assert_array_equal(saved["yhat_test_phys"], expected[:, :, None])
    metrics = pd.read_csv(tmp_path / "metrics.csv", index_col=0)
    assert metrics.loc["Chronos", "MAPE_all"] == 0
    assert metrics.loc["Other", "MAPE_all"] == 17
