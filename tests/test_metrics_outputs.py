import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_forecasting.metrics import load_metrics_outputs, parse_metric_vector, save_metrics_outputs


def test_save_metrics_outputs_writes_typed_parquet_and_compact_csv(tmp_path):
    metrics = {
        "Ridge": {
            "RMSE_horizon": np.array([0.1, 0.2]),
            "MAE_horizon": np.array([0.08, 0.15]),
            "MAPE_horizon": np.array([1.0, 2.0]),
            "MAPE_targetax": [1.25, 1.75, 2.25],
            "MAPE_boot_mean": 1.75,
            "MAPE_boot_ci95_low": 1.25,
            "MAPE_boot_ci95_high": 2.25,
            "RMSE_all": 0.16,
            "MAE_all": 0.12,
            "MAPE_all": 1.5,
        }
    }

    frame = save_metrics_outputs(metrics, tmp_path)
    parquet = pd.read_parquet(tmp_path / "results_metrics.parquet")
    csv = pd.read_csv(tmp_path / "results_metrics.csv", index_col=0)

    assert frame.index.tolist() == ["Ridge"]
    assert parquet.index.tolist() == ["Ridge"]
    np.testing.assert_allclose(parquet.loc["Ridge", "RMSE_horizon"], [0.1, 0.2])
    np.testing.assert_allclose(parquet.loc["Ridge", "MAPE_targetax"], [1.25, 1.75, 2.25])
    assert "MAPE_targetax" not in csv.columns
    assert float(csv.loc["Ridge", "MAPE_all"]) == 1.5
    assert "RMSE_horizon" in csv.columns


def test_benchmark_archive_keeps_both_metric_formats_and_public_summary(tmp_path, monkeypatch):
    import run_benchmark

    results_dir = tmp_path / "archives"
    results_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in (
        "results_metrics.csv",
        "results_metrics.parquet",
        "run_summary.json",
        "run_summary_public.json",
    ):
        (run_dir / name).write_bytes(name.encode("utf-8"))

    monkeypatch.setattr(run_benchmark, "RESULTS_DIR", results_dir)
    archive = run_benchmark.archive_current_run("20260823", "Navigation", "L50_H25", run_dir)

    for name in (
        "results_metrics.csv",
        "results_metrics.parquet",
        "run_summary_public.json",
    ):
        assert (archive / name).read_bytes() == name.encode("utf-8")
    assert not (archive / "run_summary.json").exists()


def _sample_metrics(horizon=3):
    """Small writer input shared by format and caller regression tests."""
    return {"CNN": {
        "RMSE_horizon": np.arange(1, horizon + 1) / 10,
        "MAE_horizon": np.arange(1, horizon + 1) / 20,
        "MAPE_horizon": np.arange(1, horizon + 1, dtype=float),
        "MAPE_targetax": [1.0, 2.0],
        "RMSE_all": 0.2, "MAE_all": 0.1, "MAPE_all": 2.0,
    }}


@pytest.mark.parametrize("value", ["[0.1, 0.2, 0.3]", "[0.1  0.2\n0.3]", [0.1, 0.2, 0.3]])
def test_metric_vector_preserves_every_lead(value):
    # Current CSV lists and older NumPy strings must decode identically.
    np.testing.assert_allclose(parse_metric_vector(value, 3), [0.1, 0.2, 0.3])


@pytest.mark.parametrize("value", [
    "[0.1, broken, 0.3]", "[0.1 broken]", "[0.1, ,0.3]", "[0.1 0.2",
    "[]", "", "[nan, 0.2]", "[inf 0.2]", "[[0.1, 0.2]]", 0.1, None,
])
def test_metric_vector_rejects_malformed_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        parse_metric_vector(value)


@pytest.mark.parametrize("csv_only", [False, True])
def test_metric_reader_round_trips_writer_output(tmp_path, csv_only):
    save_metrics_outputs(_sample_metrics(), tmp_path)
    if csv_only:
        (tmp_path / "results_metrics.parquet").unlink()
    else:
        # A stale CSV must not override the canonical typed artifact.
        (tmp_path / "results_metrics.csv").write_text("invalid CSV", encoding="utf-8")
    frame = load_metrics_outputs(tmp_path, expected_horizon=3)
    np.testing.assert_allclose(frame.loc["CNN", "RMSE_horizon"], [0.1, 0.2, 0.3])
    assert frame.loc["CNN", "MAPE_all"] == 2.0
    assert "MAPE_targetax" not in frame.columns
    assert frame.attrs["horizon_steps"] == 3


def test_metric_reader_rejects_wrong_expected_horizon(tmp_path):
    save_metrics_outputs(_sample_metrics(), tmp_path)
    with pytest.raises(ValueError, match="CNN/RMSE_horizon: Metric horizon 3 != expected 25"):
        load_metrics_outputs(tmp_path, expected_horizon=25)


@pytest.mark.parametrize("other_model", [False, True])
def test_metric_reader_rejects_inconsistent_vectors(tmp_path, other_model):
    metrics = _sample_metrics()
    if other_model:
        metrics["Ridge"] = _sample_metrics(2)["CNN"]
    else:
        metrics["CNN"]["MAE_horizon"] = [0.1, 0.2]
    save_metrics_outputs(metrics, tmp_path)
    with pytest.raises(ValueError, match="Metric horizon 2 != expected 3"):
        load_metrics_outputs(tmp_path)


def test_seed_aggregator_keeps_all_leads_across_both_formats(tmp_path):
    for seed in range(2):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        save_metrics_outputs(_sample_metrics(), directory)
        # Exercise discovery of both CSV-only and Parquet-only runs.
        (directory / f"results_metrics.{'parquet' if seed == 0 else 'csv'}").unlink()
    completed = subprocess.run(
        [sys.executable, "-B", "scripts/aggregate_seed_results.py", "--run-root", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(tmp_path / "aggregated_horizon.npz") as data:
        np.testing.assert_allclose(data["CNN/RMSE_horizon/mean"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(data["CNN/RMSE_horizon/std"], [0.0, 0.0, 0.0])


def test_seed_aggregator_rejects_mismatched_horizons_before_writing(tmp_path, monkeypatch):
    from scripts.aggregate_seed_results import main

    for seed, horizon in enumerate((3, 2)):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        save_metrics_outputs(_sample_metrics(horizon), directory)
    monkeypatch.setattr(sys, "argv", ["aggregate_seed_results.py", "--run-root", str(tmp_path)])
    with pytest.raises(ValueError, match="Metric horizon 2 != expected 3"):
        main()
    assert not (tmp_path / "aggregated_metrics.csv").exists()
    assert not (tmp_path / "aggregated_horizon.npz").exists()


def test_seed_figure_reader_accepts_parquet_only_run(tmp_path):
    from scripts.gen_seed_sweep_figures import load_seed_metrics

    directory = tmp_path / "seed0"
    directory.mkdir()
    save_metrics_outputs(_sample_metrics(), directory)
    (directory / "results_metrics.csv").unlink()
    summary = {"selector": "navigation", "lookback_steps": 20, "horizon_steps": 3,
               "input_files": ["fixture_leg.parquet"]}
    (directory / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    scalar, profiles = load_seed_metrics(tmp_path, "navigation", 20, 3, 1, ["CNN"])
    np.testing.assert_allclose(scalar["CNN"], [2.0])
    np.testing.assert_allclose(profiles["CNN"], [[1.0, 2.0, 3.0]])


def test_shared_point_metrics_preserve_standard_bootstrap_contract():
    from load_forecasting.metrics import bootstrap_mean_ci, compute_metrics, compute_point_metrics

    truth = np.array([[[2.], [4.]], [[4.], [8.]]])
    prediction = np.array([[[3.], [2.]], [[3.], [10.]]])
    points = compute_point_metrics(truth, prediction, 0)
    result = compute_metrics(truth, prediction, 0)
    # Existing training callers retain the same intervals and CSV column order.
    interval = bootstrap_mean_ci([50., 25.])
    assert list(result) == ["RMSE_horizon", "MAE_horizon", "MAPE_horizon", "MAPE_targetax",
                            "MAPE_boot_mean", "MAPE_boot_ci95_low", "MAPE_boot_ci95_high",
                            "RMSE_all", "MAE_all", "MAPE_all"]
    for key, value in points.items():
        np.testing.assert_array_equal(result[key], value)
    assert result["MAPE_boot_mean"] == interval["mean"]
    assert result["MAPE_boot_ci95_low"] == interval["low"]
    assert result["MAPE_boot_ci95_high"] == interval["high"]


def test_point_only_scoring_skips_bootstrap_and_preserves_scores(monkeypatch):
    from load_forecasting import metrics

    truth = np.array([[[2.], [4.]], [[4.], [8.]]])
    prediction = np.array([[[3.], [2.]], [[3.], [10.]]])
    default = metrics.compute_metrics(truth, prediction, 0)
    explicit = metrics.compute_metrics(truth, prediction, 0, include_confidence=True)
    for key in default:
        np.testing.assert_array_equal(explicit[key], default[key])

    def unexpected_bootstrap(*args, **kwargs):
        raise AssertionError("Point-only consumers must not run bootstrap")

    monkeypatch.setattr(metrics, "bootstrap_mean_ci", unexpected_bootstrap)
    points = metrics.compute_metrics(truth, prediction, 0, include_confidence=False)
    assert not any(key.startswith("MAPE_boot") for key in points)
    for key, value in points.items():
        np.testing.assert_array_equal(value, default[key])


def test_lovo_persistence_scores_do_not_compute_unused_intervals(tmp_path, monkeypatch):
    import pickle
    from load_forecasting import metrics
    from scripts.collect_lovo_results import persistence_aggregate

    # A synthetic fold exercises the caller, including inverse scaling.
    np.savez(tmp_path / "windows.npz", Xte=np.array([[[0.], [1.]], [[1.], [2.]]]),
             Yte=np.array([[[2.], [3.]], [[3.], [4.]]]))
    metadata = {"targetNames": ["power"], "MU": pd.Series({"power": 10.}),
                "SIG": pd.Series({"power": 2.})}
    with (tmp_path / "scalers_and_meta.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)

    def unexpected_bootstrap(*args, **kwargs):
        raise AssertionError("The LOVO report does not consume confidence intervals")

    monkeypatch.setattr(metrics, "bootstrap_mean_ci", unexpected_bootstrap)
    result, count = persistence_aggregate(tmp_path)
    assert count == 2
    assert result["MAE_all"] == 3.0
    assert result["RMSE_all"] == pytest.approx(np.sqrt(10))
    assert result["MAPE_all"] == pytest.approx(np.mean([2 / 14, 4 / 16, 2 / 16, 4 / 18]) * 100)
    assert not any(key.startswith("MAPE_boot") for key in result)
