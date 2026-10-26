"""Forecast evaluation metrics."""

from pathlib import Path

import numpy as np
import pandas as pd


CSV_EXCLUDED_METRICS = ("MAPE_targetax",)
HORIZON_METRICS = ("RMSE_horizon", "MAE_horizon", "MAPE_horizon")


def parse_metric_vector(value, expected_horizon=None):
    """Decode a finite 1D vector from typed metrics or either legacy CSV format."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.endswith("]"):
            if not (text.startswith("[") and text.endswith("]")):
                raise ValueError("Unmatched brackets in metric vector")
            text = text[1:-1].strip()
        # Convert every token: fromstring can silently stop at a comma or bad value.
        value = text.split(",") if "," in text else text.split()
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Metric vector contains non-numeric values") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Expected a non-empty, finite 1D metric vector")
    if expected_horizon is not None and vector.size != expected_horizon:
        raise ValueError(f"Metric horizon {vector.size} != expected {expected_horizon}")
    return vector


def load_metrics_outputs(run_dir, *, expected_horizon=None):
    """Read Parquet first, otherwise CSV, and validate all available horizon vectors."""
    run_dir = Path(run_dir)
    path = run_dir / "results_metrics.parquet"
    if path.is_file():
        from pyarrow.parquet import read_schema

        # Per-window errors can be large and are not needed for horizon reports.
        columns = [name for name in read_schema(path).names if name not in CSV_EXCLUDED_METRICS]
        frame = pd.read_parquet(path, columns=columns)
    else:
        path = run_dir / "results_metrics.csv"
        frame = pd.read_csv(path, index_col=0, usecols=lambda name: name not in CSV_EXCLUDED_METRICS)

    horizon = expected_horizon
    for metric in HORIZON_METRICS:
        if metric not in frame.columns:
            continue
        vectors = []
        for model, value in frame[metric].items():
            try:
                vector = parse_metric_vector(value, horizon)
            except ValueError as exc:
                raise ValueError(f"{path}: {model}/{metric}: {exc}") from exc
            if horizon is None:
                horizon = vector.size
            vectors.append(vector)
        frame[metric] = pd.Series(vectors, index=frame.index, dtype=object)
    # Callers without run metadata can still enforce consistency across seed files.
    frame.attrs["horizon_steps"] = horizon
    return frame


def bootstrap_mean_ci(values, n_boot=1000, ci=95, seed=42):
    """Return a bootstrap confidence interval for the sample mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap_mean_ci expects a non-empty 1D array")

    rng = np.random.default_rng(seed)
    n = values.size
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = np.mean(rng.choice(values, size=n, replace=True))

    alpha = (100.0 - ci) / 2.0
    return {
        "mean": float(np.mean(boot)),
        "low": float(np.percentile(boot, alpha)),
        "high": float(np.percentile(boot, 100.0 - alpha)),
    }


def compute_point_metrics(y_true, y_pred, d_plot):
    """
    Compute regression metrics without bootstrap intervals for a selected target.

    Inputs
    ----------
    y_true : array-like
        Ground truth values dim(y_true) = (num_windows, HorizonSteps, output_dim).
    y_pred : array-like
        Predicted values dim(y_pred) = dim(y_true).
    d_plot : int, default=0
        Target feature to evaluate.

    Returns
    -------
    dict
        Keys:
          - RMSE_horizon : np.ndarray, shape (HorizonSteps,)
          - MAE_horizon  : np.ndarray, shape (HorizonSteps,)
          - MAPE_horizon : np.ndarray, shape (HorizonSteps,)
          - MAPE_targetax: np.ndarray, shape (num_windows,)  (mean over horizon)
          - RMSE_all     : float
          - MAE_all      : float
          - MAPE_all     : float
    """
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise Warning(f"Expected (num_windows,HorizonSteps,output_dim). Got {y_true.shape} vs {y_pred.shape}")

    y_true = np.asarray(y_true[:, :, d_plot], dtype=np.float64) # (num_windows, HorizonSteps)
    y_pred = np.asarray(y_pred[:, :, d_plot], dtype=np.float64) # (num_windows, HorizonSteps)

    eps = 1e-6
    e = y_pred - y_true
    abs_pct_err = np.abs(e) / np.maximum(np.abs(y_true), eps) * 100.0
    mape_targetax = np.mean(abs_pct_err, axis=1)

    return {
        "RMSE_horizon":  np.sqrt(np.mean(e**2, axis=0)), # (HorizonSteps,) mean over num_windows
        "MAE_horizon":   np.mean(np.abs(e), axis=0),     # (HorizonSteps,) mean over num_windows
        "MAPE_horizon":  np.mean(abs_pct_err, axis=0), # (HorizonSteps, ) mean over num_windows
        "MAPE_targetax": mape_targetax.tolist(), # (num_windows, ) mean over horizon
        "RMSE_all":      np.sqrt(np.mean(e**2)),
        "MAE_all":       float(np.mean(np.abs(e))),
        "MAPE_all":      float(np.mean(abs_pct_err))
    }


def compute_metrics(y_true, y_pred, d_plot, *, include_confidence=True):
    """Return point metrics, optionally adding seeded window-level MAPE intervals.

    Confidence intervals remain enabled by default. Disable them for consumers
    that only use point scores; the three MAPE_boot keys are then omitted.
    """
    point_metrics = compute_point_metrics(y_true, y_pred, d_plot)
    if not include_confidence:
        return point_metrics
    mape_ci = bootstrap_mean_ci(point_metrics["MAPE_targetax"])
    metrics = {}
    for key, value in point_metrics.items():
        metrics[key] = value
        # Preserve the historical column order in saved metric tables.
        if key == "MAPE_targetax":
            metrics.update({
                "MAPE_boot_mean": mape_ci["mean"],
                "MAPE_boot_ci95_low": mape_ci["low"],
                "MAPE_boot_ci95_high": mape_ci["high"],
            })
    return metrics


def save_metrics_outputs(model_metrics, run_dir):
    """Write canonical typed Parquet metrics and a compact compatibility CSV."""
    rows = {
        model: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        for model, metrics in model_metrics.items()
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "model"

    run_dir = Path(run_dir)
    frame.to_parquet(run_dir / "results_metrics.parquet", index=True)
    frame.drop(columns=list(CSV_EXCLUDED_METRICS), errors="ignore").to_csv(
        run_dir / "results_metrics.csv",
        index=True,
    )
    return frame
