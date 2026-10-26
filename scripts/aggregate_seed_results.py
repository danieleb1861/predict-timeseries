"""Aggregate Parquet or legacy CSV metrics across seed run directories.

Reads <run-root>/seed*/results_metrics.{parquet,csv} and writes:
- <run-root>/aggregated_metrics.csv : per-model mean/std of the scalar metrics
  (RMSE_all, MAE_all, MAPE_all) across seeds, plus the per-seed values.
- <run-root>/aggregated_horizon.npz : per-model elementwise mean/std of the
  horizon-wise metric arrays (keys: "<model>/<metric>/mean" and ".../std").

Across-seed dispersion is reported as mean +/- sample std (ddof=1): with 3-5
seeds a bootstrap over seeds would be meaningless. Per-window bootstrap CIs
(test-set resampling) are a separate, within-run analysis that requires
persisted predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Keep direct script execution working without an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_forecasting.metrics import (  # noqa: E402
    HORIZON_METRICS,
    load_metrics_outputs,
    parse_metric_vector as parse_array_cell,
)

SCALAR_METRICS = ("RMSE_all", "MAE_all", "MAPE_all")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="Directory containing the seed*/ run dirs.")
    parser.add_argument("--pattern", default="seed*", help="Glob for the per-seed dirs (default: seed*).")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seed_dirs = sorted(d for d in run_root.glob(args.pattern)
                       if any((d / f"results_metrics.{ext}").is_file() for ext in ("parquet", "csv")))
    if len(seed_dirs) < 2:
        raise SystemExit(f"Need at least 2 seed dirs with Parquet or CSV metrics under {run_root}, found {len(seed_dirs)}.")

    # Validate every seed before writing either aggregate report.
    frames = {}
    horizon = None
    for directory in seed_dirs:
        frame = load_metrics_outputs(directory, expected_horizon=horizon)
        missing = set(SCALAR_METRICS + HORIZON_METRICS) - set(frame.columns)
        if missing:
            raise ValueError(f"{directory}: missing metrics {sorted(missing)}")
        horizon = frame.attrs["horizon_steps"]
        frames[directory.name] = frame
    model_sets = {name: set(df.index) for name, df in frames.items()}
    common_models = set.intersection(*model_sets.values())
    for name, models in model_sets.items():
        extra = models - common_models
        if extra:
            print(f"Warning: {name} has models missing from other seeds (excluded): {sorted(extra)}")
    print(f"Seeds: {', '.join(frames)} | models common to all seeds: {len(common_models)}")

    # Scalar metrics: mean/std across seeds.
    rows = []
    for model in sorted(common_models):
        row = {"model": model, "n_seeds": len(frames)}
        for metric in SCALAR_METRICS:
            values = np.array([float(frames[s].loc[model, metric]) for s in frames])
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_values"] = ";".join(f"{v:.6g}" for v in values)
        rows.append(row)
    out_csv = run_root / "aggregated_metrics.csv"
    pd.DataFrame(rows).set_index("model").to_csv(out_csv)

    # Horizon-wise arrays: elementwise mean/std across seeds.
    array_cols = HORIZON_METRICS
    arrays = {}
    for model in sorted(common_models):
        for metric in array_cols:
            stacked = np.stack([frames[s].loc[model, metric] for s in frames])
            arrays[f"{model}/{metric}/mean"] = stacked.mean(axis=0)
            arrays[f"{model}/{metric}/std"] = stacked.std(axis=0, ddof=1)
    out_npz = run_root / "aggregated_horizon.npz"
    np.savez_compressed(out_npz, **arrays)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_npz} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
