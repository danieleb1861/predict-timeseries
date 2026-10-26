"""Collect leave-one-voyage-out results into the statements the paper makes.

Reads every <run-root>/fold*/ produced by

    python scripts/run_seed_sweep.py --lovo-cross-val N ...

and answers, per held-out voyage:

1. Aggregate accuracy. Does any trained model beat the naive persistence
   forecast on aggregate MAPE? The reported single split says no; the question
   is whether that survives holding out each voyage in turn.
2. Upper-tail skill. Is the persistence-relative skill S_E = 1 - E_model/E_pers
   positive on RMSE^95 and on the other event-conditioned metrics, and in how
   many of the N held-out voyages?

Two upper-tail definitions are reported wherever both are available, because a
single voyage often sits wholly below the 95th percentile of the multi-voyage
training set, leaving the train-referenced tail empty for that fold:

- ``train`` tail (peak_metrics.csv / persistence_skill.csv): the manuscript
  definition, comparable with the reported single-split table, but estimable
  only in the folds whose held-out voyage actually exceeds the threshold.
- ``legtail`` (peak_metrics_testtail.csv / persistence_skill_testtail.csv,
  written by the two peak scripts under --tail-from test): the held-out voyage's
  own 95th percentile, so the tail is 5 % populated in every fold. This measures
  whether the model tracks each voyage's own peaks rather than whether the
  voyage is a high-load one.

``tail_frac_%`` is carried into the per-fold table so an empty or barely
populated tail is visible rather than silently dropped.

Persistence is recomputed per fold from the fold's own cached test windows, so
the baseline always sees exactly the windows the models were scored on.

Outputs, written to <run-root>:
- lovo_by_fold.csv  : one row per (fold, model), every metric
- lovo_summary.csv  : one row per (model, metric) with mean, std, and the count
                      of folds in which the model beats persistence

Example:
    python scripts/collect_lovo_results.py \
        --run-root outputs/runs/lovo_manoeuvre_benchmark_flax
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from load_forecasting.metrics import compute_metrics  # noqa: E402
from load_forecasting.scaling import inverse_scale_targets  # noqa: E402
from compute_persistence_peak_metrics import persistence_forecast  # noqa: E402

# Aggregate metrics, recomputed for persistence from the fold's own windows.
AGGREGATE_METRICS = ("MAPE_all", "RMSE_all", "MAE_all")
# Event-conditioned skills, read from the persistence_skill*.csv of each fold.
TAIL_SKILLS = ("S_E_tail_rmse_pu", "S_E_tail_mae_pu", "S_E_energy_err_mean_kwh",
               "S_E_energy_err_p95_kwh", "S_E_mape_ramp_steep_%")
# (file suffix, column suffix) for the two upper-tail definitions.
TAIL_VARIANTS = (("", ""), ("_testtail", "__legtail"))


def persistence_aggregate(fold_dir: Path) -> tuple[dict, int]:
    """Aggregate metrics of the naive forecast on this fold's test windows."""
    windows = np.load(fold_dir / "windows.npz")
    with (fold_dir / "scalers_and_meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    y_true = inverse_scale_targets(windows["Yte"], meta["targetNames"], meta["MU"], meta["SIG"])
    y_pers = persistence_forecast(windows, meta, y_true.shape[1])[:, :, None]
    # Only aggregate scores enter these reports; window-level CIs are unused.
    return compute_metrics(y_true, y_pers, 0, include_confidence=False), int(y_true.shape[0])


def fold_index(fold_dir: Path) -> int:
    """Fold index from run_summary.json, falling back to the directory name."""
    summary = fold_dir / "run_summary.json"
    if summary.exists():
        recorded = json.loads(summary.read_text(encoding="utf-8")).get("lovo_fold")
        if recorded is not None:
            return int(recorded)
    digits = "".join(c for c in fold_dir.name if c.isdigit())
    if not digits:
        raise ValueError(f"Cannot determine the fold index of {fold_dir}")
    return int(digits)


def read_optional(path: Path):
    return pd.read_csv(path, index_col=0) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="Directory containing the fold*/ run dirs.")
    parser.add_argument("--pattern", default="fold*", help="Glob for the per-fold dirs (default: fold*).")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    fold_dirs = sorted(
        (d for d in run_root.glob(args.pattern) if (d / "results_metrics.csv").exists()),
        key=fold_index,
    )
    if not fold_dirs:
        raise SystemExit(f"No fold dirs with results_metrics.csv under {run_root} (pattern {args.pattern!r}).")

    rows = []
    missing = {col_suffix: [] for _, col_suffix in TAIL_VARIANTS}
    for fold_dir in fold_dirs:
        fold = fold_index(fold_dir)
        models = pd.read_csv(fold_dir / "results_metrics.csv", index_col=0)
        pers, n_test = persistence_aggregate(fold_dir)

        variants = {}
        for file_suffix, col_suffix in TAIL_VARIANTS:
            skill = read_optional(fold_dir / f"persistence_skill{file_suffix}.csv")
            coverage = read_optional(fold_dir / f"peak_metrics_with_persistence{file_suffix}.csv")
            if skill is None:
                missing[col_suffix].append(fold_dir.name)
                continue
            variants[col_suffix] = (skill, coverage)

        for model in models.index:
            row = {"fold": fold, "held_out_leg": fold, "n_test_windows": n_test, "model": model}
            for metric in AGGREGATE_METRICS:
                row[metric] = float(models.loc[model, metric])
                row[f"pers_{metric}"] = float(pers[metric])
                row[f"S_E_{metric}"] = 1.0 - row[metric] / float(pers[metric])
            for col_suffix, (skill, coverage) in variants.items():
                if coverage is not None and model in coverage.index:
                    for col in ("tail_threshold_pu", "tail_frac_%"):
                        row[f"{col}{col_suffix}"] = float(coverage.loc[model, col])
                if model in skill.index:
                    for col in TAIL_SKILLS:
                        if col in skill.columns:
                            row[f"{col}{col_suffix}"] = float(skill.loc[model, col])
            rows.append(row)

    by_fold = pd.DataFrame(rows)
    by_fold_csv = run_root / "lovo_by_fold.csv"
    by_fold.to_csv(by_fold_csv, index=False)

    # Per-model summary: dispersion across folds, and how often the model wins.
    skill_cols = [f"S_E_{m}" for m in AGGREGATE_METRICS]
    for _, col_suffix in TAIL_VARIANTS:
        skill_cols += [f"{c}{col_suffix}" for c in TAIL_SKILLS if f"{c}{col_suffix}" in by_fold.columns]
    summary_rows = []
    for model, group in by_fold.groupby("model", sort=True):
        for col in skill_cols:
            values = group[col].dropna().to_numpy() if col in group.columns else np.array([])
            if values.size == 0:
                continue
            summary_rows.append({
                "model": model,
                "metric": col,
                "n_folds": int(values.size),
                "n_positive": int((values > 0).sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "min": float(values.min()),
                "max": float(values.max()),
            })
    summary = pd.DataFrame(summary_rows)
    summary_csv = run_root / "lovo_summary.csv"
    summary.to_csv(summary_csv, index=False)

    n_folds = len(fold_dirs)
    print(f"Folds: {n_folds} ({', '.join(d.name for d in fold_dirs)})")
    print(f"Test windows per fold: min {by_fold['n_test_windows'].min()}, "
          f"median {int(by_fold['n_test_windows'].median())}, max {by_fold['n_test_windows'].max()}")
    for col_suffix, names in missing.items():
        if names:
            label = "legtail" if col_suffix else "train-referenced"
            print(f"WARNING: no {label} persistence_skill in {len(names)} fold(s): {', '.join(names)}")

    # Upper-tail coverage, the reason the two definitions are reported side by side.
    for _, col_suffix in TAIL_VARIANTS:
        col = f"tail_frac_%{col_suffix}"
        if col not in by_fold.columns:
            continue
        per_fold = by_fold.groupby("fold")[col].first()
        label = "held-out voyage p95" if col_suffix else "training-set p95"
        print(f"\nUpper-tail coverage, {label}: "
              f"{(per_fold > 0).sum()}/{len(per_fold)} folds non-empty, "
              f"median {per_fold.median():.2f} % of test points "
              f"(range {per_fold.min():.2f}-{per_fold.max():.2f} %)")

    for col in skill_cols:
        part = summary[summary["metric"] == col]
        if part.empty:
            continue
        label = col.replace("S_E_", "")
        print(f"\nS_E on {label}: folds with positive skill, out of {n_folds}")
        for _, r in part.sort_values("n_positive", ascending=False).iterrows():
            print(f"  {r['model']:<18} {int(r['n_positive']):>2}/{int(r['n_folds']):<2}  "
                  f"mean {r['mean']:+.4f}  std {r['std']:.4f}  "
                  f"range [{r['min']:+.4f}, {r['max']:+.4f}]")

    print(f"\nWrote {by_fold_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
