"""Persistence baseline for the peak-aware/energy metrics, and skill score.

Extends compute_peak_metrics.py's tail/energy/ramp metrics to the naive
persistence forecast (y_hat(t+h) = y(t) for all h), evaluated on the exact
same cached test windows as the seven benchmarked models, then reports the
persistence-relative skill score

    S_E = 1 - E_model / E_pers

for RMSE^95, MAE^95, mean/p95 energy error and steep-ramp MAPE (manuscript
Eq. for S_E, Section "Peak-aware and energy metrics"). Positive S_E means the
model beats persistence on that event-conditioned metric.

Reuses peak_metrics() from compute_peak_metrics.py unmodified, and reads the
existing per-archive peak_metrics.csv (already computed for the seven
models) rather than recomputing them.

Example (single archive):
    python scripts/compute_persistence_peak_metrics.py \
        --archive outputs/test_results/20260721_ForecastModels_Manoeuvre_L60_H25_flax8_legsafe

Example (full legsafe grid, both regimes):
    python scripts/compute_persistence_peak_metrics.py --all-legsafe
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from load_forecasting.scaling import inverse_scale_targets  # noqa: E402
from compute_peak_metrics import default_rating_kw, peak_metrics  # noqa: E402

SKILL_COLS = ("tail_rmse_pu", "tail_mae_pu", "energy_err_mean_kwh",
              "energy_err_p95_kwh", "mape_ramp_steep_%")


def persistence_forecast(windows, meta, H):
    """(N, H) physical-unit persistence forecast from the last observed sample."""
    last_obs_z = windows["Xte"][:, -1, 0]  # last lookback sample, act_pow_pu channel, normalized
    last_obs_phys = inverse_scale_targets(
        last_obs_z[:, None], meta["targetNames"], meta["MU"], meta["SIG"])[:, 0]
    return np.repeat(last_obs_phys[:, None], H, axis=1)


def process_archive(archive: Path, rating_kw: float, tail_q: float, ramp_edges: list[float],
                    tail_from: str = "train"):
    windows = np.load(archive / "windows.npz")
    with (archive / "scalers_and_meta.pkl").open("rb") as f:
        meta = pickle.load(f)

    d = 0
    y_true = inverse_scale_targets(windows["Yte"], meta["targetNames"], meta["MU"], meta["SIG"])[:, :, d]
    y_train = inverse_scale_targets(windows["Ytr"], meta["targetNames"], meta["MU"], meta["SIG"])[:, :, d]
    H = y_true.shape[1]

    y_pers = persistence_forecast(windows, meta, H)
    y_tail_ref = y_train if tail_from == "train" else y_true
    pers_row = peak_metrics(y_true, y_pers, y_tail_ref, rating_kw, tail_q, ramp_edges)

    suffix = "" if tail_from == "train" else "_testtail"
    existing_csv = archive / f"peak_metrics{suffix}.csv"
    if not existing_csv.exists():
        raise FileNotFoundError(
            f"{existing_csv} missing -- run compute_peak_metrics.py for this archive first.")
    models_df = pd.read_csv(existing_csv, index_col=0)

    pers_df = pd.DataFrame([pers_row], index=["Persistence"])
    combined = pd.concat([models_df, pers_df])

    skill = pd.DataFrame(index=models_df.index, columns=[f"S_E_{c}" for c in SKILL_COLS], dtype=float)
    for c in SKILL_COLS:
        skill[f"S_E_{c}"] = 1.0 - models_df[c] / pers_row[c]

    out_dir = archive
    combined.to_csv(out_dir / f"peak_metrics_with_persistence{suffix}.csv")
    skill.to_csv(out_dir / f"persistence_skill{suffix}.csv")
    return combined, skill, pers_row


def latex_row(label, r, best_by_col):
    cells = []
    for c in ("tail_rmse_pu", "tail_mae_pu", "energy_err_mean_kwh", "energy_err_p95_kwh", "mape_ramp_steep_%"):
        v = f"{r[c]:.4f}" if "pu" in c else f"{r[c]:.3f}" if "kwh" in c else f"{r[c]:.2f}"
        cells.append(f"\\textbf{{{v}}}" if np.isclose(r[c], best_by_col[c]) else v)
    return f"{label} & " + " & ".join(cells) + " \\\\"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", help="Single archived benchmark config directory.")
    parser.add_argument("--all-legsafe", action="store_true",
                         help="Process every *_flax8_legsafe / *_flax7_legsafe manoeuvre+navigation archive.")
    parser.add_argument("--rating-kw", type=float, default=None, help="Per-unit base power (default: physical_system.rating_kW from the resolved system config).")
    parser.add_argument("--tail-q", type=float, default=0.95)
    parser.add_argument("--tail-from", choices=("train", "test"), default="train",
                        help="Series the upper-tail quantile is taken from; see "
                             "compute_peak_metrics.py --tail-from.")
    parser.add_argument("--ramp-edges", default="0.5,0.9")
    args = parser.parse_args()

    if args.rating_kw is None:
        args.rating_kw = default_rating_kw()
    ramp_edges = [float(v) for v in args.ramp_edges.split(",")]

    if args.all_legsafe:
        archives = sorted((REPO_ROOT / "outputs" / "test_results").glob("2026072*legsafe"))
    elif args.archive:
        archives = [Path(args.archive)]
    else:
        parser.error("pass --archive or --all-legsafe")

    for archive in archives:
        print(f"\n=== {archive.name} ===")
        try:
            combined, skill, pers_row = process_archive(archive, args.rating_kw, args.tail_q,
                                                        ramp_edges, args.tail_from)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue

        print(combined[list(SKILL_COLS)].round(4).to_string())
        print("\nSkill vs. persistence (S_E, positive = model beats persistence):")
        print(skill.round(4).to_string())

        best_by_col = {c: combined[c].min() for c in SKILL_COLS}
        print("\nLaTeX row for persistence (append to the manoeuvre/navigation table):")
        print(latex_row("Persistence", pers_row, best_by_col))


if __name__ == "__main__":
    main()
