"""Per-leg bootstrap MAPE CIs, combined by equally-weighted averaging across legs.

The navigation (and manoeuvre) test split concatenates whole voyage legs (load_forecasting/splitting.py). The per-window MAPE autocorrelation does not decay within the available test span, consistent with genuine between-leg heterogeneity rather than short-range window overlap alone -- a single pooled bootstrap over all windows therefore lets the longer legs dominate and cannot represent between-leg uncertainty.

This script instead:
- supports both legacy archives containing cross-leg windows and corrected archives whose window construction already removes them;
- computes the ordinary bootstrap CI (load_forecasting.metrics.bootstrap_mean_ci) separately within each remaining leg;
- combines the legs with equal weight: on each of n_boot iterations, draw one bootstrap resample from each leg independently and average the two resample means, so a leg's sample count does not bias the combined estimate.

Example:
    python scripts/compute_leg_bootstrap_ci.py \
        --archive outputs/test_results/20260715_ForecastModels_Navigation_L45_H25_flax8
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

from load_forecasting.config import resolve_default_config  # noqa: E402
from load_forecasting.metrics import bootstrap_mean_ci  # noqa: E402
from load_forecasting.scaling import inverse_scale_targets  # noqa: E402
from train_load_forecasters import load_and_preprocess_legs  # noqa: E402
from load_forecasting.data_prep import add_engineered  # noqa: E402
from load_forecasting.splitting import split_train_val_test  # noqa: E402

ROSTER = (
    "cnn", "tcn", "dlinear", "itransformer", "transformer",
    "convtcn_hybrid", "convlstmtcn_hybrid", "dual_stream_film",
)


def per_window_mape(y_true: np.ndarray, y_hat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    err = y_hat - y_true
    pct = np.abs(err) / np.maximum(np.abs(y_true), eps) * 100.0
    return pct.mean(axis=1)


def test_leg_boundaries(system_config_path: Path, selector: str) -> tuple[int, list[int]]:
    """Return (T, cumulative row counts per test leg) for the raw (pre-window) test frame."""
    import json

    cfg = json.loads(system_config_path.read_text(encoding="utf-8"))
    dataset_cfg = cfg["dataset"]
    physical_cfg = cfg["physical_system"]
    signal_cfg = physical_cfg["signals"]
    col_map = dataset_cfg["columns"]
    data_types = {col_map[k]: v for k, v in dataset_cfg["dtypes"].items()}
    rating_kw = float(physical_cfg["rating_kW"])
    files = dataset_cfg["selectors"][selector]
    vars_prev = signal_cfg["step_hold"]
    vars_lin = signal_cfg["resample_continuous"]
    ts = dataset_cfg.get("sampling_rate", "1s")
    max_fill_seconds = dataset_cfg.get("max_fill_seconds", 45)

    tt = load_and_preprocess_legs(files, str(REPO_ROOT / "data"), col_map, data_types,
                                   rating_kw, vars_prev, vars_lin, ts, max_fill_seconds)
    tt = add_engineered(tt)
    _, _, tt_test = split_train_val_test(tt)
    leg_sizes = tt_test.groupby("leg_id", sort=False).size().tolist()
    return len(tt_test), leg_sizes


def clean_leg_window_masks(T: int, leg_sizes: list[int], lookback: int, horizon: int,
                            stride: int, n_windows: int) -> list[np.ndarray]:
    """Boolean mask per test leg for legacy or boundary-filtered archives."""
    anchors = np.arange(lookback - 1, T - horizon, stride)

    boundaries = np.cumsum([0] + leg_sizes)  # row ranges: [boundaries[i], boundaries[i+1])
    masks = []
    for i in range(len(leg_sizes)):
        lo, hi = boundaries[i], boundaries[i + 1]
        x_start = anchors - lookback + 1
        y_end = anchors + horizon
        masks.append((x_start >= lo) & (y_end < hi))
    n_clean = sum(int(mask.sum()) for mask in masks)
    if n_windows == anchors.size:
        return masks
    if n_windows == n_clean:
        clean = np.logical_or.reduce(masks)
        return [mask[clean] for mask in masks]
    raise AssertionError(
        f"archived window count {n_windows} matches neither all anchors "
        f"({anchors.size}) nor boundary-safe anchors ({n_clean})"
    )


def combined_leg_bootstrap(leg_mapes: list[np.ndarray], n_boot: int = 1000,
                           ci: float = 95, seed: int = 42) -> dict:
    """Bootstrap an equal-weight mean across any positive number of legs."""
    if not leg_mapes or any(mape.size == 0 for mape in leg_mapes):
        raise ValueError("leg_mapes must contain one non-empty array per leg")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = np.mean([
            rng.choice(mape, size=mape.size, replace=True).mean()
            for mape in leg_mapes
        ])
    alpha = (100.0 - ci) / 2.0
    return {
        "mean": float(np.mean(boot)),
        "low": float(np.percentile(boot, alpha)),
        "high": float(np.percentile(boot, 100.0 - alpha)),
    }


def combined_two_leg_bootstrap(mape_a: np.ndarray, mape_b: np.ndarray, n_boot: int = 1000,
                                ci: float = 95, seed: int = 42) -> dict:
    """Backward-compatible two-leg wrapper."""
    return combined_leg_bootstrap([mape_a, mape_b], n_boot=n_boot, ci=ci, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--selector", default="navigation")
    parser.add_argument("--system-config", default=resolve_default_config(REPO_ROOT / "config"))
    parser.add_argument("--d-plot", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    archive = Path(args.archive)
    with (archive / "scalers_and_meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    windows = np.load(archive / "windows.npz")

    d = args.d_plot
    y_true = inverse_scale_targets(windows["Yte"], meta["targetNames"], meta["MU"], meta["SIG"])[:, :, d]
    lookback = int(meta["Lookback window L"])
    horizon = y_true.shape[1]
    stride = int(meta["WindowSlidingCoeff"])
    n_windows = y_true.shape[0]

    print("Reconstructing test-leg boundaries from the raw pipeline (no leakage into training)...")
    T, leg_sizes = test_leg_boundaries(REPO_ROOT / args.system_config, args.selector)
    print(f"Test frame rows T={T}, leg sizes={leg_sizes}")

    masks = clean_leg_window_masks(T, leg_sizes, lookback, horizon, stride, n_windows)
    n_clean = sum(m.sum() for m in masks)
    n_dropped = n_windows - n_clean
    print(f"Windows: total={n_windows}, clean-per-leg={[int(m.sum()) for m in masks]}, "
          f"dropped (cross-leg-boundary)={n_dropped}")
    rows = []
    for key in ROSTER:
        pred_file = archive / "predictions" / f"{key}.npz"
        if not pred_file.exists():
            print(f"  {key}: no cached predictions, skipped")
            continue
        payload = np.load(pred_file)
        label = str(payload["model_label"]) if "model_label" in payload else key
        y_hat = payload["yhat_test_phys"][:, :, d]
        mape_w = per_window_mape(y_true, y_hat)

        leg_mape = [mape_w[m] for m in masks]
        leg_ci = [bootstrap_mean_ci(lm, n_boot=args.n_boot) for lm in leg_mape]

        row = {"model": label}
        for i, (lm, ci) in enumerate(zip(leg_mape, leg_ci)):
            row[f"n_leg{i}"] = lm.size
            row[f"mape_leg{i}"] = float(lm.mean())
            row[f"ci_leg{i}_low"] = ci["low"]
            row[f"ci_leg{i}_high"] = ci["high"]

        if leg_mape:
            combined = combined_leg_bootstrap(leg_mape, n_boot=args.n_boot)
            row["mape_combined"] = float(np.mean([mape.mean() for mape in leg_mape]))
            row["ci_combined_low"] = combined["low"]
            row["ci_combined_high"] = combined["high"]

        rows.append(row)
        print(f"{label:<16} " + "  ".join(
            f"leg{i}(n={row[f'n_leg{i}']})={row[f'mape_leg{i}']:.4f}%[{row[f'ci_leg{i}_low']:.4f},{row[f'ci_leg{i}_high']:.4f}]"
            for i in range(len(leg_mape))
        ) + (f"  combined={row['mape_combined']:.4f}%[{row['ci_combined_low']:.4f},{row['ci_combined_high']:.4f}]"
             if "mape_combined" in row else ""))

    if not rows:
        print(f"\nNo roster model has cached predictions in {archive / 'predictions'}; nothing to write.")
        return 1

    df = pd.DataFrame(rows).set_index("model")
    out_csv = archive / "leg_bootstrap_ci.csv"
    df.to_csv(out_csv)
    print(f"\nWritten: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
