"""Load-dynamics diagnostics: per-leg and per-split statistics of the target.

Optional pipeline branch (--load-dynamics), analogous to the EDA/BESS/dPCA
branches. Operates on the preprocessed dataframes (1 Hz, PU), so the numbers
describe exactly the series the forecasters are trained on.

Diagnostics per leg, per split (all/train/val/test):
- autocorrelation decay time: first lag with r < ac_threshold (capped, so a
  value equal to the cap means ">= cap")
- ramp-rate percentiles of |dP/dt| (PU/s)
- load-peak density (peaks/min, scipy find_peaks with fixed prominence and
  minimum spacing so the definition is identical across modes)
- low/high-frequency spectral energy ratio (Welch, hann, nperseg 2048 -- same
  settings as bess.py; LF = (0, lf_split_hz))
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import signal

DEFAULT_SETTINGS = {
    "ac_threshold": 0.5,
    "ac_max_lag": 300,
    "peak_prominence": 0.05,
    "peak_distance_s": 10.0,
    "lf_split_hz": 0.05,
    "nperseg": 2048,
}

METRICS = ("ac_decay_s", "ramp_q95_pu_s", "ramp_q99_pu_s", "peaks_per_min", "lf_hf_ratio")


def autocorr_decay(y, threshold=0.5, max_lag=300):
    """First lag (samples) where autocorrelation drops below threshold; capped."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    y_c = y - y.mean()
    var = np.var(y_c)
    if var < 1e-10 or n < 8:
        return np.nan
    max_lag = min(max_lag, n // 4)
    for lag in range(1, max_lag + 1):
        r = np.mean(y_c[: n - lag] * y_c[lag:]) / var
        if r < threshold:
            return lag
    return max_lag


def ramp_abs(y, fs=1.0):
    r = np.abs(np.diff(np.asarray(y, dtype=np.float64)) * fs)
    return r[np.isfinite(r)]


def peak_density_per_min(y, fs=1.0, prominence=0.05, distance_s=10.0):
    peaks, _ = signal.find_peaks(y, prominence=prominence, distance=max(int(distance_s * fs), 1))
    minutes = len(y) / fs / 60.0
    return len(peaks) / minutes if minutes > 0 else np.nan


def lf_hf_energy_ratio(y, fs=1.0, lf_split=0.05, nperseg=2048):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 256:
        return np.nan
    nperseg = min(nperseg, len(y))
    f, pxx = signal.welch(y, fs=fs, window="hann", nperseg=nperseg,
                          noverlap=nperseg // 2, detrend="constant", scaling="density")
    trapezoid = getattr(np, "trapezoid", np.trapz)
    lf = trapezoid(pxx[(f > 0) & (f < lf_split)], f[(f > 0) & (f < lf_split)])
    hf = trapezoid(pxx[f >= lf_split], f[f >= lf_split])
    return float(lf / hf) if hf > 0 else np.nan


def leg_diagnostics(y, fs, settings):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 8:
        return None
    ramps = ramp_abs(y, fs)
    return {
        "n_samples": len(y),
        "ac_decay_s": autocorr_decay(y, settings["ac_threshold"], settings["ac_max_lag"]),
        "ramp_q95_pu_s": float(np.quantile(ramps, 0.95)) if len(ramps) >= 10 else np.nan,
        "ramp_q99_pu_s": float(np.quantile(ramps, 0.99)) if len(ramps) >= 10 else np.nan,
        "peaks_per_min": peak_density_per_min(
            y, fs, settings["peak_prominence"], settings["peak_distance_s"]
        ),
        "lf_hf_ratio": lf_hf_energy_ratio(y, fs, settings["lf_split_hz"], settings["nperseg"]),
    }


def run_load_dynamics_analysis(splits, imgs_dir, selector, fs=1.0, col="act_pow_pu", **overrides):
    """Compute per-leg diagnostics for each split and write CSV tables.

    Inputs
    ----------
    splits : dict[str, pandas.DataFrame]
        Split name -> dataframe with leg_id and the target column, e.g.
        {"all": TT, "train": TTtrain, "val": TTval, "test": TTtest}.
    imgs_dir : str
        Run image directory; tables are written to <imgs_dir>/tables.
    selector : str
        Operating-mode selector, used in output file names.

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        Per-leg table and across-legs summary (median/min/max per split).
    """
    settings = {**DEFAULT_SETTINGS, **overrides}
    tables_dir = os.path.join(imgs_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    rows = []
    for split_name, frame in splits.items():
        for leg in np.sort(frame["leg_id"].unique()):
            y = frame.loc[frame["leg_id"] == leg, col].to_numpy()
            diag = leg_diagnostics(y, fs, settings)
            if diag is None:
                continue
            rows.append({"split": split_name, "leg_id": int(leg), **diag})
    per_leg = pd.DataFrame(rows)

    summary_rows = []
    for split_name in splits:
        sub = per_leg[per_leg["split"] == split_name]
        row = {"split": split_name, "n_legs": len(sub)}
        for metric in METRICS:
            vals = sub[metric].astype(float).dropna()
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    per_leg_csv = os.path.join(tables_dir, f"load_dynamics_per_leg_{selector}.csv")
    summary_csv = os.path.join(tables_dir, f"load_dynamics_summary_{selector}.csv")
    per_leg.to_csv(per_leg_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"[load dynamics] settings: {settings}")
    print(f"[load dynamics] per-leg table -> {per_leg_csv}")
    print(f"[load dynamics] summary       -> {summary_csv}")
    print(f"[load dynamics] median [min-max] across legs ({selector}):")
    for _, row in summary.iterrows():
        parts = [f"{m}={row[f'{m}_median']:.4g} [{row[f'{m}_min']:.4g}-{row[f'{m}_max']:.4g}]"
                 for m in METRICS]
        print(f"  {row['split']:<6} (n_legs={row['n_legs']}): " + ", ".join(parts))
    print(f"[load dynamics] ac_decay_s equal to {settings['ac_max_lag']} means >= cap (censored)")
    print()
    return per_leg, summary
