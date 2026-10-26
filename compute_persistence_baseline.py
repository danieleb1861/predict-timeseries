"""
Compute persistence baseline MAPE and autocorrelation decay for
manoeuvre and navigation datasets.
Persistence forecast: y_hat(t+h) = y(t) for h in {5, 15, 25} steps.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

from load_forecasting.config import resolve_default_config

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_CONFIG = resolve_default_config(os.path.join(os.path.dirname(__file__), "config"))


def load_dataset_config(config_path):
    """Read the navigation/manoeuvre file lists and the power rating from a JSON config."""
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    selectors = config["dataset"]["selectors"]

    return selectors["navigation"], selectors["manoeuvre"], config["physical_system"]["rating_kW"]

colMap = {
    "date":      "date",
    "time":      "time",
    "act_pow":   "act_pow_kW",
    "react_pow": "react_pow_kVAR",
    "man_port":  "man_selector_port",
    "man_stbd":  "man_selector_starboard",
    "spd_port":  "lever_port",
    "spd_stbd":  "lever_starboard",
    "air_temp":  "air_temp",
    "wind_int":  "wind_intensity",
    "sea_temp":  "sea_temp",
    "sea_state": "sea_state",
}

data_types = {
    "date":                     "object",
    "time":                     "object",
    "act_pow_kW":               "float32",
    "react_pow_kVAR":           "float32",
    "man_selector_port":        "Int8",
    "man_selector_starboard":   "Int8",
    "lever_port":               "float32",
    "lever_starboard":          "float32",
    "air_temp":                 "float32",
    "wind_intensity":           "float32",
    "sea_temp":                 "float32",
    "sea_state":                "float32",
}

HORIZONS = [5, 15, 25]


def parse_horizons(raw_value):
    horizons = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("--horizons must contain at least one integer horizon.")
    return horizons


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute persistence baseline MAPE and autocorrelation decay."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON config providing the dataset selectors and the power rating.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing parquet input files.")
    parser.add_argument("--rating-kw", type=float, default=None, help="Power rating used for per-unit conversion (default: physical_system.rating_kW from --config).")
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS), help="Comma-separated forecast horizons in samples.")
    parser.add_argument("--autocorr-threshold", type=float, default=0.5, help="Autocorrelation threshold used for decay horizon.")
    return parser.parse_args()


def load_power_pu(fname, data_dir, rating_kw):
    path = os.path.join(data_dir, fname)
    T = pd.read_parquet(path).astype(data_types, errors="ignore")
    y = (T[colMap["act_pow"]] / rating_kw).values.astype(np.float32)
    return y


def persistence_mape(y, h):
    """MAPE of naive persistence over horizon h (averaged over all steps)."""
    # persistence: predict y[t] for t+1 .. t+h
    # aggregate over all anchors and all steps
    errors = []
    for step in range(1, h + 1):
        pred = y[:-step] if step > 0 else y
        true = y[step:]
        # avoid division by zero
        mask = np.abs(true) > 1e-4
        if mask.sum() > 0:
            errors.append(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100)
    return float(np.mean(errors)) if errors else np.nan


def autocorr_decay_horizon(y, threshold=0.5):
    """Lag (in samples) at which autocorrelation first drops below threshold."""
    n = len(y)
    y_c = y - np.mean(y)
    var = np.var(y_c)
    if var < 1e-10:
        return np.nan
    max_lag = min(300, n // 4)
    for lag in range(1, max_lag + 1):
        r = np.mean(y_c[:n-lag] * y_c[lag:]) / var
        if r < threshold:
            return lag
    return max_lag  # still above threshold at max_lag


def process_files(file_list, label, data_dir, rating_kw, horizons, autocorr_threshold):
    print(f"\n=== {label} ===")
    all_y = []
    leg_stats = []
    for fname in file_list:
        try:
            y = load_power_pu(fname, data_dir, rating_kw)
            all_y.append(y)
            ac_decay = autocorr_decay_horizon(y, threshold=autocorr_threshold)
            row = {"file": fname, "n": len(y),
                   "mean": np.mean(y), "std": np.std(y),
                   f"ac_decay_{autocorr_threshold:g}": ac_decay}
            for h in horizons:
                row[f"mape_pers_H{h}"] = persistence_mape(y, h)
            leg_stats.append(row)
            print(f"  {fname}: n={len(y)}, mean={np.mean(y):.3f}, std={np.std(y):.3f}, "
                  f"ac_decay={ac_decay}s, "
                  + ", ".join([f"MAPE_pers_H{h}={persistence_mape(y,h):.2f}%" for h in horizons]))
        except Exception as e:
            print(f"  {fname}: ERROR {e}")

    # pooled
    y_pool = np.concatenate(all_y) if all_y else np.array([])
    print(f"\n  --- {label} pooled (n={len(y_pool)}) ---")
    for h in horizons:
        mape = persistence_mape(y_pool, h)
        print(f"  H={h}s -> persistence MAPE = {mape:.2f}%")
    ac_pool = autocorr_decay_horizon(y_pool, threshold=autocorr_threshold)
    print(f"  Autocorr decay below {autocorr_threshold:g}: {ac_pool} s")

    return leg_stats, y_pool


def main():
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    ac_key = f"ac_decay_{args.autocorr_threshold:g}"

    nav_files, man_files, config_rating_kw = load_dataset_config(args.config)
    rating_kw = args.rating_kw if args.rating_kw is not None else config_rating_kw

    nav_stats, y_nav = process_files(nav_files, "Navigation", args.data_dir, rating_kw, horizons, args.autocorr_threshold)
    man_stats, y_man = process_files(man_files, "Manoeuvre", args.data_dir, rating_kw, horizons, args.autocorr_threshold)

    # --- summary table ---
    print("\n\n=== SUMMARY TABLE (persistence MAPE, pooled) ===")
    print(f"{'Regime':<15}" + "".join([f" {f'H={h}s':>8}" for h in horizons]))
    for label, y_pool in [("Navigation", y_nav), ("Manoeuvre", y_man)]:
        row = f"{label:<15}"
        for h in horizons:
            row += f" {persistence_mape(y_pool, h):>7.2f}%"
        print(row)

    # --- per-leg table for navigation (map to V1-V6) ---
    print("\n=== PER-LEG NAVIGATION PERSISTENCE MAPE ===")
    for i, row in enumerate(nav_stats):
        print(f"  V{i+1}: " + ", ".join([f"H{h}={row[f'mape_pers_H{h}']:.2f}%" for h in horizons])
              + f", ac_decay={row[ac_key]}s")

    print("\n=== PER-LEG MANOEUVRE PERSISTENCE MAPE (12 segments, 2 per voyage) ===")
    voyages = ['V1','V1','V2','V2','V3','V3','V4','V4','V5','V5','V6','V6']
    sides   = ['1st','2nd','1st','2nd','1st','2nd','1st','2nd','1st','2nd','1st','2nd']
    for i, row in enumerate(man_stats):
        v = voyages[i] if i < len(voyages) else f"seg{i}"
        s = sides[i] if i < len(sides) else ""
        print(f"  {v}-{s}: " + ", ".join([f"H{h}={row[f'mape_pers_H{h}']:.2f}%" for h in horizons])
              + f", ac_decay={row[ac_key]}s")


if __name__ == "__main__":
    main()
