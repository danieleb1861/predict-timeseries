"""Peak-aware and energy metrics from an archived benchmark config.

Computes, per model, on the archived test windows (physical units):
- upper-tail RMSE/MAE: errors conditional on true load exceeding the
  training-set q-quantile (default 0.95) -- peak accuracy without event
  detection machinery;
- energy-over-horizon error: |sum_h(yhat - y)| * Ts * P_rated / 3600 (kWh)
  per window, reported as mean and 95th percentile -- the EMS/BESS-facing
  quantity (dispatch commits energy, not instantaneous points);
- ramp-binned MAPE: windows grouped by their maximum true one-step ramp
  |dP/dt| into quantile bins (default edges 0.5, 0.9 -> low/mid/steep);
  fills the error-by-ramp-rate row of the load-dynamics table.

Uses archived predictions/<model>.npz when present; otherwise regenerates
them by inference from the archived Flax artifacts (deterministic) and saves
them back into the archive for later analyses (bootstrap, conformal, DM).

Example:
    python scripts/compute_peak_metrics.py \
        --archive outputs/test_results/20260714_ForecastModels_Manoeuvre_L60_H25_flax8 \
        --model-specs config/model_specs_benchmark_manoeuvre.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import ChainMap
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.config import load_model_specs, resolve_default_config  # noqa: E402
from load_forecasting.scaling import inverse_scale_targets  # noqa: E402


def default_rating_kw():
    """Per-unit base power taken from the resolved system config."""
    with open(resolve_default_config(REPO_ROOT / "config"), encoding="utf-8") as handle:
        return float(json.load(handle)["physical_system"]["rating_kW"])

ROSTER = (
    "cnn", "tcn", "dlinear", "itransformer", "transformer",
    "convtcn_hybrid", "convlstmtcn_hybrid", "dual_stream_film",
)


def artifact_name(spec: dict, key: str) -> str:
    return spec.get("artifact_key") or spec.get("artifact_name") or key


def predict_from_artifact(archive: Path, spec: dict, key: str, meta, X, specs_file, batch=512):
    """Retain single-split callers while sharing validated artifact inference."""
    from load_forecasting.deployment_flax import load_flax_run_model

    artifact_dir = archive / f"model_{artifact_name(spec, key)}_flax"
    loaded = load_flax_run_model(artifact_dir, meta, specs_file, spec=spec)
    return loaded.predict_standardized(X, batch_size=batch)


def load_or_regenerate_predictions(archive, key, spec, meta, windows, specs_file):
    """Reuse one restored model when validation and test predictions are missing."""
    pred_file = archive / "predictions" / f"{key}.npz"
    if pred_file.exists():
        # Legacy cache compatibility: validate contents without claiming freshness.
        with np.load(pred_file, allow_pickle=False) as saved:
            yhat_test = saved["yhat_test_phys"]
            if "model_label" in saved and str(saved["model_label"]) != spec["label"]:
                raise ValueError(f"{pred_file}: model label does not match {spec['label']}")
        # Targets give the sample count without decompressing feature windows.
        # Keep compatibility with callers that supply only Xte.
        sample_count = len(windows["Yte"] if "Yte" in windows else windows["Xte"])
        expected = (sample_count, int(meta["Horizon forecasting window H"]), len(meta["targetNames"]))
        if yhat_test.shape != expected or not np.all(np.isfinite(yhat_test)):
            raise ValueError(f"{pred_file}: expected finite predictions with shape {expected}")
        return yhat_test

    print(f"  predictions/{key}.npz missing -> regenerating by inference")
    from load_forecasting.deployment_flax import load_flax_run_model
    from load_forecasting.training_deep import save_predictions

    artifact_dir = archive / f"model_{artifact_name(spec, key)}_flax"
    loaded = load_flax_run_model(artifact_dir, meta, specs_file, spec=spec)

    def predict_physical(inputs):
        # Both splits share the same restored parameters, JIT callable, and scalers.
        standardized = loaded.predict_standardized(inputs)
        return inverse_scale_targets(standardized, meta["targetNames"], meta["MU"], meta["SIG"])

    yhat_test = predict_physical(windows["Xte"])
    yhat_val = predict_physical(windows["Xva"])
    save_predictions(archive, key, spec["label"], yhat_test, yhat_val)
    return yhat_test


def peak_metrics(y_true, y_hat, y_train, rating_kw, tail_q, ramp_edges):
    """All inputs (N, H) in PU; returns dict of scalar metrics."""
    prepared = _prepare_peak_metrics(y_true, y_train, tail_q, ramp_edges)
    return _score_peak_metrics(y_true, y_hat, rating_kw, prepared)


def _prepare_peak_metrics(y_true, y_train, tail_q, ramp_edges):
    """Prepare truth-only selections once for all models in an archive."""
    thr = float(np.quantile(y_train, tail_q))
    tail_mask = y_true > thr
    ramp = np.abs(np.diff(y_true, axis=1)).max(axis=1)  # PU/s per window
    edges = np.quantile(ramp, ramp_edges)
    bins = np.digitize(ramp, edges)  # 0=low, 1=mid, 2=steep
    groups = []
    for b, name in enumerate(("low", "mid", "steep")):
        m = bins == b
        yt = y_true[m]
        mask = np.abs(yt) > 1e-4
        groups.append((name, m, yt[mask], mask, int(m.sum())))
    return thr, tail_mask, 100.0 * tail_mask.mean(), groups


def _score_peak_metrics(y_true, y_hat, rating_kw, prepared):
    """Score one model using shared selections and unchanged metric formulas."""
    thr, tail_mask, tail_fraction, groups = prepared
    err = y_hat - y_true
    out = {
        "tail_threshold_pu": thr,
        "tail_frac_%": tail_fraction,
        "tail_rmse_pu": float(np.sqrt(np.mean(err[tail_mask] ** 2))),
        "tail_mae_pu": float(np.mean(np.abs(err[tail_mask]))),
    }

    e_kwh = np.abs(err.sum(axis=1)) * rating_kw / 3600.0  # Ts = 1 s
    out["energy_err_mean_kwh"] = float(e_kwh.mean())
    out["energy_err_p95_kwh"] = float(np.quantile(e_kwh, 0.95))

    for name, m, yt, mask, count in groups:
        out[f"mape_ramp_{name}_%"] = float(np.mean(np.abs((y_hat[m][mask] - yt) / yt)) * 100)
        out[f"n_windows_{name}"] = count
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True, help="Archived benchmark config directory.")
    parser.add_argument("--model-specs", required=True, help="Specs JSON matching the archive (for labels and artifact rebuild).")
    parser.add_argument("--rating-kw", type=float, default=None, help="Per-unit base power (default: physical_system.rating_kW from the resolved system config).")
    parser.add_argument("--tail-q", type=float, default=0.95, help="Quantile defining the upper tail (default 0.95).")
    parser.add_argument("--tail-from", choices=("train", "test"), default="train",
                        help="Series the upper-tail quantile is taken from: 'train' (default, the "
                             "manuscript definition) or 'test', the held-out segment itself. The latter "
                             "keeps the tail 5%% populated when the held-out segment is a single voyage, "
                             "and writes peak_metrics_testtail.csv instead of peak_metrics.csv.")
    parser.add_argument("--ramp-edges", default="0.5,0.9", help="Quantile edges for ramp bins (default 0.5,0.9).")
    parser.add_argument("--d-plot", type=int, default=0)
    args = parser.parse_args()

    if args.rating_kw is None:
        args.rating_kw = default_rating_kw()

    archive = Path(args.archive)
    windows = np.load(archive / "windows.npz")
    # Reuse targets in cache checks; feature arrays remain lazy until inference.
    windows = ChainMap({"Yte": windows["Yte"]}, windows)
    with (archive / "scalers_and_meta.pkl").open("rb") as f:
        meta = pickle.load(f)

    deep, hybrid = load_model_specs(args.model_specs)
    specs = {**deep, **hybrid}

    d = args.d_plot
    y_true = inverse_scale_targets(windows["Yte"], meta["targetNames"], meta["MU"], meta["SIG"])[:, :, d]
    y_train = inverse_scale_targets(windows["Ytr"], meta["targetNames"], meta["MU"], meta["SIG"])[:, :, d]
    ramp_edges = [float(v) for v in args.ramp_edges.split(",")]
    y_tail_ref = y_train if args.tail_from == "train" else y_true
    prepared = _prepare_peak_metrics(y_true, y_tail_ref, args.tail_q, ramp_edges)

    rows = []
    for key in ROSTER:
        if key not in specs:
            continue
        adir = archive / f"model_{artifact_name(specs[key], key)}_flax"
        if not adir.is_dir() and not (archive / "predictions" / f"{key}.npz").exists():
            print(f"  {key}: no artifact or predictions in archive, skipped")
            continue
        print(f"{specs[key]['label']} ...")
        y_hat = load_or_regenerate_predictions(archive, key, specs[key], meta, windows, args.model_specs)[:, :, d]
        rows.append({"model": specs[key]["label"], **_score_peak_metrics(
            y_true, y_hat, args.rating_kw, prepared)})

    df = pd.DataFrame(rows).set_index("model")
    out_csv = archive / ("peak_metrics.csv" if args.tail_from == "train"
                         else "peak_metrics_testtail.csv")
    df.to_csv(out_csv)
    print(f"\nWritten: {out_csv}")
    cols = ["tail_rmse_pu", "tail_mae_pu", "energy_err_mean_kwh", "energy_err_p95_kwh",
            "mape_ramp_low_%", "mape_ramp_mid_%", "mape_ramp_steep_%"]
    print(df[cols].round(4).to_string())
    print(f"\ntail threshold = {df['tail_threshold_pu'].iloc[0]:.4f} PU "
          f"({df['tail_frac_%'].iloc[0]:.1f}% of test points), "
          f"ramp bins n = {df['n_windows_low'].iloc[0]}/{df['n_windows_mid'].iloc[0]}/{df['n_windows_steep'].iloc[0]}")

    print("\nLaTeX rows (RMSE/MAE tail PU | energy kWh mean/p95 | MAPE steep %):")
    best = {c: df[c].min() for c in cols}
    for label, r in df.iterrows():
        cells = []
        for c in ("tail_rmse_pu", "tail_mae_pu", "energy_err_mean_kwh", "energy_err_p95_kwh", "mape_ramp_steep_%"):
            v = f"{r[c]:.4f}" if "pu" in c else f"{r[c]:.3f}" if "kwh" in c else f"{r[c]:.2f}"
            cells.append(f"\\textbf{{{v}}}" if np.isclose(r[c], best[c]) else v)
        print(f"{label} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
