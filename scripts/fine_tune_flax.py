"""Fine-tune saved Flax models of one run on the data of another run, and score them.

The source run provides pretrained deployment artifacts (model_<key>_flax); the
target is either one run archive or a leave-one-leg-out run root whose fold*/
directories each hold their own windows.npz, scalers and trained models. For
every target directory and model:

- the target windows are mapped into the source artifact's scale
  (load_forecasting/fine_tuning.py), so the pretrained weights see inputs in the
  units they were trained on;
- the source model is scored zero-shot, fine-tuned on the target training
  windows with early stopping on the target validation windows, and scored again;
- the fine-tuned model is saved as model_<key>_<tag>_flax (same builder,
  scalers and provenance), and its test/validation forecasts as
  predictions/<key>_<tag>.npz, the format read by fusion_mixture_replay.py;
- persistence, the target-trained model, the zero-shot and the fine-tuned model
  are scored on the same test windows with the metrics of the LOVO tables.

Outputs: <target>/fine_tune_<tag>/metrics.csv and history.json per directory;
for a run root also <run-root>/fine_tune_<tag>/by_fold.csv and summary.csv
(mean persistence-relative skill, fold-level 95 % t-interval, positive folds).

Examples:
    python scripts/fine_tune_flax.py \
        --source-run outputs/test_results/20260721_ForecastModels_Navigation_L60_H25_flax8_legsafe \
        --target outputs/test_results/20260721_ForecastModels_Manoeuvre_L60_H25_flax8_legsafe
    python scripts/fine_tune_flax.py --source-run <nav run> \
        --target outputs/runs/lovo_manoeuvre_benchmark_flax --models tcn,itransformer --frozen '^Conv_0/'
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# model key -> (artifact stem in model_<stem>_flax, prediction stem in predictions/<stem>.npz)
MODELS = {
    "cnn": ("cnn", "cnn"), "tcn": ("tcn", "tcn"), "transformer": ("tr", "transformer"),
    "itransformer": ("itransformer", "itransformer"), "dual_stream_film": ("dual_stream_film", "dual_stream_film"),
    "convtcn_hybrid": ("convtcn_hybrid", "convtcn_hybrid"), "convlstmtcn_hybrid": ("convlstmtcn_hybrid", "convlstmtcn_hybrid"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", required=True, help="Run directory holding the pretrained model_<key>_flax artifacts.")
    ap.add_argument("--target", required=True, help="Target run archive, or run root with fold*/ subdirectories.")
    ap.add_argument("--models", default=",".join(MODELS), help=f"Comma-separated keys among {', '.join(MODELS)}.")
    ap.add_argument("--tag", default="ft", help="Suffix of the saved artifacts, predictions and output folder.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate (default 1e-4).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", default="auto",
                    help="Integer, or 'auto' for the batch size of the target run's own model of the same key.")
    ap.add_argument("--trainable", nargs="*", default=None, help="Regexes of parameter paths to train (default: all).")
    ap.add_argument("--frozen", nargs="*", default=None, help="Regexes of parameter paths to freeze.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="", help="Comma-separated fold indices to run (run root only; default all).")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    return ap.parse_args()


def target_dirs(target: Path, folds: str):
    dirs = sorted(target.glob("fold*"), key=lambda p: int(p.name[4:]))
    if not dirs:
        return [(None, target)]
    keep = {int(f) for f in folds.split(",") if f.strip()}
    return [(int(d.name[4:]), d) for d in dirs if not keep or int(d.name[4:]) in keep]


def batch_size_for(tdir: Path, stem: str, requested: str) -> int:
    if requested != "auto":
        return int(requested)
    manifest = tdir / f"model_{stem}_flax" / "manifest.json"
    if manifest.is_file():
        cfg = json.loads(manifest.read_text(encoding="utf-8")).get("metadata", {}).get("config", {})
        if "batch_size" in cfg:
            return int(cfg["batch_size"])
    return 64


def run_directory(fold, tdir: Path, a, rating):
    from fusion_mixture_replay import metrics
    from load_forecasting.fine_tuning import (Scaler, fine_tune, load_source, predict_physical,
                                              save_fine_tuned, to_artifact_space)
    from load_forecasting.training_deep import DeepTrainConfig

    meta = pickle.load(open(tdir / "scalers_and_meta.pkl", "rb"))
    features, targets = list(meta["featureNames"]), list(meta["targetNames"])
    fs, ts = Scaler.from_meta(meta, features), Scaler.from_meta(meta, targets)
    w = np.load(tdir / "windows.npz")
    phys_y = lambda z: (np.asarray(z) * ts.scale + ts.mean)[..., 0]  # noqa: E731
    y_true, y_train = phys_y(w["Yte"]), phys_y(w["Ytr"])
    target_idx = features.index(targets[0])
    persistence = np.repeat((w["Xte"][:, -1:, target_idx] * fs.scale[target_idx] + fs.mean[target_idx]), y_true.shape[1], axis=1)
    rows, histories = [], {}
    base = metrics(y_true, persistence, y_train, rating)
    rows.append({"fold": fold, "model": "persistence", "variant": "persistence", **base})
    out_dir = tdir / f"fine_tune_{a.tag}"
    out_dir.mkdir(exist_ok=True)
    (tdir / "predictions").mkdir(exist_ok=True)

    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        stem, pred_stem = MODELS[key]
        art = load_source(Path(a.source_run) / f"model_{stem}_flax")
        if list(art.feature_names) != features or list(art.target_names) != targets:
            raise ValueError(f"{key}: source and target feature/target names differ")
        Xtr, Ytr = to_artifact_space(w["Xtr"], w["Ytr"], fs, ts, art)
        Xva, Yva = to_artifact_space(w["Xva"], w["Yva"], fs, ts, art)
        Xte, _ = to_artifact_space(w["Xte"], None, fs, ts, art)
        label = key
        own = tdir / "predictions" / f"{pred_stem}.npz"
        if own.is_file():
            d = np.load(own)
            label = str(d["model_label"])
            rows.append({"fold": fold, "model": label, "variant": "target-trained",
                         **metrics(y_true, d["yhat_test_phys"][..., 0], y_train, rating)})
        rows.append({"fold": fold, "model": label, "variant": "zero-shot",
                     **metrics(y_true, predict_physical(art, art.params, Xte)[..., 0], y_train, rating)})

        cfg = DeepTrainConfig(batch_size=batch_size_for(tdir, stem, a.batch_size), epochs=a.epochs, learning_rate=a.lr,
                              loss="huber", shuffle=True, patience=a.patience, seed=a.seed)
        t0 = time.perf_counter()
        res = fine_tune(art, Xtr, Ytr, Xva, Yva, cfg, trainable=a.trainable, frozen=a.frozen)
        minutes = (time.perf_counter() - t0) / 60
        yhat_te, yhat_va = predict_physical(art, res.params, Xte), predict_physical(art, res.params, Xva)
        np.savez(tdir / "predictions" / f"{pred_stem}_{a.tag}.npz", model_label=np.array(f"{label} {a.tag.upper()}"),
                 yhat_test_phys=yhat_te.astype(np.float32), yhat_val_phys=yhat_va.astype(np.float32))
        save_fine_tuned(art, res.params, tdir / f"model_{stem}_{a.tag}_flax", metadata={
            "model_key": key, "model_label": f"{label} {a.tag.upper()}", "fine_tune_config": res.config,
            "trainable": a.trainable, "frozen": a.frozen, "n_trainable": res.n_trainable,
            "epochs_run": len(res.history["loss"]), "best_val_loss": res.best_val})
        rows.append({"fold": fold, "model": label, "variant": "fine-tuned", "epochs": len(res.history["loss"]),
                     "minutes": minutes, "n_trainable": res.n_trainable,
                     **metrics(y_true, yhat_te[..., 0], y_train, rating)})
        histories[key] = res.history
        print(f"[fold {fold}] {label}: {len(res.history['loss'])} epochs, {minutes:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    for m in ("MAPE_all", "RMSE_all", "tail_rmse_pu", "tail_rmse_pu_leg", "energy_err_mean_kwh", "mape_ramp_steep_%"):
        df[f"S_{m}"] = 1 - df[m] / base[m] if base[m] else np.nan
    df.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "history.json").write_text(json.dumps({"args": vars(a), "history": histories}, indent=1), encoding="utf-8")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for (model, variant), g in df[df.variant != "persistence"].groupby(["model", "variant"], sort=False):
        r = {"model": model, "variant": variant}
        for c in [c for c in df.columns if c.startswith("S_")]:
            x = g[c].dropna().to_numpy()
            if len(x) < 2:
                r[c] = f"{x.mean():+.3f} (n={len(x)})" if len(x) else ""
                continue
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            r[c] = f"{x.mean():+.3f} [{x.mean() - h:+.3f},{x.mean() + h:+.3f}] {int((x > 0).sum())}/{len(x)}"
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    if a.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    from compute_peak_metrics import default_rating_kw

    rating = default_rating_kw()
    target = Path(a.target)
    frames = [run_directory(fold, tdir, a, rating) for fold, tdir in target_dirs(target, a.folds)]
    if frames[0]["fold"].notna().any():
        out = target / f"fine_tune_{a.tag}"
        out.mkdir(exist_ok=True)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / "by_fold.csv", index=False)
        summ = summarise(df)
        summ.to_csv(out / "summary.csv", index=False)
        pd.set_option("display.width", 250)
        print("\nPersistence-relative skill over folds, mean [95% t-CI] positive folds:")
        print(summ.to_string(index=False))
    else:
        cols = ["model", "variant", "MAPE_all"] + [c for c in frames[0].columns if c.startswith("S_")]
        pd.set_option("display.width", 250)
        print(frames[0][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
