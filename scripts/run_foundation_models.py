"""Run zero-shot foundation-model inference across benchmark archives.

Loads a pretrained foundation model once per --models entry, then iterates
every matching archive, building physical-unit inputs from windows.npz,
running batched zero-shot inference, and saving results in the same
predictions/<key>.npz + compute_metrics() schema the tuned-model pipeline
uses -- but into a SEPARATE results_metrics_foundation.csv, never touching
the trained-model results_metrics.csv (which is fully overwritten on every
run_benchmark.py/train_load_forecasters.py invocation).

Run only from the isolated foundation venv, e.g.:
    ~/predict-timeseries/.venv-foundation/bin/python scripts/run_foundation_models.py \
        --selector both --models chronos2,timesfm,tirex2_univariate --skip-existing

Example (single archive, single model, for a first end-to-end check):
    ~/predict-timeseries/.venv-foundation/bin/python scripts/run_foundation_models.py \
        --results-dir outputs/test_results \
        --archive-tag flax8 --selector manoeuvre \
        --models chronos2 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.metrics import compute_metrics  # noqa: E402
from load_forecasting.scaling import inverse_scale_targets  # noqa: E402
from load_forecasting.training_deep import save_predictions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from foundation_adapters import ADAPTER_BUILDERS  # noqa: E402

# Deliberately re-implemented rather than imported from
# scripts/gen_figures_4papers.py, to avoid pulling matplotlib into this venv.
ARCHIVE_RE = re.compile(
    r"_(?P<mode>Manoeuvre|Navigation)_L(?P<lookback>\d+)_?H(?P<horizon>\d+)"
    r"(?P<suffix>(?:_.*)?)$",
    re.IGNORECASE,
)


def discover_archives(results_dir: Path, *, selector: str, archive_tag: str | None) -> list[Path]:
    modes = ("manoeuvre", "navigation") if selector == "both" else (selector,)
    found = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        match = ARCHIVE_RE.search(path.name)
        if match is None:
            continue
        if match.group("mode").casefold() not in modes:
            continue
        suffix_tokens = match.group("suffix").strip("_").casefold().split("_")
        if archive_tag is not None and archive_tag.casefold() not in suffix_tokens:
            continue
        if not (path / "windows.npz").is_file():
            continue
        found.append(path)
    return found


def load_labels(model_specs_path: Path) -> dict[str, str]:
    data = json.loads(model_specs_path.read_text(encoding="utf-8"))
    labels = {}
    for section in ("deep_models", "hybrid_models"):
        for key, spec in data.get(section, {}).items():
            labels[key] = spec["label"]
    return labels


def timesfm_compile_dimensions(archives: list[Path]) -> tuple[int, int]:
    """Choose TimesFM compile dimensions from the selected archive names."""
    dimensions = [ARCHIVE_RE.search(archive.name) for archive in archives]
    if any(match is None for match in dimensions):
        raise ValueError("Could not infer TimesFM dimensions from an archive name")
    max_lookback = max(int(match.group("lookback")) for match in dimensions if match is not None)
    max_horizon = max(int(match.group("horizon")) for match in dimensions if match is not None)
    # TimesFM 2.5 uses a 32-step input patch and a 128-step output patch.
    return math.ceil(max_lookback / 32) * 32, math.ceil(max_horizon / 128) * 128


def build_inputs(archive: Path, covariate_aware: bool):
    """Reconstruct physical-unit history/covariates/ground-truth from an archive."""
    windows = np.load(archive / "windows.npz")
    with (archive / "scalers_and_meta.pkl").open("rb") as f:
        meta = pickle.load(f)

    feature_names = list(meta["featureNames"])
    target_names = list(meta["targetNames"])
    if len(target_names) != 1:
        raise ValueError(f"{archive}: expected a single target, got {target_names}")
    target = target_names[0]
    target_idx = feature_names.index(target)

    Xte = windows["Xte"]
    Yte = windows["Yte"]
    y_true_phys = inverse_scale_targets(Yte, target_names, meta["MU"], meta["SIG"])

    history_phys = inverse_scale_targets(
        Xte[:, :, [target_idx]], [target], meta["MU"], meta["SIG"]
    )[:, :, 0]

    covariates_phys = None
    if covariate_aware:
        covariate_names = [name for name in feature_names if name != target]
        covariate_idx = [feature_names.index(name) for name in covariate_names]
        # shape (N, L, n_covariates) -- foundation_adapters.build_tirex2_adapter
        # transposes each window's (L, n_covariates) slice to (n_covariates, L).
        covariates_phys = inverse_scale_targets(
            Xte[:, :, covariate_idx], covariate_names, meta["MU"], meta["SIG"]
        )

    horizon = Yte.shape[1]

    # Hard correctness check: the archive's own stored L/H must match the
    # actual array shapes we just built the model input from -- catches any
    # silent lookback/horizon mismatch rather than trusting it implicitly.
    meta_L = int(meta["Lookback window L"])
    meta_H = int(meta["Horizon forecasting window H"])
    if history_phys.shape[1] != meta_L or horizon != meta_H:
        raise ValueError(
            f"{archive}: shape mismatch -- history has {history_phys.shape[1]} "
            f"lookback steps (meta says L={meta_L}), horizon={horizon} "
            f"(meta says H={meta_H})"
        )
    print(f"  [inputs] N={history_phys.shape[0]} windows, lookback L={meta_L}, "
          f"horizon H={meta_H}, history_phys.shape={history_phys.shape}"
          + (f", covariates_phys.shape={covariates_phys.shape}" if covariates_phys is not None else ""))

    return history_phys, covariates_phys, y_true_phys, horizon


def upsert_metrics_row(out_csv: Path, label: str, metrics: dict) -> None:
    row = pd.DataFrame({label: metrics}).T
    if out_csv.is_file():
        existing = pd.read_csv(out_csv, index_col=0)
        existing = existing[existing.index != label]
        combined = pd.concat([existing, row])
    else:
        combined = row
    combined.to_csv(out_csv)


def _skip_existing(archive: Path, label: str, out_csv_name: str) -> bool:
    """Use the same completion check before model loading and before inference."""
    out_csv = archive / out_csv_name
    if out_csv.is_file():
        existing = pd.read_csv(out_csv, index_col=0)
        if label in existing.index:
            print(f"  [skip] {label} already in {out_csv.name}")
            return True
    return False


def run_model(adapter, archive: Path, label: str, *, batch_size: int, out_csv_name: str,
              skip_existing: bool) -> None:
    out_csv = archive / out_csv_name
    if skip_existing and _skip_existing(archive, label, out_csv_name):
        return

    history_phys, covariates_phys, y_true_phys, horizon = build_inputs(
        archive, adapter.covariate_aware
    )
    n = history_phys.shape[0]
    preds = np.empty((n, horizon), dtype=np.float32)
    n_batches = -(-n // batch_size)  # ceil division, for progress visibility
    inference_start = time.perf_counter()
    for batch_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        cov_batch = covariates_phys[start:end] if covariates_phys is not None else None
        batch_start = time.perf_counter()
        preds[start:end] = adapter.predict_batch(history_phys[start:end], horizon, cov_batch)
        batch_elapsed_s = time.perf_counter() - batch_start
        if batch_idx == 0:
            # Concrete, eyeballable proof this window's own lookback -> horizon
            # is what got fed in/out, not something trivial/degenerate.
            print(f"    [sample window 0] lookback[:5]={np.round(history_phys[0, :5], 4)} "
                  f"... lookback[-5:]={np.round(history_phys[0, -5:], 4)}")
            print(f"    [sample window 0] forecast[:5]={np.round(preds[0, :5], 4)} "
                  f"... forecast[-5:]={np.round(preds[0, -5:], 4)}")
            print(f"    [sample window 1] lookback[:5]={np.round(history_phys[1, :5], 4)}"
                  f"  (should differ from window 0's, since they're 1+ steps apart in time)")
        elapsed_s = time.perf_counter() - inference_start
        windows_per_s = end / elapsed_s
        batch_windows_per_s = (end - start) / batch_elapsed_s
        eta_s = (n - end) / windows_per_s if windows_per_s > 0 else math.inf
        print(
            f"    batch {batch_idx + 1}/{n_batches} ({end}/{n} windows)  "
            f"batch={batch_windows_per_s:.1f} windows/s  "
            f"avg={windows_per_s:.1f} windows/s  ETA={eta_s / 60:.1f} min",
            end="\r",
        )
    print()

    yhat_phys = preds[:, :, None]  # (N, H, 1) to match compute_metrics' expected shape
    save_predictions(archive, adapter.key, label, yhat_test_phys=yhat_phys)

    metrics = compute_metrics(y_true_phys, yhat_phys, d_plot=0, include_confidence=True)
    upsert_metrics_row(out_csv, label, metrics)
    print(f"  [ok] {label}: MAPE_all={metrics['MAPE_all']:.4f}%  -> {out_csv.name}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="outputs/test_results")
    parser.add_argument("--archive-tag", default="flax8")
    parser.add_argument("--selector", choices=("manoeuvre", "navigation", "both"), default="both")
    parser.add_argument("--model-specs", default="config/model_specs_foundation.json")
    parser.add_argument("--models", default="chronos2,timesfm,tirex2_univariate")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--timesfm-internal-batch-size", type=int, default=128,
        help="TimesFM per-core inference batch; tune for available VRAM (default: 128).",
    )
    parser.add_argument(
        "--timesfm-force-flip-invariance", action=argparse.BooleanOptionalAction, default=False,
        help="Run TimesFM's additional sign-flipped pass (slower, default: disabled).",
    )
    parser.add_argument(
        "--tirex2-internal-batch-size", type=int, default=None,
        help=("TiRex-2 model.forecast batch. Defaults to 512 for tirex2_univariate "
              "and 8 for dense covariate-aware tirex2."),
    )
    parser.add_argument(
        "--tirex2-force-sign-flip-invariance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run TiRex-2's second sign-flipped pass (roughly 2x cost; default: disabled).",
    )
    parser.add_argument(
        "--tirex2-differencing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable TiRex-2's per-series trend detection/differencing postprocessor "
              "(CPU-heavy for millions of windows; default: disabled)."),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-csv-name", default="results_metrics_foundation.csv")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = _project_path(args.results_dir)
    model_specs_path = _project_path(args.model_specs)

    labels = load_labels(model_specs_path)
    archives = discover_archives(results_dir, selector=args.selector, archive_tag=args.archive_tag)
    if not archives:
        raise FileNotFoundError(
            f"No archives found under {results_dir} for selector={args.selector!r} "
            f"tag={args.archive_tag!r}"
        )
    print(f"Found {len(archives)} archives.")
    timesfm_max_context, timesfm_max_horizon = timesfm_compile_dimensions(archives)
    print(
        f"TimesFM compile dimensions: max_context={timesfm_max_context}, "
        f"max_horizon={timesfm_max_horizon}"
    )

    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    for key in model_keys:
        if key not in ADAPTER_BUILDERS:
            raise ValueError(f"Unknown model key {key!r}; available: {sorted(ADAPTER_BUILDERS)}")
        if key not in labels:
            raise ValueError(f"{key!r} missing from {model_specs_path}")

        # Load a checkpoint only when this model has unfinished archives.
        label = labels[key]
        pending = [archive for archive in archives
                   if not (args.skip_existing and _skip_existing(archive, label, args.out_csv_name))]
        if not pending:
            continue

        print(f"\n=== Loading {key} ===")
        if key == "timesfm":
            adapter = ADAPTER_BUILDERS[key](
                device=args.device,
                per_core_batch_size=args.timesfm_internal_batch_size,
                max_context=timesfm_max_context,
                max_horizon=timesfm_max_horizon,
                force_flip_invariance=args.timesfm_force_flip_invariance,
            )
        elif key in ("tirex2", "tirex2_univariate"):
            adapter = ADAPTER_BUILDERS[key](
                device=args.device,
                internal_batch_size=args.tirex2_internal_batch_size,
                tta_sign_flip=args.tirex2_force_sign_flip_invariance,
                tta_diff=args.tirex2_differencing,
            )
        else:
            adapter = ADAPTER_BUILDERS[key](device=args.device)
        for archive in pending:
            print(f"[{key}] {archive.name}")
            run_model(
                adapter, archive, label,
                batch_size=args.batch_size, out_csv_name=args.out_csv_name,
                skip_existing=args.skip_existing,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
