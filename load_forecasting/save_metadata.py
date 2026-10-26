"""Run summary and metadata persistence helpers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROVENANCE_PACKAGES = (
    "load-forecasting",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "tensorflow",
    "keras",
    "xgboost",
    "jax",
    "flax",
    "optax",
)

PUBLIC_SUMMARY_KEYS = (
    "created_at",
    "preset",
    "selector",
    "models",
    "lookback_steps",
    "horizon_steps",
    "window_sliding_coeff",
    "sampling",
    "training_overrides",
    "branches",
    "features",
    "targets",
    "shapes",
    "samples",
    "metrics_schema_version",
)


def _sha256_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root):
    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _package_versions():
    versions = {}
    for name in PROVENANCE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _input_fingerprint(data_dir, input_files):
    records = []
    for name in input_files:
        path = Path(data_dir) / name
        stat = path.stat()
        records.append({
            "name": Path(name).as_posix(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "method": "sha256(name,size_bytes,mtime_ns)",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def collect_run_provenance(repo_root, config_file, preset_file, model_specs_file,
                           data_dir, input_files, backend):
    """Collect lightweight, JSON-serializable provenance for a forecasting run."""
    return {
        "git": _git_metadata(repo_root),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "backend": backend,
            "default_backend": None,
            "devices": [],
        },
        "packages": _package_versions(),
        "config_sha256": {
            "system": _sha256_file(config_file),
            "presets": _sha256_file(preset_file),
            "model_specs": _sha256_file(model_specs_file),
        },
        "input_fingerprint": _input_fingerprint(data_dir, input_files),
    }


def selected_model_names(model_labels, models_to_run):
    return [model_labels[key] for key in model_labels if key in models_to_run]


def build_run_summary(
    cli_config,
    selector,
    files,
    run_dir,
    imgs_dir,
    model_labels,
    models_to_run,
    LookbackSteps,
    HorizonSteps,
    WindowSlidingCoeff,
    Ts,
    TsH,
    max_fill_seconds,
    EDA,
    PostTraining,
    BESS,
    DPCA,
    PCAplot,
    system_config,
    extra=None,
):
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": cli_config.config,
        "preset": cli_config.preset,
        "preset_file": cli_config.preset_file,
        "model_specs_file": getattr(cli_config, "model_specs", None) or "config/model_specs.json",
        "selector": selector,
        "input_files": files,
        "run_dir": run_dir,
        "image_dir": imgs_dir,
        "models": selected_model_names(model_labels, models_to_run),
        "lookback_steps": LookbackSteps,
        "horizon_steps": HorizonSteps,
        "window_sliding_coeff": WindowSlidingCoeff,
        "sampling": {
            "Ts": Ts,
            "TsH_seconds": TsH,
            "max_fill_seconds": max_fill_seconds,
        },
        "training_overrides": {
            "epochs": cli_config.epochs,
            "batch_size": cli_config.batch_size,
            "learning_rate": cli_config.learning_rate,
            "predict_batch_size": cli_config.predict_batch_size,
        },
        "branches": {
            "eda": EDA,
            "post_training_plots": PostTraining,
            "bess": BESS,
            "dpca": DPCA,
            "pca_plot": PCAplot,
        },
        "system_config": system_config,
    }
    if extra:
        summary.update(extra)
    return summary


def print_run_summary(summary):
    print("Run summary")
    print(f"  selector: {summary['selector']}")
    print(f"  preset: {summary['preset'] or 'none'}")
    print(f"  run_dir: {summary['run_dir']}")
    print(
        "  windows: "
        f"L={summary['lookback_steps']}, H={summary['horizon_steps']}, "
        f"slide={summary['window_sliding_coeff']}"
    )
    print(f"  models: {', '.join(summary['models'])}")
    print(
        "  overrides: "
        f"epochs={summary['training_overrides']['epochs']}, "
        f"batch_size={summary['training_overrides']['batch_size']}, "
        f"lr={summary['training_overrides']['learning_rate']}, "
        f"predict_batch={summary['training_overrides']['predict_batch_size']}"
    )
    print(
        "  branches: "
        f"eda={summary['branches']['eda']}, "
        f"post_training_plots={summary['branches']['post_training_plots']}"
    )
    print()


def build_public_run_summary(summary):
    """Return an allowlisted run summary suitable for public sharing."""
    public = {"public_summary_schema_version": 1}
    public.update({key: summary[key] for key in PUBLIC_SUMMARY_KEYS if key in summary})

    provenance = summary.get("provenance", {})
    git = provenance.get("git", {})
    runtime = provenance.get("runtime", {})
    config_sha256 = provenance.get("config_sha256", {})
    input_fingerprint = provenance.get("input_fingerprint", {})
    public["provenance"] = {
        "git": {key: git[key] for key in ("commit", "dirty") if key in git},
        "runtime": {
            key: runtime[key]
            for key in (
                "python",
                "implementation",
                "platform",
                "backend",
                "default_backend",
                "devices",
            )
            if key in runtime
        },
        "packages": dict(provenance.get("packages", {})),
        "config_sha256": {
            key: config_sha256[key]
            for key in ("system", "presets", "model_specs")
            if key in config_sha256
        },
        "input_fingerprint": {
            key: input_fingerprint[key]
            for key in ("method", "sha256")
            if key in input_fingerprint
        },
    }
    return public


def save_run_summary(summary, run_dir):
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "run_summary_public.json"), "w", encoding="utf-8") as f:
        json.dump(build_public_run_summary(summary), f, indent=2)
