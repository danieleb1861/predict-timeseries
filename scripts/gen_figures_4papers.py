"""Generate paper figures from Keras or Flax benchmark archives.

The script discovers benchmark archives by geometry instead of relying on
dated directory names. Flax defaults to archives tagged ``flax8`` and supports
both legacy ``params.msgpack`` artifacts and the current Orbax deployment
schema.

Examples
--------
python scripts/gen_figures_4papers.py --backend flax
python scripts/gen_figures_4papers.py --backend keras --archive-tag any
python scripts/gen_figures_4papers.py --backend flax --selector navigation --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import re
import sys
import warnings
from zipfile import BadZipFile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.plots import set_pub_style  # noqa: E402
from load_forecasting.metrics import load_metrics_outputs, parse_metric_vector  # noqa: E402


GRID = (
    (20, 5), (20, 15), (20, 25),
    (45, 5), (45, 15), (45, 25),
    (60, 5), (60, 15), (60, 25),
)
LOOKBACKS = (20, 45, 60)
HORIZONS = (5, 15, 25)
MODE_LABELS = {"manoeuvre": "Manoeuvre", "navigation": "Navigation"}

DEFAULT_FLAX_MODELS = (
    "cnn", "tcn", "transformer", "dlinear", "itransformer",
    "dual_stream_film", "convtcn_hybrid", "convlstmtcn_hybrid",
)
DEFAULT_KERAS_MODELS = ("cnn", "tcn", "convtcn_hybrid")

COLORS = {
    "mlp": "#14b8a6",
    "gru": "#0ea5e9",
    "rnn": "#9467bd",
    "cnn": "#1f77b4",
    "tcn": "#d62728",
    "transformer": "#e377c2",
    "dlinear": "#8c8c22",
    "itransformer": "#17becf",
    "dual_stream_film": "#b44b28",
    "convlstm_hybrid": "#8c564b",
    "tcnlstm_hybrid": "#6f42c1",
    "convtcn_hybrid": "#ff7f0e",
    "convlstmtcn_hybrid": "#2ca02c",
}
LINESTYLES = {
    "mlp": "-",
    "gru": "--",
    "rnn": ":",
    "cnn": "-",
    "tcn": "--",
    "transformer": "-.",
    "dlinear": ":",
    "itransformer": "-",
    "dual_stream_film": "--",
    "convlstm_hybrid": "-",
    "tcnlstm_hybrid": "--",
    "convtcn_hybrid": "-.",
    "convlstmtcn_hybrid": ":",
}

ARCHIVE_RE = re.compile(
    r"_(?P<mode>Manoeuvre|Navigation)_L(?P<lookback>\d+)_?H(?P<horizon>\d+)"
    r"(?P<suffix>(?:_.*)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FigureModel:
    key: str
    label: str
    builder: str
    artifact_name: str
    spec: Mapping[str, object]

    @property
    def color(self) -> str:
        return COLORS.get(self.key, "#333333")

    @property
    def linestyle(self) -> str:
        return LINESTYLES.get(self.key, "-")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("flax", "keras"), default="flax")
    parser.add_argument(
        "--selector",
        choices=("manoeuvre", "navigation", "both"),
        default="both",
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("LOAD_FORECASTING_RESULTS_DIR", "outputs/test_results"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("LOAD_FORECASTING_FIGURE_DIR"),
        help="Default: outputs/figures/paper/<backend>_<archive-tag>.",
    )
    parser.add_argument(
        "--archive-tag",
        default=None,
        help="Required archive suffix. Flax defaults to 'flax8'; use 'any' to disable.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys or labels. Defaults to the backend campaign set.",
    )
    parser.add_argument(
        "--model-specs",
        default=None,
        help="Common model-specs JSON for legacy MsgPack reconstruction.",
    )
    parser.add_argument("--man-model-specs", default=None)
    parser.add_argument("--nav-model-specs", default=None)
    parser.add_argument("--focus-lookback", type=int, default=45)
    parser.add_argument("--focus-horizon", type=int, default=25)
    parser.add_argument("--segment-length", type=int, default=500)
    parser.add_argument(
        "--predict-batch-size", "--batch-size", dest="batch_size", type=int, default=512
    )
    parser.add_argument("--formats", default="eps,png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument(
        "--no-verify-golden",
        action="store_true",
        help="Skip the golden round-trip when loading current Orbax artifacts.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Resolve archives/models and validate metrics without generating figures.",
    )
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def selected_modes(selector: str) -> tuple[str, ...]:
    return tuple(MODE_LABELS) if selector == "both" else (selector,)


def effective_archive_tag(backend: str, archive_tag: str | None) -> str | None:
    if archive_tag is not None and archive_tag.casefold() == "any":
        return None
    if archive_tag is None and backend == "flax":
        return "flax8"
    return archive_tag


def default_spec_path(mode: str, backend: str) -> Path:
    if backend == "flax":
        tuned = REPO_ROOT / "config" / f"model_specs_tuned_{mode}_paper.json"
        if tuned.is_file():
            return tuned
    return REPO_ROOT / "config" / "model_specs.json"


def resolve_spec_path(args: argparse.Namespace, mode: str) -> Path:
    specific = args.man_model_specs if mode == "manoeuvre" else args.nav_model_specs
    value = specific or args.model_specs
    return _project_path(value) if value else default_spec_path(mode, args.backend)


def load_model_catalog(path: Path) -> dict[str, FigureModel]:
    if not path.is_file():
        raise FileNotFoundError(f"Model-specs file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    catalog: dict[str, FigureModel] = {}
    for section in ("deep_models", "hybrid_models"):
        specs = data.get(section, {})
        if not isinstance(specs, dict):
            raise ValueError(f"{path}: {section} must be an object")
        for key, raw_spec in specs.items():
            if not isinstance(raw_spec, dict):
                raise ValueError(f"{path}: model {key!r} must be an object")
            if section == "deep_models":
                artifact_name = str(raw_spec.get("artifact_key", key))
            else:
                artifact_name = str(raw_spec.get("artifact_name", key))
            catalog[key] = FigureModel(
                key=key,
                label=str(raw_spec["label"]),
                builder=str(raw_spec["builder"]),
                artifact_name=artifact_name,
                spec=raw_spec,
            )
    return catalog


def select_models(
    catalog: Mapping[str, FigureModel], backend: str, raw_models: str | None
) -> tuple[FigureModel, ...]:
    defaults = DEFAULT_FLAX_MODELS if backend == "flax" else DEFAULT_KERAS_MODELS
    requested = list(defaults) if raw_models is None else [
        value.strip() for value in raw_models.split(",") if value.strip()
    ]
    aliases = {key.casefold(): key for key in catalog}
    aliases.update({model.label.casefold(): key for key, model in catalog.items()})
    keys = []
    for value in requested:
        key = aliases.get(value.casefold())
        if key is None:
            raise ValueError(
                f"Unknown model {value!r}; available keys: {sorted(catalog)}"
            )
        if key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("At least one model must be selected")
    return tuple(catalog[key] for key in keys)


def parse_archive_coordinates(name: str) -> tuple[str, int, int, tuple[str, ...]] | None:
    match = ARCHIVE_RE.search(name)
    if match is None:
        return None
    mode_label = match.group("mode").casefold()
    mode = "manoeuvre" if mode_label == "manoeuvre" else "navigation"
    suffix = tuple(
        token.casefold() for token in match.group("suffix").strip("_").split("_") if token
    )
    return mode, int(match.group("lookback")), int(match.group("horizon")), suffix


def archive_supports_backend(path: Path, backend: str) -> bool:
    if backend == "keras":
        return any(path.glob("model_*.keras"))
    for artifact in path.glob("model_*_flax"):
        if not artifact.is_dir():
            continue
        if (artifact / "params.msgpack").is_file():
            return True
        if (artifact / "manifest.json").is_file():
            return True
    return False


def discover_archives(
    results_dir: Path,
    *,
    backend: str,
    modes: Iterable[str],
    archive_tag: str | None,
    grid: Sequence[tuple[int, int]] = GRID,
) -> dict[str, dict[tuple[int, int], Path]]:
    """Resolve one newest compatible archive for every requested mode/cell."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    requested_modes = tuple(modes)
    candidates: dict[tuple[str, int, int], list[Path]] = {}
    for path in results_dir.iterdir():
        if not path.is_dir() or not archive_supports_backend(path, backend):
            continue
        parsed = parse_archive_coordinates(path.name)
        if parsed is None:
            continue
        mode, lookback, horizon, suffix = parsed
        if mode not in requested_modes or (lookback, horizon) not in grid:
            continue
        if archive_tag is not None and archive_tag.casefold() not in suffix:
            continue
        candidates.setdefault((mode, lookback, horizon), []).append(path)

    resolved: dict[str, dict[tuple[int, int], Path]] = {
        mode: {} for mode in requested_modes
    }
    missing = []
    for mode in requested_modes:
        for lookback, horizon in grid:
            matches = candidates.get((mode, lookback, horizon), [])
            if not matches:
                missing.append(f"{mode}:L{lookback}/H{horizon}")
                continue
            matches.sort(key=lambda path: (path.name, path.stat().st_mtime_ns))
            resolved[mode][(lookback, horizon)] = matches[-1]
    if missing:
        tag_text = f" tagged {archive_tag!r}" if archive_tag else ""
        raise FileNotFoundError(
            f"Missing {backend}{tag_text} benchmark archives under {results_dir}: "
            + ", ".join(missing)
        )
    return resolved


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_metrics(archive: Path, models: Sequence[FigureModel], *, expected_horizon=None) -> dict[str, dict[str, object]]:
    """Adapt validated metric rows to the figure plotting keys."""
    frame = load_metrics_outputs(archive, expected_horizon=expected_horizon)
    selected_labels = {model.label for model in models}
    missing = selected_labels - {str(value).strip() for value in frame.index}
    if missing:
        raise ValueError(f"{archive} is missing selected models: {sorted(missing)}")

    result = {}
    for name, row in frame.iterrows():
        label = str(name).strip()
        if label not in selected_labels:
            continue
        result[label] = {
            "RMSE_all": float(row.get("RMSE_all", np.nan)),
            "MAE_all": float(row.get("MAE_all", np.nan)),
            "MAPE_all": float(row.get("MAPE_all", np.nan)),
            "MAPE_ci_low": float(row.get("MAPE_boot_ci95_low", np.nan)),
            "MAPE_ci_high": float(row.get("MAPE_boot_ci95_high", np.nan)),
            "RMSE_h": row.get("RMSE_horizon"),
            "MAE_h": row.get("MAE_horizon"),
            "MAPE_h": row.get("MAPE_horizon"),
        }
    return result


def _builder_definition(
    artifact_dir: Path,
    model: FigureModel,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> tuple[str, dict[str, object]]:
    """Preserve the figure helper interface using the shared legacy resolver."""
    from load_forecasting.deployment_flax import legacy_flax_builder

    return legacy_flax_builder(
        artifact_dir, {**model.spec, "builder": model.builder}, input_shape,
        output_shape, tuple(feature_names), tuple(target_names),
    )


def predict_flax_msgpack(
    artifact_dir: Path,
    model: FigureModel,
    inputs,
    *,
    output_shape: tuple[int, int],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Predict legacy standardized windows through the shared artifact loader."""
    from load_forecasting.deployment_flax import load_flax_inference

    inputs = np.asarray(inputs, dtype=np.float32)
    loaded = load_flax_inference(
        artifact_dir, spec={**model.spec, "builder": model.builder},
        input_shape=inputs.shape[1:], output_shape=output_shape,
        feature_names=feature_names, target_names=target_names,
    )
    return loaded.predict_standardized(inputs, batch_size=batch_size)


def predict_flax(
    archive: Path,
    model: FigureModel,
    inputs,
    *,
    output_shape: tuple[int, int],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    feature_mean,
    feature_scale,
    target_mean,
    target_scale,
    batch_size: int,
    verify_golden: bool,
) -> np.ndarray:
    """Validate archived preprocessing and predict with the shared Flax loader."""
    from load_forecasting.deployment_flax import load_flax_inference

    loaded = load_flax_inference(
        archive / f"model_{model.artifact_name}_flax",
        spec={**model.spec, "builder": model.builder},
        input_shape=inputs.shape[1:], output_shape=output_shape,
        feature_names=feature_names, target_names=target_names,
        feature_mean=feature_mean, feature_scale=feature_scale,
        target_mean=target_mean, target_scale=target_scale, verify_golden=verify_golden,
    )
    return loaded.predict_standardized(inputs, batch_size=batch_size)


def predict_keras(
    archive: Path,
    model: FigureModel,
    inputs,
    *,
    output_shape: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tcn import TCN  # noqa: F401

    model_path = archive / f"model_{model.artifact_name}.keras"
    require_file(model_path, "Keras model")
    # These local benchmark artifacts predate Keras safe-mode Lambda restrictions.
    loaded = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    prediction = np.asarray(
        loaded.predict(inputs, batch_size=batch_size, verbose=0), dtype=np.float32
    )
    expected = (len(inputs), *output_shape)
    if prediction.shape != expected:
        if prediction.size != int(np.prod(expected)):
            raise ValueError(
                f"{model_path}: prediction shape {prediction.shape} cannot become {expected}"
            )
        prediction = prediction.reshape(expected)
    tf.keras.backend.clear_session()
    return prediction


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("paper_figures_backend")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "figure_generation.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def parse_formats(raw_formats: str) -> tuple[str, ...]:
    formats = tuple(
        value.strip().casefold() for value in raw_formats.split(",") if value.strip()
    )
    unsupported = set(formats) - {"eps", "png", "pdf"}
    if not formats or unsupported:
        raise ValueError(
            f"--formats must contain eps, png, or pdf; unsupported: {sorted(unsupported)}"
        )
    return formats


def save_figure(
    figure,
    output_dir: Path,
    name: str,
    *,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    for figure_format in formats:
        path = output_dir / f"{name}.{figure_format}"
        with warnings.catch_warnings():
            if figure_format == "eps":
                warnings.simplefilter("ignore")
            figure.savefig(
                path,
                format=figure_format,
                dpi=dpi if figure_format == "png" else None,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        logger.info("  Saved: %s", path)
    plt.close(figure)


def best_model_per_config(
    data: Mapping[tuple[int, int], Mapping[str, Mapping[str, object]]]
) -> dict[tuple[int, int], tuple[str, Mapping[str, object]]]:
    return {
        cell: min(metrics.items(), key=lambda item: float(item[1]["MAPE_all"]))
        for cell, metrics in data.items()
    }


def make_heatmap(
    best: Mapping[tuple[int, int], tuple[str, Mapping[str, object]]],
    suffix: str,
    *,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    matrix = np.zeros((len(LOOKBACKS), len(HORIZONS)), dtype=np.float64)
    annotations = np.empty_like(matrix, dtype=object)
    for row, lookback in enumerate(LOOKBACKS):
        for column, horizon in enumerate(HORIZONS):
            label, values = best[(lookback, horizon)]
            mape = float(values["MAPE_all"])
            matrix[row, column] = mape
            annotations[row, column] = f"{mape:.2f}%\n{label}"

    colormap = LinearSegmentedColormap.from_list(
        "paper_error", ("#1a6e25", "#f9f870", "#b51c1c"), N=256
    )
    figure, axis = plt.subplots(figsize=(3.8, 2.8))
    image = axis.imshow(
        matrix,
        cmap=colormap,
        aspect="auto",
        vmin=float(matrix.min()) * 0.85,
        vmax=float(matrix.max()) * 1.05,
    )
    axis.set_xticks(range(len(HORIZONS)), [f"$H={value}$ s" for value in HORIZONS])
    axis.set_yticks(range(len(LOOKBACKS)), [f"$L={value}$ s" for value in LOOKBACKS])
    axis.set_xlabel("Forecast horizon $H$")
    axis.set_ylabel("Lookback $L$")
    for row in range(len(LOOKBACKS)):
        for column in range(len(HORIZONS)):
            axis.text(
                column,
                row,
                annotations[row, column],
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Best MAPE (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        f"heatmap_best_mape_{suffix}",
        formats=formats,
        dpi=dpi,
        logger=logger,
    )


def sensitivity_vs_horizon(
    data: Mapping[tuple[int, int], Mapping[str, Mapping[str, object]]],
    lookback: int,
    suffix: str,
    models: Sequence[FigureModel],
    *,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    """Plot horizons at a fixed lookback using the shared renderer."""
    _plot_sensitivity(
        data, [(lookback, horizon) for horizon in HORIZONS], HORIZONS, models,
        "Horizon $H$ (s)", "o", f"mape_vs_horizon_L{lookback}_{suffix}",
        output_dir, formats, dpi, logger,
    )


def _plot_sensitivity(data, cells, steps, models, xlabel, marker, filename,
                      output_dir, formats, dpi, logger) -> None:
    """Render both sensitivity axes with identical styling and export settings."""
    figure, axis = plt.subplots(figsize=(3.3, 2.6))
    for model in models:
        values = [float(data[cell][model.label]["MAPE_all"]) for cell in cells]
        axis.plot(
            steps,
            values,
            color=model.color,
            linestyle=model.linestyle,
            label=model.label,
            marker=marker,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(steps)
    axis.legend(fontsize=6.5, frameon=True, ncol=2)
    axis.grid(True)
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        filename,
        formats=formats,
        dpi=dpi,
        logger=logger,
    )


def sensitivity_vs_lookback(
    data: Mapping[tuple[int, int], Mapping[str, Mapping[str, object]]],
    horizon: int,
    suffix: str,
    models: Sequence[FigureModel],
    *,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    """Plot lookbacks at a fixed horizon using the shared renderer."""
    _plot_sensitivity(
        data, [(lookback, horizon) for lookback in LOOKBACKS], LOOKBACKS, models,
        "Lookback $L$ (s)", "s", f"mape_vs_lookback_H{horizon}_{suffix}",
        output_dir, formats, dpi, logger,
    )


def load_reference_mape(archive: Path) -> dict[str, float]:
    """Best zero-shot foundation model and last-value MAPE on the archived test windows."""
    from load_forecasting.scaling import inverse_scale_targets

    foundation = pd.read_csv(archive / "results_metrics_foundation.csv", index_col=0)
    windows = np.load(archive / "windows.npz")
    with open(archive / "scalers_and_meta.pkl", "rb") as handle:
        meta = pickle.load(handle)
    inverse = lambda z: inverse_scale_targets(z, meta["targetNames"], meta["MU"], meta["SIG"])[..., 0]  # noqa: E731
    target = inverse(windows["Yte"])
    last = np.repeat(inverse(windows["Xte"][:, -1:, :1]), target.shape[1], axis=1)
    persistence = 100.0 * np.mean(np.abs(last - target) / np.maximum(np.abs(target), 1e-6))
    return {"foundation": float(foundation["MAPE_all"].min()), "persistence": float(persistence)}


def sensitivity_summary(data, references, cells, steps, xlabel, marker, filename,
                        output_dir, formats, dpi, logger) -> None:
    """Best trained model, best foundation model and last measured value with the trained-model range."""
    trained = np.array([[float(m["MAPE_all"]) for m in data[cell].values()] for cell in cells])
    figure, axis = plt.subplots(figsize=(3.3, 2.6))
    axis.fill_between(steps, trained.min(axis=1), trained.max(axis=1),
                      color="#d2e3f0", linewidth=0, label="Trained models (range)")  # opaque: EPS has no alpha
    axis.plot(steps, trained.min(axis=1), color="#1f77b4", linestyle="-", marker=marker,
              label="Best trained model")
    axis.plot(steps, [references[cell]["foundation"] for cell in cells], color="#ff7f0e",
              linestyle="-.", marker=marker, label="Best foundation model")
    axis.plot(steps, [references[cell]["persistence"] for cell in cells], color="black",
              linestyle="--", label="Last measured value")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(steps)
    axis.legend(fontsize=6.5, frameon=True)
    axis.grid(True)
    figure.tight_layout()
    save_figure(figure, output_dir, filename, formats=formats, dpi=dpi, logger=logger)


def error_profiles(
    metrics: Mapping[str, Mapping[str, object]],
    horizon: int,
    suffix: str,
    models: Sequence[FigureModel],
    *,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    steps = np.arange(1, horizon + 1)
    profiles = (
        ("RMSE_h", "RMSE (pu)", "error_rmse"),
        ("MAE_h", "MAE (pu)", "error_mae"),
        ("MAPE_h", "MAPE (%)", "error_mape"),
    )
    for metric_key, ylabel, filename in profiles:
        figure, axis = plt.subplots(figsize=(3.3, 2.6))
        for model in models:
            values = metrics[model.label][metric_key]
            if values is None or len(values) != horizon:
                raise ValueError(
                    f"{model.label} {metric_key} has length "
                    f"{0 if values is None else len(values)}, expected {horizon}"
                )
            axis.plot(
                steps,
                values,
                color=model.color,
                linestyle=model.linestyle,
                label=model.label,
            )
        axis.set_xlabel("Horizon step (s)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, horizon)
        axis.set_xticks(np.arange(0, horizon + 1, 5))
        axis.legend(fontsize=6.5, frameon=True, ncol=2)
        axis.grid(True)
        figure.tight_layout()
        save_figure(
            figure,
            output_dir,
            f"{filename}_{suffix}",
            formats=formats,
            dpi=dpi,
            logger=logger,
        )


def bar_chart(
    metrics: Mapping[str, Mapping[str, object]],
    suffix: str,
    models: Sequence[FigureModel],
    focus: tuple[int, int],
    *,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    mapes = np.asarray([metrics[model.label]["MAPE_all"] for model in models], dtype=float)
    ci_low = np.asarray([metrics[model.label]["MAPE_ci_low"] for model in models], dtype=float)
    ci_high = np.asarray([metrics[model.label]["MAPE_ci_high"] for model in models], dtype=float)
    yerr = None
    if np.any(np.isfinite(ci_low)) and np.any(np.isfinite(ci_high)):
        yerr = np.vstack((np.maximum(0.0, mapes - ci_low), np.maximum(0.0, ci_high - mapes)))

    figure, axis = plt.subplots(figsize=(4.6, 2.8))
    bars = axis.bar(
        range(len(models)),
        mapes,
        yerr=yerr,
        capsize=3,
        color=[model.color for model in models],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )
    axis.set_xticks(range(len(models)), [model.label for model in models], rotation=30, ha="right")
    axis.set_ylabel("MAPE (%)")
    axis.grid(axis="y")
    offset = 0.02 * float(np.nanmax(mapes))
    for bar, value in zip(bars, mapes):
        if np.isfinite(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        f"bar_mape_L{focus[0]}H{focus[1]}_{suffix}",
        formats=formats,
        dpi=dpi,
        logger=logger,
    )


def validate_model_artifacts(
    archive: Path, backend: str, models: Sequence[FigureModel]
) -> None:
    missing = []
    for model in models:
        if backend == "keras":
            path = archive / f"model_{model.artifact_name}.keras"
            if not path.is_file():
                missing.append(str(path))
            continue
        artifact = archive / f"model_{model.artifact_name}_flax"
        legacy = (artifact / "params.msgpack").is_file()
        current = (artifact / "manifest.json").is_file() and (artifact / "params").is_dir()
        if not legacy and not current:
            missing.append(str(artifact))
    if missing:
        raise FileNotFoundError(
            f"{archive} is missing selected {backend} model artifacts: {missing}"
        )


def load_window_metadata(archive: Path):
    windows_path = archive / "windows.npz"
    metadata_path = archive / "scalers_and_meta.pkl"
    require_file(windows_path, "windows file")
    require_file(metadata_path, "scaler metadata")
    with np.load(windows_path, allow_pickle=False) as windows:
        Xte = np.asarray(windows["Xte"], dtype=np.float32)
        Yte = np.asarray(windows["Yte"], dtype=np.float32)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    return Xte, Yte, metadata


def _hash_prediction_files(digest, paths):
    """Hash source/artifact contents in bounded chunks, including file names."""
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)


def _prediction_context(archive, inputs, output_shape, batch_size):
    """Bind cached predictions to windows, scaling, code, and installed runtimes."""
    versions = {}
    for package in ("numpy", "jax", "jaxlib", "flax", "tensorflow", "keras", "keras-tcn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    digest = hashlib.sha256(json.dumps(
        [1, inputs.shape, str(inputs.dtype), output_shape, batch_size, versions], sort_keys=True,
    ).encode())
    # Bound copies even when callers supply noncontiguous windows.
    for start in range(0, len(inputs), 256):
        digest.update(np.ascontiguousarray(inputs[start:start + 256]).tobytes())
    _hash_prediction_files(digest, [
        archive / "scalers_and_meta.pkl", Path(__file__),
        REPO_ROOT / "load_forecasting/deployment_flax.py",
        *(REPO_ROOT / "load_forecasting/models").glob("*.py"),
    ])
    return digest


def _load_figure_predictions(path, fingerprint, count, logger):
    """Treat missing, stale, or damaged caches as a request for fresh inference."""
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["fingerprint"]) != fingerprint:
                    return None
                values = cached["physical_first_horizon"]
            if values.shape == (count,) and values.dtype == np.float64 and np.isfinite(values).all():
                return values
        except (OSError, ValueError, KeyError, EOFError, TypeError, BadZipFile):
            pass
        logger.info("  Ignoring invalid prediction cache: %s", path)
    return None


def timeseries_comparison(
    archive: Path,
    suffix: str,
    backend: str,
    models: Sequence[FigureModel],
    *,
    segment_length: int,
    batch_size: int,
    verify_golden: bool,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    logger: logging.Logger,
) -> None:
    validate_model_artifacts(archive, backend, models)
    Xte, Yte, metadata = load_window_metadata(archive)
    if Xte.ndim != 3 or Yte.ndim != 3 or len(Xte) != len(Yte):
        raise ValueError(f"Invalid test-window shapes: Xte={Xte.shape}, Yte={Yte.shape}")
    feature_names = tuple(metadata["featureNames"])
    target_names = tuple(metadata["targetNames"])
    if Xte.shape[2] != len(feature_names) or Yte.shape[2] != len(target_names):
        raise ValueError("Window channels do not match scaler metadata")
    target_mean = np.asarray(metadata["MU"].loc[list(target_names)], dtype=np.float64)
    target_scale = np.asarray(metadata["SIG"].loc[list(target_names)], dtype=np.float64)
    feature_mean = np.asarray(metadata["MU"].loc[list(feature_names)], dtype=np.float64)
    feature_scale = np.asarray(metadata["SIG"].loc[list(feature_names)], dtype=np.float64)
    stride = int(metadata["WindowSlidingCoeff"])
    if stride < 1:
        raise ValueError(f"Invalid WindowSlidingCoeff: {stride}")

    def inverse_target(values):
        return (
            np.asarray(values, dtype=np.float64)
            * target_scale.reshape((1,) * (np.ndim(values) - 1) + (-1,))
            + target_mean.reshape((1,) * (np.ndim(values) - 1) + (-1,))
        )

    y_true = inverse_target(Yte)[:, 0, 0]
    predictions = {}
    output_shape = tuple(int(value) for value in Yte.shape[1:])
    context = _prediction_context(archive, Xte, output_shape, batch_size)
    for model in models:
        # Legacy training caches lack provenance; keep verified caches with figures.
        digest = context.copy()
        digest.update(json.dumps([backend, model.key, model.label, model.builder, model.spec], sort_keys=True).encode())
        artifact = archive / (f"model_{model.artifact_name}_flax" if backend == "flax"
                              else f"model_{model.artifact_name}.keras")
        _hash_prediction_files(digest, [p for p in artifact.rglob("*") if p.is_file()]
                               if artifact.is_dir() else [artifact])
        fingerprint = digest.hexdigest()
        cache_path = output_dir / "prediction_cache" / archive.name / backend / f"{model.key}.npz"
        physical = _load_figure_predictions(cache_path, fingerprint, len(Xte), logger)
        logger.info("  %s %s from %s", "Reusing predictions for" if physical is not None else "Predicting",
                    model.label, archive.name)
        if backend == "flax":
            # Loading an empty batch preserves contract/golden checks on cache hits.
            standardized = predict_flax(
                archive,
                model,
                Xte if physical is None else Xte[:0],
                output_shape=output_shape,
                feature_names=feature_names,
                target_names=target_names,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                target_mean=target_mean,
                target_scale=target_scale,
                batch_size=batch_size,
                verify_golden=verify_golden,
            )
        elif physical is None:
            standardized = predict_keras(
                archive,
                model,
                Xte,
                output_shape=output_shape,
                batch_size=batch_size,
            )
        if physical is None:
            physical = inverse_target(standardized)[:, 0, 0]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, fingerprint=np.asarray(fingerprint),
                                physical_first_horizon=physical)
        mape = 100.0 * np.mean(
            np.abs(physical - y_true) / np.maximum(np.abs(y_true), 1e-8)
        )
        logger.info("    First-horizon MAPE=%.4f%%", mape)
        predictions[model.label] = physical

    length = min(int(segment_length), len(y_true))
    if length < 1:
        raise ValueError("Cannot plot an empty test set")
    max_start = len(y_true) - length
    starts = np.unique(np.linspace(0, max_start, num=min(20, max_start + 1), dtype=int))
    best_start = 0
    best_score = -np.inf
    for start in starts:
        segment = y_true[start : start + length]
        score = float(np.ptp(segment)) if suffix == "man" else float(np.std(segment))
        if score > best_score:
            best_start, best_score = int(start), score
    logger.info("  Segment start=%d, score=%.6f", best_start, best_score)

    time = Xte.shape[1] + (best_start + np.arange(length)) * stride
    figure, axis = plt.subplots(figsize=(6.5, 2.8))
    axis.plot(
        time,
        y_true[best_start : best_start + length],
        color="black",
        linewidth=1.8,
        label="Measured",
        zorder=10,
    )
    for model in models:
        axis.plot(
            time,
            predictions[model.label][best_start : best_start + length],
            color=model.color,
            linestyle=model.linestyle,
            linewidth=1.0,
            label=model.label,
            zorder=5,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Active power (pu)")
    axis.legend(fontsize=6.5, frameon=True, ncol=2, loc="upper right")
    axis.grid(True)
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        f"models_comparison_full_test_{suffix}",
        formats=formats,
        dpi=dpi,
        logger=logger,
    )


def _validate_catalogs(
    catalogs: Mapping[str, Mapping[str, FigureModel]], selected: Sequence[FigureModel]
) -> None:
    for mode, catalog in catalogs.items():
        for model in selected:
            if model.key not in catalog:
                raise ValueError(f"{mode} model specs do not contain {model.key!r}")
            candidate = catalog[model.key]
            if (candidate.label, candidate.artifact_name) != (
                model.label,
                model.artifact_name,
            ):
                raise ValueError(
                    f"Model identity differs across specs for {model.key!r}: "
                    f"{candidate} vs {model}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    modes = selected_modes(args.selector)
    tag = effective_archive_tag(args.backend, args.archive_tag)
    results_dir = _project_path(args.results_dir)
    output_dir = (
        _project_path(args.output_dir)
        if args.output_dir
        else REPO_ROOT
        / "outputs"
        / "figures"
        / "paper"
        / f"{args.backend}_{tag or 'any'}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    formats = parse_formats(args.formats)
    if args.batch_size < 1 or args.segment_length < 1 or args.dpi < 1:
        raise ValueError("Batch size, segment length, and DPI must be positive")

    set_pub_style(font_family="Times New Roman", fontsize=9)
    catalogs = {
        mode: load_model_catalog(resolve_spec_path(args, mode)) for mode in modes
    }
    first_catalog = catalogs[modes[0]]
    selected = select_models(first_catalog, args.backend, args.models)
    _validate_catalogs(catalogs, selected)
    mode_models = {
        mode: tuple(catalogs[mode][model.key] for model in selected) for mode in modes
    }
    archives = discover_archives(
        results_dir,
        backend=args.backend,
        modes=modes,
        archive_tag=tag,
    )

    logger.info("Backend: %s", args.backend)
    logger.info("Archive tag: %s", tag or "any")
    logger.info("Models: %s", ", ".join(model.label for model in selected))
    metrics_by_mode = {}
    for mode in modes:
        logger.info("%s archives:", MODE_LABELS[mode])
        metrics_by_mode[mode] = {}
        for cell in GRID:
            archive = archives[mode][cell]
            logger.info("  L%d/H%d -> %s", cell[0], cell[1], archive.name)
            metrics_by_mode[mode][cell] = load_metrics(archive, mode_models[mode], expected_horizon=cell[1])

    focus = (args.focus_lookback, args.focus_horizon)
    if focus not in GRID:
        raise ValueError(f"Focus cell {focus} is outside the benchmark grid {GRID}")
    for mode in modes:
        validate_model_artifacts(archives[mode][focus], args.backend, mode_models[mode])
    if args.check_only:
        logger.info("Archive and metric validation complete.")
        return 0

    suffixes = {"manoeuvre": "man", "navigation": "nav"}
    for mode in modes:
        suffix = suffixes[mode]
        models = mode_models[mode]
        data = metrics_by_mode[mode]
        make_heatmap(
            best_model_per_config(data),
            suffix,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
            logger=logger,
        )
        for lookback in LOOKBACKS:
            sensitivity_vs_horizon(
                data,
                lookback,
                suffix,
                models,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                logger=logger,
            )
        for horizon in HORIZONS:
            sensitivity_vs_lookback(
                data,
                horizon,
                suffix,
                models,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                logger=logger,
            )
        references = {cell: load_reference_mape(archives[mode][cell]) for cell in GRID}
        sensitivity_summary(
            data, references, [(focus[0], horizon) for horizon in HORIZONS], HORIZONS,
            "Horizon $H$ (s)", "o", f"mape_summary_vs_horizon_L{focus[0]}_{suffix}",
            output_dir, formats, args.dpi, logger,
        )
        sensitivity_summary(
            data, references, [(lookback, focus[1]) for lookback in LOOKBACKS], LOOKBACKS,
            "Lookback $L$ (s)", "s", f"mape_summary_vs_lookback_H{focus[1]}_{suffix}",
            output_dir, formats, args.dpi, logger,
        )
        error_profiles(
            data[focus],
            focus[1],
            suffix,
            models,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
            logger=logger,
        )
        if not args.skip_inference:
            timeseries_comparison(
                archives[mode][focus],
                suffix,
                args.backend,
                models,
                segment_length=args.segment_length,
                batch_size=args.batch_size,
                verify_golden=not args.no_verify_golden,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                logger=logger,
            )
        bar_chart(
            data[focus],
            suffix,
            models,
            focus,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
            logger=logger,
        )
    logger.info("All figures generated in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
