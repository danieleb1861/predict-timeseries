"""Persistence-relative error reduction figures.

Plots

    Delta_pers = 1 - E_model / E_pers        (reported in %)

the relative error reduction of each trained architecture with respect to the
naive persistence forecast y_hat(t+h) = y(t), evaluated on the *identical*
strided test windows.

Two products:

  (a) quadrant scatter -- aggregate MAPE Delta_pers (x) against upper-tail
      RMSE^95 Delta_pers (y), one panel per operating mode, one marker per
      (model, L, H) cell plus a labelled per-model centroid. The origin is
      persistence: the upper-right quadrant is "better than persistence on both
      the typical error and the peaks".

  (b) diverging heatmap -- upper-tail RMSE^95 Delta_pers for every
      (model, L, H) cell, one block per operating mode.

Event-conditioned Delta_pers is read from the per-archive persistence_skill.csv
written by scripts/compute_persistence_peak_metrics.py. Aggregate-MAPE
Delta_pers needs the persistence MAPE on the same window set, which is not
cached by the benchmark; it is recomputed here from windows.npz and cached to
--pers-cache so subsequent runs are fast.

Example:
    python scripts/gen_persistence_delta_figures.py
    python scripts/gen_persistence_delta_figures.py --formats png
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.scaling import inverse_scale_targets  # noqa: E402

ARCHIVE_RE = re.compile(r"ForecastModels_(?P<mode>\w+?)_L(?P<L>\d+)_H(?P<H>\d+)_")

# Paper roster, in the order used by Table "Peak-aware and energy metrics".
MODEL_ORDER = (
    "CNN", "TCN", "iTransformer", "Transformer",
    "ConvTCN-H", "ConvLSTMTCN-H", "DualStream-FiLM",
)
# Identity is carried by marker shape + direct label, never by hue alone: the
# repository COLORS registry is not colourblind-safe (ConvTCN-H vs
# ConvLSTMTCN-H separate by dE 0.7 under protanopia).
MARKERS = {
    "CNN": "o", "TCN": "s", "iTransformer": "^", "Transformer": "D",
    "ConvTCN-H": "v", "ConvLSTMTCN-H": "P", "DualStream-FiLM": "X",
}

INK = "#1a1a1a"
MUTED = "#8a8a8a"
GAIN = "#1b6ca8"   # cool pole: model better than persistence
LOSS = "#b03a2e"   # warm pole: model worse than persistence
NEUTRAL = "#f2f2f0"

PLOT_STYLE = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.3, "grid.alpha": 0.3,
    "axes.edgecolor": "#555555",
    # Match the manuscript body font; STIX is the Times-metric math companion.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # Embed TrueType rather than Type 3 so the EPS survives journal preflight.
    "ps.fonttype": 42, "pdf.fonttype": 42,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "outputs" / "test_results"))
    parser.add_argument("--archive-glob", default="*_flax8_legsafe")
    parser.add_argument("--output-dir",
                        default=str(REPO_ROOT / "outputs" / "figures" / "paper" / "persistence_delta"))
    parser.add_argument("--pers-cache",
                        default=str(REPO_ROOT / "outputs" / "test_results" / "persistence_mape_samewindows.csv"))
    parser.add_argument("--latency-dir",
                        default=str(REPO_ROOT / "outputs" / "figures" / "paper" / "latency_cpu_flax"))
    parser.add_argument("--formats", default="png,eps")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--exclude", default="DLinear",
                        help="Comma-separated model labels to omit.")
    return parser.parse_args(argv)


def archives(results_dir: Path, pattern: str):
    for path in sorted(results_dir.glob(pattern)):
        match = ARCHIVE_RE.search(path.name)
        if match:
            yield path, match.group("mode"), int(match.group("L")), int(match.group("H"))


def persistence_mape_samewindows(results_dir: Path, pattern: str, cache: Path) -> pd.DataFrame:
    """MAPE of y_hat(t+h)=y(t) on each archive's own strided test windows."""
    if cache.exists():
        return pd.read_csv(cache)

    rows = []
    for path, mode, L, H in archives(results_dir, pattern):
        windows = np.load(path / "windows.npz")
        with (path / "scalers_and_meta.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        y_true = inverse_scale_targets(
            windows["Yte"], meta["targetNames"], meta["MU"], meta["SIG"])[:, :, 0]
        last = inverse_scale_targets(
            windows["Xte"][:, -1, 0][:, None], meta["targetNames"],
            meta["MU"], meta["SIG"])[:, 0]
        y_pers = np.repeat(last[:, None], y_true.shape[1], axis=1)
        ape = np.abs(y_pers - y_true) / np.maximum(np.abs(y_true), 1e-6) * 100.0
        rows.append(dict(mode=mode, L=L, H=H, n_windows=int(y_true.shape[0]),
                         mape_pers=float(ape.mean())))
        print(f"  persistence {mode} L={L} H={H}: {ape.mean():.4f}% "
              f"({y_true.shape[0]} windows)", flush=True)

    frame = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


EVENT_METRICS = {
    "d_tail_rmse": "S_E_tail_rmse_pu",
    "d_energy_mean": "S_E_energy_err_mean_kwh",
    "d_ramp_steep": "S_E_mape_ramp_steep_%",
}
EVENT_LABELS = {
    "d_tail_rmse": r"upper-tail $\mathrm{RMSE}^{95}$",
    "d_energy_mean": r"mean energy error $\bar{\varepsilon}_E$",
    "d_ramp_steep": r"steep-ramp \mape",
}
FOUNDATION_ORDER = ("Chronos-2", "TimesFM", "TiRex-2")


def collect(results_dir: Path, pattern: str, pers: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    """One row per (mode, L, H, predictor): raw MAPE plus every Delta_pers, in %."""
    key = {(r["mode"], r.L, r.H): r.mape_pers for _, r in pers.iterrows()}
    rows = []
    for path, mode, L, H in archives(results_dir, pattern):
        metrics = pd.read_csv(path / "results_metrics.csv").set_index("Unnamed: 0")
        skill = pd.read_csv(path / "persistence_skill.csv", index_col=0)
        for model in metrics.index:
            if model in exclude or model not in skill.index:
                continue
            mape = float(metrics.loc[model, "MAPE_all"])
            row = dict(mode=mode, L=L, H=H, model=model, family="trained", mape=mape,
                       d_mape=100.0 * (1.0 - mape / key[(mode, L, H)]))
            for name, column in EVENT_METRICS.items():
                row[name] = 100.0 * float(skill.loc[model, column])
            rows.append(row)

        foundation_csv = path / "results_metrics_foundation.csv"
        if foundation_csv.exists():
            foundation = pd.read_csv(foundation_csv).set_index("Unnamed: 0")
            for label in foundation.index:
                mape = float(foundation.loc[label, "MAPE_all"])
                rows.append(dict(
                    mode=mode, L=L, H=H, model=label.split(" ")[0], family="foundation",
                    mape=mape, d_mape=100.0 * (1.0 - mape / key[(mode, L, H)])))
    return pd.DataFrame(rows)


def strip_panel(axis, data: pd.DataFrame, column: str, title: str, xlabel: str):
    """One row per (model, mode): a dot per (L, H) cell, coloured by sign."""
    rows, labels, separators = [], [], []
    for index, model in enumerate(MODEL_ORDER):
        for mode, tag in (("Manoeuvre", "Man."), ("Navigation", "Nav.")):
            rows.append((model, mode))
            labels.append(tag)
        if index < len(MODEL_ORDER) - 1:
            separators.append(len(rows) - 0.5)

    for position in separators:
        axis.axhline(position, color="#dcdcdc", linewidth=0.5, zorder=1)

    for position, (model, mode) in enumerate(rows):
        cells = data[(data.model == model) & (data["mode"] == mode)][column]
        if cells.empty:
            continue
        wins = int((cells > 0).sum())
        jitter = np.linspace(-0.16, 0.16, len(cells))
        axis.scatter(cells.values, position + jitter, s=11,
                     marker=MARKERS[model], linewidth=0.6,
                     facecolor=[GAIN if v > 0 else LOSS for v in cells.values],
                     edgecolor="white", alpha=0.95, zorder=4)
        median = float(cells.median())
        axis.plot([median, median], [position - 0.30, position + 0.30],
                  color=INK, linewidth=1.4, solid_capstyle="butt", zorder=5)
        axis.annotate(f"{wins}/9", (1.008, position), xycoords=("axes fraction", "data"),
                      va="center", fontsize=6.1, zorder=6,
                      color=GAIN if wins >= 5 else MUTED,
                      fontweight="bold" if wins == 9 else "normal")

    axis.axvline(0.0, color=INK, linewidth=0.9, zorder=3)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels, fontsize=6.4)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.invert_yaxis()
    axis.set_title(title, loc="left")
    axis.set_xlabel(xlabel)
    axis.grid(True, axis="x", linewidth=0.3, alpha=0.25, zorder=1)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def make_scatter(data: pd.DataFrame):
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 4.4), sharey=True)
    strip_panel(axes[0], data, "d_mape", "(a) Aggregate MAPE",
                r"$\Delta_{\mathrm{pers}}$ (%)")
    strip_panel(axes[1], data, "d_tail_rmse", r"(b) Upper-tail $\mathrm{RMSE}^{95}$",
                r"$\Delta_{\mathrm{pers}}$ (%)")

    # Model names sit once, centred on each two-row group, outside the left panel.
    for index, model in enumerate(MODEL_ORDER):
        axes[0].annotate(model, (-0.235, 2 * index + 0.5),
                         xycoords=("axes fraction", "data"), va="center",
                         ha="left", fontsize=7.2, color=INK)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                   markerfacecolor=GAIN, markeredgecolor="white",
                   label="better than persistence"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                   markerfacecolor=LOSS, markeredgecolor="white",
                   label="worse than persistence"),
        plt.Line2D([], [], color=INK, linewidth=1.4, label="median over the nine cells"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=6.8, bbox_to_anchor=(0.55, -0.045))
    figure.tight_layout(rect=(0.13, 0.02, 0.97, 1.0))
    figure.subplots_adjust(wspace=0.16)
    return figure


def _unused_make_heatmap(data: pd.DataFrame):
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.9),
                                gridspec_kw={"width_ratios": [1, 1.13]})
    cmap = plt.get_cmap("RdBu")
    limit = 30.0

    for index, (axis, mode) in enumerate(((axes[0], "Manoeuvre"), (axes[1], "Navigation"))):
        subset = data[data["mode"] == mode]
        grid = subset.pivot_table(index="model", columns=["L", "H"],
                                  values="d_tail_rmse").reindex(MODEL_ORDER)
        image = axis.imshow(grid.values, cmap=cmap, vmin=-limit, vmax=limit,
                            aspect="auto")
        axis.set_xticks(range(grid.shape[1]))
        axis.set_xticklabels([f"{L}/{H}" for L, H in grid.columns],
                             fontsize=6.4, rotation=45, ha="right")
        axis.set_yticks(range(grid.shape[0]))
        axis.set_yticklabels(grid.index if index == 0 else [], fontsize=6.8)
        # Panel identification is left to the LaTeX subfigure captions.
        axis.set_xlabel(r"$L/H$ (s)", fontsize=7.5)
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value = grid.values[row, col]
                if np.isnan(value):
                    continue
                text = "0" if abs(value) < 0.5 else f"{value:+.0f}"
                axis.text(col, row, text, ha="center", va="center",
                          fontsize=5.6,
                          color="white" if abs(value) > 0.62 * limit else INK)
        if index == 1:
            bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03,
                                  extend="both")
            bar.set_label(r"$\Delta_{\mathrm{pers}}$, $\mathrm{RMSE}^{95}$ (%)",
                          fontsize=7)
            bar.ax.tick_params(labelsize=6.5)
            bar.outline.set_linewidth(0.4)
    figure.tight_layout()
    return figure


def _draw_grid(axis, grid, value_fmt, text_color, best_rows):
    """Cell-grid drawing: ticks, annotations (winner in bold), hairline spines."""
    axis.set_xticks(range(grid.shape[1]))
    axis.set_xticklabels([f"{L}/{H}" for L, H in grid.columns],
                         fontsize=6.2, rotation=45, ha="right")
    axis.set_yticks(range(grid.shape[0]))
    axis.set_yticklabels(grid.index, fontsize=6.8)
    axis.set_xlabel(r"$L/H$ (s)", fontsize=7.4)
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid.values[row, col]
            if np.isnan(value):
                continue
            winner = best_rows[col] == row
            axis.text(col, row, value_fmt(value), ha="center", va="center",
                      fontsize=6.0 if winner else 5.4,
                      fontweight="bold" if winner else "normal",
                      color=text_color(value))
    for col, row in enumerate(best_rows):
        if row is not None:
            axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                         edgecolor=INK, linewidth=1.0, zorder=5))


def make_event_panel(data: pd.DataFrame, metric: str, mode: str, limit: float):
    """Single panel: Delta_pers on one event metric, one operating mode."""
    figure, axis = plt.subplots(figsize=(3.6, 2.5))
    subset = data[(data["mode"] == mode) & (data.family == "trained")]
    grid = subset.pivot_table(index="model", columns=["L", "H"],
                              values=metric).reindex(MODEL_ORDER)
    image = axis.imshow(grid.values, cmap=plt.get_cmap("RdBu"),
                        vmin=-limit, vmax=limit, aspect="auto")
    best = [int(np.nanargmax(grid.values[:, c])) for c in range(grid.shape[1])]
    _draw_grid(axis, grid, lambda v: "0" if abs(v) < 0.5 else f"{v:+.0f}",
               lambda v: "white" if abs(v) > 0.62 * limit else INK, best)
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, extend="both")
    bar.set_label(r"$\Delta_{\mathrm{pers}}$ (%)", fontsize=6.8)
    bar.ax.tick_params(labelsize=6.2)
    bar.outline.set_linewidth(0.4)
    figure.tight_layout()
    return figure


def make_mape_panel(data: pd.DataFrame, mode: str):
    """Single panel: aggregate MAPE for every predictor, trained and zero-shot."""
    order = list(MODEL_ORDER) + list(FOUNDATION_ORDER)
    figure, axis = plt.subplots(figsize=(3.9, 3.0))
    subset = data[data["mode"] == mode]
    grid = subset.pivot_table(index="model", columns=["L", "H"],
                              values="mape").reindex(order)
    top = float(np.nanpercentile(grid.values, 96))
    image = axis.imshow(grid.values, cmap=plt.get_cmap("YlGnBu"),
                        vmin=float(np.nanmin(grid.values)), vmax=top, aspect="auto")
    best = [int(np.nanargmin(grid.values[:, c])) for c in range(grid.shape[1])]
    _draw_grid(axis, grid, lambda v: f"{v:.3g}",
               lambda v: "white" if v > 0.68 * top else INK, best)
    axis.axhline(len(MODEL_ORDER) - 0.5, color=INK, linewidth=1.0, zorder=6)
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, extend="max")
    bar.set_label("aggregate MAPE (%)", fontsize=6.8)
    bar.ax.tick_params(labelsize=6.2)
    bar.outline.set_linewidth(0.4)
    figure.tight_layout()
    return figure


def make_latency(latency_dir: Path, limit_ms: float = 100.0):
    """Compiled-path batch-one latency, log axis, with the compile cost alongside."""
    jit_label = "jax.jit direct call"
    trained, foundation, compile_s = {}, {}, {}
    for mode, tag in (("Manoeuvre", "manoeuvre"), ("Navigation", "navigation")):
        frame = pd.read_csv(latency_dir / f"inference_latency_{tag}_my_models.csv")
        frame = frame[frame.method == jit_label].set_index("model")
        trained[mode] = frame
        compile_s[mode] = frame["warmup_ms"] / 1000.0
        found = pd.read_csv(latency_dir / f"inference_latency_{tag}_foundation_models.csv")
        foundation[mode] = found.assign(
            model=[label.split(" ")[0] for label in found["model"]]).set_index("model")

    order = list(MODEL_ORDER) + list(FOUNDATION_ORDER)
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.4),
                                gridspec_kw={"width_ratios": [2.5, 1]})
    axis, compile_axis = axes

    axis.axvspan(limit_ms, 1e4, facecolor=LOSS, alpha=0.07, zorder=0)
    axis.axvline(limit_ms, color=LOSS, linewidth=0.9, linestyle=":", zorder=2)

    for position, model in enumerate(order):
        for mode, marker, fill in (("Manoeuvre", "o", True), ("Navigation", "s", False)):
            table = trained[mode] if model in MODEL_ORDER else foundation[mode]
            if model not in table.index:
                continue
            mean_ms, max_ms = float(table.loc[model, "mean_ms"]), float(table.loc[model, "max_ms"])
            offset = -0.16 if mode == "Manoeuvre" else 0.16
            colour = LOSS if max_ms > limit_ms else GAIN
            axis.plot([mean_ms, max_ms], [position + offset] * 2,
                      color=colour, linewidth=0.9, alpha=0.55, zorder=3)
            axis.scatter([mean_ms], [position + offset], s=26, marker=marker, zorder=4,
                         facecolor=colour if fill else "white",
                         edgecolor=colour, linewidth=0.9)

    axis.set_xscale("log")
    axis.set_xlim(0.03, 400)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(order, fontsize=6.9)
    axis.set_ylim(-0.7, len(order) - 0.3)
    axis.invert_yaxis()
    axis.axhline(len(MODEL_ORDER) - 0.5, color=INK, linewidth=0.9, zorder=5)
    axis.set_xlabel("CPU batch-one latency (ms, log scale)")
    axis.grid(True, axis="x", which="major", linewidth=0.3, alpha=0.3)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                   markerfacecolor=INK, markeredgecolor=INK, label="manoeuvre"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=4.5,
                   markerfacecolor="white", markeredgecolor=INK, label="navigation"),
        plt.Line2D([], [], color=MUTED, linewidth=0.9, label="mean to maximum"),
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=6.3, framealpha=0.9)

    width = 0.36
    for position, model in enumerate(MODEL_ORDER):
        for mode, offset in (("Manoeuvre", -width / 2), ("Navigation", width / 2)):
            compile_axis.barh(position + offset, compile_s[mode].get(model, np.nan),
                              height=width, color=GAIN,
                              alpha=1.0 if mode == "Manoeuvre" else 0.45,
                              edgecolor="white", linewidth=0.4)
    compile_axis.set_yticks(range(len(order)))
    compile_axis.set_yticklabels([])
    compile_axis.set_ylim(-0.7, len(order) - 0.3)
    compile_axis.invert_yaxis()
    compile_axis.axhline(len(MODEL_ORDER) - 0.5, color=INK, linewidth=0.9, zorder=5)
    compile_axis.set_xlabel("one-off compile (s)")
    compile_axis.grid(True, axis="x", linewidth=0.3, alpha=0.3)
    compile_axis.set_axisbelow(True)
    for spine in ("top", "right"):
        compile_axis.spines[spine].set_visible(False)

    figure.tight_layout()
    return figure


def save(figure, output_dir: Path, name: str, formats, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        path = output_dir / f"{name}.{suffix}"
        with warnings.catch_warnings():
            if suffix == "eps":
                warnings.simplefilter("ignore")
            figure.savefig(path, format=suffix, dpi=dpi if suffix == "png" else None,
                           bbox_inches="tight", pad_inches=0.02)
        print(f"  saved {path}")
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    exclude = {item.strip() for item in args.exclude.split(",") if item.strip()}

    print("Persistence MAPE on the benchmark window set:")
    pers = persistence_mape_samewindows(results_dir, args.archive_glob, Path(args.pers_cache))

    data = collect(results_dir, args.archive_glob, pers, exclude)
    if data.empty:
        raise SystemExit("No archives matched -- check --results-dir/--archive-glob.")

    summary = data.groupby(["mode", "model"]).agg(
        cells=("d_tail_rmse", "size"),
        tail_wins=("d_tail_rmse", lambda s: int((s > 0).sum())),
        mape_wins=("d_mape", lambda s: int((s > 0).sum())),
    )
    print("\nCells with Delta_pers > 0 (out of 9):")
    print(summary.to_string())

    plt.rcParams.update(PLOT_STYLE)
    tags = {"Manoeuvre": "man", "Navigation": "nav"}
    short = {"d_tail_rmse": "tailrmse", "d_energy_mean": "energymean",
             "d_ramp_steep": "rampsteep"}

    # One symmetric scale per metric, shared by its two modes: the three metrics
    # differ by an order of magnitude, so a single global limit would saturate.
    for metric, stem in short.items():
        spread = np.abs(data[data.family == "trained"][metric].dropna())
        limit = max(20.0, float(np.ceil(np.percentile(spread, 90) / 10.0) * 10.0))
        for mode, tag in tags.items():
            save(make_event_panel(data, metric, mode, limit), output_dir,
                 f"persistence_delta_{stem}_{tag}", formats, args.dpi)

    for mode, tag in tags.items():
        save(make_mape_panel(data, mode), output_dir, f"mape_grid_{tag}", formats, args.dpi)

    save(make_scatter(data), output_dir, "persistence_delta_strip", formats, args.dpi)
    save(make_latency(Path(args.latency_dir)), output_dir, "latency_compiled", formats, args.dpi)


if __name__ == "__main__":
    main()
