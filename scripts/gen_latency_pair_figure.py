# -*- coding: utf-8 -*-
"""One full-width latency figure holding both operating modes side by side.

The manuscript previously carried one full-page figure per mode. Rendering the
existing portrait EPS at half width would scale its labels to ~29 %, so the
panels are regenerated here at their final size instead, with the type set at
generation time.

Reads the per-call CSVs written by measure_inference_latency.py:
    inference_latency_<mode>_my_models_dualpath_calls.csv       (--family trained)
    inference_latency_<mode>_<fm-stem>_calls.csv                (--family foundation)
with columns model, method, call, latency_ms.

--style violin (default) draws each distribution as a violin whose density is
estimated on log10(latency), so its shape is correct on the logarithmic axis;
--style box reproduces the earlier box plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    'font.family':      'Times New Roman',
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'xtick.labelsize':  7.5,
    'ytick.labelsize':  8,
    'legend.fontsize':  7.5,
    'axes.linewidth':   0.8,
    'lines.linewidth':  0.9,
    'grid.linewidth':   0.6,
    'grid.color':       '#ebebeb',
    'grid.alpha':       1.0,
    'ps.fonttype':      3,
    'pdf.fonttype':     42,
    'figure.constrained_layout.use': False,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.02,
})

ORDER = ["CNN", "TCN", "Transformer", "iTransformer",
         "DualStream-FiLM", "ConvTCN-H", "ConvLSTMTCN-H", "DLinear"]
DISPLAY = {"ConvTCN-H": "ConvTCN", "ConvLSTMTCN-H": "ConvLSTM-TCN",
           "convtcn_hybrid": "ConvTCN", "convlstmtcn_hybrid": "ConvLSTM-TCN"}
BUDGET_MS = 100.0
COLOURS = {"non-JIT": "#9ecae1", "JIT": "#f4a582"}
FM_ORDER = ["Chronos-2 (0-shot)", "TimesFM (0-shot)", "TiRex-2 (0-shot)"]
FM_DISPLAY = {"Chronos-2 (0-shot)": "Chronos-2", "TimesFM (0-shot)": "TimesFM",
              "TiRex-2 (0-shot)": "TiRex-2"}
FM_COLOUR = "#9ecae1"  # eager PyTorch execution, same fill as the non-JIT path


def label(m):
    return DISPLAY.get(m, m)


def draw_violin(ax, x, pos, width, colour):
    """Violin with density estimated on log10(x), plus median bar and mean diamond."""
    lx = np.log10(np.asarray(x, dtype=float))
    grid = np.linspace(lx.min(), lx.max(), 200)
    if np.ptp(lx) > 1e-9:
        dens = gaussian_kde(lx)(grid)
        half = 0.5 * width * dens / dens.max()
    else:
        half = np.full_like(grid, 0.5 * width)
    ax.fill_betweenx(10 ** grid, pos - half, pos + half, facecolor=colour,
                     edgecolor="black", linewidth=0.6, zorder=2)
    med = np.median(x)
    half_med = np.interp(np.log10(med), grid, half)
    ax.hlines(med, pos - half_med, pos + half_med, color="black", linewidth=0.9, zorder=3)
    ax.plot([pos], [np.mean(x)], marker="D", markersize=2.4, markerfacecolor="black",
            markeredgecolor="none", linestyle="none", zorder=4)


def finish_axis(ax, labels, title):
    ax.axhline(BUDGET_MS, color="#b2182b", linewidth=0.9, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def fm_panel(ax, df, title):
    present = [m for m in FM_ORDER if m in set(df["model"])]
    for i, m in enumerate(present):
        draw_violin(ax, df[df["model"] == m]["latency_ms"].to_numpy(), i, 0.5, FM_COLOUR)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylim(10, 300)  # decade below the fastest call, budget line in view
    finish_axis(ax, [FM_DISPLAY.get(m, m) for m in present], title)


def draw_range(ax, x, pos, width, colour):
    """Point-range for long call series: p1-p99 box, median bar, mean diamond, whisker to the maximum.

    With thousands of calls, 98 % of them fall within 0.01-0.13 decades while rare
    maxima reach up to 1 decade, so on a multi-decade axis a violin collapses into a
    needle; the range plot keeps the body, the median and the worst call readable.
    """
    x = np.asarray(x, dtype=float)
    p1, p99, med = np.percentile(x, 1), np.percentile(x, 99), np.median(x)
    ax.vlines(pos, p99, x.max(), color="black", linewidth=0.6, zorder=1)
    ax.plot([pos], [x.max()], marker="_", markersize=5, color="black", zorder=3)
    ax.fill_between([pos - width / 2, pos + width / 2], p1, p99, facecolor=colour,
                    edgecolor="black", linewidth=0.6, zorder=2)
    ax.hlines(med, pos - width / 2, pos + width / 2, color="black", linewidth=0.9, zorder=3)
    ax.plot([pos], [x.mean()], marker="D", markersize=2.4, markerfacecolor="black",
            markeredgecolor="none", linestyle="none", zorder=4)


def both_panel(ax, df, title, style="range"):
    """Trained (non-JIT and JIT) and foundation models on one axis, from one merged per-call CSV."""
    trained = [m for m in ORDER if m in set(df["model"])]
    fms = [m for m in FM_ORDER if m in set(df["model"])]
    width = 0.36
    draw = draw_range if style == "range" else draw_violin
    for i, m in enumerate(trained):
        for k, (meth, key) in enumerate((("module.apply (non-JIT)", "non-JIT"), ("jax.jit direct call", "JIT"))):
            x = df[(df["model"] == m) & (df["method"] == meth)]["latency_ms"].to_numpy()
            draw(ax, x, i + (k - 0.5) * width, width * 0.9, COLOURS[key])
    for j, m in enumerate(fms):
        draw(ax, df[df["model"] == m]["latency_ms"].to_numpy(), len(trained) + j, width * 0.9, FM_COLOUR)
    ax.axvline(len(trained) - 0.5, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlim(-0.6, len(trained) + len(fms) - 0.4)
    finish_axis(ax, [label(m) for m in trained] + [FM_DISPLAY.get(m, m) for m in fms], title)


def panel(ax, df, title, style="box"):
    methods = sorted(df["method"].unique())
    non_jit = [m for m in methods if "non-JIT" in m or "eager" in m][0]
    jit = [m for m in methods if m != non_jit][0]

    present = [m for m in ORDER if m in set(df["model"])]
    width = 0.36
    if style == "violin":
        for k, (meth, key) in enumerate(((non_jit, "non-JIT"), (jit, "JIT"))):
            for i, m in enumerate(present):
                x = df[(df["model"] == m) & (df["method"] == meth)]["latency_ms"].to_numpy()
                draw_violin(ax, x, i + (k - 0.5) * width, width * 0.9, COLOURS[key])
        ax.set_xlim(-0.6, len(present) - 0.4)
        finish_axis(ax, [label(m) for m in present], title)
        return
    for k, (meth, key) in enumerate(((non_jit, "non-JIT"), (jit, "JIT"))):
        data = [df[(df["model"] == m) & (df["method"] == meth)]["latency_ms"].to_numpy()
                for m in present]
        pos = np.arange(len(present)) + (k - 0.5) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markersize=2.4,
                                                       markerfacecolor="black",
                                                       markeredgecolor="none"),
                        flierprops=dict(marker="o", markersize=1.4, alpha=0.5,
                                        markeredgewidth=0.3),
                        medianprops=dict(color="black", linewidth=0.9))
        for box in bp["boxes"]:
            box.set_facecolor(COLOURS[key])
            box.set_edgecolor("black")
            box.set_linewidth(0.6)
        bp["boxes"][0].set_label(key)

    ax.axhline(BUDGET_MS, color="#b2182b", linewidth=0.9, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([label(m) for m in present], rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default="outputs/figures/paper/latency_cpu_flax")
    ap.add_argument("--out", default=None,
                    help="Output stem; default inference_latency_{models,foundation}_bothmodes in --csv-dir.")
    ap.add_argument("--style", choices=("violin", "box", "range"), default="violin",
                    help="range (only with --family both): p1-p99 box, median, mean, whisker to the maximum.")
    ap.add_argument("--family", choices=("trained", "foundation", "both"), default="trained")
    ap.add_argument("--both-stem", default="standard_1core",
                    help="--family both reads inference_latency_<mode>_<both-stem>_calls.csv "
                         "(measure_inference_latency.py --isolate --family both).")
    ap.add_argument("--fm-stem", default="foundation_models_r2",
                    help="Foundation per-call CSV is inference_latency_<mode>_<fm-stem>_calls.csv.")
    a = ap.parse_args()

    d = Path(a.csv_dir)
    if a.out is None:
        a.out = str(d / {"trained": "inference_latency_models_bothmodes",
                         "foundation": "inference_latency_foundation_bothmodes",
                         "both": f"inference_latency_all_bothmodes_{a.both_stem}"}[a.family])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (mode, title) in zip(axes, (("manoeuvre", "Manoeuvre"),
                                        ("navigation", "Navigation"))):
        if a.family == "both":
            both_panel(ax, pd.read_csv(d / f"inference_latency_{mode}_{a.both_stem}_calls.csv"), title,
                       style="range" if a.style == "range" else "violin")
        elif a.family == "foundation":
            fm_panel(ax, pd.read_csv(d / f"inference_latency_{mode}_{a.fm_stem}_calls.csv"), title)
        else:
            f = d / f"inference_latency_{mode}_my_models_dualpath_calls.csv"
            panel(ax, pd.read_csv(f), title, style=a.style)
    axes[0].set_ylabel("Latency per call (ms)")
    if a.family == "both":
        handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=lbl)
                   for c, lbl in ((COLOURS["non-JIT"], "eager"), (COLOURS["JIT"], "compiled"))]
        axes[0].legend(handles=handles, loc="lower right", frameon=True, framealpha=1.0)
    elif a.family == "trained" and a.style == "violin":
        handles = [Patch(facecolor=COLOURS[k], edgecolor="black", linewidth=0.6, label=k)
                   for k in ("non-JIT", "JIT")]
        axes[0].legend(handles=handles, loc="center left", frameon=True, framealpha=1.0)
    elif a.family == "trained":
        axes[0].legend(loc="center left", frameon=True, framealpha=1.0)
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".eps"), format="eps")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300)
    print("saved", out.with_suffix(".eps").name, "and .png")


if __name__ == "__main__":
    main()
