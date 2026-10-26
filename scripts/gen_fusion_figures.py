"""Two figures for the fusion mixture of predictors: skill per held-out leg, and weight dynamics.

    python scripts/gen_fusion_figures.py --run-root outputs/runs/lovo_manoeuvre_benchmark_flax --tag _p1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, ACCENT = "#1a1a1a", "#8a8a8a", "#1b6ca8"
STYLE = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
         "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
         "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
         "mathtext.fontset": "stix", "ps.fonttype": 3, "pdf.fonttype": 42}  # Type 3 text survives the EPS
VARIANTS = [("fusion", "Mixture"), ("prior", "Prior only"), ("mean", "Equal"), ("DLinear", "DLinear")]
PANELS = [("S_MAPE_all", "aggregate MAPE"), ("S_RMSE_all", "aggregate RMSE"),
          ("S_tail_rmse_pu", r"upper-tail RMSE$^{95}$"), ("S_mape_ramp_steep_%", "steep-ramp MAPE")]
DISPLAY = {"convtcn_hybrid": "ConvTCN", "convlstmtcn_hybrid": "ConvLSTM-TCN"}
PERS_FILL = "#d0d0d0"  # persistence, as in fusion_weights


def ems_figure(path, by_fold, out, window_s=80):
    """EMS simulation of one held-out leg, last measured value against the mixture: load and generator output, and
    BESS power, around the largest saving; cumulative throughput saved over the leg; per-leg reductions."""
    d = np.load(path)
    T = len(d["Persistence"])
    s, t = d["load"][:T], np.arange(T) / 60.0
    (gp, bp), (gm, bm) = d["Persistence"].T, d["fusion"].T
    saved = np.cumsum(np.abs(bp) - np.abs(bm))
    i0 = int(np.argmax(saved[10:] - saved[:-10])) + 5  # centre of the largest 10 s saving
    a0, a1 = max(0, i0 - window_s // 2), min(T, i0 + window_s // 2)
    z = slice(a0, a1)
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], width_ratios=[3.0, 1.0], hspace=0.55, wspace=0.32)
    axa, axb = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    axc, axd = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    axa.plot(t[z], s[z], color=INK, linewidth=1.2, label="Measured load", zorder=4)
    axa.plot(t[z], gp[z], color=MUTED, linewidth=1.1, linestyle=(0, (3, 1.5)), label="Generators, last measured value")
    axa.plot(t[z], gm[z], color=ACCENT, linewidth=1.1, label="Generators, mixture")
    axa.set_ylabel("Power (PU)")
    axa.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2)

    axb.axhline(0.0, color="#555555", linewidth=0.6)
    axb.fill_between(t[z], bp[z], color=PERS_FILL, linewidth=0, label="Last measured value")
    axb.plot(t[z], bm[z], color=ACCENT, linewidth=1.1, label="Mixture")
    axb.set_ylabel("BESS power (PU)")
    axb.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, handlelength=2.0)
    for ax in (axa, axb):
        ax.set_xlim(t[a0], t[a1 - 1])
    axb.set_xlabel("Time within the held-out manoeuvre (min)")

    axc.axvspan(t[a0], t[a1 - 1], color="#ececec", linewidth=0, zorder=0)
    axc.plot(t, saved, color=ACCENT, linewidth=1.1)
    axc.axhline(0.0, color="#555555", linewidth=0.6)
    axc.set_xlim(t[0], t[-1])
    axc.set_ylabel("Throughput saved (PU s)")
    axc.set_xlabel("Time within the held-out manoeuvre (min)")

    df = pd.read_csv(by_fold)
    base = df[df.variant == "Persistence"].set_index("fold")
    mix = df[df.variant == "fusion"].set_index("fold")
    for i, col in enumerate(("bess_throughput_pus", "bess_peak_pu")):
        x = 100 * (1 - mix[col] / base[col]).to_numpy()
        axd.scatter(np.full(x.size, i) + np.linspace(-0.16, 0.16, x.size), x, s=11,
                    facecolor="#9ecae1", edgecolor=MUTED, linewidth=0.5, zorder=3)
        axd.plot([i - 0.28, i + 0.28], [x.mean()] * 2, color=INK, linewidth=1.4, zorder=4)
    axd.axhline(0.0, color="#555555", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Throughput", "Peak power"])
    axd.set_xlim(-0.5, 1.5)
    axd.set_ylabel("Reduction per leg (%)")

    for i, ax in enumerate((axa, axb, axc, axd)):
        ax.grid(color="#dcdcdc", linewidth=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0.0, 1.03, f"({'abcd'[i]})", transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5)
    for suf in ("eps", "png"):
        fig.savefig(out / f"ems_simulation.{suf}", format=suf, dpi=300 if suf == "png" else None, bbox_inches="tight")
    print("wrote", out / "ems_simulation.eps")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / "runs" / "lovo_manoeuvre_benchmark_flax"))
    ap.add_argument("--tag", default="_p1")
    ap.add_argument("--fold", type=int, default=10)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "figures" / "paper" / "fusion"))
    ap.add_argument("--ems-tag", default="_p1_dlinear_ems", help="Tag of the --ems run whose fold traces are drawn.")
    a = ap.parse_args()
    root, out = Path(a.run_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    ems = root / f"fusion{a.ems_tag}_ems_fold{a.fold}.npz"
    if ems.exists():
        ems_figure(ems, root / f"fusion{a.ems_tag}" / "ems_by_fold.csv", out)

    df = pd.read_csv(root / f"fusion{a.tag}" / "fusion_by_fold.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, PANELS):
        for i, (key, lab) in enumerate(VARIANTS):
            x = df[df.variant == key][col].dropna().to_numpy()
            ax.scatter(np.full(x.size, i) + np.linspace(-0.16, 0.16, x.size), x, s=11,
                       facecolor="#9ecae1", edgecolor=MUTED, linewidth=0.5, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [x.mean()] * 2, color=INK, linewidth=1.4, zorder=4)
        ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([v[1] for v in VARIANTS], rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#dcdcdc", linewidth=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r"skill $S_E$ vs persistence")
    axes[0].set_ylim(-1.3, 0.12)
    axes[2].set_ylim(-1.3, 0.12)
    fig.tight_layout()
    for suf in ("eps", "png"):
        fig.savefig(out / f"fusion_skill.{suf}", format=suf, dpi=300 if suf == "png" else None, bbox_inches="tight")

    d = np.load(root / f"fusion{a.tag}_weights_fold{a.fold}.npz", allow_pickle=True)
    w, names, load = d["weights"], [str(n) for n in d["names"]], d["load"]
    t = np.arange(len(load)) * 5 / 60.0  # windows are 5 s apart
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.stackplot(t, w[:, 0], w[:, 1:].sum(1), colors=("#d0d0d0", "#9ecae1"), linewidth=0,
                 labels=("Persistence", "Learned models"))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time within the held-out manoeuvre (min)")
    ax.set_ylabel("Expert weight")
    ax2 = ax.twinx()
    ax2.plot(t, load, color=INK, linewidth=0.7)
    ax2.set_ylabel("Load (PU)")
    ax2.spines["top"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, handlelength=1.4)
    fig.tight_layout()
    for suf in ("eps", "png"):
        fig.savefig(out / f"fusion_weights.{suf}", format=suf, dpi=300 if suf == "png" else None, bbox_inches="tight")
    print("wrote", out / "fusion_skill.eps", "and", out / "fusion_weights.eps")


if __name__ == "__main__":
    main()
