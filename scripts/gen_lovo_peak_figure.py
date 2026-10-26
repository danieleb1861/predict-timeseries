"""Leave-one-leg-out upper-tail skill versus the peak load of the held-out leg.

One point per architecture and held-out manoeuvre leg: persistence-relative
skill S_E on RMSE^95 (training-set q95 threshold) against the maximum measured
load of that leg. Legs whose tail is empty under the training threshold are
omitted (4 of 16). The per-leg median over the seven architectures is drawn on
top, its marker area proportional to the tail coverage of the leg, because
legs with few tail samples give noisy skill estimates.

Inputs, both in the LOVO run root:
    lovo_by_fold.csv   from scripts/collect_lovo_results.py
    lovo_legstats.csv  per-fold held-out leg statistics (te_max: leg maximum, PU)

Example:
    python scripts/gen_lovo_peak_figure.py \
        --run-root outputs/runs/lovo_manoeuvre_benchmark_flax
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = (
    "CNN", "TCN", "iTransformer", "Transformer",
    "ConvTCN-H", "ConvLSTMTCN-H", "DualStream-FiLM",
)
LABELS = {"ConvTCN-H": "ConvTCN", "ConvLSTMTCN-H": "ConvLSTM-TCN"}
# Same marker registry as gen_persistence_delta_figures.py: identity by shape.
MARKERS = {
    "CNN": "o", "TCN": "s", "iTransformer": "^", "Transformer": "D",
    "ConvTCN-H": "v", "ConvLSTMTCN-H": "P", "DualStream-FiLM": "X",
}
INK = "#1a1a1a"
MUTED = "#8a8a8a"

PLOT_STYLE = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "ps.fonttype": 42, "pdf.fonttype": 42,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / "runs" / "lovo_manoeuvre_benchmark_flax"))
    parser.add_argument("--legstats", default=None, help="Default: <run-root>/lovo_legstats.csv")
    parser.add_argument("--metric", default="S_E_tail_rmse_pu")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "figures" / "paper" / "lovo"))
    parser.add_argument("--stem", default="lovo_tailskill_vs_peak")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    folds = pd.read_csv(run_root / "lovo_by_fold.csv")
    legs = pd.read_csv(args.legstats or run_root / "lovo_legstats.csv")[["fold", "te_max"]]
    data = folds.merge(legs, on="fold").dropna(subset=[args.metric])

    per_leg = (data.groupby("fold")
               .agg(te_max=("te_max", "first"), cover=("tail_frac_%", "first"),
                    median=(args.metric, "median"))
               .reset_index())
    rho, pval = spearmanr(per_leg["te_max"], per_leg["median"])
    print(f"legs with a populated tail: {len(per_leg)}; median skill vs leg peak: "
          f"Spearman rho={rho:+.2f}, p={pval:.3f}")
    for model in MODEL_ORDER:
        sub = data[data["model"] == model]
        r, p = spearmanr(sub["te_max"], sub[args.metric])
        print(f"  {model:16s} rho={r:+.2f} p={p:.3f} n={len(sub)}")

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2,
               label=r"persistence ($S_E=0$)")

    # Small horizontal dodge so the seven markers of one leg do not overprint.
    offsets = np.linspace(-0.009, 0.009, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        sub = data[data["model"] == model]
        ax.scatter(sub["te_max"] + offset, sub[args.metric], marker=MARKERS[model], s=16,
                   facecolor="none", edgecolor=MUTED, linewidth=0.7, zorder=3,
                   label=LABELS.get(model, model))

    sizes = 14 + 8 * per_leg["cover"].to_numpy()  # area grows with tail coverage (%)
    ax.scatter(per_leg["te_max"], per_leg["median"], s=sizes, color=INK,
               edgecolor="white", linewidth=0.8, zorder=4)

    ax.set_xlabel("Peak load of the held-out leg (PU)")
    ax.set_ylabel(r"$S_E$ on $\mathrm{RMSE}^{95}$")
    ax.grid(axis="y", color="#dcdcdc", linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.98, 0.04, rf"median vs peak load: Spearman $\rho={rho:+.2f}$ ($n={len(per_leg)}$ legs)",
            transform=ax.transAxes, fontsize=7, color=INK, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.scatter([], [], s=40, color=INK, edgecolor="white"))
    labels.append("median of the seven,\narea = tail coverage")
    for cover in (1, 5, 15):
        handles.append(plt.scatter([], [], s=14 + 8 * cover, color=INK, edgecolor="white"))
        labels.append(f"{cover}% of test points")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.10, 1.0), frameon=False,
              handletextpad=0.4, labelspacing=0.55, borderaxespad=0.0)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("eps", "png"):
        fig.savefig(out / f"{args.stem}.{suffix}", format=suffix, dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    print(f"wrote {out / args.stem}.eps and .png")


if __name__ == "__main__":
    main()
