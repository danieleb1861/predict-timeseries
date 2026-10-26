"""Pairwise Diebold-Mariano significance analysis for multi-seed sweep runs.

Post-hoc and read-only with respect to run directories: it consumes
predictions/<model>.npz, windows.npz, and scalers_and_meta.pkl from each
<run-root>/seed*/ directory (as produced by scripts/run_seed_sweep.py) and
writes results into <run-root>/significance/.

The DM test uses per-window horizon-mean squared errors on the test set with a
Newey-West (Bartlett kernel) variance and the Harvey-Leybourne-Newbold
small-sample correction. The HAC lag count defaults to ceil((L + H) / stride)
because sliding windows overlap (stride = H/5), which induces strong error
autocorrelation; a naive DM variance would overstate significance.

Example:
    python scripts/run_significance.py --run-root outputs/runs/sweep_manoeuvre_paper_flax

Outputs (in <run-root>/significance/):
    dm_pairs.csv           per-seed pairwise DM stats and Holm-adjusted p-values
    dm_summary.csv         per-pair summary across seeds
    wilcoxon_seeds.csv     paired Wilcoxon/t-test on per-seed RMSE (robustness)
    significance_table.tex booktabs table: RMSE mean +/- std with markers
    dm_heatmap.png/.eps    median DM statistic heatmap
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.scaling import inverse_scale_targets  # noqa: E402


def newey_west_variance(d, lags):
    """Newey-West (Bartlett kernel) variance of the mean of a 1D series."""
    d = np.asarray(d, dtype=np.float64)
    d = d - d.mean()
    n = d.size
    var = float(np.mean(d * d))
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * float(np.mean(d[k:] * d[:-k]))
    return max(var, 1e-300) / n


def dm_test(loss_a, loss_b, hac_lags):
    """Diebold-Mariano test with HAC variance and Harvey small-sample correction.

    Negative statistic means model A has lower loss than model B.
    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = d.size
    dm = float(d.mean()) / math.sqrt(newey_west_variance(d, hac_lags))
    h = hac_lags + 1
    dm *= math.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    p = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
    return dm, p


def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjustment; preserves input order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def load_seed_losses(seed_dir, d_plot=0):
    """Per-window horizon-mean squared error series per model, plus (L, H, stride)."""
    windows = np.load(seed_dir / "windows.npz")
    with (seed_dir / "scalers_and_meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    y_true = inverse_scale_targets(
        windows["Yte"], meta["targetNames"], meta["MU"], meta["SIG"]
    )[:, :, d_plot]

    losses = {}
    for npz_file in sorted((seed_dir / "predictions").glob("*.npz")):
        data = np.load(npz_file)
        err = data["yhat_test_phys"][:, :, d_plot] - y_true
        losses[str(data["model_label"])] = np.mean(err**2, axis=1)
    geometry = (
        int(meta["Lookback window L"]),
        int(meta["Horizon forecasting window H"]),
        int(meta["WindowSlidingCoeff"]),
    )
    return losses, geometry


def plot_dm_heatmap(median_dm, sig_frac, models, out_base, alpha):
    """Publication heatmap of median DM statistics (diverging, neutral midpoint)."""
    import matplotlib.pyplot as plt

    from load_forecasting.plots import set_pub_style

    set_pub_style()
    n = len(models)
    matrix = np.full((n, n), np.nan)
    for (a, b), value in median_dm.items():
        i, j = models.index(a), models.index(b)
        matrix[i, j] = value
        matrix[j, i] = -value

    vmax = np.nanmax(np.abs(matrix)) or 1.0
    fig, ax = plt.subplots(figsize=(0.62 * n + 1.8, 0.62 * n + 1.2))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.92")
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = tuple(sorted((models[i], models[j])))
            frac = sig_frac[pair]
            star = "*" if frac >= 0.8 else ""
            ink = "white" if abs(matrix[i, j]) > 0.55 * vmax else "0.15"
            ax.text(
                j, i, f"{matrix[i, j]:.1f}{star}",
                ha="center", va="center", fontsize=7, color=ink,
            )

    ax.set_xticks(range(n), models, rotation=45, ha="right")
    ax.set_yticks(range(n), models)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Median DM statistic\n(negative: row model better)")
    cbar.outline.set_linewidth(0.8)
    # No title: journal figures are captioned in the manuscript. Explain there:
    # "* : Holm-adjusted p < alpha in >= 80% of seeds".
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap written to {out_base}.png / .eps")


def write_latex_table(rmse_by_model, best_beats, out_file, alpha, n_seeds):
    """Booktabs table of RMSE mean +/- std across seeds, dagger when the best model wins."""
    order = sorted(rmse_by_model, key=lambda m: np.mean(rmse_by_model[m]))
    lines = [
        "% Auto-generated by scripts/run_significance.py",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & RMSE (mean $\\pm$ std) & Sig. worse than best \\\\",
        "\\midrule",
    ]
    for model in order:
        values = np.asarray(rmse_by_model[model])
        std = values.std(ddof=1) if values.size > 1 else 0.0
        wins = best_beats.get(model, 0)
        marker = "$^{\\dagger}$" if model != order[0] and wins >= 0.8 * n_seeds else ""
        lines.append(
            f"{model}{marker} & {values.mean():.4f} $\\pm$ {std:.4f} & {wins}/{n_seeds} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"% dagger: best model significantly better (DM, Holm-adj. p<{alpha:g}) in >=80% of seeds",
    ]
    Path(out_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LaTeX table written to {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="Sweep directory containing seed*/ run dirs.")
    parser.add_argument("--hac-lags", type=int, default=None, help="HAC lag count (default: ceil((L+H)/stride)).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--d-plot", type=int, default=0, help="Target channel to evaluate (default: 0).")
    parser.add_argument("--out", default=None, help="Output directory (default: <run-root>/significance).")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seed_dirs = sorted(d for d in run_root.glob("seed*") if d.is_dir())
    if not seed_dirs:
        raise SystemExit(f"No seed*/ directories found under {run_root}")
    out_dir = Path(args.out) if args.out else run_root / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    rmse_by_model: dict[str, list[float]] = {}
    hac_lags = args.hac_lags
    models = None

    for seed_dir in seed_dirs:
        losses, (L, H, stride) = load_seed_losses(seed_dir, args.d_plot)
        if hac_lags is None:
            hac_lags = math.ceil((L + H) / max(stride, 1))
            print(f"HAC lags = ceil(({L}+{H})/{stride}) = {hac_lags}")
        seed_models = sorted(losses)
        if models is None:
            models = seed_models
        elif models != seed_models:
            raise SystemExit(f"Model sets differ across seeds: {models} vs {seed_models} in {seed_dir}")

        for model, loss in losses.items():
            rmse_by_model.setdefault(model, []).append(float(np.sqrt(loss.mean())))

        pairs = list(combinations(models, 2))
        stats_p = [dm_test(losses[a], losses[b], hac_lags) for a, b in pairs]
        p_holm = holm_adjust([p for _, p in stats_p])
        for (a, b), (dm, p), ph in zip(pairs, stats_p, p_holm):
            pair_rows.append({
                "seed_dir": seed_dir.name, "model_a": a, "model_b": b,
                "dm_stat": dm, "p_value": p, "p_holm": ph,
                "significant": ph < args.alpha,
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "dm_pairs.csv", index=False)

    n_seeds = len(seed_dirs)
    summary_rows, median_dm, sig_frac = [], {}, {}
    best_model = min(rmse_by_model, key=lambda m: np.mean(rmse_by_model[m]))
    best_beats: dict[str, int] = {}
    for a, b in combinations(models, 2):
        sub = pair_df[(pair_df.model_a == a) & (pair_df.model_b == b)]
        med = float(sub.dm_stat.median())
        n_sig_a = int(((sub.p_holm < args.alpha) & (sub.dm_stat < 0)).sum())
        n_sig_b = int(((sub.p_holm < args.alpha) & (sub.dm_stat > 0)).sum())
        median_dm[(a, b)] = med
        sig_frac[(a, b)] = max(n_sig_a, n_sig_b) / n_seeds
        if a == best_model:
            best_beats[b] = n_sig_a
        elif b == best_model:
            best_beats[a] = n_sig_b
        summary_rows.append({
            "model_a": a, "model_b": b, "median_dm_stat": med,
            "seeds_a_better": n_sig_a, "seeds_b_better": n_sig_b, "n_seeds": n_seeds,
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "dm_summary.csv", index=False)

    wilcoxon_rows = []
    for a, b in combinations(models, 2):
        va, vb = np.asarray(rmse_by_model[a]), np.asarray(rmse_by_model[b])
        try:
            w_p = float(stats.wilcoxon(va, vb).pvalue) if n_seeds >= 2 else np.nan
        except ValueError:
            w_p = np.nan
        t_p = float(stats.ttest_rel(va, vb).pvalue) if n_seeds >= 2 else np.nan
        wilcoxon_rows.append({"model_a": a, "model_b": b, "wilcoxon_p": w_p, "paired_t_p": t_p})
    pd.DataFrame(wilcoxon_rows).to_csv(out_dir / "wilcoxon_seeds.csv", index=False)
    if n_seeds < 6:
        print(f"Note: with {n_seeds} seeds the two-sided Wilcoxon p-value floor is "
              f"{2 ** (1 - n_seeds):.4f}; treat it as a robustness check (DM is the primary test).")

    write_latex_table(rmse_by_model, best_beats, out_dir / "significance_table.tex", args.alpha, n_seeds)
    models_by_rmse = sorted(models, key=lambda m: np.mean(rmse_by_model[m]))
    plot_dm_heatmap(median_dm, sig_frac, models_by_rmse, out_dir / "dm_heatmap", args.alpha)

    print(f"\n{len(pair_df)} pairwise tests over {n_seeds} seed(s), HAC lags {hac_lags}.")
    print(f"Best model by mean RMSE: {best_model} "
          f"({np.mean(rmse_by_model[best_model]):.4f})")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
