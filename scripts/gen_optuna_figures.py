"""Generate publication figures from Optuna tuning studies.

Reads the sqlite storage written by scripts/tune_hyperparams.py and saves, per
study, an optimization-history PNG (best validation loss vs. trial) and, when
enough completed trials exist, a fANOVA hyperparameter-importance PNG.

Safe to run while tuning is still in progress (read-only; rerun to refresh).

Example:
    python scripts/gen_optuna_figures.py --storage sqlite:///$HOME/optuna.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "optuna_figs"


_STYLE_APPLIED = False


def _apply_pub_style():
    """Apply the repo's publication style (Times New Roman, 9 pt), also under WSL."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import sys

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    if not any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
        for fonts_dir in (Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")):
            if fonts_dir.is_dir():
                for ttf in fonts_dir.glob("times*.ttf"):
                    font_manager.fontManager.addfont(str(ttf))
    sys.path.insert(0, str(REPO_ROOT))
    from load_forecasting.plots import set_pub_style

    set_pub_style(font_family="Times New Roman", fontsize=9)
    _STYLE_APPLIED = True


def save_study_figures(study, out_dir, min_trials_for_importance=10):
    """Save history (and, if meaningful, importance) PNGs for one study.

    Returns the list of paths written; empty if the study has no completed trials.
    """
    _apply_pub_style()
    import matplotlib.pyplot as plt
    import optuna
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    written = []
    if not completed:
        return written

    base = study.study_name.replace("__", "_")
    model_name = study.study_name.split("__", 1)[-1]

    ax = plot_optimization_history(study)
    for loc in ("left", "center", "right"):
        ax.set_title("", loc=loc)
    ax.set_title(model_name)
    path = out_dir / f"{base}_history.png"
    ax.figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)
    written.append(path)

    if len(completed) >= min_trials_for_importance and len(study.best_trial.params) > 1:
        ax = plot_param_importances(study)
        for loc in ("left", "center", "right"):
            ax.set_title("", loc=loc)
        path = out_dir / f"{base}_importance.png"
        ax.figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--storage", default=f"sqlite:///{(REPO_ROOT / 'outputs' / 'tuning' / 'optuna.db').as_posix()}",
                        help="Optuna storage URL (default: the tuner's default sqlite path).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for PNGs.")
    parser.add_argument("--studies", default=None, help="Comma-separated study names (default: all in storage).")
    args = parser.parse_args()

    import optuna

    names = optuna.study.get_all_study_names(storage=args.storage)
    if args.studies is not None:
        wanted = [s.strip() for s in args.studies.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in names]
        if unknown:
            raise ValueError(f"Unknown studies: {unknown}. Available: {sorted(names)}")
        names = wanted

    for name in names:
        try:
            study = optuna.load_study(study_name=name, storage=args.storage)
            written = save_study_figures(study, args.out_dir)
            if written:
                print(f"{name}: best={study.best_value:.6f} | " + ", ".join(p.name for p in written))
            else:
                print(f"{name}: no completed trials yet, skipped")
        except Exception as exc:
            print(f"{name}: figure generation failed ({exc})")


if __name__ == "__main__":
    main()
