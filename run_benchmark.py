"""
Grid benchmark runner for manoeuvre and navigation load-forecasting modes.

Runs train_load_forecasters.py for each (LookbackSteps, HorizonSteps)
configuration and archives metrics, scalers, windows, and trained models.

Examples
--------
python run_benchmark.py
python run_benchmark.py --selector manoeuvre
python run_benchmark.py --selector navigation
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MOON_BASE = Path(__file__).resolve().parent
SCRIPT = MOON_BASE / "train_load_forecasters.py"
OUTROOT = MOON_BASE / "outputs" / "runs" / "comprehensive_benchmark"
RESULTS_DIR = MOON_BASE / "outputs" / "test_results"

CONFIGS = [
    (20, 5), (20, 15), (20, 25),
    (45, 5), (45, 15), (45, 25),
    (60, 5), (60, 15), (60, 25),
]

MODEL_PALETTE = {
    "Ridge": "#6b7280",
    "XGBoost": "#111827",
    "MLP": "#14b8a6",
    "GRU": "#0ea5e9",
    "RNN": "#a855f7",
    "CNN": "#1f77b4",
    "TCN": "#ff7f0e",
    "ConvLSTM-H": "#2ca02c",
    "ConvLSTMTCN-H": "#d62728",
    "TCN-LSTM-H": "#9467bd",
    "ConvTCN-H": "#8c564b",
    "Transformer": "#e377c2",
    "DLinear": "#bcbd22",
    "iTransformer": "#17becf",
    "DualStream-FiLM": "#e6550d",
}

MODE_CONFIG = {
    "manoeuvre": {
        "label": "Manoeuvre",
        "figure_dir": "paper_figures",
        "summary_csv": "benchmark_summary.csv",
        "best_csv": "benchmark_best_by_config.csv",
    },
    "navigation": {
        "label": "Navigation",
        "figure_dir": "paper_figures_nav",
        "summary_csv": "benchmark_summary_nav.csv",
        "best_csv": "benchmark_best_by_config_nav.csv",
    },
}


def parse_args(argv=None):
    """
    Parse command-line arguments for the benchmark runner.

    Inputs
    ----------
    argv : list[str] or None
        Optional list of command-line arguments. If None, arguments are read
        from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, including the selected operating-mode selector.
    """
    parser = argparse.ArgumentParser(
        description="Run the L/H benchmark grid for one forecasting selector."
    )
    parser.add_argument(
        "--selector",
        choices=sorted(MODE_CONFIG),
        default="manoeuvre",
        help="Dataset selector passed to train_load_forecasters.py.",
    )
    parser.add_argument(
        "--backend",
        choices=("keras", "flax"),
        default=None,
        help="Backend passed to train_load_forecasters.py (default: script default).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys passed to train_load_forecasters.py.",
    )
    parser.add_argument(
        "--model-specs",
        default=None,
        help="Model-specs JSON passed to train_load_forecasters.py (e.g. tuned specs).",
    )
    parser.add_argument(
        "--out-tag",
        default="",
        help="Suffix for the output root and archive names, to keep runs separate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to train_load_forecasters.py so models with saved predictions in the run dir are not retrained after an interrupt.",
    )
    return parser.parse_args(argv)


def save_fig(figdir, name, dpi=300):
    """
    Save the current matplotlib figure in PNG and EPS format.

    Inputs
    ----------
    figdir : pathlib.Path
        Directory where the figure files are saved.
    name : str
        Base filename without extension.
    dpi : int, optional
        Resolution used for the PNG export.

    Returns
    -------
    None
        Saves the figure and closes the active matplotlib figure.
    """
    plt.savefig(figdir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(figdir / f"{name}.eps", format="eps", bbox_inches="tight")
    plt.close()


def already_done(tag, mode_label):
    """
    Check whether a benchmark configuration has already been archived.

    Inputs
    ----------
    tag : str
        Benchmark configuration tag, such as "L45_H25" (with optional
        "_<out-tag>" suffix).
    mode_label : str
        Operating-mode label, such as "Manoeuvre" or "Navigation".

    Returns
    -------
    bool
        True if an archive matching the configuration already exists.
    """
    return any(RESULTS_DIR.glob(f"*_ForecastModels_{mode_label}_{tag}"))


def append_metrics(summary_rows, csv_path, LookbackSteps, HorizonSteps):
    """
    Append model metrics from one benchmark CSV file to the summary list.

    Inputs
    ----------
    summary_rows : list[dict]
        List updated in place with one row per model.
    csv_path : pathlib.Path
        Path to the results_metrics.csv file.
    LookbackSteps : int
        Lookback window length in seconds or samples, depending on upstream use.
    HorizonSteps : int
        Forecast horizon length in seconds or samples, depending on upstream use.

    Returns
    -------
    None
        Updates summary_rows in place.
    """
    df = pd.read_csv(csv_path, index_col=0)
    for model, row in df.iterrows():
        summary_rows.append({
            "Lookback": LookbackSteps,
            "Horizon": HorizonSteps,
            "Model": model,
            "RMSE_all": row["RMSE_all"],
            "MAE_all": row["MAE_all"],
            "MAPE_all": row["MAPE_all"],
        })


def archive_current_run(today, mode_label, tag, outroot):
    """
    Archive metrics, scalers, windows, and trained models from the latest run.

    Inputs
    ----------
    today : str
        Date string used as archive prefix, typically in YYYYMMDD format.
    mode_label : str
        Operating-mode label used in the archive name.
    tag : str
        Benchmark configuration tag, such as "L45_H25" (with optional
        "_<out-tag>" suffix).
    outroot : pathlib.Path
        Run directory whose artifacts are archived.

    Returns
    -------
    pathlib.Path
        Path to the created archive directory.
    """
    archive = RESULTS_DIR / f"{today}_ForecastModels_{mode_label}_{tag}"
    archive.mkdir(exist_ok=True)

    for fname in ("results_metrics.csv", "results_metrics.parquet", "scalers_and_meta.pkl", "windows.npz", "run_summary_public.json"):
        src = outroot / fname
        if src.exists():
            shutil.copy2(src, archive / fname)

    for pattern in ("model_*.keras", "model_*.joblib"):
        for model_path in outroot.glob(pattern):
            shutil.copy2(model_path, archive / model_path.name)

    for flax_dir in outroot.glob("model_*_flax"):
        if flax_dir.is_dir():
            shutil.copytree(flax_dir, archive / flax_dir.name, dirs_exist_ok=True)

    # Per-window predictions: required post-hoc for bootstrap CIs, conformal
    # bands, DM significance, and error-by-ramp-rate analyses.
    predictions = outroot / "predictions"
    if predictions.is_dir():
        shutil.copytree(predictions, archive / "predictions", dirs_exist_ok=True)

    return archive


def run_config(selector, mode_label, LookbackSteps, HorizonSteps, outroot,
               backend=None, models=None, model_specs=None, resume=False):
    """
    Run train_load_forecasters.py for a single benchmark configuration.

    Inputs
    ----------
    selector : str
        Dataset selector passed to the training script.
    mode_label : str
        Human-readable operating-mode label. Kept for call-site consistency.
    LookbackSteps : int
        Lookback window length passed to the training script.
    HorizonSteps : int
        Forecast horizon length passed to the training script.
    outroot : pathlib.Path
        Run directory passed as --run-dir.
    backend : str or None
        Optional --backend override ("keras" or "flax").
    models : str or None
        Optional comma-separated --models override.
    model_specs : str or None
        Optional --model-specs JSON path (e.g. tuned specs).
    resume : bool
        If True, pass --resume so models with saved predictions are skipped.

    Returns
    -------
    None
        Executes the training script and raises an error if it fails. Anyway, you can just import the SCRIPT as it is a .py program.
    """
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--selector",
        selector,
        "--lookback",
        str(LookbackSteps),
        "--horizon",
        str(HorizonSteps),
        "--run-dir",
        str(outroot),
        "--skip-eda",
        "--skip-plots",
    ]
    if backend:
        cmd += ["--backend", backend]
    if models:
        cmd += ["--models", models]
    if model_specs:
        cmd += ["--model-specs", model_specs]
    if resume:
        cmd += ["--resume"]
    subprocess.run(
        cmd,
        cwd=str(MOON_BASE),
        check=True,
    )


def write_summary_figures(summary, best, figdir, mode_label):
    """
    Generate benchmark summary figures for one operating mode.

    Inputs
    ----------
    summary : pandas.DataFrame
        Full benchmark table with one row per model and configuration.
    best : pandas.DataFrame
        Best-performing model for each lookback/horizon configuration.
    figdir : pathlib.Path
        Directory where summary figures are saved.
    mode_label : str
        Human-readable operating-mode label used in figure titles.

    Returns
    -------
    None
        Saves heatmaps, winner tables, and MAPE trend figures.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    heat = best.pivot(index="Lookback", columns="Horizon", values="MAPE_all").sort_index()
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        linewidths=0.5,
        cbar_kws={"label": "MAPE (%)"},
    )
    plt.title(f"{mode_label} - best MAPE by lookback / horizon")
    plt.xlabel("Horizon (s)")
    plt.ylabel("Lookback (s)")
    plt.tight_layout()
    save_fig(figdir, "heatmap_best_mape")

    winner = best.pivot(index="Lookback", columns="Horizon", values="Model").sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=winner.values,
        rowLabels=winner.index,
        colLabels=winner.columns,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.5)
    plt.title(f"{mode_label} - best model by lookback / horizon")
    plt.tight_layout()
    save_fig(figdir, "winner_table")

    for LookbackSteps in sorted(summary["Lookback"].unique()):
        dfL = summary[summary["Lookback"] == LookbackSteps]
        plt.figure(figsize=(6.5, 4))
        sns.lineplot(
            data=dfL,
            x="Horizon",
            y="MAPE_all",
            hue="Model",
            style="Model",
            markers=True,
            dashes=False,
            palette=MODEL_PALETTE,
        )
        plt.title(f"{mode_label} MAPE vs horizon  (L={LookbackSteps} s)")
        plt.xlabel("Horizon (s)")
        plt.ylabel("MAPE (%)")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        save_fig(figdir, f"mape_vs_horizon_L{LookbackSteps}")

    for HorizonSteps in sorted(summary["Horizon"].unique()):
        dfH = summary[summary["Horizon"] == HorizonSteps]
        plt.figure(figsize=(6.5, 4))
        sns.lineplot(
            data=dfH,
            x="Lookback",
            y="MAPE_all",
            hue="Model",
            style="Model",
            markers=True,
            dashes=False,
            palette=MODEL_PALETTE,
        )
        plt.title(f"{mode_label} MAPE vs lookback  (H={HorizonSteps} s)")
        plt.xlabel("Lookback (s)")
        plt.ylabel("MAPE (%)")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        save_fig(figdir, f"mape_vs_lookback_H{HorizonSteps}")


def run_benchmark(selector, backend=None, models=None, model_specs=None, out_tag="", resume=False):
    """
    Run the full lookback/horizon benchmark grid for one operating mode.

    Inputs
    ----------
    selector : str
        Operating-mode selector, either "manoeuvre" or "navigation".
    backend : str or None
        Optional --backend override passed to the training script.
    models : str or None
        Optional comma-separated --models override.
    model_specs : str or None
        Optional --model-specs JSON path (e.g. tuned specs).
    out_tag : str
        Optional suffix isolating this campaign's output root and archives
        from previous runs (e.g. "flax7").
    resume : bool
        If True, forward --resume to the training script so models already
        trained in an interrupted configuration are not retrained.

    Returns
    -------
    None
        Runs missing configurations, archives outputs, writes CSV summaries,
        and generates summary figures.
    """
    mode = MODE_CONFIG[selector]
    mode_label = mode["label"]
    outroot = OUTROOT.with_name(f"{OUTROOT.name}_{out_tag}") if out_tag else OUTROOT
    figdir = outroot / mode["figure_dir"]

    outroot.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    summary_rows = []
    today = datetime.now().strftime("%Y%m%d")

    for LookbackSteps, HorizonSteps in CONFIGS:
        tag = f"L{LookbackSteps}_H{HorizonSteps}" + (f"_{out_tag}" if out_tag else "")

        if already_done(tag, mode_label):
            print(f"SKIP {tag} - archive found, loading metrics")
            existing = sorted(RESULTS_DIR.glob(f"*_ForecastModels_{mode_label}_{tag}"))[-1]
            csv_path = existing / "results_metrics.csv"
            if csv_path.exists():
                append_metrics(summary_rows, csv_path, LookbackSteps, HorizonSteps)
            continue

        print(
            f"\n{'=' * 60}\n"
            f"=== {mode_label.upper()}  L={LookbackSteps}s  H={HorizonSteps}s ===\n"
            f"{'=' * 60}",
            flush=True,
        )

        # Run the main training script as a subprocess. Anyway, you can just import the SCRIPT as it is a .py program.
        run_config(
            selector,
            mode_label,
            LookbackSteps,
            HorizonSteps,
            outroot,
            backend=backend,
            models=models,
            model_specs=model_specs,
            resume=resume,
        )
        archive = archive_current_run(today, mode_label, tag, outroot)
        print(f"  -> {archive.name}")

        append_metrics(
            summary_rows,
            outroot / "results_metrics.csv",
            LookbackSteps,
            HorizonSteps,
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError(f"No benchmark metrics found for selector={selector!r}.")

    summary.to_csv(outroot / mode["summary_csv"], index=False)

    best = summary.loc[
        summary.groupby(["Lookback", "Horizon"])["MAPE_all"].idxmin()
    ].copy()
    best.to_csv(outroot / mode["best_csv"], index=False)

    write_summary_figures(summary, best, figdir, mode_label)
    print(f"\nDone. Figures -> {figdir}")


def main(argv=None):
    """
    Run the benchmark command-line entry point.

    Inputs
    ----------
    argv : list[str] or None
        Optional list of command-line arguments. If None, arguments are read
        from sys.argv.

    Returns
    -------
    None
        Parses command-line arguments and runs the selected benchmark.
    """
    args = parse_args(argv)
    run_benchmark(
        args.selector,
        backend=args.backend,
        models=args.models,
        model_specs=args.model_specs,
        out_tag=args.out_tag,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
