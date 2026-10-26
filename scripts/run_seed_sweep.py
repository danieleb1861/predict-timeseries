"""Run the training pipeline over multiple seeds, one run directory per seed.

Example:
    python scripts/run_seed_sweep.py --preset manoeuvre_paper --backend flax --seeds 0,1,2

Each seed trains into <run-root>/seed<k>/ so artifacts and metrics never
overwrite each other. Aggregate afterwards with scripts/aggregate_seed_results.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, help="Named preset from config/presets.json.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (default: 0,1,2).")
    parser.add_argument("--lovo-cross-val", type=int, default=None, metavar="N",
                        help="Run an N-fold leave-one-voyage-out cross-validation instead of the configured "
                             "split: fold k holds out leg #k as the test set. N must equal the number of leg "
                             "files of the selector. Runs land in <run-root>/fold<k>/, or "
                             "<run-root>/fold<k>/seed<s>/ when several seeds are requested.")
    parser.add_argument("--backend", default="flax", choices=("keras", "flax"))
    parser.add_argument("--models", default=None, help="Optional --models override passed through.")
    parser.add_argument("--model-specs", default=None, help="Optional --model-specs override passed through to each seed run.")
    parser.add_argument("--config", default=None, help="Optional --config override passed through to each seed run (dataset/physical-system JSON). Without it every seed uses config/default_config.json, which points at the synthetic demo data.")
    parser.add_argument(
        "--run-root",
        default=None,
        help="Parent directory for the per-seed run dirs (default: outputs/runs/sweep_<preset>_<backend>).",
    )
    parser.add_argument("--skip-plots", action="store_true", default=True, help="Skip per-seed plots (default on).")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    run_root = (
        Path(args.run_root)
        if args.run_root
        else REPO_ROOT / "outputs" / "runs" / f"sweep_{args.preset}_{args.backend}"
    )

    folds = list(range(args.lovo_cross_val)) if args.lovo_cross_val else [None]
    jobs = []
    for fold in folds:
        for seed in seeds:
            if fold is None:
                run_dir = run_root / f"seed{seed}"
            elif len(seeds) == 1:
                run_dir = run_root / f"fold{fold}"
            else:
                run_dir = run_root / f"fold{fold}" / f"seed{seed}"
            jobs.append((fold, seed, run_dir))

    for fold, seed, run_dir in jobs:
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "train_load_forecasters.py"),
            "--preset", args.preset,
            "--backend", args.backend,
            "--seed", str(seed),
            "--run-dir", str(run_dir),
        ]
        if args.skip_plots:
            cmd.append("--skip-plots")
        if args.models:
            cmd += ["--models", args.models]
        if args.model_specs:
            cmd += ["--model-specs", args.model_specs]
        if args.config:
            cmd += ["--config", args.config]
        if fold is not None:
            cmd += ["--lovo-fold", str(fold)]

        label = f"seed {seed}" if fold is None else f"fold {fold} / seed {seed}"
        print(f"\n=== {label} -> {run_dir} ===")
        print(">>", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            sys.exit(f"{label} failed with exit code {result.returncode}; sweep aborted.")

    pattern = "seed*" if args.lovo_cross_val is None else "fold*"
    print(f"\nSweep complete ({len(jobs)} runs). Aggregate with:")
    print(f"  python scripts/aggregate_seed_results.py --run-root {run_root} --pattern '{pattern}'")


if __name__ == "__main__":
    main()
