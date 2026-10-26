"""Optuna hyperparameter tuning for Flax forecasting models (validation loss objective).

Reuses windows.npz and scalers_and_meta.pkl from an existing run directory, so
data preparation runs zero times during tuning. Create the windows cheaply with:

    python train_load_forecasters.py --preset <preset> --backend flax --models ridge \
        --run-dir outputs/runs/windows_<preset>

Example:
    python scripts/tune_hyperparams.py --preset manoeuvre_paper \
        --windows-dir outputs/runs/windows_manoeuvre_paper \
        --models dlinear,tcn,itransformer --n-trials 40

Each (preset, model) pair gets its own resumable Optuna study (TPE sampler,
median pruner) stored in sqlite; on WSL prefer a WSL-native --storage path
(e.g. sqlite:////home/<user>/optuna.db) to avoid DrvFS locking issues.
Best parameters are exported to a tuned specs JSON consumable via
train_load_forecasters.py --model-specs / scripts/run_seed_sweep.py --model-specs.
The objective is the best early-stopped validation loss in each model's own
training loss (mse or huber per config/model_specs.json), which is comparable
across trials of the same study.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.config import load_model_specs  # noqa: E402

TUNING_EPOCHS_CAP = 60


def suggest_params(trial, model_key, lookback):
    """Suggest learning rate, batch size, and per-family builder_args for a trial."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    builder_args = {}
    if model_key == "dlinear":
        kernels = list(range(5, min(lookback - 1, 25) + 1, 2))
        builder_args["moving_avg_kernel"] = trial.suggest_categorical("moving_avg_kernel", kernels)
    elif model_key in ("transformer", "itransformer"):
        builder_args["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
        builder_args["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
        builder_args["ff_dim"] = trial.suggest_categorical("ff_dim", [64, 128, 256])
        builder_args["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        if model_key == "transformer":
            builder_args["num_transformer_blocks"] = trial.suggest_int("num_transformer_blocks", 1, 4)
        else:
            builder_args["num_blocks"] = trial.suggest_int("num_blocks", 1, 3)
    elif model_key == "tcn":
        builder_args["nb_filters"] = trial.suggest_categorical("nb_filters", [32, 64, 96, 128])
        builder_args["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
        dilations = trial.suggest_categorical("dilations", ["1,2,4", "1,2,4,8"])
        builder_args["dilations"] = [int(d) for d in dilations.split(",")]
        builder_args["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.3)
    elif model_key in ("gru", "rnn"):
        builder_args["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128])
    elif model_key == "dual_stream_film":
        builder_args["dynamics_width"] = trial.suggest_categorical(
            "dynamics_width", [32, 48, 64]
        )
        builder_args["decoder_width"] = trial.suggest_categorical(
            "decoder_width", [32, 64]
        )
        builder_args["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.2)
    return params, builder_args


def build_module(
    builders,
    spec,
    input_shape,
    output_shape,
    builder_args,
    feature_names,
    target_names,
    target_indices,
):
    args = dict(spec.get("builder_args", {}))
    args.update(builder_args)
    if "mlp_units" in args:
        args["mlp_units"] = tuple(args["mlp_units"])
    if "dilations" in args:
        args["dilations"] = tuple(args["dilations"])
    if spec["builder"] == "itransformer":
        args.setdefault("target_indices", target_indices)
    if spec["builder"] == "dual_stream_film":
        from load_forecasting.models.dual_stream_flax import resolve_feature_group_indices

        args.update(
            resolve_feature_group_indices(
                feature_names,
                target_names,
                spec.get("feature_groups"),
            )
        )
    return builders[spec["builder"]](input_shape=input_shape, output_shape=output_shape, **args)


def make_objective(model_key, spec, data, feature_names, target_names, target_indices, seed):
    import optuna

    from load_forecasting.models.registry_flax import FLAX_MODEL_BUILDERS
    from load_forecasting.training_deep import DeepTrainConfig
    from load_forecasting.training_flax import fit_flax_model

    Xtr, Ytr, Xva, Yva = data
    input_shape = Xtr.shape[1:]
    output_shape = (Ytr.shape[1], Ytr.shape[2])
    lookback = Xtr.shape[1]

    def objective(trial):
        params, builder_args = suggest_params(trial, model_key, lookback)
        module = build_module(
            FLAX_MODEL_BUILDERS,
            spec,
            input_shape,
            output_shape,
            builder_args,
            feature_names,
            target_names,
            target_indices,
        )
        config = DeepTrainConfig(
            batch_size=params["batch_size"],
            epochs=TUNING_EPOCHS_CAP,
            learning_rate=params["learning_rate"],
            shuffle=spec.get("shuffle", False),
            loss="huber" if spec["loss"] == "huber" else "mse",
            seed=seed,
        )

        def epoch_callback(epoch, val_loss):
            trial.report(val_loss, epoch)
            return trial.should_prune()

        _state, _history, best_val = fit_flax_model(
            module, Xtr, Ytr, Xva, Yva, config, epoch_callback=epoch_callback
        )
        if trial.should_prune():
            raise optuna.TrialPruned()
        trial.set_user_attr("builder_args", builder_args)
        return best_val

    return objective


def export_tuned_specs(base_specs_file, out_file, best_by_model, meta):
    """Deep-copy the base specs and override lr/batch/builder_args with tuned values."""
    with open(base_specs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for model_key, tuned in best_by_model.items():
        section = "deep_models" if model_key in data.get("deep_models", {}) else "hybrid_models"
        spec = data[section][model_key]
        spec["learning_rate"] = tuned["learning_rate"]
        spec["batch_size"] = tuned["batch_size"]
        if tuned["builder_args"]:
            spec.setdefault("builder_args", {}).update(tuned["builder_args"])
    data["tuning_meta"] = meta
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Tuned specs written to {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, help="Preset name (study naming and export filename).")
    parser.add_argument("--windows-dir", required=True, help="Run dir containing windows.npz and scalers_and_meta.pkl.")
    parser.add_argument("--models", default=None, help="Comma-separated model keys (default: all Flax models in the specs file).")
    parser.add_argument("--n-trials", type=int, default=40, help="Trials per model (default: 40).")
    parser.add_argument("--seed", type=int, default=0, help="Training seed fixed during the search (default: 0).")
    parser.add_argument("--model-specs", default=str(REPO_ROOT / "config" / "model_specs.json"), help="Base specs file.")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (default: sqlite in outputs/tuning/).")
    parser.add_argument("--export", default=None, help="Output tuned specs JSON (default: config/model_specs_tuned_<preset>.json).")
    args = parser.parse_args()

    import optuna

    windows_dir = Path(args.windows_dir)
    windows = np.load(windows_dir / "windows.npz")
    data = tuple(windows[k].astype(np.float32) for k in ("Xtr", "Ytr", "Xva", "Yva"))
    with (windows_dir / "scalers_and_meta.pkl").open("rb") as f:
        meta_pkl = pickle.load(f)
    feature_names = tuple(meta_pkl["featureNames"])
    target_names = tuple(meta_pkl["targetNames"])
    target_indices = tuple(feature_names.index(target) for target in target_names)

    deep_specs, hybrid_specs = load_model_specs(args.model_specs)
    all_specs = {**deep_specs, **hybrid_specs}
    model_keys = list(all_specs) if args.models is None else [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in all_specs]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {sorted(all_specs)}")

    if args.storage is None:
        storage_dir = REPO_ROOT / "outputs" / "tuning"
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(storage_dir / 'optuna.db').as_posix()}"
    else:
        storage = args.storage

    best_by_model = {}
    for model_key in model_keys:
        spec = all_specs[model_key]
        study = optuna.create_study(
            study_name=f"{args.preset}__{model_key}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        )
        remaining = max(args.n_trials - len(study.trials), 0)
        print(f"\n=== {model_key}: {len(study.trials)} existing trials, running {remaining} more ===")
        if remaining:
            study.optimize(
                make_objective(
                    model_key,
                    spec,
                    data,
                    feature_names,
                    target_names,
                    target_indices,
                    args.seed,
                ),
                n_trials=remaining,
            )
        best = study.best_trial
        best_by_model[model_key] = {
            "learning_rate": best.params["learning_rate"],
            "batch_size": best.params["batch_size"],
            "builder_args": best.user_attrs.get("builder_args", {}),
            "best_val_loss": best.value,
        }
        print(f"{model_key} best val loss: {best.value:.6f} | params: {best.params}")

    export_file = args.export or (REPO_ROOT / "config" / f"model_specs_tuned_{args.preset}.json")
    export_tuned_specs(
        args.model_specs,
        export_file,
        best_by_model,
        meta={
            "preset": args.preset,
            "windows_dir": str(windows_dir),
            "n_trials": args.n_trials,
            "seed": args.seed,
            "objective": "best early-stopped validation loss (per-model training loss)",
            "backend": "flax",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "best_val_loss": {k: v["best_val_loss"] for k, v in best_by_model.items()},
        },
    )
    try:  # best-effort figures; must never fail the tuning run
        from gen_optuna_figures import save_study_figures

        figs_dir = REPO_ROOT / "outputs" / "optuna_figs"
        for model_key in model_keys:
            study = optuna.load_study(study_name=f"{args.preset}__{model_key}", storage=storage)
            for fig_path in save_study_figures(study, figs_dir):
                print(f"Figure written: {fig_path}")
    except Exception as exc:
        print(f"Figure generation skipped: {exc}")

    print("\nNext step (5-seed final runs):")
    print(f"  python scripts/run_seed_sweep.py --preset {args.preset} --backend flax --seeds 0,1,2,3,4 --model-specs {export_file}")


if __name__ == "__main__":
    main()
