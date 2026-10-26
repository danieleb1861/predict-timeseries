"""
Regression using Keras on load datasets
End-to-end pipeline for timeseries analysis (.parquet databases):
load files - resample - normalize - windowing - optimize features - train models - evaluate - post processing

Prediction of active power consumption using other measured variables as features.

"""

# %% SETUP
import json
import os
import numpy as np
import pandas as pd
from time import time as tmer
from datetime import datetime

import pickle # Implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” (or “serialization”, “marshalling,” or “flattening”) is the process whereby a Python object hierarchy is converted into a binary file or bytes-like object, and “unpickling” is the inverse operation.

from load_forecasting.pca_dpca import run_pca_dpca_analysis
from load_forecasting.data_prep import add_engineered, cached_preprocess_dataset
from load_forecasting.config import (
    cfg_or,
    load_model_specs,
    load_system_config,
    parse_cli_config,
    parse_selected_models,
    resolve_default_config,
    resolve_project_path,
    validate_inputs,
)
from load_forecasting.metrics import compute_metrics, save_metrics_outputs
from load_forecasting.plots import (
    plot_full_test,
    plot_metric_over_horizon,
    plot_scatter_grid,
    save_plot,
    set_pub_style,
)
from load_forecasting.postprocessing import stitch_prediction_dict
from load_forecasting.save_metadata import collect_run_provenance, build_run_summary, build_public_run_summary, print_run_summary, save_run_summary
from load_forecasting.scaling import inverse_scale_targets, scale_full_dataset_for_diagnostics, scale_train_val_test
from load_forecasting.splitting import lovo_split, split_train_val_test
from load_forecasting.training import train_eval_flat_model
from load_forecasting.windowing import make_windows


def load_and_preprocess_legs(files, DATA_DIR, colMap, data_types, rating_kW, varsPrev, varsLin, Ts, max_fill_seconds):
    legs_res = []
    for k, fname in enumerate(files):
        path = os.path.join(DATA_DIR, fname)
        # append to legs
        legs_res.append(cached_preprocess_dataset(path, k, colMap, data_types, DATA_DIR, rating_kW, varsPrev, varsLin, Ts=Ts, max_fill_seconds=max_fill_seconds))

    return pd.concat(legs_res, ignore_index=True)


def build_forecasting_windows(TTtrainZ, TTvalZ, TTtestZ, featureNames, targetNames, LookbackSteps, HorizonSteps, WindowSlidingCoeff):
    group_col = "leg_id" if all("leg_id" in frame.columns for frame in (TTtrainZ, TTvalZ, TTtestZ)) else None
    print("Training windows:")
    Xtr, Ytr = make_windows(TTtrainZ, featureNames, targetNames, LookbackSteps, HorizonSteps, WindowSlidingCoeff=WindowSlidingCoeff, group_col=group_col)
    print("Validation windows:")
    Xva, Yva = make_windows(TTvalZ,   featureNames, targetNames, LookbackSteps, HorizonSteps, WindowSlidingCoeff=WindowSlidingCoeff, group_col=group_col)
    print("Test windows:")
    Xte, Yte = make_windows(TTtestZ,  featureNames, targetNames, LookbackSteps, HorizonSteps, WindowSlidingCoeff=WindowSlidingCoeff, group_col=group_col)

    output_dim = len(targetNames)
    num_features = Xtr.shape[2]                 # num features
    input_shape = (LookbackSteps, num_features) # Input shape: (lookback horizon=LookbackSteps, features=num_features)
    output_shape = (HorizonSteps, output_dim)  # Output shape: (forecast horizon=HorizonSteps, output_dim)

    print("Xtr", Xtr.shape, "Ytr", Ytr.shape)
    print("Xva", Xva.shape, "Yva", Yva.shape)
    print("Xte", Xte.shape, "Yte", Yte.shape)

    return Xtr, Ytr, Xva, Yva, Xte, Yte, input_shape, output_shape


def resolve_optional_branch_flags(cli_config):
    return {
        "EDA": not cli_config.skip_eda,
        "PostTraining": not cli_config.skip_plots,
        "LoadDynamics": bool(cli_config.load_dynamics),
    }


def main():
    print()
    print(5*"-", " Hi there! Test run at", datetime.now(), 5*"-")
    print()

    PIPELINE_START_TIME = tmer()

    # ====================================================
    # %% -- Functions

    def log_time(label, start_time):
        """
        Helper function to log elapsed time for a given stage of the pipeline.
        """
        elapsed = tmer() - start_time
        total = tmer() - PIPELINE_START_TIME
        print(f"[timing] {label}: {elapsed:.2f} s (total {total:.2f} s)")
        print()

    def inv_scale(z):
        """
        Inverse scaling of predictions using stored mean and std.
    
        Inputs
        ----------
        z : array-like
            Normalised predictions dim(z) = (num_windows, HorizonSteps, output_dim).
    
        Returns
        -------
            array of predictions in the original scale (num_windows, HorizonSteps, output_dim)
        """

        return inverse_scale_targets(z, targetNames, MU, SIG)

    set_pub_style(font_family="Times New Roman", fontsize=9)

    # ====================================================
    # %% -- Dataset loading and preprocessing
    # ====================================================

    # Directory setup. We are on the MOON!
    MOON_BASE = os.path.dirname(os.path.abspath(__file__))

    MODEL_LABELS = {
        "ridge": "Ridge",
        "xgb": "XGBoost",
        "mlp": "MLP",
        "gru": "GRU",
        "rnn": "RNN",
        "cnn": "CNN",
        "tcn": "TCN",
        "transformer": "Transformer",
        "dlinear": "DLinear",
        "itransformer": "iTransformer",
        "dual_stream_film": "DualStream-FiLM",
        "convlstm_hybrid": "ConvLSTM-H",
        "tcnlstm_hybrid": "TCN-LSTM-H",
        "convtcn_hybrid": "ConvTCN-H",
        "convlstmtcn_hybrid": "ConvLSTMTCN-H",
    }

    DEFAULT_CONFIG_FILE = resolve_default_config(os.path.join(MOON_BASE, "config"))
    DEFAULT_PRESET_FILE = os.path.join(MOON_BASE, "config", "presets.json")
    DEFAULT_MODEL_SPECS_FILE = os.path.join(MOON_BASE, "config", "model_specs.json")
    PRESET_KEYS = {
        "selector",
        "lookback",
        "horizon",
        "run_dir",
        "backend",
        "skip_eda",
        "skip_plots",
        "load_dynamics",
        "models",
        "epochs",
        "batch_size",
        "learning_rate",
        "predict_batch_size",
        "seed",
        "model_specs",
        "resume",
    }
    CLI_DEFAULTS = {
        "selector": "manoeuvre",
        "lovo_fold": None,
        "lookback": None,
        "horizon": None,
        "run_dir": None,
        "backend": "keras",
        "skip_eda": True,
        "skip_plots": True,
        "load_dynamics": False,
        "models": "all",
        "epochs": None,
        "batch_size": None,
        "learning_rate": None,
        "predict_batch_size": 256,
        "seed": 0,
        "model_specs": None,
        "resume": False,
    }


    CLI_CONFIG = parse_cli_config(DEFAULT_CONFIG_FILE, DEFAULT_PRESET_FILE, MODEL_LABELS, PRESET_KEYS, CLI_DEFAULTS)
    SYSTEM_CONFIG = load_system_config(CLI_CONFIG.config)
    MODEL_SPECS_FILE = resolve_project_path(CLI_CONFIG.model_specs, MOON_BASE) if CLI_CONFIG.model_specs else DEFAULT_MODEL_SPECS_FILE
    DEEP_MODEL_SPECS, HYBRID_MODEL_SPECS = load_model_specs(MODEL_SPECS_FILE)
    configured_model_keys = {"ridge", "xgb", *DEEP_MODEL_SPECS, *HYBRID_MODEL_SPECS}
    MODEL_LABELS = {
        key: label for key, label in MODEL_LABELS.items() if key in configured_model_keys
    }
    if CLI_CONFIG.model_specs and any(v is not None for v in (CLI_CONFIG.epochs, CLI_CONFIG.batch_size, CLI_CONFIG.learning_rate)):
        print("WARNING: global epochs/batch_size/learning_rate overrides (CLI or preset) take precedence over the per-model values in --model-specs.")

    MODELS_TO_RUN = parse_selected_models(CLI_CONFIG.models, MODEL_LABELS)
    PREDICT_BATCH_SIZE = CLI_CONFIG.predict_batch_size
    BACKEND = CLI_CONFIG.backend
    print("Selected models:", ", ".join(MODEL_LABELS[k] for k in MODEL_LABELS if k in MODELS_TO_RUN))
    print("Deep-learning backend:", BACKEND)
    print()

    # Folder names. Benchmarks pass --run-dir explicitly.
    PATH_CONFIG = SYSTEM_CONFIG["paths"]
    DATASET_CONFIG = SYSTEM_CONFIG["dataset"]
    PHYSICAL_CONFIG = SYSTEM_CONFIG["physical_system"]
    SIGNAL_CONFIG = PHYSICAL_CONFIG["signals"]

    DATA_DIR = resolve_project_path(PATH_CONFIG.get("data_dir", "data"), MOON_BASE)
    RUN_DIR = CLI_CONFIG.run_dir or resolve_project_path(PATH_CONFIG.get("run_dir", "outputs/runs"), MOON_BASE)
    IMGS_DIR = os.path.join(RUN_DIR, "img")
    os.makedirs(IMGS_DIR, exist_ok=True)

    selector = CLI_CONFIG.selector
    FILES_BY_SELECTOR = DATASET_CONFIG["selectors"]
    if selector not in FILES_BY_SELECTOR:
        raise ValueError(f"Selector '{selector}' is not defined in config selectors: {sorted(FILES_BY_SELECTOR)}")
    files = FILES_BY_SELECTOR[selector]

    # Dataset and physical-system settings come from --config.
    colMap = DATASET_CONFIG["columns"]
    data_types = {colMap[key]: dtype for key, dtype in DATASET_CONFIG["dtypes"].items()}
    rating_kW = float(PHYSICAL_CONFIG["rating_kW"])

    # ====================================================
    # %% -- Parameters for windowing and models
    # ====================================================
    if selector == "navigation":
        numGens = 3
        TsH = 1.0                                   # seconds (since Ts = 1S)
        LookbackSteps = int(round((CLI_CONFIG.lookback or 45) / TsH))        # lookback steps
        HorizonSteps = int(round((CLI_CONFIG.horizon or 15) / TsH))         # horizon steps
        WindowSlidingCoeff = int(HorizonSteps/5)    # forecasting windows
    elif selector == "manoeuvre":
        numGens = 4
        TsH = 1.0                                   # seconds (since Ts = 1S)
        LookbackSteps = int(round((CLI_CONFIG.lookback or 45) / TsH))        # lookback steps
        HorizonSteps = int(round((CLI_CONFIG.horizon or 15) / TsH))         # horizon steps
        WindowSlidingCoeff = int(HorizonSteps/5)    # forecasting windows
    else:
        numGens = np.nan
        TsH = 1.0                                   # seconds (since Ts = 1S)
        LookbackSteps = int(round((CLI_CONFIG.lookback or 40) / TsH))        # lookback steps
        HorizonSteps = int(round((CLI_CONFIG.horizon or 15) / TsH))         # horizon steps
        WindowSlidingCoeff = int(HorizonSteps/5)    # forecasting windows

    if WindowSlidingCoeff < 1:
        raise ValueError("WindowSlidingCoeff must be >= 1")

    if LookbackSteps < 1 or HorizonSteps < 1:
        raise ValueError("LookbackSteps and HorizonSteps must be positive")

    branch_flags = resolve_optional_branch_flags(CLI_CONFIG)
    EDA = branch_flags["EDA"]
    BESS = False
    DPCA = False
    PCAplot = False
    LOAD_DYNAMICS = branch_flags["LoadDynamics"]
    PostTraining = branch_flags["PostTraining"]

    # ===========================================================
    # %% -- Load, concatenate all legs and resample to fixed cadence
    # ===========================================================
    Ts = DATASET_CONFIG.get("sampling_rate", "1s")
    max_fill_seconds = DATASET_CONFIG.get("max_fill_seconds", 45)

    validate_inputs(files, DATA_DIR, colMap)

    RUN_PROVENANCE = collect_run_provenance(
        MOON_BASE,
        CLI_CONFIG.config,
        CLI_CONFIG.preset_file,
        MODEL_SPECS_FILE,
        DATA_DIR,
        files,
        BACKEND,
    )

    summary_kwargs = {
        "cli_config": CLI_CONFIG,
        "selector": selector,
        "files": files,
        "run_dir": RUN_DIR,
        "imgs_dir": IMGS_DIR,
        "model_labels": MODEL_LABELS,
        "models_to_run": MODELS_TO_RUN,
        "LookbackSteps": LookbackSteps,
        "HorizonSteps": HorizonSteps,
        "WindowSlidingCoeff": WindowSlidingCoeff,
        "Ts": Ts,
        "TsH": TsH,
        "max_fill_seconds": max_fill_seconds,
        "EDA": EDA,
        "PostTraining": PostTraining,
        "BESS": BESS,
        "DPCA": DPCA,
        "PCAplot": PCAplot,
        "system_config": SYSTEM_CONFIG,
    }
    RUN_SUMMARY = build_run_summary(**summary_kwargs,
                                    extra={"provenance": RUN_PROVENANCE,
                                           "lovo_fold": CLI_CONFIG.lovo_fold})
    print_run_summary(RUN_SUMMARY)
    save_run_summary(RUN_SUMMARY, RUN_DIR)

    varsPrev = SIGNAL_CONFIG["step_hold"]
    varsLin = SIGNAL_CONFIG["resample_continuous"]

    stage_start = tmer()
    TT = load_and_preprocess_legs(files, DATA_DIR, colMap, data_types, rating_kW, varsPrev, varsLin, Ts, max_fill_seconds)
    log_time("load_preprocess_concat", stage_start)

    print("TT details:")
    TT.info()
    print()

    print("Group the number of samples of refined datasets by leg_id")
    print(TT.groupby("leg_id").size())
    print()

    # plt.figure(figsize=(8,4))
    # plt.plot(TT["dtime"], TT["act_pow_pu"])
    # plt.xlabel("Time")
    # plt.ylabel("Active Power (PU)")
    # plt.title("Active Power Profile")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # ====================================================
    # %% -- Feature engineering
    # ====================================================

    stage_start = tmer()
    TT = add_engineered(TT)
    log_time("feature_engineering", stage_start)

    # ====================================================
    # %% -- Train / Validation / Test splits
    # ====================================================

    stage_start = tmer()
    # "balanced" whole-leg assignment (matches sample fractions closely,
    # order not preserved) is opt-in per selector so existing archived
    # splits (e.g. manoeuvre) stay reproducible; navigation already adopted
    # it (see load_forecasting/splitting.py mode="balanced" docstring).
    split_mode = "balanced" if selector == "navigation" else "ordered"
    if CLI_CONFIG.lovo_fold is None:
        TTtrain, TTval, TTtest = split_train_val_test(TT, mode=split_mode)
    else:
        # Leave-one-voyage-out fold: one whole leg is the test set (see
        # load_forecasting/splitting.py lovo_split docstring).
        TTtrain, TTval, TTtest = lovo_split(TT, CLI_CONFIG.lovo_fold)

    log_time("train_val_test_split", stage_start)

    # ====================================================
    # %% -- Normalization (z-score, train stats)
    # ====================================================

    stage_start = tmer()
    print("TTtrainZ stats:", TTtrain.drop('dtime', axis=1).describe())
    print()

    print("TTvalZ stats:", TTval.drop('dtime', axis=1).describe())
    print()

    print("TTtestZ stats:", TTtest.drop('dtime', axis=1).describe())
    print()

    varsCont = SIGNAL_CONFIG["standardize_continuous"]
    TTtrainZ, TTvalZ, TTtestZ, MU, SIG = scale_train_val_test(TTtrain, TTval, TTtest, varsCont)
    print("Nan values in TTtrainZ:", np.isnan(TTtrainZ).sum())
    print()

    # Only PCA plotting needs the full-fit copy; dPCA uses TTtrainZ directly.
    TTZ = None
    if PCAplot:
        TTZ = scale_full_dataset_for_diagnostics(TT, varsCont)
        print("TTz details:")
        TTZ.info()
        print()
    log_time("normalization", stage_start)

    # ====================================================
    # %% -- Exploratory Data Analysis (EDA)
    # ====================================================
    if EDA is True:
        from load_forecasting.eda import run_eda_analysis

        run_eda_analysis(TT, IMGS_DIR, selector, numGens)
    else:
        print("EDA skipped")
        print()
    
    # ====================================================
    # %% -- BESS sizing analysis
    # ====================================================

    # %% Run EDA suite and sizing
    if BESS == True:
        from load_forecasting.bess import run_bess_sizing_analysis

        TTp = run_bess_sizing_analysis(TT, IMGS_DIR, selector, numGens, rating_kW)  # noqa: F841
    else:
        print("Skipping frequency analysis")
        print()

    # ====================================================
    # %% -- Load-dynamics diagnostics (per leg, per split)
    # ====================================================
    if LOAD_DYNAMICS:
        from load_forecasting.load_dynamics import run_load_dynamics_analysis

        stage_start = tmer()
        run_load_dynamics_analysis(
            {"all": TT, "train": TTtrain, "val": TTval, "test": TTtest},
            IMGS_DIR,
            selector,
            fs=1.0 / TsH,
        )
        log_time("load_dynamics", stage_start)
    else:
        print("Skipping load-dynamics diagnostics")
        print()
    
    # ====================================================
    # %% -- Dinamic PCA (dPCA) - Singular Spectrum Analysis (SSA)
    # ====================================================
    pca_cols = run_pca_dpca_analysis(TT, TTtrainZ, TTvalZ, TTtestZ, TTZ, IMGS_DIR, DPCA, PCAplot)
    
    # ====================================================
    # %% -- Windowing
    # ====================================================

    stage_start = tmer()
    # -- Features and targets (feature selection should be done through PCA + KMeans)
    featureNames = pca_cols
    targetNames = SIGNAL_CONFIG["targets"]

    # dimension axis value of targetNames for plots
    d_plot = 0                # 0=act, 1=react (if output_dim=2)

    Xtr, Ytr, Xva, Yva, Xte, Yte, input_shape, output_shape = build_forecasting_windows(
        TTtrainZ, TTvalZ, TTtestZ, featureNames, targetNames, LookbackSteps, HorizonSteps, WindowSlidingCoeff
    )
    log_time("windowing", stage_start)

    # ====================================================
    # %% -- Baseline and legacy deep models
    # ====================================================

    y_true_phys = inv_scale(Yte)
    model_metrics = {}
    model_predictions = {}

    RESUME = bool(CLI_CONFIG.resume)

    def maybe_resume_model(model_key, model_label):
        """
        Load saved per-window predictions for one model when resuming.

        Inputs
        ----------
        model_key : str
            Key used by save_predictions(...) for <run-dir>/predictions/<key>.npz.
        model_label : str
            Human-readable model label used in log messages.

        Returns
        -------
        np.ndarray or None
            Test predictions in physical units when --resume is active and the
            saved file matches the current window configuration; otherwise
            None, which triggers a normal training run.
        """
        if not RESUME:
            return None
        pred_path = os.path.join(RUN_DIR, "predictions", f"{model_key}.npz")
        if not os.path.exists(pred_path):
            return None
        try:
            with np.load(pred_path) as data:
                if "yhat_val_phys" not in data.files:
                    return None
                yhat_phys = data["yhat_test_phys"]
                yhat_val_phys = data["yhat_val_phys"]
        except Exception as exc:
            print(f"RESUME {model_label}: unreadable {pred_path} ({exc}); retraining.")
            return None
        # Shape guard against stale files from another lookback/horizon/selector.
        if yhat_phys.shape != y_true_phys.shape or yhat_val_phys.shape != (Xva.shape[0], *y_true_phys.shape[1:]):
            print(f"RESUME {model_label}: stale prediction shapes in {pred_path}; retraining.")
            return None
        print(f"RESUME {model_label}: loaded saved predictions, skipping training.")
        return yhat_phys

    flat_train_kwargs = {
        "run_dir": RUN_DIR,
        "inv_scale": inv_scale,
        "y_true_phys": y_true_phys,
        "d_plot": d_plot,
        "log_time": log_time,
        "Xva": Xva,
    }
    # Baseline-only runs keep the configured backend in provenance without loading it.
    if not MODELS_TO_RUN.intersection(DEEP_MODEL_SPECS.keys() | HYBRID_MODEL_SPECS.keys()):
        print("Deep-learning backend skipped (baseline-only selection).")
    elif BACKEND == "flax":
        from load_forecasting.models.registry_flax import (
            FLAX_DEEP_BUILDERS,
            FLAX_HYBRID_BUILDERS,
        )
        from load_forecasting.training_deep import DeepTrainConfig
        from load_forecasting.training_flax import train_eval_flax_model
        import jax
        print("JAX devices:", jax.devices(), "| default backend:", jax.default_backend())
        RUN_PROVENANCE["runtime"]["default_backend"] = jax.default_backend()
        RUN_PROVENANCE["runtime"]["devices"] = [str(device) for device in jax.devices()]

        def flax_artifact_spec(builder_key, builder_args):
            return {
                "builder_key": builder_key,
                "builder_args": dict(builder_args or {}),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "feature_names": featureNames,
                "target_names": targetNames,
                "feature_mean": MU.loc[featureNames].to_numpy(dtype=np.float32),
                "feature_scale": SIG.loc[featureNames].to_numpy(dtype=np.float32),
                "target_mean": MU.loc[targetNames].to_numpy(dtype=np.float32),
                "target_scale": SIG.loc[targetNames].to_numpy(dtype=np.float32),
                "sampling_rate": Ts,
            }

        def flax_config(batch_size=128, epochs=40, shuffle=False, lr=1e-3, loss="mse"):
            return DeepTrainConfig(
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=lr,
                predict_batch_size=PREDICT_BATCH_SIZE,
                shuffle=shuffle,
                loss="huber" if loss == "huber" else "mse",
                seed=CLI_CONFIG.seed,
            )

        def train_deep_model(model_key, model_label, model, batch_size=128, epochs=40, shuffle=False, lr=1e-3, loss="mse", artifact_key=None, plot_key=None, log_key=None, builder_key=None, builder_args=None):
            del plot_key, log_key
            result = train_eval_flax_model(
                model_key,
                model_label,
                model,
                Xtr, Ytr, Xva, Yva, Xte, Yte,
                run_dir=RUN_DIR,
                inv_scale=inv_scale,
                y_true_phys=y_true_phys,
                d_plot=d_plot,
                log_time=log_time,
                config=flax_config(batch_size=batch_size, epochs=epochs, shuffle=shuffle, lr=lr, loss=loss),
                artifact_key=artifact_key,
                artifact_spec=flax_artifact_spec(builder_key, builder_args),
            )
            return result.yhat_phys, result.metrics

        def train_hybrid_model(model, model_name, batch_size=128, epochs=40, lr=7e-4, loss=None, shuffle=False, builder_key=None, builder_args=None):
            result = train_eval_flax_model(
                model_name,
                model_name,
                model,
                Xtr, Ytr, Xva, Yva, Xte, Yte,
                run_dir=RUN_DIR,
                inv_scale=inv_scale,
                y_true_phys=y_true_phys,
                d_plot=d_plot,
                log_time=log_time,
                config=flax_config(batch_size=batch_size, epochs=epochs, shuffle=shuffle, lr=lr, loss="huber" if loss is None else loss),
                artifact_spec=flax_artifact_spec(builder_key, builder_args),
            )
            return result.history, result.yhat_phys, result.metrics

        huber_loss = "huber"
        deep_builders = FLAX_DEEP_BUILDERS
        hybrid_builders = FLAX_HYBRID_BUILDERS
    else:
        import tensorflow as tf
        from tensorflow import keras

        RUN_PROVENANCE["runtime"]["default_backend"] = "tensorflow"
        RUN_PROVENANCE["runtime"]["devices"] = [
            device.name for device in tf.config.list_physical_devices()
        ]

        from load_forecasting.models.deep import (
            build_cnn_model,
            build_gru_model,
            build_mlp_model,
            build_rnn_model,
            build_tcn_model,
            build_transformer_forecaster,
        )
        from load_forecasting.models.hybrid import (
            build_conv_lstm_hybrid,
            build_conv_lstm_tcn_parallel,
            build_conv_tcn_parallel,
            build_tcn_lstm_hybrid,
        )
        from load_forecasting.training import train_eval_hybrid, train_eval_keras_model

        # Keras EarlyStopping has no relative min_delta; absolute floor only.
        # The Flax trainer additionally applies a 1% relative threshold.
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-6,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1 # Print message when reducing learning rate
        )

        cb = [early_stop, reduce_lr]
        keras_train_kwargs = {
            "run_dir": RUN_DIR,
            "inv_scale": inv_scale,
            "y_true_phys": y_true_phys,
            "d_plot": d_plot,
            "callbacks": cb,
            "predict_batch_size": PREDICT_BATCH_SIZE,
            "imgs_dir": IMGS_DIR,
            "save_plot": save_plot,
            "log_time": log_time,
        }
        huber_loss = keras.losses.Huber(delta=1.0)

        def train_deep_model(model_key, model_label, model, batch_size=128, epochs=40, shuffle=False, lr=1e-3, loss="mse", artifact_key=None, plot_key=None, log_key=None, builder_key=None, builder_args=None):
            del builder_key, builder_args
            return train_eval_keras_model(
                model_key, model_label, model,
                Xtr, Ytr, Xva, Yva, Xte, Yte,
                batch_size=batch_size, epochs=epochs,
                shuffle=shuffle, lr=lr, loss=loss,
                artifact_key=artifact_key, plot_key=plot_key, log_key=log_key,
                **keras_train_kwargs
            )

        def train_hybrid_model(model, model_name, batch_size=128, epochs=40, lr=7e-4, loss=None, shuffle=False, builder_key=None, builder_args=None):
            del builder_key, builder_args
            return train_eval_hybrid(
                model, model_name, Xtr, Ytr, Xva, Yva, Xte, Yte,
                batch_size=batch_size, epochs=epochs,
                lr=lr, loss=loss, shuffle=shuffle,
                **keras_train_kwargs
            )

        deep_builders = {
            "mlp": build_mlp_model,
            "gru": build_gru_model,
            "rnn": build_rnn_model,
            "cnn": build_cnn_model,
            "tcn": build_tcn_model,
            "transformer": build_transformer_forecaster,
        }
        hybrid_builders = {
            "convlstm_hybrid": build_conv_lstm_hybrid,
            "tcnlstm_hybrid": build_tcn_lstm_hybrid,
            "convtcn_hybrid": build_conv_tcn_parallel,
            "convlstmtcn_hybrid": build_conv_lstm_tcn_parallel,
        }

    if "ridge" in MODELS_TO_RUN:
        from load_forecasting.models.baselines import build_ridge_baseline

        y_ridge_phys, metrics_ridge = train_eval_flat_model(
            "ridge", "Ridge", build_ridge_baseline(random_state=CLI_CONFIG.seed),
            Xtr, Ytr, Xte, Yte, output_shape,
            **flat_train_kwargs
        )
        model_metrics["Ridge"] = metrics_ridge
        model_predictions["Ridge"] = y_ridge_phys
    else:
        print("Skipping Ridge (--models).")

    if "xgb" in MODELS_TO_RUN:
        from load_forecasting.models.baselines import build_xgb_baseline

        y_xgb_phys, metrics_xgb = train_eval_flat_model(
            "xgb", "XGBoost", build_xgb_baseline(random_state=CLI_CONFIG.seed),
            Xtr, Ytr, Xte, Yte, output_shape,
            **flat_train_kwargs
        )
        model_metrics["XGBoost"] = metrics_xgb
        model_predictions["XGBoost"] = y_xgb_phys
    else:
        print("Skipping XGBoost (--models).")

    # ====================================================
    # %% -- Configured deep models
    # ====================================================

    for model_key, spec in DEEP_MODEL_SPECS.items():
        model_label = spec["label"]
        if model_key not in MODELS_TO_RUN:
            print(f"Skipping {model_label} (--models).")
            continue
        if spec["builder"] not in deep_builders:
            print(f"Skipping {model_label} (not available in {BACKEND} backend).")
            continue

        resumed_yhat = maybe_resume_model(model_key, model_label)
        if resumed_yhat is not None:
            model_metrics[model_label] = compute_metrics(y_true_phys, resumed_yhat, d_plot, include_confidence=True)
            model_predictions[model_label] = resumed_yhat
            continue

        loss = huber_loss if spec["loss"] == "huber" else spec["loss"]
        builder_args = dict(spec.get("builder_args", {}))
        if "mlp_units" in builder_args:
            builder_args["mlp_units"] = tuple(builder_args["mlp_units"])
        if spec["builder"] == "itransformer":
            builder_args.setdefault(
                "target_indices",
                tuple(featureNames.index(t) for t in targetNames),
            )
        if spec["builder"] == "dual_stream_film":
            from load_forecasting.models.dual_stream_flax import resolve_feature_group_indices

            builder_args.update(
                resolve_feature_group_indices(
                    featureNames,
                    targetNames,
                    spec.get("feature_groups"),
                )
            )
        model = deep_builders[spec["builder"]](
            input_shape=input_shape,
            output_shape=output_shape,
            **builder_args
        )
        yhat_phys, metrics = train_deep_model(
            model_key, model_label, model,
            batch_size=cfg_or(CLI_CONFIG.batch_size, spec["batch_size"]),
            epochs=cfg_or(CLI_CONFIG.epochs, spec["epochs"]),
            shuffle=spec["shuffle"],
            lr=cfg_or(CLI_CONFIG.learning_rate, spec["learning_rate"]),
            loss=loss,
            artifact_key=spec.get("artifact_key"),
            plot_key=spec.get("plot_key"),
            log_key=spec.get("log_key"),
            builder_key=spec["builder"],
            builder_args=builder_args,
        )
        model_metrics[model_label] = metrics
        model_predictions[model_label] = yhat_phys

    # ====================================================
    # %% -- Hybrid architectures benchmark
    # ====================================================

    for model_key, spec in HYBRID_MODEL_SPECS.items():
        model_label = spec["label"]
        if model_key not in MODELS_TO_RUN:
            print(f"Skipping {model_label} (--models).")
            continue

        resumed_yhat = maybe_resume_model(spec["artifact_name"], model_label)
        if resumed_yhat is not None:
            model_metrics[model_label] = compute_metrics(y_true_phys, resumed_yhat, d_plot, include_confidence=True)
            model_predictions[model_label] = resumed_yhat
            continue

        print()
        print(f"Training {model_label}...")
        print()
        _history, yhat_phys, metrics = train_hybrid_model(
            hybrid_builders[spec["builder"]](input_shape, output_shape), spec["artifact_name"],
            batch_size=cfg_or(CLI_CONFIG.batch_size, spec["batch_size"]),
            epochs=cfg_or(CLI_CONFIG.epochs, spec["epochs"]),
            lr=cfg_or(CLI_CONFIG.learning_rate, spec["learning_rate"]),
            loss=spec["loss"],
            shuffle=spec.get("shuffle", False),
            builder_key=spec["builder"],
            builder_args={},
        )
        model_metrics[model_label] = metrics
        model_predictions[model_label] = yhat_phys

    # ====================================================
    # %% -- Save models and scalers
    # ====================================================

    stage_start = tmer()
    # Clustering = {
    #     "idx": idx_clust_all,
    #     "K": Kbest,
    #     "km": km_models,
    #     "sil": silhouette_score
    # }

    # RuleLabels = {
    #     "y": y_mode,
    #     "names": modeNames
    # }

    to_save = {
        # "Clustering": Clustering,
        # "RuleLabels": RuleLabels,
        "MU": MU, "SIG": SIG,
        "varsCont": varsCont,
        "featureNames": featureNames,
        "targetNames": targetNames,
        "Sampling rate Ts": Ts,
        "Lookback window L": LookbackSteps,
        "Horizon forecasting window H": HorizonSteps,
        "WindowSlidingCoeff": WindowSlidingCoeff,
        "models_selected": sorted(MODELS_TO_RUN),
        "epochs_override": CLI_CONFIG.epochs,
        "batch_size_override": CLI_CONFIG.batch_size,
        "learning_rate_override": CLI_CONFIG.learning_rate,
        "predict_batch_size": PREDICT_BATCH_SIZE,
        "run_summary": RUN_SUMMARY,
    }

    RUN_SUMMARY = build_run_summary(**summary_kwargs, extra={
        "provenance": RUN_PROVENANCE,
        "metrics_schema_version": 1,
        "features": featureNames,
        "targets": targetNames,
        "shapes": {
            "train_X": list(Xtr.shape),
            "train_Y": list(Ytr.shape),
            "validation_X": list(Xva.shape),
            "validation_Y": list(Yva.shape),
            "test_X": list(Xte.shape),
            "test_Y": list(Yte.shape),
        },
        "samples": {
            "train": int(len(TTtrainZ)),
            "validation": int(len(TTvalZ)),
            "test": int(len(TTtestZ)),
        },
    })
    to_save["run_summary"] = build_public_run_summary(RUN_SUMMARY)
    save_run_summary(RUN_SUMMARY, RUN_DIR)

    with open(os.path.join(RUN_DIR, "scalers_and_meta.pkl"), "wb") as f:
        pickle.dump(to_save, f)

    np.savez_compressed(
        os.path.join(RUN_DIR, "windows.npz"),
        Xtr=Xtr, Ytr=Ytr,
        Xva=Xva, Yva=Yva,
        Xte=Xte, Yte=Yte
    )
    # np.savez(
    #     "windows.npz",
    #     Xtr=Xtr, Ytr=Ytr,
    #     Xva=Xva, Yva=Yva,
    #     Xte=Xte, Yte=Yte
    # )

    print("Saved models_and_scalers.pkl and windows.npz")
    log_time("save_metadata_windows", stage_start)

    # ====================================================
    # %% -- Post-training: Results aggregation

    stage_start = tmer()
    # time_length_test = len(TTtestZ)
    time_length_test = len(TTtest)
    anchors = np.arange(LookbackSteps - 1,      # first anchor with full lookback available
                        time_length_test - HorizonSteps,   # last anchor so that t+H <= T-1
                        WindowSlidingCoeff,     # step size between windows (e.g. HorizonSteps/5 means 80% overlap)
                        # dtype=np.int64
                        )                       # array of time indices for windowing - same as make_windows

    all_metrics = model_metrics

    # DataFrame containing metrics for all models, indexed by model names.
    data_frame_results = save_metrics_outputs(all_metrics, RUN_DIR)
    print(data_frame_results[["RMSE_horizon", "MAE_horizon", "MAPE_horizon"]])
    print("Saved results_metrics.parquet and compact results_metrics.csv")
    log_time("results_aggregation", stage_start)

    # ====================================================
    # %% -- Post-training: Plots

    if PostTraining == True:
        stage_start = tmer()
        model_order = [label for label in MODEL_LABELS.values() if label in data_frame_results.index]

        plot_metric_over_horizon(data_frame_results, "RMSE_horizon", ylabel="RMSE (PU)", model_order=model_order)
        plot_name = "error_rmse"
        save_plot(IMGS_DIR, plot_name)

        plot_metric_over_horizon(data_frame_results, "MAE_horizon", ylabel="MAE (PU)", model_order=model_order)
        plot_name = "error_mae"
        save_plot(IMGS_DIR, plot_name)

        plot_metric_over_horizon( data_frame_results, "MAPE_horizon", ylabel="MAPE (%)", model_order=model_order)
        plot_name = "error_mape"
        save_plot(IMGS_DIR, plot_name)

        # Predictions on full TEST set by stitching
        y_true_full = TTtest[targetNames[d_plot]].to_numpy(dtype=np.float64)   # example: act_pow_pu

        # (num_windows, HorizonSteps, output_dim = 1 if single target)
        models_phys = model_predictions

        preds_full, boundaries = stitch_prediction_dict(
            models_phys, anchors, time_length_test, HorizonSteps, d_plot, mode="horizon_weighted"
        )

        plot_full_test(TTtest, y_true_full, preds_full, boundaries,
                title="Test set - true vs forecasts",
                downsample=WindowSlidingCoeff,
                boundary_every_seconds=None,
                seg=(0, len(TTtest)),
                zoom="auto",
                zoom_loc="auto",
                gap="auto",
            )
        plot_name = "models_comparison_full_test"
        save_plot(IMGS_DIR, plot_name)

        preds = model_predictions

        plot_scatter_grid(y_true_phys, preds, HorizonSteps, d_plot, subplot_cols=3)

        plot_name = "test_distribution_scatter"
        save_plot(IMGS_DIR, plot_name)
        log_time("post_training_plots", stage_start)
    else:
        print("Post-training plots skipped")
        print()


if __name__ == "__main__":
    main()
