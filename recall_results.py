"""
Regression using Keras on load datasets
End-to-end pipeline for timeseries analysis (.parquet databases):
load files - resample - normalize - windowing - optimize features - train models - evaluate - post processing

Prediction of active power consumption using other measured variables as features.

"""

# %% SETUP
import json
import os
from time import time as tmer
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.show(block=False)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import pickle

from tcn import TCN

from load_forecasting.config import resolve_default_config
# Reuse package helpers; adapters below preserve historical recall conventions.
from load_forecasting.data_prep import (
    add_engineered, make_datetime_from_date_time,
    preprocess_dataset as _preprocess_dataset,
)
from load_forecasting.metrics import compute_point_metrics
from load_forecasting.plots import (
    get_horizon_matrix as _get_horizon_matrix,
    plot_scatter_grid as _plot_scatter_grid,
    set_pub_style as _set_pub_style,
)
from load_forecasting.scaling import fit_train_scaler, inverse_scale_targets, transform_with_scaler
from load_forecasting.windowing import flatten_XY, make_windows as _make_windows, unflatten_Y

from tensorflow import keras

print()
print("----- Hi there! Test run at", {np.datetime64('now')}, "-----")
print()

# ====================================================
# %% -- Functions


def make_windows(df, featureNames, targetNames, LookbackSteps, HorizonSteps, stride=None, valid_col=None):
    """Use shared windowing while preserving the legacy stride and sample order."""
    # Historical recall windows cross leg boundaries; changing this invalidates alignment.
    X, Y = _make_windows(
        df, featureNames, targetNames, LookbackSteps, HorizonSteps,
        WindowSlidingCoeff=2 if stride is None else stride, valid_col=valid_col,
        group_col=None,
    )
    # The old stack-based helper returned writable arrays to interactive callers.
    if not X.flags.writeable:
        X = X.copy()
    if not Y.flags.writeable:
        Y = Y.copy()
    return X, Y

def compute_metrics(y_true, y_pred, d_plot):
    """Keep the legacy metric keys, list types, and shape error contract."""
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError(f"Expected (N,H,D). Got {y_true.shape} vs {y_pred.shape}")
    # Recall reports historically contain point estimates, without bootstrap intervals.
    metrics = compute_point_metrics(y_true, y_pred, d_plot)
    metrics["MAPE_horizon"] = metrics["MAPE_horizon"].tolist()
    return metrics

def inv_scale(z):
    """Restore targets using the legacy caller's scaling metadata."""
    return inverse_scale_targets(z, targetNames, MU, SIG)

def plot_scatter_grid(y_true, preds, k, d=0, subplot_cols=4, figsize=(14, 7)):
    """Adapt the legacy zero-based horizon index to the package plotter."""
    _plot_scatter_grid(y_true, preds, k + 1, d=d, subplot_cols=subplot_cols, figsize=figsize)

def stitch_by_stride(y_pred, anchors, T, H, stride):
    """
    Stitch windowed multi-step predictions onto a full-length timeline by copying only the first `stride` horizon steps from each window (no overlap averaging).

    Inputs
    ------
    y_pred : np.ndarray, shape (num_windows, HorizonSteps, output_dim)
        Windowed multi-step predictions.
    anchors : array-like, length num_windows
        Anchor indices t_n used for windowing.
    timeline_length : int
        Length of the full timeline to reconstruct.
    HorizonSteps : int
        Forecast horizon length (must match y_pred.shape[1]).
    stride : int
        Number of horizon steps copied from each window (often equals the window sliding step).

    Returns
    -------
    y_stitch : np.ndarray, shape (timeline_length, output_dim)
        Stitched prediction timeline (NaN where no prediction exists).
    boundaries : np.ndarray
        Start indices start_n = t_n + 1 for each window (useful for plotting boundaries).
    """
    N, H2, D = y_pred.shape
    assert H2 == H
    y_stitch = np.full((T, D), np.nan, dtype=np.float64)
    boundaries = []

    for n, t in enumerate(anchors):
        t = int(t)
        start = t + 1
        if start >= T: 
            break
        boundaries.append(start)

        m = min(stride, H, T - start) # number of steps to copy
        y_stitch[start:start+m, :] = y_pred[n, 0:m, :]

    # fill in the tail if needed
    if len(anchors) > 0:
        n_last = len(anchors) - 1
        t_last = int(anchors[n_last])
        start = t_last + 1
        for k in range(H):
            j = start + k
            if 0 <= j < T and np.isnan(y_stitch[j, 0]):
                y_stitch[j, :] = y_pred[n_last, k, :]

    return y_stitch, np.array(boundaries, dtype=np.int64)

def compress_time_gaps(t, gap="30min"):
    """
    Compress time series by removing datetime gaps larger than the specified duration.
    Add a vline in the plot to indicate where gaps were removed.
    """
    t = pd.to_datetime(t)
    t_ns = t.view("i8").copy()

    dt = np.diff(t_ns)
    gap_ns = pd.Timedelta(gap).value

    shift = np.zeros(len(t_ns), dtype=np.int64)
    total_shift = 0

    for i, delta in enumerate(dt, start=1):
        if delta > gap_ns:
            total_shift += delta - gap_ns
        shift[i] = total_shift

    t_comp_ns = t_ns - shift
    return pd.to_datetime(t_comp_ns)

def plot_full_test(TTtest, y_true_full, preds_full_dict, boundaries=None, title=None, downsample=1, boundary_every_seconds=None,
        seg=None, zoom=None, zoom_loc="lower center", gap="30min"
        ):
    """
    Prototype for plotting full test set with model predictions and window boundaries.
    
    - TTtest: DataFrame with "dtime" column
    - y_true_full: (T,) ground truth full test signal
    - preds_full_dict: dict{name: (T,) predicted}
    - boundaries: array of indices (T) where new windows start
    - downsample: 1=no downsampling, 10=1 point every 10 seconds
    - boundary_every_seconds: boundary every tot seconds
    - seg: subwindow of the test (i0, i1) for plotting and to anchor zooming
    - zoom: subwindow of the seg subwindow (z0, z1) to select smaples to be zoomed-in, starting from i0. Zooming window is hardcoded
    - zoom_loc: description of zoom window location
    - gap_mult: multiplier for gap detection in time axis (to avoid connecting points across gaps)
    """
    t_real = TTtest["dtime"].values
    t_plot = compress_time_gaps(t_real, gap).to_numpy()
    dt = np.diff(t_plot.astype("datetime64[ns]"))
    gap_td = pd.Timedelta(gap).to_numpy()
    shift_idx = np.where(dt >= gap_td)[0] + 1

    T = len(t_plot)

    i0, i1 = (0, T) if seg is None else seg
    i0 = max(0, int(i0)); i1 = min(T, int(i1))
    
    # idx = np.arange(i0, i1, downsample, dtype=np.int64)
    idx = np.arange(0, T, downsample, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t_plot[idx], y_true_full[idx], color="k", linewidth=1.2, label="True")
    ax.vlines(t_plot[shift_idx],
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        colors="red",
        linestyles="dashed",
        linewidth=1.2
    )

    for name, yhat in preds_full_dict.items():
        ax.plot(t_plot[idx], yhat[idx], linewidth=1.0, label=name)

    if boundary_every_seconds is not None and boundaries is not None:
        step = max(1, int(boundary_every_seconds / max(1, downsample)))
        for b in boundaries[::step]:
            if i0 <= b < i1:
                ax.axvline(t_plot[b], linestyle="--", linewidth=0.6, alpha=0.6)

    ax.grid(True)
    ax.legend(ncol=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Active power (PU)")
    if title: ax.set_title(title)

    # ---- zooming
    if zoom is not None:
        z0, z1 = zoom
        z0 = i0 + int(z0); z1 = i0 + int(z1)
        z0 = max(i0, z0); z1 = min(i1, z1)

        axins = inset_axes(ax, width="35%", height="35%", loc=zoom_loc)
        axins.plot(t_plot[z0:z1], y_true_full[z0:z1], color="k", linewidth=1.4)
        for name, yhat in preds_full_dict.items():
            axins.plot(t_plot[z0:z1], yhat[z0:z1], linewidth=1.0)
        axins.grid(True, lw=0.4, alpha=0.4)
        axins.tick_params(labelsize=6)
        axins.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(axins.xaxis.get_majorticklabels(), rotation=30, ha="right")
        axins.set_ylabel("PU", fontsize=6)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=0.8)

def save_plot(FolderName, plot_name):
    """ 
    Save current plot as EPS and PNG in specified folder.
    """
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "outputs", "figures", "eps", f"{plot_name}.eps"), format="eps")#, bbox_inches="tight")
    plt.savefig(os.path.join(FolderName, f"{plot_name}.png"), dpi=400, bbox_inches="tight")
    # plt.show()



def set_pub_style(font_family="Times New Roman", fontsize=9):
    """Share publication styling without changing the caller's figure resolution."""
    # The package sets DPI explicitly; the legacy helper leaves those settings alone.
    dpi = {key: mpl.rcParams[key] for key in ("figure.dpi", "savefig.dpi")}
    _set_pub_style(font_family=font_family, fontsize=fontsize)
    mpl.rcParams.update(dpi)

set_pub_style(font_family="Times New Roman", fontsize=9)

# Test for imgs.
# x = np.array(range(0, 10))
# y = np.random.rand(10,1)
# plt.scatter(x,y)
# fig = plt.gcf()

# fig.savefig("test.pdf")
# fig.savefig(os.path.join(os.getcwd(), "test.eps"), format="eps")

# ====================================================
# %% -- Dataset loading and preprocessing
# ====================================================

# Folder name for plots
FolderName = os.path.join(os.getcwd(), "outputs", "figures", "diagnostic")
os.makedirs(FolderName, exist_ok=True)

# Input files: taken from the selector lists of the JSON config.
# Override with LOAD_FORECASTING_SELECTOR (all | manoeuvre | navigation)
# and LOAD_FORECASTING_CONFIG (path to a different config file).
selector = os.environ.get("LOAD_FORECASTING_SELECTOR", "navigation")
config_path = os.environ.get(
    "LOAD_FORECASTING_CONFIG", resolve_default_config(os.path.join(os.getcwd(), "config"))
)
with open(config_path, encoding="utf-8") as handle:
    system_config = json.load(handle)
files = system_config["dataset"]["selectors"][selector]

# Trained artifacts recalled below (models, scalers, cached windows). They are
# git-ignored run outputs, so a fresh checkout has none: point
# LOAD_FORECASTING_ARTIFACT_DIR at a run directory produced by
# train_load_forecasters.py.
artifact_dir = os.environ.get("LOAD_FORECASTING_ARTIFACT_DIR", os.getcwd())


def artifact_path(name):
    """Resolve a recalled run artifact, failing with an actionable message."""
    path = os.path.join(artifact_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing run artifact: {path}. Train the models first with "
            "train_load_forecasters.py, or set LOAD_FORECASTING_ARTIFACT_DIR "
            "to an existing run directory."
        )
    return path


# Column mapping for consistency
colMap = {
    "date":      "date",                   # YYYY-MM-DD
    "time":      "time",                   # HH:MM:SS
    "act_pow":   "act_pow_kW",             # active power [kW]
    "react_pow": "react_pow_kVAR",         # reactive power [kVAr]
    "man_port":  "man_selector_port",      # 0/1
    "man_stbd":  "man_selector_starboard", # 0/1
    "spd_port":  "lever_port",             # lever
    "spd_stbd":  "lever_starboard",        # lever
    "air_temp":  "air_temp",               # °C
    "wind_int":  "wind_intensity",         # Douglas scale
    "sea_temp":  "sea_temp",               # °C
    "sea_state": "sea_state"               # Beaufort scale
}

data_types = {
        colMap["date"]:      "object",
        colMap["time"]:      "object",
        colMap["act_pow"]:   "float32",
        colMap["react_pow"]: "float32",
        colMap["man_port"]:  "Int8",
        colMap["man_stbd"]:  "Int8",
        colMap["spd_port"]:  "float32",
        colMap["spd_stbd"]:  "float32",
        colMap["air_temp"]:  "float32",
        colMap["wind_int"]:  "float32",
        colMap["sea_temp"]:  "float32",
        colMap["sea_state"]: "float32",
}

# Power base for per-unit conversion
rating_kW = system_config["physical_system"]["rating_kW"]

# ====================================================
# %% -- Parameters for windowing and models
# ====================================================
if selector == "navigation":
    numGens = 3
    TsH = 1.0                             # seconds (since Ts = 1S)
    LookbackSteps = int(round(50 / TsH))  # lookback steps
    HorizonSteps = int(round(25 / TsH))   # horizon steps
    stride = int(HorizonSteps/5)          # forecasting windows
elif selector == "manoeuvre":
    numGens = 4
    TsH = 1.0                            # seconds (since Ts = 1S)
    LookbackSteps = int(round(20 / TsH)) # lookback steps
    HorizonSteps = int(round(5 / TsH))   # horizon steps
    stride = int(HorizonSteps/5)         # forecasting windows
else:
    numGens = np.nan
    TsH = 1.0                            # seconds (since Ts = 1S)
    LookbackSteps = int(round(40 / TsH)) # lookback steps
    HorizonSteps = int(round(15 / TsH))   # horizon steps
    stride = int(HorizonSteps/5)         # forecasting windows

# ===========================================================
# %% -- Load, concatenate all legs and resample to fixed cadence
# ===========================================================
Ts = "1s"

varsPrev = ["man_port", "man_stbd", "spd_port", "spd_stbd"]  # step-hold
varsLin  = ["act_pow_pu", "react_pow_pu", "air_temp",
            "wind_intensity", "sea_temp", "sea_state"]       # linear

def preprocess_dataset(path, leg_id, colMap, data_types, Ts="1s", max_fill_seconds=None):
    """Pass legacy globals explicitly to the shared preprocessing implementation."""
    return _preprocess_dataset(
        path, leg_id, colMap, data_types, rating_kW, varsPrev, varsLin,
        Ts=Ts, max_fill_seconds=max_fill_seconds,
    )

legs_res = []
for k, fname in enumerate(files):
    path = os.path.join("data", fname)
    # append to legs
    legs_res.append(preprocess_dataset(path, k, colMap, data_types, Ts=Ts, max_fill_seconds=45))

TT = pd.concat(legs_res, ignore_index=True)

print("TT details:")
TT.info()
print()

print("Group the number of samples of refined datasets by leg_id")
print(TT.groupby("leg_id").size())
print()

# ====================================================
# %% -- Train / Validation / Test splits
# ====================================================

leg_ids = TT["leg_id"].unique()
num_legs = len(leg_ids)
num_samples = len(TT)


if num_legs > 1:
    tr_legs = leg_ids[:int(0.70*num_legs)]
    va_legs = leg_ids[int(0.70*num_legs):int(0.85*num_legs)]
    te_legs = leg_ids[int(0.85*num_legs):]

    TTtrain = TT[TT.leg_id.isin(tr_legs)].reset_index(drop=True)
    TTval   = TT[TT.leg_id.isin(va_legs)].reset_index(drop=True)
    TTtest  = TT[TT.leg_id.isin(te_legs)].reset_index(drop=True)
else:
    # Alternative: chronological split
    train_end = int(np.floor(0.7 * num_samples))
    val_end   = int(np.floor(0.85 * num_samples))

    TTtrain = TT.iloc[:train_end].reset_index(drop=True)
    TTval   = TT.iloc[train_end:val_end].reset_index(drop=True)
    TTtest  = TT.iloc[val_end:].reset_index(drop=True)

# ====================================================
# %% -- Feature engineering and normalization (z-score, train stats)
# ====================================================


TT      = add_engineered(TT)
TTtrain = add_engineered(TTtrain)
TTval   = add_engineered(TTval)
TTtest  = add_engineered(TTtest)


print("TTtrainZ stats:", TTtrain.drop('dtime', axis=1).describe())
print()

print("TTvalZ stats:", TTval.drop('dtime', axis=1).describe())
print()

print("TTtestZ stats:", TTtest.drop('dtime', axis=1).describe())
print()

varsCont    = ["act_pow_pu", "react_pow_pu",
               "air_temp", "sea_temp",
               "wind_intensity", "sea_state",
               "lever_diff", "pfactor",
               "spd_port", "spd_stbd",
               "dP",
               "moving_avg_5", "moving_avg_10", "moving_avg_20",
               "moving_std_5", "moving_std_10", "moving_std_20",
               "dQ",
               "moving_avg_5_Q", "moving_avg_10_Q", "moving_avg_20_Q",
               "moving_std_5_Q", "moving_std_10_Q", "moving_std_20_Q"]
varsBinary  = ["man_port", "man_stbd"]

# Fit on training rows only; validation and test use the same training statistics.
scaler, MU, SIG = fit_train_scaler(TTtrain, varsCont)
TTtrainZ = transform_with_scaler(TTtrain, varsCont, scaler)
print("Nan values in TTtrainZ:", np.isnan(TTtrainZ.to_numpy()).sum())
print()
TTvalZ = transform_with_scaler(TTval, varsCont, scaler)
TTtestZ = transform_with_scaler(TTtest, varsCont, scaler)

# Full-fit diagnostics are separate; MU and SIG remain train-fitted.
scaler.fit(TT[varsCont])
TTZ = transform_with_scaler(TT, varsCont, scaler)

print("TTz details:")
TTZ.info()
print()


set_pub_style(font_family="Times New Roman", fontsize=9)

pca_cols = ["act_pow_pu", "react_pow_pu",
            "air_temp", "sea_temp",
            "wind_intensity", "sea_state",
            "lever_diff", "pfactor",
            # "man_port", "man_stbd", # binary vars not used in PCA
            "spd_port", "spd_stbd",
            "dP",
            "moving_avg_5", "moving_avg_10", "moving_avg_20",
            "moving_std_5", "moving_std_10", "moving_std_20",
            "dQ",
            "moving_avg_5_Q", "moving_avg_10_Q", "moving_avg_20_Q",
            "moving_std_5_Q", "moving_std_10_Q", "moving_std_20_Q"
            ]

# ====================================================
# %% -- Windowing
# ====================================================

# -- Features and targets (feature selection should be done through PCA + KMeans)
featureNames = pca_cols
targetNames = ["act_pow_pu"]
output_dim = len(targetNames)

# dimension axis value of targetNames for plots
d_plot = 0                # 0=act, 1=react (if output_dim=2)

print("Training windows:")
Xtr, Ytr = make_windows(TTtrainZ, featureNames, targetNames, LookbackSteps, HorizonSteps, stride=stride)
print("Validation windows:")
Xva, Yva = make_windows(TTvalZ,   featureNames, targetNames, LookbackSteps, HorizonSteps, stride=stride)
print("Test windows:")
Xte, Yte = make_windows(TTtestZ,  featureNames, targetNames, LookbackSteps, HorizonSteps, stride=stride)
numFeatures = Xtr.shape[2]                 # num features
input_shape = (LookbackSteps, numFeatures) # Input shape: (lookback horizon=LookbackSteps, features=numFeatures)
output_shape = (HorizonSteps, output_dim)  # Output shape: (forecast horizon=HorizonSteps, output_dim)

print("Xtr", Xtr.shape, "Ytr", Ytr.shape)
print("Xva", Xva.shape, "Yva", Yva.shape)
print("Xte", Xte.shape, "Yte", Yte.shape)

# ====================================================
# %% -- Callbacks
# ====================================================

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    min_delta=1e-4,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

cb = [early_stop, reduce_lr]

# ====================================================
# %% -- Linear model
# ====================================================

Xtr2d, Ytr2d = flatten_XY(Xtr, Ytr)
Xva2d, Yva2d = flatten_XY(Xva, Yva)
Xte2d, Yte2d = flatten_XY(Xte, Yte)

start_time = tmer()
model_ridge = joblib.load(artifact_path("model_ridge.joblib"))

Yh_lin2d = model_ridge.predict(Xte2d)           # (N, HorizonSteps*output_dim)
Yh_lin   = unflatten_Y(Yh_lin2d, output_shape)  # (N, HorizonSteps, output_dim)

y_true_phys = inv_scale(Yte)
y_lin_phys  = inv_scale(Yh_lin)

metrics_ridge = compute_metrics(y_true_phys, y_lin_phys, d_plot)
print("Ridge RMSE (PU):", metrics_ridge["RMSE_horizon"])
print("Ridge RMSE all (PU):", metrics_ridge["RMSE_all"])
print("Ridge MAE all (PU):", metrics_ridge["MAE_all"])
print("Ridge MAPE all (%):", metrics_ridge["MAPE_all"])
print()

print("Ridge load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- XGBoost - eXtreme Gradient Boosting (multi-output)
# ====================================================

# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# xgb_base = XGBRegressor(
#     n_estimators=600,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     reg_lambda=1.0,
#     objective="reg:squarederror",
#     n_jobs=-1,
#     random_state=0
# )

# model_xgb = MultiOutputRegressor(xgb_base)

# model_xgb.fit(Xtr2d, Ytr2d)

# model_xgb.save = joblib.dump(model_xgb, "model_xgb.joblib")

start_time = tmer()
model_xgb = joblib.load(artifact_path("model_xgb.joblib"))

Yh_xgb2d = model_xgb.predict(Xte2d)             # (N, HorizonSteps*output_dim)
Yh_xgb   = unflatten_Y(Yh_xgb2d, output_shape)  # (N, HorizonSteps, output_dim)

y_xgb_phys  = inv_scale(Yh_xgb)

metrics_xgb = compute_metrics(y_true_phys, y_xgb_phys, d_plot)
print("XGBoost RMSE:", metrics_xgb["RMSE_horizon"])
print("XGBoost RMSE all (PU):", metrics_xgb["RMSE_all"])
print("XGBoost MAE all (PU):", metrics_xgb["MAE_all"])
print("XGBoost MAPE all (%):", metrics_xgb["MAPE_all"])
print()

print("XGBoost load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- Multi-layer Perceptron (MLP) model
# ====================================================

start_time = tmer()

model_mlp = keras.models.load_model("model_mlp.keras", compile=True)

mse_test_mlp = model_mlp.evaluate(Xte, Yte, batch_size=256)
print("MLP - test MSE:", mse_test_mlp)

y_mlp = model_mlp.predict(Xte, batch_size=256)   # (N, H, output_dim)
y_mlp_phys  = inv_scale(y_mlp)
metrics_mlp = compute_metrics(y_true_phys, y_mlp_phys, d_plot)
print("MLP RMSE:", metrics_mlp["RMSE_horizon"])
print("MLP RMSE all (PU):", metrics_mlp["RMSE_all"])
print("MLP MAE all (PU):", metrics_mlp["MAE_all"])
print("MLP MAPE all (%):", metrics_mlp["MAPE_all"])
print()

print("MLP load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- GRU Gated Recurrent Unit (whole horizon)
# ====================================================

start_time = tmer()

model_gru = keras.models.load_model("model_gru.keras", compile=True)

Yh_gru = model_gru.predict(Xte, batch_size=256)   # (N, HorizonSteps, output_dim)
y_gru_phys  = inv_scale(Yh_gru)

metrics_gru = compute_metrics(y_true_phys, y_gru_phys, d_plot)
print("GRU RMSE:", metrics_gru["RMSE_horizon"])
print("GRU RMSE all (PU):", metrics_gru["RMSE_all"])
print("GRU MAE all (PU):", metrics_gru["MAE_all"])
print("GRU MAPE all (%):", metrics_gru["MAPE_all"])
print()

print("GRU load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- Recurrent Neural Network (w LSTM)
# ====================================================

start_time = tmer()

model_rnn = keras.models.load_model("model_rnn.keras", compile=True)

y_rnn_phys = model_rnn.predict(Xte, batch_size=256)   # (N, HorizonSteps, output_dim)
y_rnn_phys  = inv_scale(y_rnn_phys)

metrics_rnn = compute_metrics(y_true_phys, y_rnn_phys, d_plot)
print("RNN RMSE:", metrics_rnn["RMSE_horizon"])
print("RNN RMSE all (PU):", metrics_rnn["RMSE_all"])
print("RNN MAE all (PU):", metrics_rnn["MAE_all"])
print("RNN MAPE all (%):", metrics_rnn["MAPE_all"])
print()

print("RNN load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- 1-D Convolutional Neural Network ---------- ENSEMBLE LEARNING (?)
# ====================================================

start_time = tmer()

model_cnn = keras.models.load_model("model_cnn.keras", compile=True)

y_cnn_phys = model_cnn.predict(Xte, batch_size=256)   # (N, HorizonSteps, output_dim)
y_cnn_phys  = inv_scale(y_cnn_phys)

metrics_cnn = compute_metrics(y_true_phys, y_cnn_phys, d_plot)
print("CNN RMSE:", metrics_cnn["RMSE_horizon"])
print("CNN RMSE all (PU):", metrics_cnn["RMSE_all"])
print("CNN MAE all (PU):", metrics_cnn["MAE_all"])
print("CNN MAPE all (%):", metrics_cnn["MAPE_all"])
print()

print("CNN load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- Temporal Convolutional Network
# ====================================================

start_time = tmer()

model_tcn = keras.models.load_model("model_tcn.keras", compile=True, custom_objects={"TCN": TCN})

y_tcn_phys = model_tcn.predict(Xte, batch_size=256)   # (N, HorizonSteps, output_dim)
y_tcn_phys  = inv_scale(y_tcn_phys)

metrics_tcn = compute_metrics(y_true_phys, y_tcn_phys, d_plot)
print("TCN RMSE:", metrics_tcn["RMSE_horizon"])
print("TCN RMSE all (PU):", metrics_tcn["RMSE_all"])
print("TCN MAE all (PU):", metrics_tcn["MAE_all"])
print("TCN MAPE all (%):", metrics_tcn["MAPE_all"])
print()

print("TCN load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- Transformer (encoder-only)
# ====================================================

start_time = tmer()

model_tr = keras.models.load_model("model_tr.keras", compile=True)

Yh_tr = model_tr.predict(Xte, batch_size=256)   # (N, HorizonSteps, output_dim)
y_true_phys = inv_scale(Yte)
y_tr_phys   = inv_scale(Yh_tr)

metrics_tr = compute_metrics(y_true_phys, y_tr_phys, d_plot)
print("Transformer RMSE_horizon:", metrics_tr["RMSE_horizon"])
print("Transformer RMSE all (PU):", metrics_tr["RMSE_all"])
print("Transformer MAE all (PU):", metrics_tr["MAE_all"])
print("Transformer MAPE all (%):", metrics_tr["MAPE_all"])
print()

print("Transformer load/predict/score time (s):", tmer() - start_time)
print()

# ====================================================
# %% -- Pang et al. TCN+BiLSTM (2025)
# ====================================================

# start_time = tmer()

# model_tcn_bilstm = keras.models.load_model("model_tcn_bilstm.keras", compile=True)

# y_tcn_bilstm_phys = model_tcn_bilstm.predict(Xte, batch_size=256)   # (N, HorizonSteps, output_dim)
# y_tcn_bilstm_phys  = inv_scale(y_tcn_bilstm_phys)

# metrics_tcn_bilstm = compute_metrics(y_true_phys, y_tcn_bilstm_phys, d_plot)
# print("TCN-BiLSTM RMSE:", metrics_tcn_bilstm["RMSE_horizon"])
# print("TCN-BiLSTM RMSE all (PU):", metrics_tcn_bilstm["RMSE_all"])
# print("TCN-BiLSTM MAE all (PU):", metrics_tcn_bilstm["MAE_all"])
# print("TCN-BiLSTM MAPE all (%):", metrics_tcn_bilstm["MAPE_all"])
# print()

# print("TCN-BiLSTM prediction time (s):", tmer() - start_time)
# print()

# ====================================================
# %% -- Save models and scalers
# ====================================================

to_load = {
    # "Clustering": Clustering,
    # "RuleLabels": RuleLabels,
    "MU": MU, "SIG": SIG,
    "varsCont": varsCont,
    "featureNames": featureNames,
    "targetNames": targetNames,
    "Sampling rate Ts": Ts,
    "Lookback window L": LookbackSteps,
    "Horizon forecasting window H": HorizonSteps,
    "stride": stride,
}

with open(artifact_path("scalers_and_meta.pkl"), "rb") as f:
    scalers_and_meta = pickle.load(f)

np.load(artifact_path("windows.npz"))

# np.savez_compressed(
#     "windows.npz",
#     Xtr=Xtr, Ytr=Ytr,
#     Xva=Xva, Yva=Yva,
#     Xte=Xte, Yte=Yte
# )

# ====================================================
# %% -- Post-training: Results aggregation

# Ttest = len(TTtestZ)
Ttest = len(TTtest)
anchors = np.arange(LookbackSteps - 1, Ttest - HorizonSteps, stride, dtype=np.int64) # same as make_windows

import numpy as np
all_metrics = {
    "Ridge": metrics_ridge,
    "MLP": metrics_mlp,
    "XGBoost": metrics_xgb,
    "GRU": metrics_gru,
    "RNN": metrics_rnn,
    "CNN": metrics_cnn,
    "TCN": metrics_tcn,
    "Transformer": metrics_tr,
    # "TCN-BiLSTM": metrics_tcn_bilstm
}

# DataFrame containing metrics for all models, indexed by model names
df_results = pd.DataFrame(all_metrics).T
print(df_results[["RMSE_horizon", "MAE_horizon", "MAPE_horizon"]])
df_results.to_csv("results_metrics.csv", index=True)

# ====================================================
# %% -- Post-training: Plots

def get_horizon_matrix(df_results, metric_key, model_order=None):
    """Preserve the legacy keyword name while sharing horizon validation."""
    return _get_horizon_matrix(df_results, metric_key, model_order=model_order)

def plot_metric_over_horizon(df_results, metric_key, ylabel, model_order=None, xlabel=r"Forecast horizon steps (steps)", figsize=(6.9, 3.0)):
    
    models, Y = get_horizon_matrix(df_results, metric_key, model_order=model_order)
    HorizonSteps = Y.shape[1]
    k = np.arange(HorizonSteps)

    # Linestyles/markers: also for black&white printing
    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 2)), (0, (3, 1, 1, 1))]
    markers = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, m in enumerate(models):
        ax.plot(
            k, Y[i],
            label=m,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markevery=max(1, HorizonSteps // 8)  # limit to the marker number
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, HorizonSteps - 1)
    ax.grid(True)

    plt.legend(ncol=3)
    # # Legend outside the plot
    # ax.legend(
    #     ncols=2,
    #     frameon=False,
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1.0),
    #     borderaxespad=0.0
    # )

model_order = ["Ridge", "MLP", "XGBoost", "GRU", "RNN", "CNN", "TCN", "Transformer"]#, "TCN-BiLSTM"]

plot_metric_over_horizon(df_results, "RMSE_horizon", ylabel="RMSE", model_order=model_order)
plot_name = "error_rmse"
save_plot(FolderName, plot_name)

plot_metric_over_horizon(df_results, "MAE_horizon", ylabel="MAE", model_order=model_order)
plot_name = "error_mae"
save_plot(FolderName, plot_name)

plot_metric_over_horizon( df_results, "MAPE_horizon", ylabel="MAPE (\%)", model_order=model_order)
plot_name = "error_mape"
save_plot(FolderName, plot_name)

# Predictions on full TEST set by stitching
y_true_full = TTtest[targetNames[d_plot]].to_numpy(dtype=np.float64)   # example: act_pow_pu

# (N,H,1)
models_phys = {
    # "Ridge": y_lin_phys,
    # "XGBoost": y_xgb_phys,
    "MLP": y_mlp_phys,
    "GRU": y_gru_phys,
    "RNN": y_rnn_phys,
    "CNN": y_cnn_phys,
    "TCN": y_tcn_phys,
    "Transformer": y_tr_phys,
    # "TCN-BiLSTM": y_tcn_bilstm_phys
}

preds_full = {}
boundaries = None

for name, y_phys in models_phys.items():
    yhat_full, b = stitch_by_stride(y_phys, anchors, Ttest, HorizonSteps, stride)
    preds_full[name] = yhat_full[:, d_plot]
    if b is not None:
        boundaries = b

plot_full_test(TTtest, y_true_full, preds_full, boundaries,
            title="Test set - true vs forecasts time series",
            downsample=stride,
            boundary_every_seconds=None,
            seg=(0, len(TTtest)),
            zoom=(round(len(TTtest)/4), round(len(TTtest)/3)),
            zoom_loc="lower left",
            gap="30min"
        )

plot_name = "models_comparison_full_test"
save_plot(FolderName, plot_name)

preds = {
    # "Ridge": y_lin_phys,
    # "XGBoost":   y_xgb_phys,
    "MLP":   y_mlp_phys,
    "GRU":   y_gru_phys,
    "RNN":   y_rnn_phys,
    "CNN":   y_cnn_phys,
    "TCN":   y_tcn_phys,
    "TR":    y_tr_phys,
    # "TCN-BiLSTM": y_tcn_bilstm_phys
}

plot_scatter_grid(y_true_phys, preds, k=HorizonSteps-1, d=d_plot, subplot_cols=3)

plot_name = "test_distribution_scatter"
save_plot(FolderName, plot_name)
