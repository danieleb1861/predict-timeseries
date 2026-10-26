"""
gen_inference_figures.py
========================
Standalone script that reproduces two diagnostic figures from raw parquet data and trained ConvLSTM-H model weights.

Figures produced
----------------
1. rolling_forecast_ramp.png
   Rolling multi-step forecast during a sustained rising manoeuvring ramp.
   The first anchor is placed at the ramp midpoint (half rise already visible in the 30-s context window, half still in the 15-s horizon). Seven consecutive predictions at Delta_t = 3 s illustrate the model's adaptation time as the ramp enters the lookback window.

2. single_window_inference.png
   Four representative single-window inference panels spanning the full operational load range. Model input and measured signal are derived directly from the raw parquet files (no pre-computed windows.npz).
   Panels cover: (a) mid-low transient, (b) onset of rising ramp,
   (c) steep-ramp lookback with near-steady horizon, (d) full-range excursion.

Usage
-----
    python postprocess_figures.py --kind inference

All paths are configured in the CONFIG section below.  The script requires only the trained .keras model, the scalers_and_meta.pkl file, and the raw manoeuvre parquet files listed in MAN_FILES.

Dependencies: numpy, pandas, matplotlib, keras (tensorflow backend)

Feature engineering
-------------------
The script replicates add_engineered() from train_load_forecasters.py.
Static features: act_pow_pu, react_pow_pu, spd_port, spd_stbd, air_temp,
  wind_intensity, sea_temp, sea_state, lever_diff, pfactor.
Dynamic causal features (active power): dP, moving_avg_*, moving_std_*.
Dynamic causal features (reactive power): dQ, moving_avg_*, moving_std_*.
Only the subset listed in meta["featureNames"] is fed to the model.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from load_forecasting.config import resolve_default_config  # noqa: E402

# Directory containing .keras model files, windows.npz, scalers_and_meta.pkl.
# The default names a local publication archive, which is git-ignored: point
# LOAD_FORECASTING_INFERENCE_RUN_DIR at your own archive on a fresh checkout.
RUN_DIR = Path(os.environ.get(
    "LOAD_FORECASTING_INFERENCE_RUN_DIR",
    "outputs/test_results/20260422_7Models_Manoeuvre_L45_H25",
))

# Directory containing raw parquet files
DATA_DIR = Path(r"data")

# Model file name inside RUN_DIR
MODEL_FILE = "model_convtcn_hybrid.keras"

# Output directory for generated figures (paper img directory)
OUT_DIR = Path(os.environ.get("LOAD_FORECASTING_FIGURE_DIR", "outputs/figures/paper"))

# Dataset config providing the manoeuvre file list and the per-unit base power
CONFIG_PATH = Path(os.environ.get("LOAD_FORECASTING_CONFIG", resolve_default_config("config")))
with open(CONFIG_PATH, encoding="utf-8") as _handle:
    _CONFIG = json.load(_handle)

# Manoeuvre parquet files searched for fingerprint matching
MAN_FILES = _CONFIG["dataset"]["selectors"]["manoeuvre"]

# Raw voyage parquet used for the rolling forecast measured signal.
# Defaults to the last manoeuvre leg; set LOAD_FORECASTING_ROLLING_PARQUET to the
# leg that contains the ramp event.
RAW_VOYAGE_PARQUET = DATA_DIR / os.environ.get("LOAD_FORECASTING_ROLLING_PARQUET", MAN_FILES[-1])

# Figure DPI
DPI = 200

# PU base power [kW]
BASE_KW = _CONFIG["physical_system"]["rating_kW"]

# Extra rows loaded before the lookback window to warm up rolling features.
# Must be >= max rolling window in add_engineered (currently 20).
ROLLING_CONTEXT = 20

import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import keras
from tcn import TCN

# ------------------ Style ------------------
plt.rcParams.update({
    'font.family':      'Times New Roman',
    'font.size':        9,
    'axes.titlesize':   10,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'axes.linewidth':   0.8,
    'lines.linewidth':  1.2,
    'lines.markersize': 4,
    'grid.linewidth':   0.6,
    'grid.alpha':       0.25,
    'xtick.direction':  'out',
    'ytick.direction':  'out',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    'ps.fonttype':      3,       # Type-3 → text visible in all EPS viewers
    'pdf.fonttype':     42,
    'figure.constrained_layout.use': False,
    'figure.dpi':       200,
    'savefig.dpi':      200,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.02,
})

# ------------------ Functions ------------------

class PositionalEmbedding(keras.layers.Layer):
    """Required to load the Transformer model; not used by ConvLSTM-H."""
    def __init__(self, seq_len, d_model, **kw):
        super().__init__(**kw)
        self.seq_len = seq_len
        self.d_model = d_model
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = keras.ops.convert_to_tensor(pe[None])

    def call(self, x):
        return x + self.pe[:, :keras.ops.shape(x)[1], :]

    def get_config(self):
        c = super().get_config()
        c.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return c


def load_meta_and_model(run_dir, model_file):
    """Load trusted local scaler metadata and a Keras model."""
    with open(run_dir / "scalers_and_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model = keras.models.load_model(
        run_dir / model_file, compile=False,
        custom_objects={"PositionalEmbedding": PositionalEmbedding, "TCN": TCN},
    )
    return meta, model

def validate_artifacts(run_dir, model_file, data_dir, raw_voyage_parquet, man_files):
    required = [
        run_dir / "scalers_and_meta.pkl",
        run_dir / "windows.npz",
        run_dir / model_file,
        raw_voyage_parquet,
    ]
    required.extend(data_dir / fname for fname in man_files)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inference artifacts: {missing}")

def build_feature_matrix(df_src, row_start, n_rows):
    """
    Compute all model features from a raw parquet DataFrame, mirroring the
    add_engineered() function in train_load_forecasters.py.

    Static algebraic features (always computed):
        act_pow_pu   = act_pow_kW  / BASE_KW
        react_pow_pu = react_pow_kVAR / BASE_KW
        spd_port     = lever_port
        spd_stbd     = lever_starboard
        lever_diff   = spd_port - spd_stbd
        pfactor      = act_pow_pu / sqrt(act_pow_pu^2 + react_pow_pu^2)
        air_temp, wind_intensity, sea_temp, sea_state  -> direct columns

    Causal dynamic features (add_engineered, computed over rolling context):
        dP             = first difference of act_pow_pu
        moving_avg_w   = rolling mean of act_pow_pu  (w in [5, 10, 20])
        moving_std_w   = rolling std  of act_pow_pu  (w in [5, 10, 20])
        dQ             = first difference of react_pow_pu
        moving_avg_w_Q = rolling mean of react_pow_pu (w in [5, 10, 20])
        moving_std_w_Q = rolling std  of react_pow_pu (w in [5, 10, 20])

    ROLLING_CONTEXT extra rows before row_start are loaded so that rolling
    features at the first lookback step are correctly initialised (same
    behaviour as the training preprocessing which computes features over the
    full leg before windowing).
    """
    # Load extra context rows for warm-up of rolling features
    ctx   = ROLLING_CONTEXT
    start = max(0, row_start - ctx)
    seg   = df_src.iloc[start : row_start + n_rows]
    skip  = len(seg) - n_rows   # rows to discard after feature computation

    P  = seg["act_pow_kW"].astype(float).values / BASE_KW
    Q  = seg["react_pow_kVAR"].astype(float).values / BASE_KW
    S  = np.sqrt(P**2 + Q**2)
    pf = np.where(S > 0, P / S, 0.0).astype(np.float32)
    sp = seg["lever_port"].astype(float).values
    ss = seg["lever_starboard"].astype(float).values
    ld = (sp - ss).astype(np.float32)
    at   = seg["air_temp"].astype(float).values
    wind = seg["wind_intensity"].astype(float).values
    st   = seg["sea_temp"].astype(float).values
    sst  = seg["sea_state"].astype(float).values

    # Causal dynamic features (pandas Series for rolling ops)
    P_s  = pd.Series(P)
    Q_s  = pd.Series(Q)
    dP   = P_s.diff().fillna(0.0).values.astype(np.float32)
    dQ   = Q_s.diff().fillna(0.0).values.astype(np.float32)

    col_map = {
        "act_pow_pu": P, "react_pow_pu": Q,
        "air_temp": at, "sea_temp": st, "wind_intensity": wind,
        "sea_state": sst, "lever_diff": ld, "pfactor": pf,
        "spd_port": sp, "spd_stbd": ss,
        "dP": dP, "dQ": dQ,
    }
    # Active power rolling features
    for w in [5, 10, 20]:
        col_map[f"moving_avg_{w}"] = (
            P_s.rolling(w, min_periods=1).mean().values.astype(np.float32)
        )
        col_map[f"moving_std_{w}"] = (
            P_s.rolling(w, min_periods=1).std().fillna(0.0).values.astype(np.float32)
        )
    # Reactive power rolling features
    for w in [5, 10, 20]:
        col_map[f"moving_avg_{w}_Q"] = (
            Q_s.rolling(w, min_periods=1).mean().values.astype(np.float32)
        )
        col_map[f"moving_std_{w}_Q"] = (
            Q_s.rolling(w, min_periods=1).std().fillna(0.0).values.astype(np.float32)
        )

    # Discard the warm-up rows, keep only the requested n_rows
    return {k: v[skip:] for k, v in col_map.items()}


def normalise(col_map, feat_names, MU, SIG, n_rows):
    """
    Z-score normalise and stack into (n_rows, n_features) array.
    Only features present in feat_names (from meta) are included,
    so the function works for both the legacy 10-feature set and the
    updated 18-feature set produced by add_engineered().
    """
    mat = np.zeros((n_rows, len(feat_names)), dtype=np.float32)
    for fi, fname in enumerate(feat_names):
        if fname not in col_map:
            raise KeyError(
                f"Feature '{fname}' required by the model is not computed "
                f"by build_feature_matrix(). Check add_engineered() mapping."
            )
        mat[:, fi] = (col_map[fname] - float(MU[fname])) / float(SIG[fname])
    return mat


def fingerprint_match(raw_pu, fp, fp_len=15, tol=0.02):
    """Return row index of best match of fp in raw_pu, or None if tol exceeded."""
    best_row, best_diff = None, np.inf
    for i in range(len(raw_pu) - fp_len):
        d = np.abs(raw_pu[i:i + fp_len] - fp).max()
        if d < best_diff:
            best_diff, best_row = d, i
            if d < 1e-4:
                break
    return (best_row, best_diff) if best_diff < tol else (None, best_diff)

#  Figure 1: rolling forecast

def make_rolling_forecast(x_pow, y_true, Xte, model,
                          parquet_cache, MU_P, SIG_P, L, H, STRIDE,
                          raw_voyage_parquet, out_dir):
    """
    Rolling multi-step forecast with 7 anchors during a sustained rising ramp.
    First anchor at the ramp onset: lookback flat, horizon shows the upcoming ramp.
    As k advances, the ramp enters the lookback and predictions converge.

    Window selection: K_ANCHOR=8919 (RAW_VOYAGE_PARQUET row 4543, diff=0.008).
    L=45 H=25 STRIDE=5 — windows 8919..8925 span the full ramp entry sequence.
    """
    M = 7
    # Hardcoded anchor verified against RAW_VOYAGE_PARQUET row 4543 (diff=0.008)
    K_ANCHOR = 8919
    k = K_ANCHOR

    # Run model on all M windows
    batch = Xte[k : k + M]
    preds = model.predict(batch, batch_size=M, verbose=0)[:, :, 0] * SIG_P + MU_P

    for j in range(M):
        fwd = y_true[k + j]
        pred = preds[j]
        mape = np.mean(np.abs(fwd - pred) / np.maximum(np.abs(fwd), 0.05)) * 100
        print(f"  anchor t=+{j*STRIDE:2d}s  MAPE={mape:.1f}%  "
              f"lb_net={x_pow[k+j,-1]-x_pow[k+j,0]:+.3f} PU  "
              f"fwd_net={fwd[-1]-fwd[0]:+.3f} PU")

    # Load continuous measured signal from raw parquet
    # Verified: K_ANCHOR=8919 corresponds to row 4543 in RAW_VOYAGE_PARQUET
    # (fingerprint diff=0.008, L=45, H=25, STRIDE=5)
    df_raw = pd.read_parquet(raw_voyage_parquet)
    ROW_LB = 4543
    n_raw  = L + M * STRIDE + (H - STRIDE)    # 45 + 35 + 20 = 100 samples
    raw_pu = df_raw["act_pow_kW"].iloc[ROW_LB : ROW_LB + n_raw].values.astype(float) / BASE_KW
    t_raw  = np.arange(n_raw) - L             # t=0 at first anchor's lb end

    fig, ax = plt.subplots(figsize=(8.5, 3.8))

    # Background shading: context (grey) vs horizon (blue-tint)
    ax.axvspan(t_raw[0], 0, color="#f2f2f2", zorder=0)
    ax.axvspan(0, t_raw[-1], color="#ecf3fb", zorder=0)
    ax.axvline(0, color="#888", lw=0.8, ls="--", zorder=3)

    # Measured signal (raw parquet, step plot to show 1-s discrete nature)
    ax.step(t_raw, raw_pu, where="mid", color="#1a1a1a", lw=1.2,
            label="Measured load", zorder=5)

    # Rolling predictions (light -> dark blue)
    cmap   = cm.get_cmap("Blues")
    colors = [cmap(0.35 + 0.65 * j / (M - 1)) for j in range(M)]
    for j in range(M):
        anchor_t = j * STRIDE
        conn_val  = raw_pu[j * STRIDE + L - 1]   # last measured point before anchor
        pred_conn = np.concatenate([[conn_val], preds[j]])
        t_pred    = np.arange(-1, H) + anchor_t
        fwd       = y_true[k + j]
        pred      = preds[j]
        mape      = np.mean(np.abs(fwd - pred) / np.maximum(np.abs(fwd), 0.05)) * 100
        lbl = (f"Forecast $t={anchor_t:+d}$ s  (MAPE={mape:.0f}%)"
               if j in (0, 3, 6) else None)
        ax.plot(t_pred, pred_conn, color=colors[j], lw=1.1, ls="--",
                alpha=0.85, zorder=4 + j, label=lbl)
        ax.axvline(anchor_t, color=colors[j], lw=0.5, ls=":", alpha=0.4, zorder=2)

    ax.text(-L * 0.5, raw_pu.max() * 0.999,
            f"$L = {L}$ s context", ha="center", va="top",
            fontsize=7.5, color="#555", style="italic")
    ax.text(M * STRIDE * 0.5, raw_pu.max() * 0.999,
            f"Rolling forecast ($H = {H}$ s, $\\Delta t = {STRIDE}$ s)",
            ha="center", va="top", fontsize=7.5, color="#2166ac", style="italic")

    ax.set_xlabel("Time relative to first forecast anchor (s)")
    ax.set_ylabel("Active power (PU)")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_xlim(t_raw[0] - 0.5, t_raw[-1] + 0.5)
    ax.grid(True)
    fig.tight_layout()

    out = out_dir / "rolling_forecast_ramp.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), format="eps", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved at {out}")

#  Figure 2: four-panel single-window inference

def make_four_panel(x_pow, y_true, raw_inference, FEAT, MU, SIG,
                    L, H, PWR, out_dir):
    """
    Four representative inference panels.
    Model input is constructed from raw parquet features (no windows.npz).

    Panel selection (verified fingerprint matches against the manoeuvre parquets):
      (a) win=8919  — mid-low transient, moderate lookback excursion
      (b) win=4240  — onset of rising ramp (midpoint: lb rises, fwd continues)
      (c) win=200   — steep-ramp lookback, near-steady-state horizon
      (d) win=561   — flat lookback with rising horizon (anticipation scenario)
    """
    PANELS = [8919, 4240, 200, 567]
    LABELS = ["(a)", "(b)", "(c)", "(d)"]

    results = []
    for widx in PANELS:
        print(f"  win={widx} ...", end=" ", flush=True)
        lb_pu, fwd_pu, pred_pu, mape, fname, row_lb = raw_inference(widx)
        mean_pow = np.concatenate([lb_pu, fwd_pu]).mean()
        results.append({
            "win": widx, "fname": fname,
            "lb_pu": lb_pu, "fwd_pu": fwd_pu, "pred_pu": pred_pu,
            "mape": mape, "mean_pow": mean_pow,
        })
        print(f"MAPE={mape:.1f}%  mean={mean_pow:.3f} PU  ({fname})")

    # Global y-axis (same for all panels)
    all_vals = np.concatenate([
        np.concatenate([r["lb_pu"], r["fwd_pu"], r["pred_pu"]]) for r in results
    ])
    pad  = 0.08 * (all_vals.max() - all_vals.min())
    y_lo = all_vals.min() - pad
    y_hi = all_vals.max() + pad
    print(f"  Global y-axis: [{y_lo:.3f}, {y_hi:.3f}] PU")

    t_lb        = np.arange(-L, 0)       # -L .. -1
    t_fwd       = np.arange(0, H)        # 0 .. H-1
    t_pred_conn = np.arange(-1, H)       # -1 .. H-1  (lb[-1] prepended)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True)
    axes = axes.flatten()

    for ax, r, lbl in zip(axes, results, LABELS):
        ax.axvspan(-L, 0, color="#f6f6f6", zorder=0)
        ax.axvspan(0,  H, color="#ecf3fb", zorder=0)
        ax.axvline(0, color="#888", lw=0.8, ls=":", zorder=2)

        # Measured lookback (raw)
        ax.step(t_lb, r["lb_pu"], where="mid",
                color="#1a1a1a", lw=1.1, zorder=4, label="Measured load")
        # True horizon (dashed grey)
        ax.step(t_fwd, r["fwd_pu"], where="mid",
                color="#555", lw=0.9, ls="--", alpha=0.7, zorder=3,
                label="True horizon")
        # Prediction connected at t=-1 to t=H-1
        pred_conn = np.concatenate([[r["lb_pu"][-1]], r["pred_pu"]])
        ax.plot(t_pred_conn, pred_conn, color="#2166ac", lw=1.3, zorder=5,
                label=f"Prediction (MAPE = {r['mape']:.1f} %)")

        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-L - 0.5, H + 0.5)
        ax.set_title(
            f"{lbl}  $\\bar{{P}}$ = {r['mean_pow']:.2f} PU  |  MAPE = {r['mape']:.1f} %",
            fontsize=8.5,
        )
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left", framealpha=0.85, fontsize=7)
        ax.grid(True)

    # Shared y-label
    fig.text(0.02, 0.5, "Active power (PU)", va="center",
             rotation="vertical", fontsize=9)

    # Context / forecast labels on top row panels
    for ax in axes[:2]:
        ax.text(-L * 0.5, y_hi * 0.998, f"$L = {L}$ s lookback",
                ha="center", va="top", fontsize=7, color="#555", style="italic")
        ax.text(H * 0.5,  y_hi * 0.998, f"$H = {H}$ s forecast",
                ha="center", va="top", fontsize=7, color="#2166ac", style="italic")

    plt.tight_layout(rect=[0.04, 0, 1, 1])

    out = out_dir / "single_window_inference.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), format="eps", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved at {out}")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "lines.linewidth": 1.1, "axes.linewidth": 0.8,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_artifacts(RUN_DIR, MODEL_FILE, DATA_DIR, RAW_VOYAGE_PARQUET, MAN_FILES)

    print("Loading metadata and model ...")
    meta, model = load_meta_and_model(RUN_DIR, MODEL_FILE)
    MU   = meta["MU"]
    SIG  = meta["SIG"]
    FEAT = list(meta["featureNames"])
    L    = int(meta["Lookback window L"])
    H    = int(meta["Horizon forecasting window H"])
    STRIDE = int(meta["WindowSlidingCoeff"])
    PWR  = FEAT.index("act_pow_pu")
    MU_P = float(MU["act_pow_pu"])
    SIG_P = float(SIG["act_pow_pu"])

    # Load windows.npz for fingerprint reference only (x_pow in PU)
    print("Loading windows.npz (for fingerprint matching) ...")
    npz   = np.load(RUN_DIR / "windows.npz", allow_pickle=True)
    Xte   = npz["Xte"].astype(np.float32)
    Yte   = npz["Yte"].astype(np.float32)
    x_pow = Xte[:, :, PWR] * SIG_P + MU_P   # (N, L) power in PU
    y_true = Yte[:, :, 0]  * SIG_P + MU_P   # (N, H) power in PU

    # Pre-load all manoeuvre parquets for fingerprint search
    print("Pre-loading parquet files ...")
    parquet_cache = {}
    for fname in MAN_FILES:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df["ts"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str)
            )
            parquet_cache[fname] = df

    def find_source(win_idx):
        """Locate (fname, row_lb) of window win_idx in the parquet cache."""
        fp = x_pow[win_idx, :15]
        best = (None, None, np.inf)
        for fname, df in parquet_cache.items():
            raw = df["act_pow_kW"].values.astype(float) / BASE_KW
            row, diff = fingerprint_match(raw, fp)
            if row is not None and diff < best[2]:
                best = (fname, row, diff)
                if diff < 1e-4:
                    break
        return best   # (fname, row_lb, diff)

    def raw_inference(win_idx):
        """
        Run model on raw parquet features for one window.
        Returns (lb_pu, fwd_pu, pred_pu, mape, fname, row_lb).
        """
        fname, row_lb, diff = find_source(win_idx)
        if fname is None:
            raise RuntimeError(f"Window {win_idx}: source parquet not found")
        df_src = parquet_cache[fname]
        col_lb  = build_feature_matrix(df_src, row_lb,     L)
        col_fwd = build_feature_matrix(df_src, row_lb + L, H)
        mat_lb  = normalise(col_lb,  FEAT, MU, SIG, L)   # (L, 10)
        x_in    = mat_lb[None]                             # (1, L, 10)
        y_pred_norm = model.predict(x_in, verbose=0)       # (1, H, 1)
        pred_pu = y_pred_norm[0, :, 0] * SIG_P + MU_P
        lb_pu   = col_lb["act_pow_pu"]
        fwd_pu  = col_fwd["act_pow_pu"]
        mape    = (np.abs(fwd_pu - pred_pu) /
                   np.maximum(np.abs(fwd_pu), 0.05)).mean() * 100
        return lb_pu, fwd_pu, pred_pu, mape, fname, row_lb

    # Figure 1: rolling forecast during rising ramp
    print("\n[Fig 1] Rolling forecast during a rising manoeuvring ramp ...")
    make_rolling_forecast(
        x_pow, y_true, Xte, model, parquet_cache,
        MU_P, SIG_P, L, H, STRIDE,
        RAW_VOYAGE_PARQUET, OUT_DIR,
    )

    # Figure 2: four-panel single-window inference
    print("\n[Fig 2] Four-panel single-window inference ...")
    make_four_panel(
        x_pow, y_true, raw_inference, FEAT, MU, SIG,
        L, H, PWR, OUT_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
