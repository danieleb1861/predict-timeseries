"""
User-facing dispatcher for optional figure post-processing.

Examples:
    python postprocess_figures.py --kind figures_4papers
    python postprocess_figures.py --kind inference
    python postprocess_figures.py --kind foundation_full_test
    python postprocess_figures.py --kind data_characterisation [--data-proposals]
    python postprocess_figures.py --kind all
"""

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats


ROOT = Path(__file__).resolve().parent

POSTPROCESS_SCRIPTS = {
    "figures_4papers": ROOT / "scripts" / "gen_figures_4papers.py",
    "foundation_full_test": ROOT / "scripts" / "gen_foundation_full_test_figures.py",
    "inference": ROOT / "scripts" / "gen_inference_figures.py",
}


def parse_args(argv=None):
    """
    Parse command-line arguments for the post-processing launcher.

    Inputs
    ----------
    argv : list[str] or None
        Optional list of command-line arguments. If None, arguments are read
        from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, including the selected post-processing workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run optional post-processing figure workflows."
    )
    parser.add_argument(
        "--kind",
        choices=("figures_4papers", "foundation_full_test", "inference", "data_characterisation", "all"),
        required=True,
        help="Which figure workflow to run.",
    )
    parser.add_argument(
        "--foundation-archive-tag",
        default="flax8_legsafe",
        help="Archive suffix used by --kind foundation_full_test (default: flax8_legsafe).",
    )
    parser.add_argument(
        "--foundation-lookback",
        type=int,
        default=45,
        help="Foundation full-test archive lookback (default: 45).",
    )
    parser.add_argument(
        "--foundation-horizon",
        type=int,
        default=25,
        help="Foundation full-test archive horizon (default: 25).",
    )
    parser.add_argument(
        "--data-figures-dir",
        default=str(ROOT / "outputs" / "figures" / "data_characterisation"),
        help="Output folder of --kind data_characterisation (EPS panels, legs cache).",
    )
    parser.add_argument(
        "--data-proposals",
        action="store_true",
        help="Also draw the unused proposal figures (context, traces, campaign, leadlag) as PDF.",
    )
    return parser.parse_args(argv)


def run_script(script_path, extra_args=()):
    """
    Execute a single post-processing script.

    Inputs
    ----------
    script_path : pathlib.Path
        Path to the Python script to execute.

    Returns
    -------
    None
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Post-processing script not found: {script_path}")

    print(f"[postprocess] running {script_path.name}")
    subprocess.run(
        [sys.executable, str(script_path), *extra_args], cwd=str(ROOT), check=True
    )


def discover_foundation_archives(tag, lookback, horizon):
    """Return the matching manoeuvre and navigation publication archives."""
    results_dir = ROOT / "outputs" / "test_results"
    archives = {}
    for mode in ("manoeuvre", "navigation"):
        pattern = f"*_{mode.title()}_L{lookback}_H{horizon}_{tag}"
        matches = sorted(path for path in results_dir.glob(pattern) if path.is_dir())
        if not matches:
            raise FileNotFoundError(
                f"No {mode} archive matching {pattern!r} under {results_dir}"
            )
        archives[mode] = matches[-1]
    return archives


# ---------------------------------------------------------------------------
# Data characterisation figures (BBF paper): one EPS per panel, no titles or
# panel letters (subfigures and captions are set in LaTeX). EPS has no
# transparency, so translucent fills are composited over white.
# ---------------------------------------------------------------------------
SPLIT_C = {"train": "#9ecae1", "val": "#fdae6b", "test": "#d6604d"}
SPLIT_L = {"train": "Train", "val": "Validation", "test": "Test"}
MODE_C = {"manoeuvre": "#b2182b", "navigation": "#2166ac"}
MODE_L = {"manoeuvre": "Manoeuvre", "navigation": "Navigation"}
MODE_SUF = {"manoeuvre": "man", "navigation": "nav"}
Q95 = {"manoeuvre": 2.007, "navigation": 1.568}  # training p95, Table tab:split_stats
Y_COL = "act_pow_pu"
DATA_STYLE = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
              "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
              "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
              "mathtext.fontset": "stix"}


def build_legs(cache):
    """Rebuild the 1 s PU series of every leg with the training preprocessing (read-only on raw data)."""
    if cache.exists():
        return pd.read_parquet(cache)
    from load_forecasting.data_prep import preprocess_dataset, make_datetime_from_date_time
    from load_forecasting.splitting import split_train_val_test

    cfg = json.load(open(ROOT / "config" / "my_system_config.json"))
    ds, ph = cfg["dataset"], cfg["physical_system"]
    voyages = ds["voyage_pseudonyms"]  # private map from file stem to V1..V8
    col_map = ds["columns"]
    dtypes = {col_map[k]: v for k, v in ds["dtypes"].items()}
    frames = []
    for mode in ("manoeuvre", "navigation"):
        legs = []
        for k, f in enumerate(ds["selectors"][mode]):
            path = ROOT / "data" / f
            with contextlib.redirect_stdout(io.StringIO()):
                d = preprocess_dataset(str(path), k, col_map, dtypes, ph["rating_kW"], ph["signals"]["step_hold"],
                                       ph["signals"]["resample_continuous"], ds["sampling_rate"], ds["max_fill_seconds"])
            raw = pd.read_parquet(path, columns=[col_map["date"], col_map["time"]])
            obs = pd.DatetimeIndex(make_datetime_from_date_time(raw[col_map["date"]], raw[col_map["time"]])).floor("s")
            d["observed"] = d["dtime"].isin(obs)
            stem = f.replace(".parquet", "")
            base = stem.rsplit("_", 1)[0] if stem.endswith(("_1st", "_2nd", "_NAV")) else stem
            d["voyage"] = voyages[base]
            d["leg_name"] = voyages[base] +{"_1st": "a", "_2nd": "b"}.get(stem[-4:], "")
            legs.append(d)
        tt = pd.concat(legs, ignore_index=True)
        tr, va, te = split_train_val_test(tt, mode="balanced" if mode == "navigation" else "ordered")
        split = {**{l: "train" for l in tr.leg_id.unique()}, **{l: "val" for l in va.leg_id.unique()},
                 **{l: "test" for l in te.leg_id.unique()}}
        tt["split"] = tt["leg_id"].map(split)
        tt["mode"] = mode
        frames.append(tt)
    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def _blend(c, a):
    return tuple(a * np.array(to_rgb(c)) + (1 - a))


def _composite(ax, x, bands, zorder=0):
    """Exact alpha-over-white compositing of translucent bands [(lo, hi, colour, alpha)] on a shared x grid."""
    x = np.asarray(x)
    lo = np.array([np.broadcast_to(b[0], x.shape) for b in bands], float)
    hi = np.array([np.broadcast_to(b[1], x.shape) for b in bands], float)
    lo, hi = 0.5 * (lo[:, 1:] + lo[:, :-1]), 0.5 * (hi[:, 1:] + hi[:, :-1])
    verts, cols = [], []
    for i in range(len(x) - 1):
        edges = np.unique(np.r_[lo[:, i], hi[:, i]])
        for a, b in zip(edges[:-1], edges[1:]):
            on = [k for k in range(len(bands)) if lo[k, i] <= 0.5 * (a + b) <= hi[k, i]]
            if not on:
                continue
            c = np.ones(3)
            for k in on:
                c = bands[k][3] * np.array(to_rgb(bands[k][2])) + (1 - bands[k][3]) * c
            verts.append([(x[i], a), (x[i + 1], a), (x[i + 1], b), (x[i], b)])
            cols.append(c)
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors=cols, linewidths=0.25, zorder=zorder))


def _leg_order(d):
    return sorted(d.leg_name.unique(), key=lambda s: (int(s[1]), s[2:]))


def _acf(v, nlag=300):
    v = v - v.mean()
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    return np.fft.irfft(f * np.conj(f))[: nlag + 1] / (v.var() * np.arange(n, n - nlag - 1, -1))


def _save(fig, out, name, fmt="eps"):
    fig.savefig(out / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def fig_ridge(df, out, mode):
    """Per-leg kernel density of active power, coloured by split (ridge_man/nav)."""
    fig, ax = plt.subplots(figsize=(3.4, 4.0))
    d = df[df["mode"] == mode]
    legs = _leg_order(d)
    grid = np.linspace(0.7, 2.8, 500) if mode == "manoeuvre" else np.linspace(0.8, 1.9, 500)
    for i, leg in enumerate(legs[::-1]):
        v = d.loc[d.leg_name == leg, Y_COL].dropna().to_numpy()
        sp = d.loc[d.leg_name == leg, "split"].iloc[0]
        k = stats.gaussian_kde(v[:: max(1, len(v) // 20000)], bw_method=0.25)(grid)
        k = k / k.max() * 1.6
        ax.fill_between(grid, i, i + k, color=_blend(SPLIT_C[sp], 0.9), lw=0, zorder=len(legs) - i)
        ax.plot(grid, i + k, color="#333333", lw=0.5, zorder=len(legs) - i)
        ax.plot([np.median(v)] * 2, [i, i + 0.25], color="#1a1a1a", lw=0.8, zorder=99)
        ax.text(grid[0] - 0.02, i + 0.15, leg, ha="right", va="bottom", fontsize=7)
    ax.axvline(Q95[mode], color="#1a1a1a", ls="--", lw=0.7, zorder=100)
    ax.text(Q95[mode], len(legs) + 1.0, r"train $q_{95}$", ha="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.2, len(legs) + 1.5)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Active power (PU)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_blend(c, 0.9)) for c in SPLIT_C.values()]
    ax.legend(handles, SPLIT_L.values(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    _save(fig, out, f"ridge_{MODE_SUF[mode]}")


def fig_split(df, out, mode):
    """Kernel density of active power per split, training q95 dashed (split_man/nav)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    d = df[df["mode"] == mode]
    curves = []
    for sp in ("train", "val", "test"):
        v = d.loc[d.split == sp, Y_COL].dropna().to_numpy()
        sns.kdeplot(v[:: max(1, len(v) // 50000)], ax=ax, color=SPLIT_C[sp], lw=1.0, bw_adjust=0.8, label=SPLIT_L[sp])
        curves.append((ax.get_lines()[-1].get_data(), SPLIT_C[sp]))
    xs = np.linspace(min(c[0][0].min() for c in curves), max(c[0][0].max() for c in curves), 800)
    _composite(ax, xs, [(0.0, np.interp(xs, *xy, left=0, right=0), c, 0.25) for xy, c in curves])
    ax.axvline(Q95[mode], color="#1a1a1a", ls="--", lw=0.7)
    ax.set_xlabel("Active power (PU)")
    ax.set_ylabel("Density")
    ax.set_ylim(top=1.2 * ax.get_ylim()[1])
    sns.despine(ax=ax)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", columnspacing=1.0)
    _save(fig, out, f"split_{MODE_SUF[mode]}")


def fig_dynamics(df, out):
    """Autocorrelation, Welch spectrum and ramp-rate exceedance per leg, medians in bold (acf, psd, ramp)."""
    figs = {n: plt.subplots(figsize=(2.33, 2.3)) for n in ("acf", "psd", "ramp")}
    a0, a1, a2 = (figs[n][1] for n in ("acf", "psd", "ramp"))
    lags, fq, grid = np.arange(301), np.linspace(0, 0.5, 400), np.geomspace(1e-4, 1.0, 200)
    bands = []
    for mode in ("manoeuvre", "navigation"):
        d = df[df["mode"] == mode]
        c = MODE_C[mode]
        R, P, C, q = [], [], [], []
        for leg in _leg_order(d):
            v = d.loc[d.leg_name == leg, Y_COL].dropna().to_numpy().astype(float)
            r = _acf(v)
            R.append(r)
            a0.plot(lags, r, color=_blend(c, 0.35), lw=0.4)
            f, p = signal.welch(v, fs=1.0, window="hann", nperseg=min(2048, len(v)), detrend="constant")
            P.append(np.interp(fq, f, p))
            ramp = np.sort(np.abs(np.diff(v)))
            ccdf = 1 - np.searchsorted(ramp, grid, side="right") / len(ramp)
            C.append(ccdf)
            q.append(np.quantile(ramp, 0.95))
            a2.plot(grid[ccdf > 0], ccdf[ccdf > 0], color=_blend(c, 0.35), lw=0.4)
        a0.plot(lags, np.median(R, axis=0), color=c, lw=1.6, label=MODE_L[mode])
        P = np.array(P)
        bands.append((np.quantile(P, 0.25, axis=0)[1:], np.quantile(P, 0.75, axis=0)[1:], c, 0.2))
        a1.plot(fq[1:], np.median(P, axis=0)[1:], color=c, lw=1.2, label=MODE_L[mode], zorder=3)
        med = np.median(C, axis=0)
        a2.plot(grid[med > 0], med[med > 0], color=c, lw=1.6, label=MODE_L[mode])
        qm = np.median(q)  # median over legs of the per-leg 95th percentile, as in the text
        left = mode == "navigation"  # labels apart: navigation lies left of manoeuvre
        a2.plot([qm], [0.05], "o", ms=3.5, color=c, mec="white", mew=0.5, zorder=5)
        a2.annotate(f"{qm:.3f}", (qm, 0.05), xytext=(-3 if left else 3, 4), textcoords="offset points",
                    fontsize=6.5, color=c, ha="right" if left else "left")
    _composite(a1, fq[1:], bands)
    a0.axhline(0.5, color="#555555", ls="--", lw=0.6)
    a0.set(xlabel="Lag (s)", ylabel="Autocorrelation", xlim=(0, 300))
    a1.set(yscale="log", xscale="log", xlabel="Frequency (Hz)", ylabel=r"PSD (PU$^2$/Hz)")
    a1.axvline(0.05, color="#555555", ls="--", lw=0.6)
    a2.axhline(0.05, color="#555555", ls=":", lw=0.6)
    a2.set(xscale="log", yscale="log", xlabel=r"Ramp rate $|y_{t+1}-y_t|/\Delta t$ (PU/s)", ylabel="Exceedance probability",
           xlim=(1e-4, 1))
    a0.legend(frameon=False)
    for name, (fig, ax) in figs.items():
        sns.despine(ax=ax)
        _save(fig, out, name)


def fig_lead(df, out):
    """Load change over the lead h, i.e. the persistence error (lead_man/nav, lead_quant)."""
    leads, bins, rows = (1, 5, 15, 25), np.linspace(-0.8, 0.8, 161), []
    for mode in ("manoeuvre", "navigation"):
        fig, ax = plt.subplots(figsize=(2.33, 2.3))
        d = df[df["mode"] == mode]
        pal = sns.light_palette(MODE_C[mode], n_colors=len(leads) + 1)[1:]
        for h, col in zip(leads, pal):
            dv = []
            for leg in _leg_order(d):
                v = d.loc[d.leg_name == leg, Y_COL].to_numpy().astype(float)
                x = v[h:] - v[:-h]
                dv.append(x[np.isfinite(x)])
            dv = np.concatenate(dv)
            hist, e = np.histogram(dv, bins=bins, density=True)
            ax.step(0.5 * (e[1:] + e[:-1]), np.where(hist > 0, hist, np.nan), where="mid", color=col, lw=1.0,
                    label=f"$h$ = {h} s")
            a = np.abs(dv)
            rows.append({"mode": mode, "h": h, "q50": np.quantile(a, 0.5), "q95": np.quantile(a, 0.95),
                         "q99": np.quantile(a, 0.99)})
        ax.set(yscale="log", xlabel=r"$y_{t+h}-y_t$ (PU)", ylabel="Density", xlim=(-0.8, 0.8), ylim=(1e-3, 3e3))
        ax.legend(frameon=False, fontsize=6.3)
        sns.despine(ax=ax)
        _save(fig, out, f"lead_{MODE_SUF[mode]}")
    s = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(2.33, 2.3))
    hh = np.linspace(leads[0], leads[-1], 300)  # log-linear upsampling, as fill_between draws on the log axis
    up = lambda g, k: np.exp(np.interp(hh, g.h, np.log(g[k])))  # noqa: E731
    _composite(ax, hh, [(up(g, "q50"), up(g, "q99"), MODE_C[m], 0.12) for m, g in s.groupby("mode")])
    for mode, g in s.groupby("mode"):
        ax.plot(g.h, g.q95, "-o", ms=3, color=MODE_C[mode], label=f"{MODE_L[mode]}, $q_{{95}}$")
        ax.plot(g.h, g.q50, "--", color=MODE_C[mode], lw=0.8)
    ax.set(yscale="log", xlabel="Lead $h$ (s)", ylabel=r"$|y_{t+h}-y_t|$ (PU)")
    ax.legend(frameon=False, fontsize=6.3)
    sns.despine(ax=ax)
    _save(fig, out, "lead_quant")
    s.to_csv(out / "lead_quantiles.csv", index=False)


def fig_proposals(df, out):
    """Unused proposal figures, multi-panel PDF: context, traces, campaign, leadlag."""
    out.mkdir(parents=True, exist_ok=True)
    d = df.assign(lever=0.5 * (df.spd_port + df.spd_stbd))
    # propulsion context
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, mode in zip(axes[:2], ("manoeuvre", "navigation")):
        s = d[d["mode"] == mode].dropna(subset=["lever", Y_COL])
        ax.hexbin(s.lever, s[Y_COL], gridsize=45, bins="log", cmap="Blues" if mode == "navigation" else "Reds",
                  mincnt=1, linewidths=0)
        ax.set(xlabel="Mean lever position (-)", ylabel="Active power (PU)")
        ax.set_title(f"({'ab'[mode == 'navigation']}) {MODE_L[mode]}", loc="left")
        rho = stats.spearmanr(s.lever[::10], s[Y_COL][::10]).statistic
        ax.text(0.03, 0.95, rf"Spearman $\rho$ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=7)
        sns.despine(ax=ax)
    s = d[d["mode"] == "navigation"].dropna(subset=["sea_state", Y_COL])
    s = s.assign(SS=s.sea_state.round().astype(int))
    sns.boxenplot(data=s.iloc[::20], x="SS", y=Y_COL, color=MODE_C["navigation"], ax=axes[2], linewidth=0.5,
                  showfliers=False, width=0.7)
    axes[2].set(xlabel="Sea state (Douglas)", ylabel="Active power (PU)")
    axes[2].set_title("(c) Navigation vs sea state", loc="left")
    sns.despine(ax=axes[2])
    fig.tight_layout()
    _save(fig, out, "context", "pdf")
    # example traces
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 0.01], hspace=0.6, wspace=0.25)
    nav = df[(df["mode"] == "navigation") & (df.leg_name == "V4")]
    man = df[(df["mode"] == "manoeuvre") & (df.leg_name == "V4a")]
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(nav.t_seconds / 3600, nav[Y_COL], color=MODE_C["navigation"], lw=0.3)
    ax.set(xlabel="Time (h)", ylabel="P (PU)")
    ax.set_title("(a) Navigation leg V4 (test)", loc="left")
    sns.despine(ax=ax)
    ax = fig.add_subplot(gs[1, 0])
    t, v = man.t_seconds / 60, man[Y_COL].to_numpy()
    ax.plot(t, v, color=MODE_C["manoeuvre"], lw=0.6)
    pk, _ = signal.find_peaks(np.nan_to_num(v), prominence=0.05, distance=10)
    ax.plot(t.iloc[pk], v[pk], "v", ms=2.5, color="#1a1a1a", label="peaks (prominence 0.05 PU)")
    dv = np.abs(np.r_[0, np.diff(v)])
    ax.fill_between(t, v.min(), v.max(), where=dv > np.nanquantile(dv, 0.95), color="#fdae6b", alpha=0.5, lw=0,
                    label=r"$|\Delta P|>q_{95}$")
    ax.axhline(Q95["manoeuvre"], color="#1a1a1a", ls="--", lw=0.6)
    ax.set(xlabel="Time (min)", ylabel="P (PU)")
    ax.set_title("(b) Manoeuvre leg V4a (train)", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    sns.despine(ax=ax)
    ax = fig.add_subplot(gs[1, 1])
    for mode in ("manoeuvre", "navigation"):
        sns.ecdfplot(df.loc[df["mode"] == mode, Y_COL].dropna().to_numpy(), ax=ax, color=MODE_C[mode],
                     label=MODE_L[mode])
    ax.set(xlabel="P (PU)", ylabel="ECDF")
    ax.set_title("(c) Pooled distribution", loc="left")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    _save(fig, out, "traces", "pdf")
    # campaign overview
    g = df.groupby(["mode", "leg_name", "voyage", "split"]).agg(
        start=("dtime", "min"), hours=("dtime", "size"), obs=("observed", "mean")).reset_index()
    g["hours"] /= 3600
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    vo = [f"V{i}" for i in range(1, 9)]
    for _, r in g.iterrows():
        nav_leg = r["mode"] == "navigation"
        axes[0].barh(vo.index(r.voyage) + (0.22 if nav_leg else -0.22), r.hours if nav_leg else r.hours * 10,
                     left=0, height=0.38, color=SPLIT_C[r.split], edgecolor=MODE_C[r["mode"]], lw=0.6)
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels(vo)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Duration on the 1 s grid (h; manoeuvre legs x10)")
    axes[0].set_title("(a) Legs per voyage and split", loc="left")
    sns.despine(ax=axes[0])
    for mode in ("manoeuvre", "navigation"):
        s = g[g["mode"] == mode]
        axes[1].scatter(s.hours, 100 * s.obs, color=[SPLIT_C[x] for x in s.split], edgecolor=MODE_C[mode], lw=0.8,
                        s=22, marker="o" if mode == "navigation" else "s", label=MODE_L[mode])
    axes[1].set(xscale="log", xlabel="Leg duration (h)", ylabel="Grid points with a record (%)", ylim=(0, 100))
    axes[1].set_title("(b) Sampling density", loc="left")
    axes[1].legend(frameon=False)
    sns.despine(ax=axes[1])
    fig.tight_layout()
    _save(fig, out, "campaign", "pdf")
    g.to_csv(out / "campaign_legs.csv", index=False)
    # lever-load lead-lag
    K = np.arange(-30, 61)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for mode in ("manoeuvre", "navigation"):
        dm, curves = d[d["mode"] == mode], []
        for leg in _leg_order(dm):
            s = dm[dm.leg_name == leg]
            u, p = s.lever.diff().to_numpy(), s[Y_COL].diff().to_numpy()
            ok = np.isfinite(u) & np.isfinite(p)
            u, p = np.where(ok, u, 0.0), np.where(ok, p, 0.0)
            if u.std() == 0:
                continue
            u, p = (u - u.mean()) / u.std(), (p - p.mean()) / p.std()
            n = len(u)
            c = [np.mean(u[max(0, -k): n - max(0, k)] * p[max(0, k): n - max(0, -k)]) for k in K]
            curves.append(c)
            ax.plot(K, c, color=MODE_C[mode], lw=0.3, alpha=0.3)
        ax.plot(K, np.median(curves, axis=0), color=MODE_C[mode], lw=1.5, label=MODE_L[mode])
    ax.axvline(0, color="#555555", lw=0.6, ls="--")
    ax.set(xlabel=r"Lag $k$ (s): corr$(\Delta u_t,\ \Delta P_{t+k})$", ylabel="Cross-correlation")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out, "leadlag", "pdf")


def run_data_characterisation(out_dir, proposals=False):
    """Draw the data characterisation panels of the BBF paper into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[postprocess] data characterisation -> {out}")
    df = build_legs(out / "legs.parquet")
    with plt.rc_context(DATA_STYLE):
        for mode in ("manoeuvre", "navigation"):
            fig_ridge(df, out, mode)
            fig_split(df, out, mode)
        fig_dynamics(df, out)
        fig_lead(df, out)
        if proposals:
            fig_proposals(df, out / "proposals")


def main(argv=None):
    """
    Run the selected post-processing workflow.

    Inputs
    ----------
    argv : list[str] or None
        Optional list of command-line arguments. If None, arguments are read
        from sys.argv.

    Returns
    -------
    None
    """
    args = parse_args(argv)

    if args.kind == "foundation_full_test":
        archives = discover_foundation_archives(
            args.foundation_archive_tag,
            args.foundation_lookback,
            args.foundation_horizon,
        )
        script_path = POSTPROCESS_SCRIPTS[args.kind]
        for mode, archive in archives.items():
            run_script(
                script_path,
                ("--mode", mode, "--archive", str(archive)),
            )
        return

    if args.kind == "data_characterisation":
        run_data_characterisation(args.data_figures_dir, args.data_proposals)
        return

    kinds = (
        POSTPROCESS_SCRIPTS
        if args.kind == "all"
        else {args.kind: POSTPROCESS_SCRIPTS[args.kind]}
    )

    for script_path in kinds.values():
        run_script(script_path)

if __name__ == "__main__":
    main()
