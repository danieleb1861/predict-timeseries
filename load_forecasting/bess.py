"""BESS and frequency-domain sizing analysis workflow."""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal

from load_forecasting.plots import save_plot


def run_bess_sizing_analysis(TT, IMGS_DIR, selector, numGens, rating_kW):
    sns.set_theme(style="whitegrid", context="paper", font="Times New Roman")
    sns.set_palette(sns.color_palette("Greys", 10))

    def compute_ecdfFunc(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / float(len(x))
        return x, y

    def plot_ecdf(TT, col, IMGS_DIR, name, xlabel=None, q_levels=(0.95,0.9985)):
        """
        ECDF plot per leg.
        """

        legs = np.sort(TT["leg_id"].unique())
        linestyles = ["-","--","-.",":",(0,(3,1,1,1)),(0,(5,2)),(0,(1,1)),(0,(7,2,2,2))]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        # grayscale mapping
        shades = sns.color_palette("Greys", n_colors=max(3, len(legs) + 2))[2:]

        for i, leg in enumerate(legs):
            d = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            d = d[np.isfinite(d)]
            if len(d) < 10:
                continue

            x, y = compute_ecdfFunc(d)
            ax.plot(x, y,
                    color=shades[i % len(shades)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.2,
                    label=f"leg {leg}")

            # quantile markers
            for q in q_levels:
                xq = np.quantile(d, q)
                ax.axvline(xq,
                        color=shades[i % len(shades)],
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=0.8,
                        alpha=0.4)


        ax.grid(True)
        ax.set_ylabel("ECDF")
        ax.set_xlabel(xlabel if xlabel else col)
        ax.legend(ncol=2, frameon=False, loc="lower right")
        save_plot(IMGS_DIR, name)

    def _ecdf_on_grid(x_leg, x_grid):
        """ECDF right-continuous evaluated on a common x_grid."""
        x_leg = np.asarray(x_leg, dtype=float)
        x_leg = x_leg[np.isfinite(x_leg)]
        x_leg.sort()
        n = len(x_leg)
        if n == 0:
            return np.full_like(x_grid, np.nan, dtype=float)
        return np.searchsorted(x_leg, x_grid, side="right") / float(n)

    def plot_ecdf_fanchart(TT, col, IMGS_DIR, name, xlabel=None,
        bands=(10, 20, 30, 40),    # meaning: 10–90, 20–80, 30–70, 40–60
        n_grid=700, min_n=10, show_legs=False
    ):
        legs = np.sort(TT["leg_id"].unique())

        # Collect samples per leg
        samples = []
        for leg in legs:
            d = TT.loc[TT["leg_id"] == leg, col].to_numpy(float)
            d = d[np.isfinite(d)]
            if len(d) >= min_n:
                samples.append(d)

        if len(samples) < 2:
            return None

        # Common x grid (use global min/max across legs)
        xmin = min(np.min(s) for s in samples)
        xmax = max(np.max(s) for s in samples)
        x_grid = np.linspace(xmin, xmax, int(n_grid))

        # Build ECDF matrix: (n_leg, n_grid)
        Y = np.vstack([_ecdf_on_grid(s, x_grid) for s in samples])

        # Median ECDF across legs (pointwise)
        y_med = np.nanmedian(Y, axis=0)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        # Optional: faint individual legs (if you really want)
        if show_legs:
            for j in range(Y.shape[0]):
                ax.plot(x_grid, Y[j], color="0.85", linewidth=0.8, alpha=0.5, zorder=1)

        # Fan chart: nested percentile bands (outer light -> inner dark)
        # Example bands: (10,20,30,40) => fill 10–90, 20–80, 30–70, 40–60
        # Grayscale mapping: outer is lighter (higher gray value), inner darker (lower gray value)
        # You can tweak these to taste.
        gray_outer, gray_inner = 0.95, 0.50
        grays = np.linspace(gray_outer, gray_inner, num=len(bands))

        legend_patches = []
        for p, g in zip(bands, grays):
            lo = np.nanquantile(Y, p/100.0, axis=0)
            hi = np.nanquantile(Y, 1.0 - p/100.0, axis=0)
            ax.fill_between(x_grid, lo, hi, color=str(g), linewidth=0.0, zorder=2)
            legend_patches.append(Patch(facecolor=str(g), edgecolor="none", label=f"{p}-{100-p}%"))

        # Median on top
        ax.plot(x_grid, y_med, color="0.05", linewidth=2.0, zorder=3)

        ax.set_ylabel("ECDF")
        ax.set_xlabel(xlabel if xlabel else col)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.35)

        # Legend: Median line + Pct10/20/etc.
        handles = [Line2D([0],[0], color="0.05", lw=2.0, label="Median")] + legend_patches
        ax.legend(handles=handles, frameon=True, loc="lower right")

        return save_plot(IMGS_DIR, name)

    def table_tail(TT, col, q_levels=(0.95,0.99,0.995,0.9985)):
        """
        Returns a dataframe with quantiles per leg + global.
        """
        legs = np.sort(TT["leg_id"].unique())
        rows = []
        for leg in legs:
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < 10:
                continue
            row = {"leg_id": int(leg), "N": int(len(x))}
            for q in q_levels:
                row[f"q{q:.4f}"] = float(np.quantile(x, q))
            rows.append(row)

        # global
        xg = TT[col].to_numpy(np.float64)
        xg = xg[np.isfinite(xg)]
        row = {"leg_id": -1, "N": int(len(xg))}
        for q in q_levels:
            row[f"q{q:.4f}"] = float(np.quantile(xg, q))
        rows.append(row)

        return pd.DataFrame(rows)

    def stats_ramp_rate(TT, col="act_pow_pu", fs=1.0):
        """
        Ramp rate computed per leg to avoid boundary artefacts:
        dX/dt approx = diff * fs
        Returns:
        - per-leg dataframe of ramp quantiles
        - concatenated ramp array (finite)
        """
        q_levels = [0.50, 0.90, 0.95, 0.99, 0.995]
        legs = np.sort(TT["leg_id"].unique())
        all_r = []

        rows = []
        for leg in legs:
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < 3:
                continue
            r = np.diff(x) * fs  # PU/s
            r = r[np.isfinite(r)]
            if len(r) < 10:
                continue
            all_r.append(r)

            row = {"leg_id": int(leg), "N": int(len(r))}
            a = np.abs(r)
            row["mean_abs"] = float(np.mean(a))
            row["std_abs"]  = float(np.std(a))
            for q in q_levels:
                row[f"|r|_q{q:.3f}"] = float(np.quantile(a, q))
            rows.append(row)

        rall = np.concatenate(all_r) if len(all_r) else np.array([], dtype=np.float64)
        return pd.DataFrame(rows), rall

    def stats_event_duration(TT, col="act_pow_pu", fs=1.0, thr_quantile=0.99):
        """
        Computes durations (seconds) of events where |dX/dt| exceeds a threshold.
        threshold is set from GLOBAL ramp distribution quantile.
        """
        _, rall = stats_ramp_rate(TT, col=col, fs=fs)
        if len(rall) < 10:
            return pd.DataFrame(), np.nan

        thr = float(np.quantile(np.abs(rall), thr_quantile))

        rows = []
        legs = np.sort(TT["leg_id"].unique())
        for leg in legs:
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < 3:
                continue
            r = np.diff(x) * fs
            mask = (np.abs(r) >= thr).astype(np.int8)

            # run-length encoding on mask
            durations = []
            run = 0
            for m in mask:
                if m == 1:
                    run += 1
                elif run > 0:
                    durations.append(run / fs)
                    run = 0
            if run > 0:
                durations.append(run / fs)

            if len(durations) == 0:
                continue

            d = np.array(durations, dtype=np.float64)
            row = {
                "leg_id": int(leg),
                "thr_abs_ramp": thr,
                "num_events": int(len(d)),
                "dur_mean_s": float(d.mean()),
                "dur_q50_s": float(np.quantile(d, 0.50)),
                "dur_q90_s": float(np.quantile(d, 0.90)),
                "dur_q99_s": float(np.quantile(d, 0.99)),
                "dur_max_s": float(d.max()),
            }
            rows.append(row)

        return pd.DataFrame(rows), thr

    # Power Spectral Density (Welch) + bandpower + decomposition into bands
    def welch_psd(x, fs=1.0, nperseg=2048, noverlap=None):
        """
        Requires SciPy
        Returns f, Pxx (density).
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if noverlap is None:
            noverlap = nperseg // 2

        nperseg = min(nperseg, len(x))
        noverlap = min(noverlap, nperseg - 1)

        f, Pxx = signal.welch(
            x, fs=fs, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            detrend="constant", scaling="density"
        )
        return f, Pxx

    def integrate_band(f, Pxx, f_lo, f_hi):
        """
        Integrate PSD over [f_lo, f_hi] to get band power (variance contribution).
        """
        if f is None or Pxx is None:
            return np.nan
        mask = (f >= f_lo) & (f < f_hi)
        if not np.any(mask):
            return np.nan
        return float(np.trapz(Pxx[mask], f[mask]))

    def table_psd_bandpower(TT, col="act_pow_pu", fs=1.0, bands=None, nperseg=2048):
        """
        Compute PSD bandpower per leg.
        bands: dict name->(f_lo,f_hi)
        """
        if bands is None:
            bands = {
                "slow_<0.005": (0.0, 0.005),
                "med_0.005-0.05": (0.005, 0.05),
                "fast_0.05-0.25": (0.05, 0.25),
                "noise_>0.25": (0.25, 0.5*fs),
            }

        rows = []
        for leg in np.sort(TT["leg_id"].unique()):
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < 256:
                continue
            f, Pxx = welch_psd(x, fs=fs, nperseg=nperseg)
            if f is None:
                continue
            row = {"leg_id": int(leg), "N": int(len(x))}
            total = integrate_band(f, Pxx, 0.0, 0.5*fs)
            row["total_power"] = total
            for name, (flo, fhi) in bands.items():
                row[name] = integrate_band(f, Pxx, flo, fhi)
                row[name+"_frac"] = float(row[name] / total) if np.isfinite(total) and total > 0 else np.nan
            rows.append(row)

        # global
        xg = TT[col].to_numpy(np.float64)
        xg = xg[np.isfinite(xg)]
        if len(xg) >= 256:
            f, Pxx = welch_psd(xg, fs=fs, nperseg=nperseg)
            if f is not None:
                row = {"leg_id": -1, "N": int(len(xg))}
                total = integrate_band(f, Pxx, 0.0, 0.5*fs)
                row["total_power"] = total
                for name, (flo, fhi) in bands.items():
                    row[name] = integrate_band(f, Pxx, flo, fhi)
                    row[name+"_frac"] = float(row[name] / total) if np.isfinite(total) and total > 0 else np.nan
                rows.append(row)

        return pd.DataFrame(rows)

    def plot_psd_by_leg(TT, col, IMGS_DIR, name, fs=1.0, nperseg=2048, fmax=0.3):
        """
        PSD plot in log scale, B/W-friendly (grayscale + linestyles).
        """

        legs = np.sort(TT["leg_id"].unique())
        linestyles = ["-","--","-.",":",(0,(3,1,1,1)),(0,(5,2)),(0,(1,1)),(0,(7,2,2,2))]
        shades = sns.color_palette("Greys", n_colors=max(3, len(legs) + 2))[2:]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        for i, leg in enumerate(legs):
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < 256:
                continue
            f, Pxx = welch_psd(x, fs=fs, nperseg=nperseg)
            if f is None:
                continue
            mask = (f > 0) & (f <= fmax)
            ax.plot(f[mask], Pxx[mask],
                    color=shades[i % len(shades)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.1,
                    label=f"leg {leg}")

        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (PU$^2$/Hz)")
        ax.grid(True, which="both")
        ax.legend(ncol=2, frameon=False, loc="upper right")
        save_plot(IMGS_DIR, name)

    def plot_psd_fanchart(
        TT, col, IMGS_DIR, name,
        fs=1.0, nperseg=2048, noverlap=None, fmax=0.3,
        bands=(10, 20, 30, 40),     # 10–90, 20–80, 30–70, 40–60
        min_n=256, show_legs=False,
        log_quantiles=True, eps=1e-18
    ):
        """
        PSD fan chart across legs:
        - compute Welch PSD per leg
        - align on a common frequency grid (fixed nperseg across included legs)
        - compute pointwise median + quantile bands across legs
        - plot in log-y with grayscale fan (outer light -> inner dark)

        log_quantiles=True:
        quantiles are computed on log10(PSD).
        """

        legs = np.sort(TT["leg_id"].unique())

        # --- collect valid legs
        series = []
        lengths = []
        for leg in legs:
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) >= min_n:
                series.append(x)
                lengths.append(len(x))

        if len(series) < 2:
            return None

        # --- enforce a common nperseg (so frequency grid is identical)
        nperseg_common = int(min(nperseg, min(lengths)))
        if nperseg_common < 8:
            return None
        if noverlap is None:
            noverlap_common = nperseg_common // 2
        else:
            noverlap_common = int(min(noverlap, nperseg_common - 1))

        # --- reference frequency grid
        f_ref, P_ref = signal.welch(
            series[0], fs=fs, window="hann",
            nperseg=nperseg_common, noverlap=noverlap_common,
            detrend="constant", scaling="density"
        )

        # frequency mask
        mask_f = (f_ref >= 0.0) & (f_ref <= fmax)
        f_grid = f_ref[mask_f]

        # --- build PSD matrix (n_leg, n_freq)
        P = []
        for x in series:
            f, Pxx = signal.welch(
                x, fs=fs, window="hann",
                nperseg=nperseg_common, noverlap=noverlap_common,
                detrend="constant", scaling="density"
            )
            Pxx = np.asarray(Pxx, dtype=np.float64)

            # align (should already match; keep safe)
            if len(f) != len(f_ref) or np.any(f != f_ref):
                Pxx = np.interp(f_ref, f, Pxx, left=np.nan, right=np.nan)

            Pxx = Pxx[mask_f]
            Pxx = np.maximum(Pxx, eps)  # avoid zeros for log-scale
            P.append(Pxx)

        P = np.vstack(P)  # shape: (n_legs, n_freq)

        # --- quantiles across legs (pointwise)
        if log_quantiles:
            L = np.log10(P)
            med = 10.0 ** np.nanmedian(L, axis=0)
        else:
            med = np.nanmedian(P, axis=0)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        if show_legs:
            for j in range(P.shape[0]):
                ax.plot(f_grid, P[j], color="0.85", linewidth=0.8, alpha=0.45, zorder=1)

        # grayscale fan: outer lighter -> inner darker
        gray_outer, gray_inner = 0.95, 0.50
        grays = np.linspace(gray_outer, gray_inner, num=len(bands))

        legend_patches = []
        for p, g in zip(bands, grays):
            if log_quantiles:
                lo = 10.0 ** np.nanquantile(L, p/100.0, axis=0)
                hi = 10.0 ** np.nanquantile(L, 1.0 - p/100.0, axis=0)
            else:
                lo = np.nanquantile(P, p/100.0, axis=0)
                hi = np.nanquantile(P, 1.0 - p/100.0, axis=0)

            lo = np.maximum(lo, eps)
            hi = np.maximum(hi, eps)

            ax.fill_between(f_grid, lo, hi, color=str(g), linewidth=0.0, zorder=2)
            legend_patches.append(Patch(facecolor=str(g), edgecolor="none", label=f"{p}-{100-p}%"))

        # median on top
        ax.plot(f_grid, np.maximum(med, eps), color="0.05", linewidth=2.0, zorder=3)

        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (PU$^2$/Hz)")
        ax.grid(True, which="both", alpha=0.35)

        handles = [Line2D([0], [0], color="0.05", lw=2.0, label="Median")] + legend_patches
        ax.legend(handles=handles, frameon=True, loc="upper right")

        return save_plot(IMGS_DIR, name)

    # --- Filters / time series decomposition
    def _safe_filtfilt(b, a, x):
        """
        filtfilt with guard for short signals.
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 20:
            return x
        padlen = 3 * max(len(a), len(b))
        if len(x) <= padlen + 1:
            # too short for filtfilt padding -> fallback to lfilter
            return signal.lfilter(b, a, x)
        return signal.filtfilt(b, a, x)

    def butter_filter(x, fs, ftype, fc=None, band=None, order=4):
        """
        IIR Butterworth (fc=0.005Hz would require huge FIR).
        ftype: "lowpass" / "bandpass" / "highpass"
        """

        nyq = 0.5 * fs
        if ftype == "lowpass":
            wn = fc / nyq
            b, a = signal.butter(order, wn, btype="low")
        elif ftype == "highpass":
            wn = fc / nyq
            b, a = signal.butter(order, wn, btype="high")
        elif ftype == "bandpass":
            lo, hi = band
            wn = [lo/nyq, hi/nyq]
            b, a = signal.butter(order, wn, btype="band")
        else:
            raise ValueError("Unknown filter type")
        return _safe_filtfilt(b, a, x)

    def decompose_bands(x, fs=1.0, f1=0.005, f2=0.05, f3=0.25, order=4):
        """
        Returns slow, medium, fast, noise components (sum approx x).
        Uses IIR for practicality.
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        slow = butter_filter(x, fs, "lowpass", fc=f1, order=order)
        med  = butter_filter(x, fs, "bandpass", band=(f1, f2), order=order)
        fast = butter_filter(x, fs, "bandpass", band=(f2, f3), order=order)
        # noise = residual above f3
        noise = x - (slow + med + fast)

        return slow, med, fast, noise

    def iqrm(x, qlo=0.0015, qhi=0.9985):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return np.nan
        return float(np.quantile(x, qhi) - np.quantile(x, qlo))

    def component_stats(x, fs=1.0):
        """
        Return key stats for sizing: std, RMS, IQRm, range, ramp quantiles.
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return {}

        dx = np.diff(x) * fs
        ax = np.abs(dx[np.isfinite(dx)])

        out = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "rms": float(np.sqrt(np.mean(x**2))),
            "iqrm": iqrm(x),
            "q0.95": float(np.quantile(x, 0.95)),
            "q0.99": float(np.quantile(x, 0.99)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "range": float(np.max(x) - np.min(x)),
        }
        if len(ax) >= 10:
            out["|ramp|_q0.95"] = float(np.quantile(ax, 0.95))
            out["|ramp|_q0.99"] = float(np.quantile(ax, 0.99))
            out["|ramp|_max"]   = float(np.max(ax))
        else:
            out["|ramp|_q0.95"] = np.nan
            out["|ramp|_q0.99"] = np.nan
            out["|ramp|_max"]   = np.nan
        return out

    def table_band_kpi(TT, col="act_pow_pu", fs=1.0, f1=0.005, f2=0.05, f3=0.25, order=4):
        """
        Per leg: compute decomposition and sizing-relevant KPI for each band.
        """
        rows = []
        for leg in np.sort(TT["leg_id"].unique()):
            x = TT.loc[TT["leg_id"] == leg, col].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < 512:
                continue
            comp = decompose_bands(x, fs=fs, f1=f1, f2=f2, f3=f3, order=order)
            if comp is None:
                continue
            slow, med, fast, noise = comp

            for name, sig in [("slow", slow), ("medium", med), ("fast", fast), ("noise", noise)]:
                st = component_stats(sig, fs=fs)
                st["leg_id"] = int(leg)
                st["band"] = name
                rows.append(st)

        # global aggregate (optional)
        xg = TT[col].to_numpy(np.float64)
        xg = xg[np.isfinite(xg)]
        if len(xg) >= 512:
            comp = decompose_bands(xg, fs=fs, f1=f1, f2=f2, f3=f3, order=order)
            if comp is not None:
                slow, med, fast, noise = comp
                for name, sig in [("slow", slow), ("medium", med), ("fast", fast), ("noise", noise)]:
                    st = component_stats(sig, fs=fs)
                    st["leg_id"] = -1
                    st["band"] = name
                    rows.append(st)

        return pd.DataFrame(rows)

    def plot_band_timeseries(TT, col, IMGS_DIR, name, fs=1.0, leg_id=None, t0_idx=0, length_s=600,
                            f1=0.005, f2=0.05, f3=0.25, order=4):
        """
        Plot a short segment showing decomposition components
        """

        if leg_id is None:
            leg_id = int(np.sort(TT["leg_id"].unique())[0])

        df = TT.loc[TT["leg_id"] == leg_id].reset_index(drop=True)
        x = df[col].to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 512:
            return

        i0 = int(t0_idx)
        i1 = min(len(df), i0 + int(length_s * fs))
        xs = df[col].iloc[i0:i1].to_numpy(np.float64)
        ts = df["dtime"].iloc[i0:i1].to_numpy()

        comp = decompose_bands(xs, fs=fs, f1=f1, f2=f2, f3=f3, order=order)
        if comp is None:
            return
        slow, med, fast, noise = comp

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(ts, xs,   linestyle="-",  linewidth=1.2, label="total")
        ax.plot(ts, slow, linestyle="--", linewidth=1.1, label="slow")
        ax.plot(ts, med,  linestyle="-.", linewidth=1.1, label="medium")
        ax.plot(ts, fast, linestyle=":",  linewidth=1.1, label="fast")
        # noise can clutter; plot only if needed
        # ax.plot(ts, noise, linestyle=(0,(1,1)), linewidth=1.0, label="noise")

        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend(frameon=False, ncol=2)
        save_plot(IMGS_DIR, name)

    # %% BESS and genset sizing section
    def convert_pu_kW(TT, rating_kW, col_pu="act_pow_pu"):
        """P_tot[kW] from pu base = rating_kW (single genset base)."""
        x = TT[col_pu].to_numpy(dtype=float)
        return x * float(rating_kW)

    def compute_percLoad(TT, col_pu="act_pow_pu", numGens=3):
        """
        %MCR per genset assuming equal sharing and pu base = single genset rating.
        %MCR = 100 * (P_tot / numGens) / P_rated  = 100 * act_pow_pu / numGens
        """
        x = TT[col_pu].to_numpy(dtype=float)
        return 100.0 * x / float(numGens)

    def compute_genset_rating_kW(Pgen_per_genset_kW, q=0.99, percLoad_at_q=0.90):
        """
        Choose rating so that q-quantile of P_gen_per_genset hits percLoad_at_q of rating.
        rating_new = P_q / percLoad_at_q
        """
        x = np.asarray(Pgen_per_genset_kW, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return np.nan
        Pq = np.quantile(x, q)
        return float(Pq / float(percLoad_at_q))


    # Plot: PDF + CDF with peaks
    def plot_pdf_cdf_mcr(mcr_pct, IMGS_DIR, name, q_peak=0.99, designed_peak_pct=100.0, bins=100, xlabel="Required active power (% MCR)",
        pf=None,                 # (array-like, same length as mcr_pct)
        pf_bins=30,              # how many x-bins for PF curve
        pf_stat="mean",          # "mean" or "median"
        pf_min_count=50          # min samples per bin to plot
        ):
        mcr = np.asarray(mcr_pct, dtype=float)
        mcr = mcr[np.isfinite(mcr)]
        if len(mcr) < 20:
            return None

        fig, ax_pdf = plt.subplots(figsize=(7.2, 3.4))
        ax_cdf = ax_pdf.twinx()

        # # operating band shading
        # ax_pdf.axvspan(op_band[0], op_band[1], color="0.9", alpha=0.6, zorder=0)
        # ax_pdf.text(op_band[0], ax_pdf.get_ylim()[1]*0.95 if ax_pdf.get_ylim()[1] > 0 else 0.01,
        #             "recommended\nband", color="0.25", fontsize=9)

        # PDF histogram
        ax_pdf.hist(mcr, bins=bins, density=True, color="0.35", alpha=0.75, edgecolor="0.35", linewidth=0.3)
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_xlabel(xlabel)

        # CDF (ECDF)
        x, y = compute_ecdfFunc(mcr)
        ax_cdf.plot(x, y, color="0.05", linewidth=1.3)
        ax_cdf.set_ylabel("ECDF")
        ax_cdf.set_ylim(0, 1)

        # spot the 0.99 value
        idx = int(np.searchsorted(y, float(q_peak), side="left"))
        idx = np.clip(idx, 0, len(x) - 1)
        xq = float(x[idx])   # power (%MCR) at CDF=q
        yq = float(y[idx])

        ax_cdf.plot([xq], [yq], marker="o", markersize=6, color="0.05", zorder=5)

        # label next to the dot
        if selector == "manoeuvre":
            ax_cdf.annotate(
                f"ECDF = {q_peak:.2f}\n%Power = {xq:.1f}%",
                xy=(xq, yq),
                xytext=(xq - 0.3*(np.nanmax(x)-np.nanmin(x)), 0.7),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.05"),
                fontsize=9, color="0.05"
            )
        else:
            ax_cdf.annotate(
                f"ECDF = {q_peak:.2f}\n%Power = {xq:.1f}%",
                xy=(xq, yq),
                xytext=(xq + 0.3*(np.nanmax(x)-np.nanmin(x)), 0.8),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.05"),
                fontsize=9, color="0.05"
            )

        # Mark quantile peak
        real_peak = float(np.quantile(mcr, 0.9995))
        ax_pdf.axvline(real_peak, color="0.05", linestyle="--", linewidth=1.1)
        if selector == "manoeuvre":
            ax_pdf.annotate(
                f"Real peak = {real_peak:.1f}%",
                xy=(real_peak, ax_pdf.get_ylim()[1]*0.65),
                xytext=(real_peak + 0.1*(np.max(mcr)-np.min(mcr)), ax_pdf.get_ylim()[1]*0.65),
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.05"),
                fontsize=9, color="0.05"
            )
        else:
            ax_pdf.annotate(
                f"Real peak = {real_peak:.1f}%",
                xy=(real_peak, ax_pdf.get_ylim()[1]*0.65),
                xytext=(real_peak + 0.2*(np.max(mcr)-np.min(mcr)), ax_pdf.get_ylim()[1]*0.65),
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.05"),
                fontsize=9, color="0.05"
            )

        # Mark designed peak
        ax_pdf.axvline(designed_peak_pct, color="0.2", linestyle=(0, (4, 4)), linewidth=1.1)
        if selector == "manoeuvre":
            ax_pdf.annotate(
                f"Max (as-built) = {designed_peak_pct:.0f}%",
                xy=(designed_peak_pct, ax_pdf.get_ylim()[1]*0.5),
                xytext=(designed_peak_pct - 0.5*(np.max(mcr)-np.min(mcr)), ax_pdf.get_ylim()[1]*0.5),
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.2"),
                fontsize=9, color="0.2"
            )
        else:
            ax_pdf.annotate(
                f"Max (as-built) = {designed_peak_pct:.0f}%",
                xy=(designed_peak_pct, ax_pdf.get_ylim()[1]*0.5),
                xytext=(designed_peak_pct - 0.8*(np.max(mcr)-np.min(mcr)), ax_pdf.get_ylim()[1]*0.5),
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.2"),
                fontsize=9, color="0.2"
            )

        # ---------- PF line (binned) ----------
        if pf is not None:
            pf = np.asarray(pf, dtype=float)
            pf = pf[np.isfinite(pf)]

            # pf must align with mcr_pct sample-by-sample.
            # If you filtered mcr, you must apply the same mask to PF upstream.

            # Create a third axis for PF
            ax_pf = ax_pdf.twinx()
            ax_pf.spines["right"].set_position(("axes", 1.10))  # shift to the right
            ax_pf.set_frame_on(True)
            ax_pf.patch.set_visible(False)

            # Bin PF vs MCR
            edges = np.linspace(0.0, 100.0, pf_bins + 1)
            bin_id = np.digitize(mcr, edges) - 1
            centers = 0.5 * (edges[:-1] + edges[1:])

            pf_curve = np.full(pf_bins, np.nan)
            counts = np.zeros(pf_bins, dtype=int)

            for b in range(pf_bins):
                mask_b = bin_id == b
                counts[b] = int(np.sum(mask_b))
                if counts[b] < pf_min_count:
                    continue
                vals = pf[mask_b]
                pf_curve[b] = np.nanmedian(vals) if pf_stat == "median" else np.nanmean(vals)

            ok = np.isfinite(pf_curve)

            ax_pf.plot(
                centers[ok], pf_curve[ok],
                color="0.55", linewidth=1.3, linestyle="-", zorder=4
            )
            ax_pf.set_ylim(0, 1.0)
            ax_pf.set_ylabel("Power factor", color="0.55")
            ax_pf.tick_params(axis="y", colors="0.55")

        ax_pdf.grid(True, which="both", alpha=0.4)
        return save_plot(IMGS_DIR, name)

    # %% Plot: As-built vs NEW design with SFOC curve

    def plot_old_new_design(mcr_old_pct, mcr_new_pct, IMGS_DIR, name,
                            bins=120, xlim=(0, 100),
                            bess_pct=None, bess_kW=None, bess_label="BESS"):

        old = np.asarray(mcr_old_pct, dtype=float)
        new = np.asarray(mcr_new_pct, dtype=float)
        old = old[np.isfinite(old)]
        new = new[np.isfinite(new)]
        if len(old) < 20 or len(new) < 20:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

        # ax, data, title, color
        panels = [(axes[0], old, "As-built operation", "0.35"),
                (axes[1], new, "New design", "0.70")]

        for ax, data, title, color in panels:
            ax_sfoc = ax.twinx()

            # PDF
            ax.hist(
                data, bins=bins, density=True,
                color=color, alpha=0.75, edgecolor=color, linewidth=0.3
            )
            ax.set_ylabel("PDF")
            ax.set_title(title)

            # SFOC curve (placeholder)
            perc = np.array([0, 20, 40, 60, 75, 85, 95, 100], dtype=float)
            sfoc = np.array([300, 250, 230, 215, 211, 208, 210, 213], dtype=float)

            ax_sfoc.plot(perc, sfoc, color="0.05", linestyle="--", linewidth=1.3)
            ax_sfoc.set_ylabel("SFOC (g/kWh)")

            # mark median load (vertical line on PDF axis)
            med = float(np.median(data))
            ax.axvline(med, color="0.05", linestyle=":", linewidth=1.0)
            ax.annotate(
                f"median = {med:.1f}%",
                xy=(med, ax.get_ylim()[1]*0.75),
                xytext=(med + 0.1*(np.max(data)-np.min(data)), ax.get_ylim()[1]*0.75),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.05"),
                fontsize=9, color="0.05"
            )

            # # Plot BESS range
            # if (ax is axes[1]) and (bess_pct is not None) and np.isfinite(bess_pct):
            #     x0 = float(med)  # centro della freccia (uso la mediana del NEW)
            #     xL = float(np.clip(x0 - bess_pct, xlim[0], xlim[1]))
            #     xR = float(np.clip(x0 + bess_pct, xlim[0], xlim[1]))

            #     # y stabile (non dipende dalla PDF): frazione dell'asse
            #     yfrac = 0.18

            #     ax.annotate(
            #         "", xy=(xR, yfrac), xytext=(xL, yfrac),
            #         xycoords=("data", "axes fraction"),
            #         textcoords=("data", "axes fraction"),
            #         arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.05")
            #     )

            #     txt = f"{bess_label}: ±{bess_pct:.1f}% MCR"
            #     if bess_kW is not None and np.isfinite(bess_kW):
            #         txt += f"\n(±{bess_kW:.0f} kW)"

            #     ax.text(
            #         0.5*(xL+xR), yfrac+0.03, txt,
            #         transform=ax.get_xaxis_transform(),
            #         ha="center", va="bottom",
            #         fontsize=9, color="0.05"
            #     )

            # dot at intersection (median, SFOC(median))
            med_clip = float(np.clip(med, perc.min(), perc.max()))
            sfoc_med = float(np.interp(med_clip, perc, sfoc))  # linear interpolation

            ax_sfoc.plot(
                [med_clip], [sfoc_med],
                marker="o", markersize=5.5,
                color="0.05", zorder=6
            )

            # label near the dot
            ax_sfoc.annotate(
                f"{sfoc_med:.0f} g/kWh",
                xy=(med_clip, sfoc_med),
                xytext=(med_clip + 5, sfoc_med + 5),
                textcoords="data",
                fontsize=9,
                color="0.05",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.05"),
            )

            ax.grid(True, alpha=0.4)

        axes[1].set_xlabel("Required active power (% MCR)")
        axes[1].set_xlim(*xlim)
        return save_plot(IMGS_DIR, name)

    def compute_pf(TT):
        S = np.sqrt(TT["act_pow_pu"]**2 + TT["react_pow_pu"]**2)
        pf = np.where(S > 0, TT["act_pow_pu"] / S, 0.0).astype(np.float32)
        return pf

    # End-to-end: build both plots + BESS sizing numbers
    def plot_design_figs(TT, IMGS_DIR, selector_label, fs, numGens, rating_kW, fc_slow=0.01, q_peak=0.99, percLoad_at_q_resize=0.90, designed_peak_pct=100.0):
        """
        Produces:
        1) PDF+CDF on %MCR per genset (OLD)
        2) OLD vs NEW design plot (with smoothing + resized rating)

        Design logic:
        - P_tot[kW] from pu
        - P_slow_tot[kW] = lowpass(P_tot, fc_slow)
        - gensets supply P_slow (equal sharing): Pgen_per_genset = P_slow_tot/numGens
        - rating_new chosen so that q_peak of Pgen_per_genset hits percLoad_at_q_resize
        Also prints BESS sizing from residual P_bess = P_tot - P_slow.
        """
        # OLD: %MCR directly from pu / numGens
        mcr_old = compute_percLoad(TT, col_pu="act_pow_pu", numGens=numGens)
        pf = compute_pf(TT)

        # Convert to kW (total)
        actpow_tot_kW = convert_pu_kW(TT, col_pu="act_pow_pu", rating_kW=rating_kW)

        # Slow component (total)
        actpow_slow_kW = butter_filter(actpow_tot_kW, fs, "lowpass", fc=fc_slow, order=4)

        # NEW: gensets track slow only, equal sharing
        actpow_gen_kW = actpow_slow_kW / float(numGens)

        # Resized genset rating
        rating_new_kW = compute_genset_rating_kW(actpow_gen_kW, q=q_peak, percLoad_at_q=percLoad_at_q_resize)

        # NEW %MCR time series under new rating
        mcr_new = 100.0 * actpow_gen_kW / float(rating_new_kW) if np.isfinite(rating_new_kW) else np.full_like(actpow_gen_kW, np.nan)

        # BESS residual (total)
        P_bess_kW = actpow_tot_kW - actpow_slow_kW
        P_bess_q99 = float(np.quantile(np.abs(P_bess_kW[np.isfinite(P_bess_kW)]), 0.99))
        P_bess_max = float(np.max(np.abs(P_bess_kW[np.isfinite(P_bess_kW)])))
        dt = 1.0 / float(fs)
        e_kWh = np.cumsum(np.nan_to_num(P_bess_kW, nan=0.0)) * dt / 3600.0
        E_swing = float(np.max(e_kWh) - np.min(e_kWh))

        # BESS envelope in %MCR per genset
        if np.isfinite(rating_new_kW) and rating_new_kW > 0:
            bess_pct_q99 = 100.0 * P_bess_q99 / float(rating_new_kW)
        else:
            bess_pct_q99 = np.nan

        print(f"[{selector_label}] rating_old = {rating_kW:.1f} kW per genset")
        print(f"[{selector_label}] rating_new (q={q_peak:.2f} -> {percLoad_at_q_resize*100:.0f}%MCR) = {rating_new_kW:.1f} kW per genset")
        print(f"[{selector_label}] BESS power: q_0.99 = {P_bess_q99:.1f} kW, max = {P_bess_max:.1f} kW")
        print(f"[{selector_label}] BESS energy swing (raw) = {E_swing:.2f} kWh")
        print()

        os.makedirs(IMGS_DIR, exist_ok=True)

        # Plot 1: PDF + CDF peaks (OLD)
        plot_pdf_cdf_mcr(mcr_old, IMGS_DIR, f"mcr_pdf_cdf_{selector_label}", q_peak, designed_peak_pct, bins=120, xlabel="Required active power (% MCR)", pf=pf, pf_bins=600, pf_stat="mean", pf_min_count=50)

        # Plot 2: As-built vs new design
        plot_old_new_design(mcr_old, mcr_new, IMGS_DIR, f"new_design_{selector_label}",
            bins=120, xlim=(0, 100),
            bess_pct=bess_pct_q99,     # or bess_pct_max
            bess_kW=P_bess_q99,
            bess_label="BESS (q0.99)"
        )

        return {
            "rating_new_kW": rating_new_kW,
            "P_bess_q99_kW": P_bess_q99,
            "P_bess_max_kW": P_bess_max,
            "E_swing_kWh": E_swing
        }

    def compute_reserves(TT, numGens, pf_rated=0.8):
        """
        Physically consistent reserve computation with an apparent-power constraint.

        Inputs:
        TT must contain act_pow_pu, react_pow_pu
        numGens: number of online gensets in this dataset slice (e.g., 3 nav, 4 man)
        pf_rated: rated power factor of genset (design value)

        Assumptions (your current scaling):
        act_pow_pu   = P[kW] / rating_kW
        react_pow_pu = Q[kVAr] / rating_kW   (same P_base; OK if stated explicitly)

        Total apparent capability in the same base:
        Smax_pu = numGens / pf_rated
        Qmax(P) = sqrt(Smax^2 - P^2)
        Q_reserve = Qmax(P) - |Q|
        """
        out = TT.copy()

        P = out["act_pow_pu"].to_numpy(np.float64)
        Q = out["react_pow_pu"].to_numpy(np.float64)

        # Active reserve
        out["P_reserve_pu"] = (float(numGens) - out["act_pow_pu"]).astype(np.float32)

        # Reactive reserve from S-limit
        Smax = float(numGens) / float(pf_rated)
        Qmax = np.sqrt(np.maximum(0.0, Smax**2 - P**2))
        out["Q_reserve_pu"] = (Qmax - np.abs(Q)).astype(np.float32)

        # Diagnostics
        out["S_pu_Pbase"] = np.sqrt(out["act_pow_pu"]**2 + out["react_pow_pu"]**2).astype(np.float32)
        out["pfactor"] = compute_pf(out)

        return out

    # %% Run EDA suite and sizing
    # Group the run in one function
    def run_eda_suite(
        TT,
        IMGS_DIR,
        selector_label="navigation",
        fs=1.0,
        numGens=np.nan,
        pf_rated=0.8,
        do_reserve=True,
        f1=0.005, f2=0.05, f3=0.25
    ):
        """
        Run the whole EDA suite (navigation or manoeuvre) for BESS sizing and load analyses.
        TT must include: dtime, leg_id, act_pow_pu, react_pow_pu, spd_*, man_*, meteo, valid(optional)
        """

        os.makedirs(os.path.join(IMGS_DIR, "tables"), exist_ok=True)

        # %% Basic sanity & reserves
        if do_reserve and np.isfinite(numGens):
            TTp = compute_reserves(TT, numGens=int(numGens), pf_rated=pf_rated)
        else:
            TTp = TT.copy()
            TTp["pfactor"] = compute_pf(TTp)

        # %% --- Design figures: genset resizing + BESS sizing (paper-style)
        _ = plot_design_figs(
            TTp,
            IMGS_DIR=IMGS_DIR,
            selector_label=selector_label,
            fs=fs,
            numGens=int(numGens),
            rating_kW=rating_kW,
            fc_slow=0.01,
            q_peak=0.99,
            percLoad_at_q_resize=0.90,
            designed_peak_pct=100.0
        )

        # %% Empirical Cumulative Distribution Function (ECDF) + tails
        plot_ecdf(TTp, "act_pow_pu", IMGS_DIR, f"ecdf_P_{selector_label}", xlabel="Active power (PU)")
        plot_ecdf(TTp, "react_pow_pu", IMGS_DIR, f"ecdf_Q_{selector_label}", xlabel="Reactive power (PU)")
        plot_ecdf_fanchart(TTp, "act_pow_pu", IMGS_DIR, f"ecdfBand_P_{selector_label}",
                            xlabel="Active power (PU)", bands=(10,20,30,40))
        plot_ecdf_fanchart(TTp, "react_pow_pu", IMGS_DIR, f"ecdfBand_Q_{selector_label}",
                                xlabel="Reactive power (PU)", bands=(10,20,30,40))


        tP = table_tail(TTp, "act_pow_pu")
        tQ = table_tail(TTp, "react_pow_pu")
        tPF = table_tail(TTp, "pfactor")

        tP.to_csv(os.path.join(IMGS_DIR, "tables", f"tails_P_{selector_label}.csv"), index=False)
        tQ.to_csv(os.path.join(IMGS_DIR, "tables", f"tails_Q_{selector_label}.csv"), index=False)
        tPF.to_csv(os.path.join(IMGS_DIR, "tables", f"tails_pf_{selector_label}.csv"), index=False)

        # %% Ramp stats + event durations
        rampP_leg, rampP_all = stats_ramp_rate(TTp, col="act_pow_pu", fs=fs)
        rampQ_leg, rampQ_all = stats_ramp_rate(TTp, col="react_pow_pu", fs=fs)
        rampP_leg.to_csv(os.path.join(IMGS_DIR, "tables", f"ramp_P_{selector_label}.csv"), index=False)
        rampQ_leg.to_csv(os.path.join(IMGS_DIR, "tables", f"ramp_Q_{selector_label}.csv"), index=False)

        durP_leg, thrP = stats_event_duration(TTp, col="act_pow_pu", fs=fs, thr_quantile=0.99)
        durP_leg.to_csv(os.path.join(IMGS_DIR, "tables", f"dur_events_P_{selector_label}.csv"), index=False)

        # %% PSD + bandpowers
        # plot_psd_by_leg(TTp, "act_pow_pu", IMGS_DIR, f"psd_P_{selector_label}", fs=fs, nperseg=2048, fmax=0.3)
        # plot_psd_by_leg(TTp, "react_pow_pu", IMGS_DIR, f"psd_Q_{selector_label}", fs=fs, nperseg=2048, fmax=0.3)

        # PSD fan chart (median + bands)
        plot_psd_fanchart(TTp, "act_pow_pu", IMGS_DIR, f"psdBand_P_{selector_label}",
                        fs=fs, nperseg=2048, fmax=0.3, bands=(10,20,30,40),
                        min_n=256, show_legs=False, log_quantiles=True)

        plot_psd_fanchart(TTp, "react_pow_pu", IMGS_DIR, f"psdBand_Q_{selector_label}",
                        fs=fs, nperseg=2048, fmax=0.3, bands=(10,20,30,40),
                        min_n=256, show_legs=False, log_quantiles=True)

        bpP = table_psd_bandpower(TTp, col="act_pow_pu", fs=fs, nperseg=2048)
        bpQ = table_psd_bandpower(TTp, col="react_pow_pu", fs=fs, nperseg=2048)
        bpP.to_csv(os.path.join(IMGS_DIR, "tables", f"bandpower_P_{selector_label}.csv"), index=False)
        bpQ.to_csv(os.path.join(IMGS_DIR, "tables", f"bandpower_Q_{selector_label}.csv"), index=False)

        # %% Band decomposition KPI (slow/medium/fast/noise)
        kpiP = table_band_kpi(TTp, col="act_pow_pu", fs=fs, f1=f1, f2=f2, f3=f3, order=4)
        kpiQ = table_band_kpi(TTp, col="react_pow_pu", fs=fs, f1=f1, f2=f2, f3=f3, order=4)
        kpiP.to_csv(os.path.join(IMGS_DIR, "tables", f"kpi_bands_P_{selector_label}.csv"), index=False)
        kpiQ.to_csv(os.path.join(IMGS_DIR, "tables", f"kpi_bands_Q_{selector_label}.csv"), index=False)

        # %% Example time-series decomposition plot (first leg, first 10 minutes)
        first_leg = int(np.sort(TTp["leg_id"].unique())[0])
        plot_band_timeseries(TTp, "act_pow_pu", IMGS_DIR, f"bands_example_P_{selector_label}",
                            fs=fs, leg_id=first_leg, t0_idx=0, length_s=6000,
                            f1=f1, f2=f2, f3=f3)

        # %% Reserve plots/tables (only if numGens known)
        if do_reserve and np.isfinite(numGens):
            tPR = table_tail(TTp, "P_reserve_pu")
            tQR = table_tail(TTp, "Q_reserve_pu")
            tPR.to_csv(os.path.join(IMGS_DIR, "tables", f"tails_Preserve_{selector_label}.csv"), index=False)
            tQR.to_csv(os.path.join(IMGS_DIR, "tables", f"tails_Qreserve_{selector_label}.csv"), index=False)
            plot_ecdf(TTp, "P_reserve_pu", IMGS_DIR, f"ecdf_Preserve_{selector_label}", xlabel="P reserve (PU)")
            plot_ecdf(TTp, "Q_reserve_pu", IMGS_DIR, f"ecdf_Qreserve_{selector_label}", xlabel="Q reserve (PU)")

        print(f"[EDA suite] Completed for {selector_label}. Outputs in: {IMGS_DIR}")
        return TTp

    TTp = run_eda_suite(TT, IMGS_DIR=IMGS_DIR, selector_label=selector, fs=1.0, numGens=numGens, pf_rated=0.8,
                        f1=0.005, f2=0.05, f3=0.25)
    return TTp
