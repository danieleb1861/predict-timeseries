"""Plotting helpers for load forecasting runs."""

import os
import warnings as _w

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_scatter_grid(y_true, preds, HorizonSteps, d=0, subplot_cols=4, figsize=(14, 7)):
    """
    Plot scatter grid of true vs predicted values for multiple models.

    Inputs
    ----------
    y_true : array-like
        Ground truth, shape (num_windows, HorizonSteps, output_dim).
    preds : dict[str, array-like]
        Model predictions, shape (num_windows, HorizonSteps, output_dim).
    HorizonSteps : int
        Horizon step index (0-based) to plot.
    d : int, default=0
        Target dimension index to plot.
    subplot_cols : int, default=4
        Number of columns in the subplot grid.
    figsize : tuple, default=(14, 7)
        Figure size.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and axes array with a grid of scatter plots.
    """

    k = HorizonSteps-1  # adjust for 0-based indexing to access arrays at t+1, t+2, ... t+HorizonSteps

    names = list(preds.keys())
    M = len(names)
    cols = min(subplot_cols, M)
    rows = int(np.ceil(M / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    yt = y_true[:, k, d]
    lo = yt.min()
    hi = yt.max()

    for i, name in enumerate(names):
        div, mod = divmod(i, cols)
        ax = axes[div, mod]

        yp = preds[name][:, k, d]
        lo_i = min(lo, yp.min())
        hi_i = max(hi, yp.max())

        ax.scatter(yt, yp, s=3, alpha=0.25)
        ax.plot([lo_i, hi_i], [lo_i, hi_i], linewidth=1)
        ax.grid(True)
        ax.set_title(f"{name} @ t+{HorizonSteps}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")

    # turn off empty axes
    for j in range(M, rows * cols):
        div, mod = divmod(j, cols)
        axes[div, mod].axis("off")


def _time_ns(time_real):
    return pd.to_datetime(time_real).to_numpy(dtype="datetime64[ns]").astype(
        np.int64,
        copy=False,
    )


def _resolve_gap_ns(time_ns, gap):
    if isinstance(gap, str) and gap == "auto":
        positive_steps = np.diff(time_ns)
        positive_steps = positive_steps[positive_steps > 0]
        cadence_ns = int(np.median(positive_steps)) if len(positive_steps) else 10**9
        return 5 * cadence_ns

    gap_ns = pd.Timedelta(gap).value
    if gap_ns <= 0:
        raise ValueError("gap must be positive or 'auto'")
    return gap_ns


def compress_time_gaps(time_real, gap="auto"):
    """
    Compress large gaps occurring in the test set in a datetime sequence for better visualisation.

    Inputs
    ------
    time_real : array-like
        Datetime-like sequence (pandas Series/Index, numpy datetime64, list of strings, etc.).
        Forward and backward discontinuities are treated as separate segments.
    gap : str or pandas.Timedelta, default="auto"
        Maximum displayed gap. ``"auto"`` uses five inferred sampling intervals.

    Results
    -------
    pandas.DatetimeIndex
        Compressed datetime sequence with the same length as t.
        Note: returned timestamps are not physical time anymore and should be used
        for plotting only.
    """
    time_ns = _time_ns(time_real)
    if len(time_ns) < 2:
        return pd.to_datetime(time_ns)
    dt = np.diff(time_ns)
    gap_ns = _resolve_gap_ns(time_ns, gap)
    discontinuities = (dt <= 0) | (dt > gap_ns)
    plot_dt = np.where(discontinuities, gap_ns, dt)
    time_comp_ns = time_ns[0] + np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(plot_dt, dtype=np.int64)]
    )
    return pd.to_datetime(time_comp_ns)


def _time_discontinuity_indices(time_real, gap):
    time_ns = _time_ns(time_real)
    dt = np.diff(time_ns)
    gap_ns = _resolve_gap_ns(time_ns, gap)
    return np.flatnonzero((dt <= 0) | (dt > gap_ns)) + 1


def _break_series_at_gaps(values, sample_indices, gap_indices):
    """Return sampled values with plot-only breaks at time discontinuities."""
    sampled = np.asarray(values, dtype=np.float64)[sample_indices].copy()
    if len(gap_indices):
        break_positions = np.searchsorted(sample_indices, gap_indices)
        break_positions = break_positions[break_positions < len(sample_indices)]
        sampled[break_positions] = np.nan
    return sampled


def _auto_zoom_bounds(length, discontinuity_indices):
    """Select a bounded zoom window that cannot cross a time discontinuity."""
    if length < 1:
        return 0, 0

    span = int(np.clip(round(0.02 * length), 300, 1200))
    edges = np.unique(
        np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.asarray(discontinuity_indices, dtype=np.int64),
                np.array([length], dtype=np.int64),
            ]
        )
    )
    center = length // 3
    segment_index = max(0, np.searchsorted(edges, center, side="right") - 1)
    segment_start = int(edges[segment_index])
    segment_end = int(edges[segment_index + 1])
    span = min(span, segment_end - segment_start)
    start = int(np.clip(center - span // 2, segment_start, segment_end - span))
    return start, start + span


def plot_full_test(TTtest, y_true_full, preds_full_dict, boundaries=None, title="Test set - true vs forecasts time series", downsample=1, boundary_every_seconds=None, seg=None, zoom=None, zoom_loc="auto", gap="auto"):
    """
    Plot the full test timeline (true signal + multiple model forecasts), with optional:
      - time-gap compression for cleaner visualisation,
      - window-boundary markers,
      - segment selection,
      - zoomed inset, for better visualisation of the local patterns of each prediction.
    """
    time_real = TTtest["dtime"].values
    shift_idx = _time_discontinuity_indices(time_real, gap)
    time_plot = compress_time_gaps(time_real, gap).to_numpy()
    T = len(time_plot)

    i0, i1 = (0, T) if seg is None else seg
    i0 = max(0, int(i0)); i1 = min(T, int(i1))
    idx = np.arange(i0, i1, downsample, dtype=np.int64)
    gap_marks = shift_idx[(shift_idx >= i0) & (shift_idx < i1)]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(
        time_plot[idx],
        _break_series_at_gaps(y_true_full, idx, gap_marks),
        color="k",
        linewidth=1.4,
        label="Measured",
    )

    for name, yhat in preds_full_dict.items():
        ax.plot(
            time_plot[idx],
            _break_series_at_gaps(yhat, idx, gap_marks),
            linewidth=1.0,
            label=name,
        )

    if len(shift_idx):
        ymin = np.nanmin(y_true_full[idx])
        ymax = np.nanmax(y_true_full[idx])
        if len(gap_marks):
            ax.vlines(
                time_plot[gap_marks],
                ymin,
                ymax,
                colors="#c62828",
                linestyles="dashed",
                linewidth=1.0,
                label="Voyage-leg boundary",
                zorder=1,
            )

    if boundary_every_seconds is not None and boundaries is not None:
        step = max(1, int(boundary_every_seconds / max(1, downsample)))
        for b in boundaries[::step]:
            if i0 <= b < i1:
                ax.axvline(time_plot[b], linestyle="--", linewidth=0.6, alpha=0.6)

    ax.grid(True)
    ax.legend(
        ncol=3,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.75",
        fontsize=8,
        loc="upper right",
        handlelength=2.6,
        columnspacing=1.4,
    )
    ax.set_xlabel("Compressed timeline" if len(shift_idx) else "Time")
    ax.set_ylabel("Active power (pu)")
    if title:
        ax.set_title(title)

    if isinstance(zoom, str):
        if zoom != "auto":
            raise ValueError("zoom must be a (start, end) pair, 'auto', or None")
        segment_gaps = shift_idx[(shift_idx > i0) & (shift_idx < i1)] - i0
        zoom = _auto_zoom_bounds(i1 - i0, segment_gaps)

    if zoom is not None:
        z0, z1 = zoom
        z0 = i0 + int(z0); z1 = i0 + int(z1)
        z0 = max(i0, z0); z1 = min(i1, z1)
        zoom_idx = np.arange(z0, z1, dtype=np.int64)
        zoom_gap_marks = shift_idx[(shift_idx >= z0) & (shift_idx < z1)]
        if zoom_loc == "auto":
            zoom_center = ((z0 + z1) / 2.0 - i0) / max(i1 - i0, 1)
            resolved_zoom_loc = "center right" if zoom_center <= 0.5 else "center left"
            inset_anchor = {
                "bbox_to_anchor": (0.0, -0.20, 1.0, 1.0),
                "bbox_transform": ax.transAxes,
            }
        else:
            resolved_zoom_loc = zoom_loc
            inset_anchor = {}
        axins = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc=resolved_zoom_loc,
            borderpad=2.0,
            **inset_anchor,
        )
        axins.plot(
            time_plot[zoom_idx],
            _break_series_at_gaps(y_true_full, zoom_idx, zoom_gap_marks),
            color="k",
            linewidth=1.4,
        )
        for name, yhat in preds_full_dict.items():
            axins.plot(
                time_plot[zoom_idx],
                _break_series_at_gaps(yhat, zoom_idx, zoom_gap_marks),
                linewidth=1.0,
            )
        axins.grid(True, lw=0.4, alpha=0.4)
        axins.tick_params(labelsize=6)
        axins.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(axins.xaxis.get_majorticklabels(), rotation=30, ha="right")
        axins.set_ylabel("PU", fontsize=6)
        connector_locs = (2, 3) if resolved_zoom_loc.endswith("right") else (1, 4)
        mark_inset(
            ax,
            axins,
            loc1=connector_locs[0],
            loc2=connector_locs[1],
            fc="none",
            ec="0.3",
            lw=0.8,
        )


def save_plot(IMGS_DIR, plot_name, dpi=400):
    """Save current plot as EPS (vector, journal-quality) and PNG (raster).
    Suppresses PostScript transparency warnings (EPS backend limitation).
    """
    plt.tight_layout()
    eps_path = os.path.join(IMGS_DIR, "eps", f"{plot_name}.eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        plt.savefig(eps_path, format="eps", bbox_inches="tight",
                    pad_inches=0.02)
    plt.savefig(os.path.join(IMGS_DIR, f"{plot_name}.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close()


def set_pub_style(font_family="Times New Roman", fontsize=9):
    mpl.rcParams.update({
        # Text dimensions
        "font.family": font_family,      # or "serif" + "DejaVu Serif"
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,

        # Lines and ticks
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Grid
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,

        # Layout
        "figure.constrained_layout.use": False,

        # EPS/PDF font type
        "ps.fonttype": 3,
        "pdf.fonttype": 42,

        # Resolution
        "figure.dpi": 200,
        "savefig.dpi": 200,

        # Save
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def get_horizon_matrix(data_frame_results, metric_key, model_order=None):
    """
    Extract a horizon-metric matrix from a results DataFrame for a chosen metric.
    """
    if model_order is None:
        model_order = list(data_frame_results.index)

    Ys = []
    for m in model_order:
        y = np.asarray(data_frame_results.loc[m, metric_key], dtype=float).ravel()
        Ys.append(y)

    # check consistent length
    lengths = [len(y) for y in Ys]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent horizon lengths for {metric_key}: {dict(zip(model_order, lengths))}")

    Y = np.vstack(Ys)  # (n_models, H)
    return model_order, Y


def plot_metric_over_horizon(data_frame_results, metric_key, ylabel, model_order=None, xlabel=r"Forecast horizon steps (steps)", figsize=(6.9, 3.0)):
    """
    Plot a horizon-dependent error metric for multiple models on the same axes.
    """
    models, Y = get_horizon_matrix(data_frame_results, metric_key, model_order=model_order)
    HorizonSteps = Y.shape[1]
    k = np.arange(HorizonSteps)

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 2)), (0, (3, 1, 1, 1))]
    markers = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, m in enumerate(models):
        ax.plot(
            k, Y[i],
            label=m,
            color=colors[i % len(colors)] if colors else None,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markevery=max(1, HorizonSteps // 8),
            linewidth=1.6,
            markersize=4
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, HorizonSteps - 1)
    ax.grid(True)

    plt.legend(ncol=3)


def plot_data_characterisation(records, arrays, summaries):
    """Return a composite and four panels for reconstructed benchmark data.

    Each record supplies an anonymous leg key, mode, split, mean and std;
    arrays maps '<key>_y' and '<key>_ramps' to finite one-dimensional arrays.
    ECDFs weight grid samples equally; mode ramp curves weight legs equally.
    Thresholds are training-grid quantiles, not overlapping-window quantiles.
    """
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter

    font_manager.findfont('Times New Roman', fallback_to_default=False)
    split_names = ('train', 'validation', 'test')
    split_labels = {'train': 'Training', 'validation': 'Validation', 'test': 'Test'}
    colours = {'train': '#294F6F', 'validation': '#238477', 'test': '#BC6248'}
    styles = {'train': '-', 'validation': '--', 'test': '-.'}
    modes = ('manoeuvre', 'navigation')
    mode_colours = {'manoeuvre': '#292929', 'navigation': '#A17A27'}
    mode_pale = {'manoeuvre': '#D2D2D2', 'navigation': '#E3D8BB'}
    mode_styles = {'manoeuvre': '-', 'navigation': '--'}
    mode_markers = {'manoeuvre': 'o', 'navigation': '^'}
    grouped, ordered_ramps, derived = {}, {}, {}
    for mode in modes:
        for split in split_names:
            selected = [r for r in records if r['mode'] == mode and r['split'] == split]
            grouped[mode, split] = np.sort(np.concatenate([arrays[r['key'] + '_y'] for r in selected]))
        threshold = float(np.quantile(grouped[mode, 'train'], .95))
        test = grouped[mode, 'test']
        derived[mode] = {
            'training_grid_q95': threshold,
            'test_grid_exceedance_pct': float(100 * np.mean(test > threshold)),
            'test_minus_train_mean_pu': float(test.mean() - grouped[mode, 'train'].mean()),
        }
        ordered_ramps[mode] = [np.sort(arrays[r['key'] + '_ramps']) for r in records if r['mode'] == mode]

    def finish(ax, letter):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#777777')
        ax.tick_params(colors='#333333')
        ax.set_axisbelow(True)

    def ecdf(ax, mode, letter):
        for split in split_names:
            values = grouped[mode, split]
            # Preserve the empirical ranks; thin display vertices only.
            indices = np.unique(np.linspace(0, len(values) - 1, min(len(values), 2200)).astype(int))
            count = sum(r['mode'] == mode and r['split'] == split for r in records)
            ax.plot(values[indices], (indices + 1) / len(values), color=colours[split],
                    linestyle=styles[split], linewidth=1.55,
                    label=f'{split_labels[split]} ({count} legs)')
        threshold = derived[mode]['training_grid_q95']
        ax.axvline(threshold, color='#929292', linestyle=(0, (2, 2)), linewidth=.85, zorder=0)
        ax.axhline(.95, color='#DEDEDE', linewidth=.65, zorder=0)
        ax.text(.98, .59, f'Training-grid 95th percentile\n{threshold:.3f} PU',
                transform=ax.transAxes, ha='right', va='center', fontsize=7.3,
                color='#666666', linespacing=1.4)
        delta = derived[mode]['test_minus_train_mean_pu']
        exceedance = derived[mode]['test_grid_exceedance_pct']
        ax.text(.98, .08, f'Test above threshold: {exceedance:.1f}%\n'
                f'Test mean shift: {delta:+.3f} PU', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, linespacing=1.5,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
        ax.set(xlim=(.7, 2.8), ylim=(0, 1.025), xlabel='Active power (PU)',
               ylabel='Cumulative probability')
        ax.set_xticks([.8, 1.2, 1.6, 2., 2.4, 2.8])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis='y', color='#E8E8E8', linewidth=.5)
        ax.legend(loc='upper left' if mode == 'manoeuvre' else 'upper right',
                  frameon=False, fontsize=7.5, handlelength=2.5,
                  title=mode.capitalize(), title_fontsize=8.5)
        finish(ax, letter)

    def ramps(ax):
        positive = [r[r > 0] for mode in modes for r in ordered_ramps[mode]]
        maximum = max(float(r[-1]) for r in positive)
        minimum = min(float(r[0]) for r in positive)
        xmin = max(minimum, 1e-6)
        grid = np.geomspace(xmin, maximum * 1.05, 550)
        for mode in modes:
            curves = []
            for values in ordered_ramps[mode]:
                survival = (len(values) - np.searchsorted(values, grid, side='right')) / len(values)
                curves.append(survival)
                ax.plot(grid, np.where(survival > 0, survival, np.nan),
                        color=mode_pale[mode], linestyle=mode_styles[mode], linewidth=.55, zorder=1)
            mean_curve = np.mean(curves, axis=0)
            ax.plot(grid, np.where(mean_curve > 0, mean_curve, np.nan),
                    color=mode_colours[mode], linestyle=mode_styles[mode], linewidth=1.8,
                    label=f'{mode.capitalize()} ({len(curves)} legs)', zorder=3)
        ax.set(xscale='log', yscale='log', xlim=(xmin, maximum * 1.05), ylim=(1e-4, 1),
               xlabel='Ramp rate (PU/s)', ylabel='Exceedance probability')
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which='major', color='#E8E8E8', linewidth=.5)
        ax.legend(loc='lower left', frameon=False, fontsize=7.5, handlelength=2.5)
        ax.text(.98, .97, 'Thin: individual legs\nBold: equal-leg average',
                transform=ax.transAxes, ha='right', va='top', fontsize=7.3,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
        finish(ax, 'c')

    def variability(ax):
        for mode in modes:
            for split in split_names:
                selected = [r for r in records if r['mode'] == mode and r['split'] == split]
                ax.scatter([r['mean'] for r in selected], [r['std'] for r in selected],
                           marker=mode_markers[mode], s=34, color=colours[split],
                           edgecolor='white', linewidth=.65, zorder=3)
        handles = [Line2D([], [], linestyle='none', marker=mode_markers[m], color='#444444',
                          markersize=5, label=m.capitalize()) for m in modes]
        legend_modes = ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=7.4)
        ax.add_artist(legend_modes)
        handles = [Line2D([], [], linestyle='none', marker='o', color=colours[s], markersize=4.5,
                          label=split_labels[s]) for s in split_names]
        ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=7.4)
        ax.set(xlabel='Mean active power (PU)', ylabel='Within-leg standard deviation (PU)',
               xlim=(1.05, 1.85), ylim=(.035, .32))
        ax.grid(color='#E8E8E8', linewidth=.5)
        finish(ax, 'd')

    rc = {
        'font.family': 'Times New Roman', 'font.size': 9,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.usetex': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'axes.formatter.use_mathtext': True, 'grid.alpha': 1.,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    }
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(7., 5.65))
        fig.subplots_adjust(left=.10, right=.985, bottom=.10, top=.95, wspace=.32, hspace=.40)
        ecdf(axes[0, 0], 'manoeuvre', 'a')
        ecdf(axes[0, 1], 'navigation', 'b')
        ramps(axes[1, 0])
        variability(axes[1, 1])
        figures = {'data_characterisation': fig}
        for suffix, draw in (
                ('a_manoeuvre', lambda ax: ecdf(ax, 'manoeuvre', 'a')),
                ('b_navigation', lambda ax: ecdf(ax, 'navigation', 'b')),
                ('c_ramps', ramps), ('d_leg_variability', variability)):
            single, ax = plt.subplots(figsize=(3.5, 2.85))
            single.subplots_adjust(left=.20, right=.98, bottom=.19, top=.94)
            draw(ax)
            figures['data_characterisation_' + suffix] = single
    return figures, derived


def plot_forecast_change_distributions(records, arrays, horizons=(5, 15, 25)):
    """Seaborn split violins and per-leg boxes for forecast-lead increments.

    Deterministic observed-rank subsamples give each leg equal density weight.
    All per-leg statistics are computed from the complete valid increments.
    """
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    import seaborn as sns

    font_manager.findfont('Times New Roman', fallback_to_default=False)
    modes = ('manoeuvre', 'navigation')
    colours = {'manoeuvre': '#38484E', 'navigation': '#A17A27'}
    pale = {'manoeuvre': '#BFCFD3', 'navigation': '#E5C985'}
    groups, rows = {}, []
    for mode in modes:
        for horizon in horizons:
            increments = []
            for record in records:
                if record['mode'] != mode:
                    continue
                values = arrays[record['key'] + '_y']
                seconds = arrays[record['key'] + '_seconds']
                valid = np.isclose(seconds[horizon:] - seconds[:-horizon], horizon)
                delta = np.abs(values[horizon:] - values[:-horizon])[valid]
                if not len(delta) or not np.isfinite(delta).all():
                    raise ValueError('Every leg must have finite contiguous increments.')
                increments.append(delta)
                rows.append(dict(mode=mode, leg_id=record['leg_id'], split=record['split'],
                                 horizon_s=horizon, n_origins=len(delta), excluded_gap_origins=int((~valid).sum()),
                                 zero_fraction=float(np.mean(delta == 0)),
                                 median=float(np.median(delta)), q95=float(np.quantile(delta, .95)),
                                 maximum=float(delta.max())))
            groups[mode, horizon] = increments
    samples, zero_pct = [], {}
    for (mode, horizon), increments in groups.items():
        for values in increments:
            positive = np.sort(values[values > 0])
            if not len(positive):
                raise ValueError('A fully constant leg cannot define a positive-change violin.')
            indices = np.linspace(0, len(positive) - 1, 2048).astype(int)
            samples.append(pd.DataFrame({'mode': mode, 'horizon_s': horizon, 'change': positive[indices]}))
        zero_pct[mode, horizon] = 100 * np.mean([np.mean(values == 0) for values in increments])
    violin_data = pd.concat(samples, ignore_index=True)
    leg_data = pd.DataFrame(rows)
    smallest, largest = violin_data.change.min(), violin_data.change.max()

    def style(ax, letter):
        ax.set_xlabel('Lead $h$ (s)')
        ax.set_axisbelow(True)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', color='#E4E6E7', linewidth=.6)
        sns.despine(ax=ax, offset=4)

    def violins(ax):
        sns.violinplot(data=violin_data, x='horizon_s', y='change', hue='mode',
                       order=list(horizons), hue_order=list(modes), palette=pale,
                       split=True, inner='quart', cut=0, bw_method='scott', bw_adjust=.75,
                       density_norm='area', common_norm=True, log_scale=(False, True),
                       linewidth=.8, saturation=1, width=.85, gap=.04, gridsize=160,
                       legend=False, ax=ax)
        ax.set_ylim(smallest * .75, largest * 1.7)
        ax.set_ylabel(r'$|y_{t+h}-y_t|$ (PU)')
        style(ax, 'a')

    def boxes(ax):
        sns.boxplot(data=leg_data, x='horizon_s', y='q95', hue='mode',
                    order=list(horizons), hue_order=list(modes), palette=pale,
                    width=.7, gap=.2, whis=1.5, showfliers=False, linewidth=.85,
                    saturation=1, legend=False, ax=ax)
        state = np.random.get_state()
        try:
            np.random.seed(24601)
            sns.stripplot(data=leg_data, x='horizon_s', y='q95', hue='mode',
                          order=list(horizons), hue_order=list(modes), palette=colours,
                          dodge=True, jitter=.13, size=3.1, linewidth=.35,
                          edgecolor='white', legend=False, ax=ax)
        finally:
            np.random.set_state(state)
        ax.set_ylim(0, leg_data.q95.max() * 1.12)
        ax.set_ylabel('95th percentile of\n' + r'$|y_{t+h}-y_t|$ (PU)')
        style(ax, 'b')

    def legend(figure):
        handles = [Line2D([], [], marker='s', markerfacecolor=pale[mode],
                          markeredgecolor=colours[mode], linestyle='none', markersize=6,
                          label=f'{mode.capitalize()} ({len(groups[mode, horizons[0]])} legs)')
                   for mode in modes]
        figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, 1.005),
                      ncol=2, frameon=False, fontsize=8, handletextpad=.4, columnspacing=1.7)

    rc = {'font.family': 'Times New Roman', 'font.size': 9, 'axes.labelsize': 9,
          'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': .65,
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'grid.alpha': 1.,
          'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
          'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
          'text.usetex': False, 'axes.formatter.use_mathtext': True}
    with sns.axes_style('whitegrid'), mpl.rc_context(rc):
        combined, axes = plt.subplots(1, 2, figsize=(7., 3.35))
        combined.subplots_adjust(left=.10, right=.985, bottom=.17, top=.87, wspace=.32)
        violins(axes[0])
        boxes(axes[1])
        legend(combined)
        figures = {'forecast_change_distributions': combined}
        for name, draw in (('forecast_change_violins', violins), ('forecast_change_boxplots', boxes)):
            figure, ax = plt.subplots(figsize=(3.6, 3.35))
            figure.subplots_adjust(left=.20, right=.98, bottom=.17, top=.87)
            draw(ax)
            legend(figure)
            figures[name] = figure
    notes = {'seaborn_version': sns.__version__, 'bw_method': 'scott', 'bw_adjust': .75,
             'positive_observed_ranks_per_leg': 2048, 'leg_weighting': 'equal; positive changes conditional within each leg',
             'zero_percentage': {f'{mode}_{horizon}s': value for (mode, horizon), value in zero_pct.items()}}
    return figures, rows, notes






def plot_load_dynamics_diagnostics(per_leg):
    """Compact horizontal boxes and observed legs for the four paper diagnostics."""
    import seaborn as sns
    from matplotlib import font_manager
    from matplotlib.ticker import ScalarFormatter, FixedLocator, FuncFormatter

    font_manager.findfont('Times New Roman', fallback_to_default=False)
    modes = ['manoeuvre', 'navigation']
    colours = {'manoeuvre': '#4C72B0', 'navigation': '#DD8452'}
    definitions = [
        ('ac_decay_s', 'Autocorrelation decay time (s)'),
        ('ramp_q95_pu_s', '95th-percentile ramp rate (PU/s)'),
        ('peaks_per_min', 'Peak density (peaks/min)'),
        ('lf_hf_ratio', 'Low/high-frequency energy ratio')]
    if len(per_leg) != 24 or per_leg.groupby('mode').size().to_dict() != {'manoeuvre': 16, 'navigation': 8}:
        raise ValueError('The figure requires all 24 benchmark legs.')
    per_leg = per_leg.copy()
    per_leg['row'] = per_leg['mode'].map({'manoeuvre': .19, 'navigation': 1.19})

    def observed_swarm(ax, frame, column, native=False):
        if len(frame):
            sns.swarmplot(data=frame, x=column, y='row' if native else 'mode',
                          hue='mode', hue_order=modes, palette=colours, orient='h',
                          native_scale=native, order=None if native else modes,
                          size=3.0, edgecolor='white', linewidth=.3, legend=False,
                          warn_thresh=.01, ax=ax, zorder=4)

    def panel(ax, column, label):
        if column == 'lf_hf_ratio':
            ax.set_xscale('log')
        if column == 'ac_decay_s':
            uncensored = per_leg.loc[per_leg[column] < 300]
            observed_swarm(ax, uncensored, column)
            for index, mode in enumerate(modes):
                capped = per_leg.loc[(per_leg['mode'] == mode) & (per_leg[column] >= 300)]
                positions = index + np.linspace(-.29, .29, len(capped))
                ax.scatter(np.full(len(capped), 300.), positions, marker='>', s=11,
                           color=colours[mode], edgecolor='white', linewidth=.25,
                           zorder=5, clip_on=False)
            ax.axvline(300, color='#B7BEC4', linewidth=.65,
                       linestyle=(0, (2.5, 2.5)), zorder=0)
            ax.set_xlim(90, 316)
            ax.set_xticks([100, 200, 300], ['100', '200', r'$\geq 300$'])
            ax.set_yticks([0, 1], ['Manoeuvre', 'Navigation'])
        else:
            sns.boxplot(data=per_leg, x=column, y='mode', hue='mode',
                        hue_order=modes, order=modes, palette=colours,
                        orient='h', width=.20, fill=False, showfliers=False,
                        linewidth=.75, legend=False, whis=1.5,
                        medianprops={'linewidth': 1.25},
                        capprops={'linewidth': .65},
                        whiskerprops={'linewidth': .65}, ax=ax, zorder=2)
            observed_swarm(ax, per_leg, column, native=True)
            ax.set_yticks([.08, 1.08], ['Manoeuvre', 'Navigation'])
            if column == 'ramp_q95_pu_s':
                ax.set_xlim(0, .0225)
                ax.set_xticks([0, .005, .010, .015, .020])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: '0' if value == 0 else f'{value:.3f}'))
            elif column == 'peaks_per_min':
                ax.set_xlim(0, .61)
                ax.set_xticks([0, .2, .4, .6])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}'))
            else:
                ax.set_xlim(16, 1100)
                ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, _: ''))
        ax.set_ylim(1.48, -.48)
        ax.set_xlabel(label, fontsize=8.2, labelpad=6)
        ax.set_ylabel('')
        ax.set_axisbelow(True)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color='#EEF0F2', linewidth=.45)
        ax.tick_params(axis='x', labelsize=7.5, length=2.5, width=.5, color='#9AA4AD')
        ax.tick_params(axis='x', which='minor', length=1.5, width=.4, color='#B5BDC4')
        ax.tick_params(axis='y', labelsize=7.8, length=0, pad=6)
        sns.despine(ax=ax, left=True, offset={'bottom': 3})
        ax.spines['bottom'].set_color('#BAC2C8')
        ax.spines['bottom'].set_linewidth(.55)

    rc = {'font.family': 'Times New Roman', 'font.size': 8,
          'axes.labelsize': 8.2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.8,
          'axes.linewidth': .55, 'pdf.fonttype': 42, 'ps.fonttype': 42,
          'grid.alpha': 1., 'mathtext.fontset': 'custom',
          'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
          'mathtext.bf': 'Times New Roman:bold', 'text.usetex': False,
          'figure.facecolor': 'white', 'axes.facecolor': 'white',
          'savefig.facecolor': 'white'}
    with sns.axes_style('white'), mpl.rc_context(rc):
        figure, axes = plt.subplots(2, 2, figsize=(7., 3.7))
        figure.subplots_adjust(left=.105, right=.985, bottom=.13, top=.97,
                               wspace=.39, hspace=.58)
        for ax, (column, label) in zip(axes.ravel(), definitions):
            panel(ax, column, label)
    return figure


def plot_campaign_composition(records, arrays, leg_metadata, figsize=(7., 6.5)):
    """Display every reconstructed load profile in manuscript voyage order.

    leg_metadata maps (mode, leg_id) to voyage_index and manoeuvre_order using
    the unique native record counts in the manuscript Appendix. Plot vertices
    retain extrema in consecutive bins; timestamp gaps are never connected.
    """
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator, NullLocator
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    if len(leg_metadata) != 24 or leg_metadata.voyage_index.nunique() != 8:
        raise ValueError('The campaign overview requires all 24 legs and 8 voyages.')
    lookup = {(r['mode'], int(r['leg_id'])): r['key'] for r in records}
    columns = ['Manoeuvre - first', 'Navigation', 'Manoeuvre - second']
    colours = {'manoeuvre': '#4C72B0', 'navigation': '#DD8452'}

    def retained_vertices(times, values, max_points=500):
        breaks = np.r_[0, np.flatnonzero(~np.isclose(np.diff(times), 1.)) + 1, len(times)]
        result_x, result_y = [], []
        for start, stop in zip(breaks[:-1], breaks[1:]):
            length = stop - start
            bins = min(length, max(1, int(np.ceil(max_points * length / len(times) / 2))))
            edges = np.linspace(start, stop, bins + 1).astype(int)
            indices = [start, stop - 1]
            for left, right in zip(edges[:-1], edges[1:]):
                if right > left:
                    segment = values[left:right]
                    indices.extend((left + int(np.argmin(segment)), left + int(np.argmax(segment))))
            indices = np.unique(indices)
            result_x.extend(times[indices] / 3600.)
            result_y.extend(values[indices])
            result_x.append(np.nan)
            result_y.append(np.nan)
        return np.asarray(result_x), np.asarray(result_y)

    all_min = min(float(np.min(arrays[r['key'] + '_y'])) for r in records)
    all_max = max(float(np.max(arrays[r['key'] + '_y'])) for r in records)
    lower = np.floor(all_min * 10) / 10
    upper = np.ceil(all_max * 10) / 10
    rc = {'font.family': 'Times New Roman', 'font.size': 9, 'axes.labelsize': 9,
          'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': .45,
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.usetex': False,
          'figure.facecolor': 'white', 'savefig.facecolor': 'white'}
    with mpl.rc_context(rc):
        figure, axes = plt.subplots(8, 3, figsize=figsize, sharey=True)
        figure.subplots_adjust(left=.13, right=.99, bottom=.075, top=.93,
                               wspace=.12, hspace=.24)
        for column, header in enumerate(columns):
            bounds = axes[0, column].get_position()
            figure.text(bounds.x0 + bounds.width / 2, .966, header,
                        ha='center', va='center', fontsize=9)
        for row in leg_metadata.itertuples(index=False):
            column = 1 if row.mode == 'navigation' else (0 if row.manoeuvre_order == 1 else 2)
            ax = axes[int(row.voyage_index) - 1, column]
            key = lookup[row.mode, int(row.leg_id)]
            times = arrays[key + '_seconds']
            values = arrays[key + '_y']
            x, y = retained_vertices(times, values)
            duration = float(times[-1]) / 3600.
            ax.plot(x, y, color=colours[row.mode], linewidth=.45,
                    solid_capstyle='round', solid_joinstyle='round')
            ax.set(xlim=(0, duration), ylim=(lower, upper))
            ax.xaxis.set_major_locator(FixedLocator([duration]))
            ax.set_xticklabels([f'{duration:.1f}'])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(FixedLocator([1., 2.]))
            ax.tick_params(axis='x', length=2, width=.45, pad=1.5, color='#8B8B8B')
            ax.tick_params(axis='y', length=2 if column == 0 else 0, width=.45,
                           pad=2, labelleft=column == 0, color='#8B8B8B')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines['left'].set_visible(column == 0)
            ax.spines['bottom'].set_color('#B8B8B8')
            ax.spines['left'].set_color('#B8B8B8')
            ax.grid(False)
        for row, voyage in enumerate(range(1, 9)):
            axes[row, 0].text(-.24, .5, f'V{voyage}', transform=axes[row, 0].transAxes,
                              ha='right', va='center', fontsize=8)
        figure.text(.018, .515, 'Active power (PU)', rotation=90,
                    ha='center', va='center', fontsize=9)
        figure.text(.56, .022, 'Time from leg start (h)', ha='center', va='center', fontsize=9)
    return figure
