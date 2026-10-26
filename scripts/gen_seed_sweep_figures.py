"""Generate multi-seed aggregate and horizon-wise MAPE paper figures.

Ported from the iSCSS-2026 figure framework (ML_load-profiles
scripts/_regen_bar_error_mape.py) and adapted to the APEN benchmark roster.

Two figures per operating mode, in the publication style used across the paper
(Times New Roman 9 pt, EPS + PNG):

- seed_bar_mape_<suffix>.eps   : per-model mean MAPE with across-seed sample
  std as error bars, individual seeds overplotted as points.
- seed_error_mape_<suffix>.eps : horizon-wise MAPE, mean line with +/- 1 std
  band, one series per architecture.

The seed_ prefix is required, not cosmetic: the manuscript img/ directory already
holds error_mape_{man,nav}.eps from the benchmark error-profile figure. The
lookback/horizon are carried by the caption and by the summary CSVs, not by the
filename, because each sweep writes into its own output directory.

Two summary tables per mode:

- mape_seed_summary_<suffix>.csv    : per-model mean/std of aggregate MAPE.
- mape_horizon_summary_<suffix>.csv : per-model, per-lead mean/std plus the raw
  per-seed values.

Every seed run is validated against run_summary.json before use: the selector,
lookback and horizon must match, and --require-model-specs asserts that the run
used the intended specs file. That last check exists because a previous sweep
was executed with config/model_specs_tuned_*_paper.json while the reported
benchmark grid used config/model_specs_benchmark_*.json; the two are not
comparable and the mismatch was invisible in the metrics alone.

Example
-------
python scripts/gen_seed_sweep_figures.py \\
    --run-root outputs/runs/sweep_manoeuvre_benchmark_flax \\
    --selector manoeuvre --lookback 60 --horizon 25 --expected-seeds 5 \\
    --require-model-specs config/model_specs_benchmark_manoeuvre.json \\
    --out-dir outputs/figures/paper/seed_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the package reader when launched directly from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_forecasting.metrics import load_metrics_outputs, parse_metric_vector as parse_vec  # noqa: E402

# Publication style, identical to load_forecasting.plots.set_pub_style and to
# the iSCSS-2026 figure set. Kept inline so this script is self-contained.
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
    'grid.color':       '#ebebeb',
    'grid.alpha':       1.0,
    'ps.fonttype':      3,
    'pdf.fonttype':     42,
    'figure.constrained_layout.use': False,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.02,
})

# Row labels as written by train_load_forecasters.py into results_metrics.csv.
# DLinear is trained by the benchmark but is deliberately not reported in the
# APEN manuscript; it is excluded here so the figures match the paper roster.
MODELS = [
    'CNN',
    'TCN',
    'Transformer',
    'iTransformer',
    'DualStream-FiLM',
    'ConvTCN-H',
    'ConvLSTMTCN-H',
]

COLORS = {
    'CNN':             '#1f77b4',
    'TCN':             '#d62728',
    'Transformer':     '#2ca02c',
    'iTransformer':    '#9467bd',
    'DualStream-FiLM': '#8c564b',
    'ConvTCN-H':       '#ff7f0e',
    'ConvLSTMTCN-H':   '#17becf',
    'DLinear':         '#8c8c22',
}

LSTYLES = {
    'CNN':             '-',
    'TCN':             '--',
    'Transformer':     '-.',
    'iTransformer':    ':',
    'DualStream-FiLM': (0, (3, 1, 1, 1)),
    'ConvTCN-H':       (0, (5, 2)),
    'ConvLSTMTCN-H':   (0, (1, 1)),
    'DLinear':         (0, (6, 2, 1, 2)),
}

# Axis and legend labels. The keys above are the row labels written into
# results_metrics.csv, which come from the "label" field of the model specs and
# still carry the -H hybrid suffix. The manuscript dropped that suffix, so the
# figures are relabelled at plotting time rather than by editing the specs,
# which would break the match against every archived results_metrics.csv.
DISPLAY = {
    'ConvTCN-H':     'ConvTCN',
    'ConvLSTMTCN-H': 'ConvLSTM-TCN',
}


def label(model):
    return DISPLAY.get(model, model)


SUFFIXES = {'manoeuvre': 'man', 'navigation': 'nav'}


def load_seed_metrics(run_root, selector, lookback, horizon, expected_seeds, models,
                      require_model_specs=None):
    seed_dirs = [run_root / f'seed{seed}' for seed in range(expected_seeds)]
    missing = [str(path) for path in seed_dirs
               if not any((path / f'results_metrics.{ext}').is_file() for ext in ('parquet', 'csv'))]
    if missing:
        raise FileNotFoundError(f'Missing completed seed results: {missing}')

    scalar = {model: [] for model in models}
    profiles = {model: [] for model in models}
    dataset_seen = {}
    for seed_dir in seed_dirs:
        summary_path = seed_dir / 'run_summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'Missing run summary: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        observed = (summary.get('selector'), summary.get('lookback_steps'), summary.get('horizon_steps'))
        expected = (selector, lookback, horizon)
        if observed != expected:
            raise ValueError(f'{seed_dir.name} configuration {observed} != expected {expected}')

        if require_model_specs is not None:
            used = str(summary.get('model_specs_file', ''))
            if Path(used).name != Path(require_model_specs).name:
                raise ValueError(
                    f'{seed_dir.name} was trained with model_specs_file={used!r}, '
                    f'expected {require_model_specs!r}. Refusing to plot a sweep that '
                    f'does not match the benchmark configuration.'
                )

        # Dataset provenance. The model-specs check above does NOT catch a run
        # trained on the synthetic demo data, because the demo runs use the very
        # same specs file. Reject demo data outright, and require every seed to
        # share one dataset: a run root that was not deleted before relaunching
        # silently mixes generations, and the mixture is invisible in the metrics.
        input_files = tuple(summary.get('input_files') or ())
        if not input_files:
            raise ValueError(f'{seed_dir.name}: run_summary.json records no input_files.')
        if any('demo' in f for f in input_files):
            raise ValueError(
                f'{seed_dir.name} was trained on the synthetic demo dataset '
                f'({list(input_files)}). Refusing to plot it.'
            )
        dataset_seen.setdefault(input_files, []).append(seed_dir.name)

        frame = load_metrics_outputs(seed_dir, expected_horizon=horizon)
        for model in models:
            if model not in frame.index:
                raise ValueError(f'{seed_dir.name} is missing model {model}')
            mape_all = float(frame.loc[model, 'MAPE_all'])
            mape_horizon = frame.loc[model, 'MAPE_horizon']
            if not np.isfinite(mape_all):
                raise ValueError(f'{seed_dir.name}/{model} has non-finite MAPE_all')
            if mape_horizon.shape != (horizon,) or not np.all(np.isfinite(mape_horizon)):
                raise ValueError(
                    f'{seed_dir.name}/{model} MAPE_horizon shape={mape_horizon.shape}; '
                    f'expected ({horizon},) with finite values'
                )
            scalar[model].append(mape_all)
            profiles[model].append(mape_horizon)

    if len(dataset_seen) > 1:
        detail = '; '.join(
            f'{seeds} -> {len(files)} files starting {files[0]}'
            for files, seeds in dataset_seen.items()
        )
        raise ValueError(
            f'Seed runs under {run_root} were trained on different datasets: {detail}. '
            f'Delete the run root and retrain the whole sweep; a partially '
            f'overwritten directory cannot be aggregated.'
        )

    scalar = {model: np.asarray(values, dtype=np.float64) for model, values in scalar.items()}
    profiles = {model: np.stack(values) for model, values in profiles.items()}
    return scalar, profiles


def sample_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def save_summaries(scalar, profiles, out, suffix, models):
    scalar_rows = []
    horizon_rows = []
    for model in models:
        mean, std = sample_mean_std(scalar[model])
        scalar_rows.append({'Model': model, 'Seeds': len(scalar[model]), 'MAPE_mean': mean, 'MAPE_std': std})
        mean_h, std_h = sample_mean_std(profiles[model])
        for step, (step_mean, step_std) in enumerate(zip(mean_h, std_h), start=1):
            row = {'Model': model, 'HorizonStep': step, 'MAPE_mean': step_mean, 'MAPE_std': step_std}
            row.update({f'seed{seed}': value for seed, value in enumerate(profiles[model][:, step - 1])})
            horizon_rows.append(row)
    pd.DataFrame(scalar_rows).to_csv(out / f'mape_seed_summary_{suffix}.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(out / f'mape_horizon_summary_{suffix}.csv', index=False)


def save_figure(fig, out, stem):
    fig.savefig(out / f'{stem}.eps', format='eps')
    fig.savefig(out / f'{stem}.png', format='png', dpi=300)
    plt.close(fig)
    print('saved', f'{stem}.eps/.png')


def bar_chart(scalar, out, lookback, horizon, suffix, models):
    means, stds = zip(*(sample_mean_std(scalar[model]) for model in models))
    colors = [COLORS[m] for m in models]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    bars = ax.bar(
        range(len(models)), means, yerr=stds, capsize=4,
        color=colors, edgecolor='black', linewidth=0.5,
        error_kw={'ecolor': 'black', 'elinewidth': 0.9, 'capthick': 0.9},
    )
    offsets = np.linspace(-0.13, 0.13, len(next(iter(scalar.values()))))
    for position, model in enumerate(models):
        ax.scatter(
            position + offsets, scalar[model], s=14, color=COLORS[model],
            edgecolor='black', linewidth=0.45, zorder=3,
        )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([label(m) for m in models], rotation=30, ha='right')
    ax.set_ylabel('MAPE (%)')
    ax.grid(axis='y')
    ceiling = max(
        float(np.max(np.asarray(means) + np.asarray(stds))),
        float(max(np.max(scalar[model]) for model in models)),
    )
    label_offset = 0.03 * ceiling
    for bar, model, mean, std in zip(bars, models, means, stds):
        # Clear both the error-bar cap and the highest individual seed point.
        top = max(mean + std, float(np.max(scalar[model])))
        ax.text(
            bar.get_x() + bar.get_width() / 2, top + label_offset,
            f'{mean:.2f} $\\pm$ {std:.2f}%', ha='center', va='bottom', fontsize=7,
        )
    # Headroom so the mean +/- std annotations never collide with a seed point.
    ax.set_ylim(0.0, ceiling * 1.18)
    fig.tight_layout()
    # Stems match the filenames used by the manuscript img/ directory. They are
    # deliberately prefixed: img/ already contains error_mape_{man,nav}.eps from
    # the benchmark error-profile figure, and an unprefixed name would overwrite it.
    save_figure(fig, out, f'seed_bar_mape_{suffix}')


def error_profile_mape(profiles, out, horizon, suffix, models):
    steps = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for model in models:
        mean, std = sample_mean_std(profiles[model])
        shade = tuple(
            0.92 + 0.08 * channel
            for channel in matplotlib.colors.to_rgb(COLORS[model])
        )
        ax.fill_between(
            steps, mean - std, mean + std, color=shade,
            alpha=1.0, linewidth=0,
        )
        ax.plot(steps, mean, color=COLORS[model], ls=LSTYLES[model], label=label(model))
    ax.set_xlabel('Horizon step (s)')
    ax.set_ylabel('MAPE (%)')
    ax.set_xlim(1, horizon)
    ax.set_xticks(np.arange(0, horizon + 1, 5))
    # Curves rise monotonically with the lead, so the lower-right corner is the
    # only region the seven series and their bands never occupy.
    ax.legend(fontsize=6.5, frameon=True, framealpha=1.0, ncol=2, loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, out, f'seed_error_mape_{suffix}')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--selector', choices=tuple(SUFFIXES), required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=25)
    parser.add_argument('--expected-seeds', type=int, default=5)
    parser.add_argument(
        '--models', default=None,
        help='Comma-separated results_metrics.csv row labels (default: the paper roster).',
    )
    parser.add_argument(
        '--require-model-specs', default=None,
        help='Assert every seed run recorded this model-specs filename in run_summary.json.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(',')] if args.models else list(MODELS)
    unknown = [m for m in models if m not in COLORS]
    if unknown:
        raise SystemExit(f'No colour/linestyle defined for: {unknown}')

    out = args.out_dir or args.run_root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    suffix = SUFFIXES[args.selector]
    scalar, profiles = load_seed_metrics(
        args.run_root, args.selector, args.lookback, args.horizon,
        args.expected_seeds, models, args.require_model_specs,
    )
    save_summaries(scalar, profiles, out, suffix, models)
    bar_chart(scalar, out, args.lookback, args.horizon, suffix, models)
    error_profile_mape(profiles, out, args.horizon, suffix, models)
    print('done')


if __name__ == '__main__':
    main()
