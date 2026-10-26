# Script workflows

The scripts are grouped by purpose. Run commands from the repository root.

## Publication benchmark

```bash
python scripts/tune_hyperparams.py --preset manoeuvre_paper --windows-dir outputs/runs/windows_manoeuvre_paper --n-trials 40
python scripts/run_seed_sweep.py --preset manoeuvre_paper --backend flax --seeds 0,1,2,3,4 --model-specs config/model_specs_tuned_manoeuvre_paper.json
python scripts/aggregate_seed_results.py --run-root outputs/runs/sweep_manoeuvre_paper_flax
python scripts/run_significance.py --run-root outputs/runs/sweep_manoeuvre_paper_flax
```

Repeat the commands with the navigation preset where applicable.

## Leave-one-voyage-out cross-validation

One fold per leg: leg `k` alone is the test set, the last two of the remaining
legs are validation, the rest is training. Used for the manoeuvre regime, where
the reported single split draws its upper-tail metrics from only three legs.

```bash
bash scripts/lovo_manoeuvre.sh                    # 16 folds, 7 models, ~1 h on GPU
LOVO_FOLDS=2 LOVO_MODELS=tcn bash scripts/lovo_manoeuvre.sh    # rehearsal
LOVO_STAGE=post bash scripts/lovo_manoeuvre.sh    # re-derive the tables only
```

The driver wraps `run_seed_sweep.py --lovo-cross-val N`, then the per-fold peak
and persistence metrics, then `collect_lovo_results.py`, which reports for each
model in how many held-out voyages it beats persistence.

## Foundation models

Foundation inference uses the optional foundation-model environment described in
the main README. The default roster is Chronos-2, TimesFM, and univariate TiRex-2.

```bash
python scripts/run_foundation_models.py --selector both --models chronos2,timesfm,tirex2_univariate --skip-existing
python postprocess_figures.py --kind foundation_full_test
python scripts/smoke_test_foundation.py
```

The full-test figure command uses the publication `L45/H25` archives by default.
Override the geometry or suffix when using another archive set:

```bash
python postprocess_figures.py --kind foundation_full_test --foundation-lookback 60 --foundation-horizon 25 --foundation-archive-tag flax8_legsafe
```

## Figures

```bash
python postprocess_figures.py --kind figures_4papers
python postprocess_figures.py --kind inference
python postprocess_figures.py --kind all
```

`gen_optuna_figures.py` is run separately after an Optuna study:

```bash
python scripts/gen_optuna_figures.py --storage sqlite:///outputs/tuning/optuna.db
```

## Statistical and operational diagnostics

```bash
python scripts/compute_leg_bootstrap_ci.py --archive <archive>
python scripts/compute_peak_metrics.py --archive <archive> --model-specs <model-specs.json>
python scripts/compute_persistence_peak_metrics.py --all-legsafe
```

The remaining scripts are specialized diagnostics and are not required for the publication path above.

## Paper figures and tables

Every command needs the restricted operational data and the run archives of the
commands above. The synthetic demo files only exercise the code path.

| Paper content | Command |
|---|---|
| Data characterisation: split, autocorrelation, spectrum, ramps, lead time | `python postprocess_figures.py --kind data_characterisation` |
| Error per horizon, lookback/horizon heatmaps, MAPE summaries, full-test trajectories | `python postprocess_figures.py --kind figures_4papers` |
| Foundation-model full-test trajectories | `python postprocess_figures.py --kind foundation_full_test` |
| Seed-sweep spread | `python scripts/gen_seed_sweep_figures.py` |
| Inference latency | `python measure_inference_latency.py --threads 1core`, then `python postprocess_figures.py --kind inference` |
| COAST weights and skill | `python scripts/fusion_mixture_replay.py`, then `python scripts/gen_fusion_figures.py` |
| Diebold-Mariano tests | `python scripts/run_significance.py` |
| Leave-one-voyage-out tables | `bash scripts/lovo_manoeuvre.sh` |
| Bootstrap confidence intervals and peak metrics | `compute_leg_bootstrap_ci.py`, `compute_peak_metrics.py`, `compute_persistence_peak_metrics.py` |
