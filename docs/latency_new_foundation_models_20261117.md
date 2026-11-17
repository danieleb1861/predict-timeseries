# New foundation models: latency

Native Windows, Python 3.11.9, PyTorch 2.9.1+cpu, float32; same paper script and test windows. L=60, H=25, batch=1. Each model in a fresh process; 100 warm-up and 3600 timed calls. One-core runs pinned to logical CPU 2, one intra-op thread. No compilation or sign-flip augmentation. Host input to host median forecast in PU, including native preprocessing and transfers. Framework-default thread runs are reported separately.

| Configuration | Model | Mean ms | p99 ms | Max ms | >=100 ms | CPU ms |
|---|---|---:|---:|---:|---:|---:|
| manoeuvre_1core | Timestral-209M (0-shot) | 173.247 | 191.475 | 197.846 | 3600/3600 | 160.156 |
| manoeuvre_1core | t0-gamma (0-shot) | 215.830 | 240.173 | 249.875 | 3533/3600 | 214.062 |
| navigation_1core | Timestral-209M (0-shot) | 175.369 | 200.252 | 204.738 | 3551/3600 | 191.406 |
| navigation_1core | t0-gamma (0-shot) | 240.621 | 251.942 | 258.651 | 3600/3600 | 237.500 |
| manoeuvre_default | Timestral-209M (0-shot) | 101.780 | 107.172 | 115.307 | 2880/3600 | 1197.656 |
| manoeuvre_default | t0-gamma (0-shot) | 115.838 | 121.386 | 126.673 | 3591/3600 | 1396.875 |
| navigation_default | Timestral-209M (0-shot) | 101.839 | 106.914 | 112.065 | 2956/3600 | 1259.375 |
| navigation_default | t0-gamma (0-shot) | 115.927 | 120.280 | 126.637 | 3600/3600 | 1414.062 |

Verification: 28,800 finite positive raw call times; per-model call indices, sample counts, warm-up counts, window dimensions, timing flags, platform and one-core affinity/thread metadata checked; summary means and maxima recomputed from raw CSVs. Full commands, runtime versions, source hashes and checkpoint revisions are in campaign.json. The original paper result tables are preserved; combined tables and the thesis figure are generated separately.

TimeH2O is excluded because access to the checkpoint is not authorised. Timings do not establish forecasting accuracy or decision value. Hardware/OS load can affect latency; these are empirical measurements, not worst-case execution-time guarantees.

## Reproduction and provenance

Copy `outputs/latency/new_fm_20261117/run_campaign.py` into a fresh directory under `outputs/latency`, then run that copy with the native Windows environment `.venv-latency-fm2026`. It inherits PyTorch 2.9.1+cpu and NumPy 2.1.3 from the original paper environment and isolates the new model runtimes. It is dedicated to these two models; the original environment remains available for the older foundation models. Local checkpoint cache: `D:/Programs/hf-cache/hub`.

The 15 November campaign is excluded because concurrent workload was reported. Only the new 17 November campaign is used here. Raw call times, per-model metadata, package versions, source hashes and commands are retained in `outputs/latency/new_fm_20261117/`.

The combined tables use the existing paper measurements for the original eleven predictors and the new measurements for Timestral-209M and t0-gamma. The new filenames end in `_fm20261117`; the original tables are preserved. Each combined metadata file records the constituent CSVs and their hashes.

Figure command: `python scripts/gen_latency_pair_figure.py --family both --style range --both-stem standard_1core_fm20261117 --out outputs/figures/paper/latency_cpu_flax/inference_latency_all_bothmodes_standard_1core_fm20261117`.

Prediction sanity checks passed for both models in both modes: finite float32 arrays of shape (1, 25), with covariate input rejected. Input-window and prediction hashes are saved in `prediction_checks.json`; verification scripts are retained alongside the results.
