#!/usr/bin/env bash
# Leave-one-voyage-out cross-validation on the manoeuvre regime, for the APEN
# manuscript (reviewer request: "leave-one-voyage-out, reduced to a few models;
# the test windows are few and heavily overlapping").
#
# Why manoeuvre only
# ------------------
# The reported manoeuvre split has 4183 test windows drawn from 3 legs, and the
# upper-tail metrics rest on the excursions inside those 3 legs alone. Holding
# out one leg at a time turns "positive skill on this split" into "positive
# skill on k of N held-out voyages", which is the claim the manuscript needs.
# Navigation is not cross-validated here: its training set is 30x larger per
# fold, and its aggregate result (persistence is not beaten) does not hinge on
# which legs are held out.
#
# Cost. One fold trains on ~15/16 of the manoeuvre record instead of 11/16, so
# a fold costs roughly 1.4x a seed of sweep_benchmark_seeds.sh: ~3.5 min for the
# seven architectures on an RTX-class GPU, hence ~1 h for 16 folds, plus ~10 min
# of post-processing. Disk: each fold keeps its own windows.npz and model
# artifacts, so budget ~1-2 GB for the whole run root.
#
# Execution is WSL-native. Never point training at /mnt/d: the 9p mount makes
# checkpoint and window I/O the bottleneck. Sync artifacts back afterwards with
# scripts/sync_outputs_to_windows.sh.
#
# Usage
# -----
#   cd ~/predict-timeseries
#   bash scripts/lovo_manoeuvre.sh                    # all folds, all 7 models, GPU
#   LOVO_MODELS=tcn,dual_stream_film bash scripts/lovo_manoeuvre.sh
#   LOVO_FOLDS=4 bash scripts/lovo_manoeuvre.sh       # short rehearsal
#   LOVO_STAGE=post bash scripts/lovo_manoeuvre.sh    # re-derive tables only
#
# Detached:
#   nohup bash scripts/lovo_manoeuvre.sh > outputs/runs/lovo_manoeuvre.log 2>&1 &
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------ single-instance
# Concurrent invocations would train into the same fold directory and interleave
# the same log. A separate lock from sweep_benchmark_seeds.sh, so the two can be
# chained in one overnight command without either blocking the other.
LOVO_LOCK="${LOVO_LOCK:-$REPO_ROOT/outputs/runs/.lovo_manoeuvre.lock}"
mkdir -p "$(dirname "$LOVO_LOCK")"
exec 9>"$LOVO_LOCK"
if ! flock -n 9; then
    echo "FATAL: another LOVO run already holds $LOVO_LOCK."
    pgrep -af 'run_seed_sweep|train_load_forecasters' || true
    exit 1
fi

# ---------------------------------------------------------------- configuration
LOVO_MODE="${LOVO_MODE:-manoeuvre}"
LOVO_TAG="${LOVO_TAG:-benchmark}"
LOVO_DEVICE="${LOVO_DEVICE:-gpu}"          # gpu | cpu
LOVO_STAGE="${LOVO_STAGE:-all}"            # all | train | post
LOVO_SEEDS="${LOVO_SEEDS:-0}"              # one seed per fold; the fold is the replicate
LOVO_DETERMINISTIC="${LOVO_DETERMINISTIC:-1}"
LOVO_FOLDS="${LOVO_FOLDS:-}"               # empty: one fold per leg file of the selector
LOVO_CONFIG="${LOVO_CONFIG:-$REPO_ROOT/config/my_system_config.json}"
LOVO_ROOT="${LOVO_ROOT:-$REPO_ROOT/outputs/runs/lovo_${LOVO_MODE}_${LOVO_TAG}_flax}"
LOVO_MODELS="${LOVO_MODELS:-cnn,tcn,transformer,itransformer,dual_stream_film,convtcn_hybrid,convlstmtcn_hybrid}"
SPECS="config/model_specs_${LOVO_TAG}_${LOVO_MODE}.json"

PY="$REPO_ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1

case "$LOVO_DEVICE" in
    gpu)
        export JAX_PLATFORMS=cuda
        EXPECTED_BACKEND=gpu
        if [ "$LOVO_DETERMINISTIC" = "1" ]; then
            # --xla_gpu_enable_triton_gemm=false is required alongside the
            # deterministic flag: deterministic ops force the Triton GEMM
            # emitter, which cannot handle the dot shapes the TCN produces on
            # the ragged final batch. See sweep_benchmark_seeds.sh.
            export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false"
        fi
        ;;
    cpu)
        export JAX_PLATFORMS=cpu
        EXPECTED_BACKEND=cpu
        ;;
    *)
        echo "FATAL: LOVO_DEVICE must be gpu or cpu, got '$LOVO_DEVICE'"; exit 1 ;;
esac

# ------------------------------------------------------------------- preflight
echo "=== preflight ==="
[ -x "$PY" ] || { echo "FATAL: $PY not found. Create the venv first."; exit 1; }
[ -f "$SPECS" ] || { echo "FATAL: missing $SPECS"; exit 1; }

# Dataset guard. config/default_config.json points at the SYNTHETIC demo data,
# so a run without --config would produce plausible numbers unrelated to the
# vessel. The real leg mapping lives in the gitignored my_system_config.json.
[ -f "$LOVO_CONFIG" ] || {
    echo "FATAL: dataset config not found: $LOVO_CONFIG"
    echo "       It is gitignored. Copy it into this checkout, or set LOVO_CONFIG=<path>."
    exit 1
}
LEG_COUNT="$("$PY" - "$LOVO_CONFIG" "$LOVO_MODE" <<'PY'
import json, sys
cfg_path, mode = sys.argv[1], sys.argv[2]
files = json.load(open(cfg_path))["dataset"]["selectors"].get(mode, [])
if not files:
    raise SystemExit(f"FATAL: selector {mode!r} lists no files in {cfg_path}")
if any("demo" in f for f in files):
    raise SystemExit(f"FATAL: refusing to train on synthetic demo data: {files}")
print(len(files))
PY
)" || exit 1
LOVO_FOLDS="${LOVO_FOLDS:-$LEG_COUNT}"
if [ "$LOVO_FOLDS" -gt "$LEG_COUNT" ]; then
    echo "FATAL: LOVO_FOLDS=$LOVO_FOLDS exceeds the $LEG_COUNT legs of '$LOVO_MODE'."; exit 1
fi

# Split-path assertions. Cheap, and they catch the two ways the --lovo-fold
# patch could go wrong: a changed default split, or a fold that leaks a leg
# across splits.
"$PY" - <<'PY' || { echo "FATAL: split assertions failed."; exit 1; }
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from load_forecasting.splitting import lovo_split, split_train_val_test

frame = pd.DataFrame({
    "leg_id": np.repeat(np.arange(16), 50),
    "act_pow_pu": np.random.default_rng(0).normal(size=800),
})
# Default path untouched: 16 legs, ordered -> 11/2/3 as the manuscript reports.
tr, va, te = split_train_val_test(frame, mode="ordered")
counts = tuple(len(part["leg_id"].unique()) for part in (tr, va, te))
assert counts == (11, 2, 3), f"default ordered split changed: {counts}"
# Every fold: disjoint legs, one test leg, all 16 legs used.
for fold in range(16):
    tr, va, te = lovo_split(frame, fold)
    legs = [set(part["leg_id"].unique()) for part in (tr, va, te)]
    assert legs[2] == {fold}, f"fold {fold}: test legs {legs[2]}"
    assert not (legs[0] & legs[1]) and not (legs[0] & legs[2]) and not (legs[1] & legs[2]), fold
    assert set.union(*legs) == set(range(16)), fold
    assert (len(legs[0]), len(legs[1])) == (13, 2), (fold, len(legs[0]), len(legs[1]))
print("splits    : default ordered 11/2/3 preserved; 16 folds disjoint at 13/2/1")
PY

echo "repo      : $REPO_ROOT"
echo "commit    : $(git rev-parse --short HEAD 2>/dev/null || echo '(not a git checkout)')"
echo "dirty     : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "python    : $("$PY" -c 'import sys; print(sys.version.split()[0])')"

if [ "$LOVO_DEVICE" = "gpu" ]; then
    command -v nvidia-smi >/dev/null || {
        echo "FATAL: LOVO_DEVICE=gpu but nvidia-smi is not on PATH inside WSL."; exit 1; }
    echo "gpu       : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
fi

# Hard assertion: abort rather than cross-validate a whole night on the wrong device.
"$PY" - "$EXPECTED_BACKEND" <<'PY' || { echo "FATAL: JAX backend assertion failed."; exit 1; }
import sys
import jax
expected = sys.argv[1]
backend = jax.default_backend()
print(f"jax       : {jax.__version__}  backend={backend}  devices={jax.devices()}")
if backend != expected:
    print(f"ASSERTION: expected backend {expected!r}, got {backend!r}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "device    : $LOVO_DEVICE (JAX_PLATFORMS=$JAX_PLATFORMS)"
if [ "$LOVO_DEVICE" = "gpu" ]; then echo "xla_flags : ${XLA_FLAGS:-<none>}"; fi
echo "mode      : $LOVO_MODE ($LEG_COUNT legs)"
echo "folds     : $LOVO_FOLDS"
echo "seeds     : $LOVO_SEEDS"
echo "models    : $LOVO_MODELS"
echo "specs     : $SPECS"
echo "run root  : $LOVO_ROOT"
echo "stage     : $LOVO_STAGE"
echo

# ----------------------------------------------------------------------- stages
stage_train() {
    echo "=== train: $LOVO_FOLDS folds -> $LOVO_ROOT ==="
    "$PY" scripts/run_seed_sweep.py \
        --preset "${LOVO_MODE}_paper" \
        --backend flax \
        --seeds "$LOVO_SEEDS" \
        --lovo-cross-val "$LOVO_FOLDS" \
        --models "$LOVO_MODELS" \
        --model-specs "$SPECS" \
        --config "$LOVO_CONFIG" \
        --run-root "$LOVO_ROOT"
}

stage_post() {
    for fold_dir in "$LOVO_ROOT"/fold*; do
        [ -f "$fold_dir/results_metrics.csv" ] || { echo "skip $(basename "$fold_dir"): no metrics"; continue; }
        echo "=== peak metrics: $(basename "$fold_dir") ==="
        # Two upper-tail definitions. The manuscript threshold is the training-set
        # p95, but a single held-out voyage can sit wholly below it, leaving the
        # tail empty for that fold; the held-out voyage's own p95 is 5 % populated
        # by construction. The second pass reuses the cached predictions of the
        # first, so it costs only the metric arithmetic.
        "$PY" scripts/compute_peak_metrics.py --archive "$fold_dir" --model-specs "$SPECS"
        "$PY" scripts/compute_persistence_peak_metrics.py --archive "$fold_dir"
        "$PY" scripts/compute_peak_metrics.py --archive "$fold_dir" --model-specs "$SPECS" --tail-from test
        "$PY" scripts/compute_persistence_peak_metrics.py --archive "$fold_dir" --tail-from test
    done
    echo "=== across-fold dispersion of the aggregate metrics ==="
    "$PY" scripts/aggregate_seed_results.py --run-root "$LOVO_ROOT" --pattern 'fold*'
    echo "=== leave-one-voyage-out summary ==="
    "$PY" scripts/collect_lovo_results.py --run-root "$LOVO_ROOT"
}

case "$LOVO_STAGE" in
    all)   stage_train; stage_post ;;
    train) stage_train ;;
    post)  stage_post ;;
    *) echo "FATAL: unknown LOVO_STAGE=$LOVO_STAGE"; exit 1 ;;
esac

echo
echo "=== done ==="
echo "Per-fold rows : $LOVO_ROOT/lovo_by_fold.csv"
echo "Summary       : $LOVO_ROOT/lovo_summary.csv"
echo "Sync          : PREDICT_TS_WINDOWS_DIR=/mnt/<drive>/path/to/predict-timeseries \\"
echo "                  bash scripts/sync_outputs_to_windows.sh"
