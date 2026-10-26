#!/usr/bin/env bash
# Multi-seed sweep on the *benchmark* configuration, for the APEN manuscript.
#
# Why this script exists
# ----------------------
# The 126-model benchmark grid was trained with config/model_specs_benchmark_*.json.
# An earlier 5-seed sweep was trained with config/model_specs_tuned_*_paper.json:
# a different architecture and optimiser configuration for every model. Its mean
# and std therefore cannot be used as the uncertainty yardstick for the benchmark
# ranking. This script re-runs the sweep on the benchmark specs so that the two
# are on the same basis, and records the specs file in every run_summary.json so
# the mismatch cannot recur silently.
#
# Execution is WSL-native. Never point training at /mnt/d: the 9p mount makes
# checkpoint and window I/O the bottleneck. Sync artifacts back afterwards with
# scripts/sync_outputs_to_windows.sh.
#
# Usage
# -----
#   cd ~/predict-timeseries
#   bash scripts/sweep_benchmark_seeds.sh                 # both modes, seeds 0-4, GPU
#   SWEEP_SEEDS=0,1,2 bash scripts/sweep_benchmark_seeds.sh
#   SWEEP_SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/sweep_benchmark_seeds.sh   # see note below
#   SWEEP_MODES=navigation bash scripts/sweep_benchmark_seeds.sh
#   SWEEP_DEVICE=cpu bash scripts/sweep_benchmark_seeds.sh   # fall back to CPU
#   SWEEP_STAGE=figures bash scripts/sweep_benchmark_seeds.sh # skip training
#
# The run aborts before training if the requested device is not the one JAX
# actually initialises, so a broken CUDA plugin can never yield a silently
# CPU-trained sweep. The device and XLA flags are echoed in the preflight and
# recorded per seed in run_summary.json by collect_run_provenance().
#
# Detached:
#   nohup bash scripts/sweep_benchmark_seeds.sh > outputs/runs/sweep_benchmark.log 2>&1 &
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------ single-instance
# Concurrent invocations train into the same seed directory and interleave the
# same log (the log becomes binary, and no artifact in the run root can be
# trusted). The lock makes a second launch fail immediately instead of colliding
# silently. Held on fd 9 for the life of the script; released on exit.
SWEEP_LOCK="${SWEEP_LOCK:-$REPO_ROOT/outputs/runs/.sweep_benchmark.lock}"
mkdir -p "$(dirname "$SWEEP_LOCK")"
exec 9>"$SWEEP_LOCK"
if ! flock -n 9; then
    echo "FATAL: another sweep already holds $SWEEP_LOCK."
    echo "       Running processes:"
    pgrep -af 'run_seed_sweep|train_load_forecasters' || true
    echo "       Wait for it, or stop it before launching again."
    exit 1
fi

# ---------------------------------------------------------------- configuration
# Seed count note: at n=5 the minimum attainable exact two-sided Wilcoxon
# signed-rank p is 0.0625, so no pairwise comparison can reach p<0.05 whatever
# the effect size. This sweep therefore puts the seed dispersion on the same
# configuration as the reported benchmark, but it does not by itself resolve the
# significance-threshold objection; report the paired t-test p-values that
# scripts/run_significance.py emits alongside wilcoxon_p. n=10 lowers the floor
# to 0.002 at twice the compute.
SWEEP_SEEDS="${SWEEP_SEEDS:-0,1,2,3,4}"
SWEEP_MODES="${SWEEP_MODES:-manoeuvre navigation}"
SWEEP_DEVICE="${SWEEP_DEVICE:-gpu}"        # gpu | cpu
SWEEP_STAGE="${SWEEP_STAGE:-all}"          # all | train | post | figures
# Deterministic GPU reductions: makes each seed bit-reproducible so the reported
# across-seed std is attributable to initialisation and shuffling alone rather
# than to kernel scheduling. Costs some throughput; set to 0 to disable.
SWEEP_DETERMINISTIC="${SWEEP_DETERMINISTIC:-1}"
# Optional VRAM cap (fraction of total). Unset uses JAX's default preallocation
# of ~75%. The benchmark models are small; lower this only to share the GPU.
SWEEP_GPU_MEM_FRACTION="${SWEEP_GPU_MEM_FRACTION:-}"
SWEEP_TAG="${SWEEP_TAG:-benchmark}"
# Dataset configuration. config/default_config.json points at the SYNTHETIC demo
# data since commit f5afdc8 ("prepare public forecasting platform"), so a run
# that does not pass --config trains on demo_manoeuvre.parquet and produces
# numbers that look plausible but are unrelated to the vessel. The real leg
# mapping lives in config/my_system_config.json, which is gitignored and must be
# copied into each checkout by hand.
SWEEP_CONFIG="${SWEEP_CONFIG:-$REPO_ROOT/config/my_system_config.json}"
SWEEP_LOOKBACK="${SWEEP_LOOKBACK:-60}"
SWEEP_HORIZON="${SWEEP_HORIZON:-25}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/outputs/figures/paper/seed_sweep}"

# DLinear is trained by the benchmark but is not reported in the manuscript.
SWEEP_MODELS="${SWEEP_MODELS:-cnn,tcn,transformer,itransformer,dual_stream_film,convtcn_hybrid,convlstmtcn_hybrid}"

PY="$REPO_ROOT/.venv/bin/python"

# Unbuffered Python for every child process. Without this, stdout is block
# buffered as soon as the sweep is piped (tee) or redirected to a file, so the
# seed headers and progress lines appear in bursts minutes late, or not at all
# until a stage ends. train_load_forecasters.py is already launched with -u by
# run_seed_sweep.py; this covers run_seed_sweep.py itself and the post/figure
# stages.
export PYTHONUNBUFFERED=1

# JAX platform. JAX_PLATFORMS pins the backend so a missing or broken CUDA
# plugin raises instead of silently falling back to CPU, which would produce a
# valid-looking sweep on the wrong device. Note the value must be "cuda":
# JAX_PLATFORMS=gpu is not a known backend in jax 0.6.x and aborts on 'rocm'.
case "$SWEEP_DEVICE" in
    gpu)
        export JAX_PLATFORMS=cuda
        EXPECTED_BACKEND=gpu
        if [ "$SWEEP_DETERMINISTIC" = "1" ]; then
            # --xla_gpu_enable_triton_gemm=false is REQUIRED alongside the
            # deterministic flag, not optional. Deterministic ops force the
            # Triton GEMM emitter, which cannot handle the dot shapes the TCN
            # produces on a ragged batch: iter_batches() yields a final batch of
            # N % batch_size (47 of 12079 in manoeuvre training, 16 of 2576 in
            # validation), and Triton then aborts with "Contracting dimension is
            # too fragmented" (batch 47) or "Too small divisible part of the
            # contracting dimension" (batch 16). Disabling the Triton path falls
            # back to cuBLAS and every model compiles at every batch size, with
            # determinism preserved. Verified over all seven architectures at
            # batch sizes {spec, 47, 16, 7, 1} in both operating modes.
            export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false"
        fi
        if [ -n "$SWEEP_GPU_MEM_FRACTION" ]; then
            export XLA_PYTHON_CLIENT_MEM_FRACTION="$SWEEP_GPU_MEM_FRACTION"
        fi
        ;;
    cpu)
        export JAX_PLATFORMS=cpu
        EXPECTED_BACKEND=cpu
        ;;
    *)
        echo "FATAL: SWEEP_DEVICE must be gpu or cpu, got '$SWEEP_DEVICE'"; exit 1 ;;
esac

# ------------------------------------------------------------------- preflight
echo "=== preflight ==="
[ -x "$PY" ] || { echo "FATAL: $PY not found. Create the venv first."; exit 1; }

for mode in $SWEEP_MODES; do
    specs="config/model_specs_${SWEEP_TAG}_${mode}.json"
    [ -f "$specs" ] || { echo "FATAL: missing $specs"; exit 1; }
done

# Dataset guard. Refuse to burn a sweep on the synthetic demo data.
[ -f "$SWEEP_CONFIG" ] || {
    echo "FATAL: dataset config not found: $SWEEP_CONFIG"
    echo "       It is gitignored, so a fresh clone does not have it. Copy it in:"
    echo "         cp /mnt/d/.../predict-timeseries/config/my_system_config.json config/"
    echo "       Or set SWEEP_CONFIG=<path> explicitly."
    exit 1
}
"$PY" - "$SWEEP_CONFIG" $SWEEP_MODES <<'PY' || exit 1
import json, sys
cfg_path, modes = sys.argv[1], sys.argv[2:]
sel = json.load(open(cfg_path))["dataset"]["selectors"]
bad = []
for mode in modes:
    files = sel.get(mode, [])
    if not files:
        bad.append(f"{mode}: no files listed")
    elif any("demo" in f for f in files):
        bad.append(f"{mode}: synthetic demo data -> {files}")
if bad:
    print("FATAL: refusing to train on this dataset configuration:", file=sys.stderr)
    for b in bad:
        print(f"       {b}", file=sys.stderr)
    raise SystemExit(1)
for mode in modes:
    print(f"dataset   : {mode} -> {len(sel[mode])} leg files ({sel[mode][0]}, ...)")
PY

echo "repo      : $REPO_ROOT"
echo "commit    : $(git rev-parse --short HEAD 2>/dev/null || echo '(not a git checkout)')"
echo "dirty     : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "python    : $("$PY" -c 'import sys; print(sys.version.split()[0])')"

if [ "$SWEEP_DEVICE" = "gpu" ]; then
    command -v nvidia-smi >/dev/null || {
        echo "FATAL: SWEEP_DEVICE=gpu but nvidia-smi is not on PATH inside WSL."; exit 1; }
    echo "gpu       : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
fi

# Hard assertion: abort rather than train a whole sweep on the wrong device.
"$PY" - "$EXPECTED_BACKEND" <<'PY' || { echo "FATAL: JAX backend assertion failed."; exit 1; }
import sys
import jax, flax, optax
expected = sys.argv[1]
backend = jax.default_backend()
devices = jax.devices()
print(f"jax       : {jax.__version__}  backend={backend}  devices={devices}")
print(f"flax      : {flax.__version__}   optax: {optax.__version__}")
if backend != expected:
    print(f"ASSERTION: expected backend {expected!r}, got {backend!r}", file=sys.stderr)
    raise SystemExit(1)
if expected == "gpu" and not any(d.platform == "gpu" for d in devices):
    print(f"ASSERTION: no GPU device visible: {devices}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "device    : $SWEEP_DEVICE (JAX_PLATFORMS=$JAX_PLATFORMS)"
if [ "$SWEEP_DEVICE" = "gpu" ]; then
    echo "xla_flags : ${XLA_FLAGS:-<none>}"
fi
echo "seeds     : $SWEEP_SEEDS"
echo "modes     : $SWEEP_MODES"
echo "models    : $SWEEP_MODELS"
echo "stage     : $SWEEP_STAGE"
echo

if [ "$(git status --porcelain 2>/dev/null | wc -l)" -ne 0 ]; then
    echo "WARNING: working tree is dirty; the sweep provenance will not match any commit."
fi

# ----------------------------------------------------------------------- stages
run_root_for() { echo "$REPO_ROOT/outputs/runs/sweep_${1}_${SWEEP_TAG}_flax"; }

stage_train() {
    for mode in $SWEEP_MODES; do
        root="$(run_root_for "$mode")"
        echo "=== train: $mode -> $root ==="
        "$PY" scripts/run_seed_sweep.py \
            --preset "${mode}_paper" \
            --backend flax \
            --seeds "$SWEEP_SEEDS" \
            --models "$SWEEP_MODELS" \
            --model-specs "config/model_specs_${SWEEP_TAG}_${mode}.json" \
            --config "$SWEEP_CONFIG" \
            --run-root "$root"
    done
}

stage_post() {
    for mode in $SWEEP_MODES; do
        root="$(run_root_for "$mode")"
        echo "=== aggregate: $mode ==="
        "$PY" scripts/aggregate_seed_results.py --run-root "$root"
        echo "=== significance: $mode ==="
        "$PY" scripts/run_significance.py --run-root "$root"
    done
}

stage_figures() {
    mkdir -p "$FIG_DIR"
    n_seeds=$(awk -F, '{print NF}' <<< "$SWEEP_SEEDS")
    for mode in $SWEEP_MODES; do
        root="$(run_root_for "$mode")"
        echo "=== figures: $mode ==="
        "$PY" scripts/gen_seed_sweep_figures.py \
            --run-root "$root" \
            --selector "$mode" \
            --lookback "$SWEEP_LOOKBACK" \
            --horizon "$SWEEP_HORIZON" \
            --expected-seeds "$n_seeds" \
            --require-model-specs "config/model_specs_${SWEEP_TAG}_${mode}.json" \
            --out-dir "$FIG_DIR"
    done
}

case "$SWEEP_STAGE" in
    all)     stage_train; stage_post; stage_figures ;;
    train)   stage_train ;;
    post)    stage_post ;;
    figures) stage_figures ;;
    *) echo "FATAL: unknown SWEEP_STAGE=$SWEEP_STAGE"; exit 1 ;;
esac

echo
echo "=== done ==="
echo "Figures : $FIG_DIR"
echo "Sync    : PREDICT_TS_WINDOWS_DIR=/mnt/<drive>/path/to/predict-timeseries \\"
echo "            bash scripts/sync_outputs_to_windows.sh"
