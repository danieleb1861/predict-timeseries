#!/usr/bin/env bash
# Watch a running (or finished) seed sweep without touching it.
#
# The sweep is launched detached with nohup, so attaching and detaching a viewer
# has no effect on it: Ctrl-C stops the viewer, never the run.
#
# Usage
# -----
#   bash scripts/watch_sweep.sh            # live filtered feed (Ctrl-C to leave)
#   bash scripts/watch_sweep.sh --status   # one-shot snapshot, then exit
#   SWEEP_LOG=/path/to.log bash scripts/watch_sweep.sh
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_LOG="${SWEEP_LOG:-$REPO_ROOT/outputs/runs/sweep_benchmark.log}"

# Lines worth seeing: stage banners, per-seed headers, model start/end, epoch
# ticks (proof of life), and anything that looks like a failure.
FILTER='^=== |Training .* \(Flax\)|parameters:|MAPE all|^epoch |Early stopping|Reducing learning rate|failed|Error|Traceback'

status() {
    echo "=== processes ==="
    pgrep -af 'run_seed_sweep|train_load_forecasters' | grep -v pgrep \
        || echo "  none running (sweep finished, or not started)"

    if command -v nvidia-smi >/dev/null; then
        echo "=== gpu ==="
        echo "  $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
              --format=csv,noheader)"
    fi

    echo "=== completed seeds ==="
    local found=0
    # Run roots live beside the log, so derive them from SWEEP_LOG rather than
    # from this script's repo: the two differ whenever you watch one checkout's
    # log using another checkout's copy of this script.
    local runs_dir; runs_dir="$(dirname "$SWEEP_LOG")"
    for root in "$runs_dir"/sweep_*_benchmark_flax; do
        [ -d "$root" ] || continue
        for d in "$root"/seed*; do
            [ -d "$d" ] || continue
            found=1
            local label="$(basename "$root")/$(basename "$d")"
            if [ -f "$d/results_metrics.csv" ]; then
                printf '  %-46s done\n' "$label"
            else
                printf '  %-46s in progress\n' "$label"
            fi
        done
        [ -f "$root/aggregated_metrics.csv" ] && echo "  ^ aggregated_metrics.csv present"
    done
    [ "$found" -eq 0 ] && echo "  (no seed directories yet)"

    echo "=== last activity ==="
    if [ -f "$SWEEP_LOG" ]; then
        grep -a -E "$FILTER" "$SWEEP_LOG" | tail -6 | sed 's/^/  /'
        if grep -aq '^=== done ===' "$SWEEP_LOG"; then
            echo "  >>> SWEEP COMPLETE (=== done === present in the log)"
        fi
    else
        echo "  no log at $SWEEP_LOG"
    fi
}

if [ "${1:-}" = "--status" ]; then
    status
    exit 0
fi

if [ ! -f "$SWEEP_LOG" ]; then
    echo "No log at $SWEEP_LOG. Start the sweep first, or set SWEEP_LOG."
    exit 1
fi

status
echo
echo "=== live feed (Ctrl-C leaves the viewer; the sweep keeps running) ==="
tail -f -n 0 "$SWEEP_LOG" | grep --line-buffered -a -E "$FILTER"
