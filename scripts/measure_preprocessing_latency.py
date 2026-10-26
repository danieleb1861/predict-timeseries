"""Per-tick preprocessing latency, for the manuscript's end-to-end budget claim.

measure_inference_latency.py times the forward pass. Online, the forecaster must
also rebuild its input window every second: the engineered channels (dP, power
factor, lever asymmetry, 5/10/20 s rolling mean and std of active power), the
standardisation, and the (L, F) tensor. Two implementations are timed: recompute
over the trailing window, and O(1) running sums. Both are checked to produce the
same feature vector before either is timed.

    python scripts/measure_preprocessing_latency.py
    python scripts/measure_preprocessing_latency.py --ticks 3600 --warmup 100 --threads 1core
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

W, L, F, TICKS, WARMUP = (5, 10, 20), 60, 17, 2000, 200
DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_manoeuvre.parquet"


def load():
    d = pd.read_parquet(DEMO).tail(TICKS + WARMUP + 2 * L).reset_index(drop=True)
    base = float(np.abs(d["act_pow_kW"]).max()) or 1.0
    return (d["act_pow_kW"].to_numpy(float) / base,
            d["react_pow_kVAR"].to_numpy(float) / base,
            d["lever_port"].to_numpy(float), d["lever_starboard"].to_numpy(float))


def run(path, P, Q, up, us):
    """Returns (per-tick times in ms, last feature vector)."""
    mu, sig = np.zeros(F, np.float32), np.ones(F, np.float32)
    ring = np.zeros((L, F), np.float32)
    f = np.empty(F, np.float32)
    s = {w: 0.0 for w in W}
    ss = {w: 0.0 for w in W}
    n = 0
    times = []
    for k, t in enumerate(range(max(W), max(W) + WARMUP + TICKS)):
        t0 = time.perf_counter()
        S = np.hypot(P[t], Q[t])
        f[0], f[1] = P[t], Q[t]
        f[2:6] = 0.0                                    # metocean channels, held
        f[6] = up[t] - us[t]
        f[7] = P[t] / S if S else 0.0
        f[8], f[9] = up[t], us[t]
        f[10] = P[t] - P[t - 1]
        if path == "recompute":
            for j, w in enumerate(W):
                seg = P[t - w + 1:t + 1]
                f[11 + 2 * j] = seg.mean()
                f[12 + 2 * j] = seg.std(ddof=1)
        else:
            n += 1
            for j, w in enumerate(W):
                old = P[t - w] if n > w else 0.0
                s[w] += P[t] - old
                ss[w] += P[t] * P[t] - old * old
                m = min(n, w)
                avg = s[w] / m
                var = (ss[w] - m * avg * avg) / (m - 1) if m > 1 else 0.0
                f[11 + 2 * j] = avg
                f[12 + 2 * j] = np.sqrt(var) if var > 0 else 0.0
        g = (f - mu) / sig
        ring = np.roll(ring, -1, axis=0)
        ring[-1] = g
        _ = ring[None, ...]
        if k >= WARMUP:
            times.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(times), f.copy()


def main():
    global TICKS, WARMUP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=TICKS, help="Timed ticks.")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="Untimed warm-up ticks.")
    ap.add_argument("--threads", choices=("default", "1core"), default="default",
                    help="1core: pin to the performance core used by measure_inference_latency.py --threads 1core.")
    a = ap.parse_args()
    TICKS, WARMUP = a.ticks, a.warmup
    if a.threads == "1core":
        import psutil

        psutil.Process().cpu_affinity([2])
    P, Q, up, us = load()
    out = {}
    for path in ("recompute", "incremental"):
        out[path] = run(path, P, Q, up, us)
    assert np.allclose(out["recompute"][1], out["incremental"][1], atol=1e-4), \
        np.abs(out["recompute"][1] - out["incremental"][1]).max()

    print("%-12s %8s %8s %8s   (ms, %d ticks)" % ("path", "mean", "p95", "max", TICKS))
    for path, (a, _) in out.items():
        print("%-12s %8.4f %8.4f %8.4f" % (path, a.mean(), np.percentile(a, 95), a.max()))


if __name__ == "__main__":
    main()
