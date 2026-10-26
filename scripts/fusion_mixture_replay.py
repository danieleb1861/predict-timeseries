"""Fusion mixture of predictors: persistence-anchored online aggregation, replayed on LOVO folds.

For every held-out leg of a leave-one-leg-out run root, the saved test predictions of
the trained models and the persistence forecast are combined per lead h,

    y_hat(t+h|t) = sum_k w_kh(t) * y_hat_k(t+h|t),     w_h(t) in the simplex,

with exponentially weighted updates under a tail-weighted loss
omega(y) * |y_hat - y|^p (p = 2 by default), omega(y) = 1 + lam * 1[y > q95_train], and a fixed-share step
towards a context prior pi_h(c):

    w_h <- normalise(w_h * exp(-eta * loss_h)),   w_h <- (1 - alpha) * w_h + alpha * pi_h(c_t).

Feedback is delayed and causal: the forecast of window n for lead h is scored only
from window n + ceil(h / stride) on, when y(t_n + h) has been observed. The context c_t
uses the last observed load (3 bands, training quantiles 0.5/0.9) and its 5 s change
(2 bands, training quantile 0.9). The prior and (eta, alpha, kappa) are fitted on the
fold's validation legs only; the test leg is never used for any choice.

Variants reported against persistence, on the paper's metrics:
    mean      equal weights over all experts (static ensemble)
    prior     context prior only (alpha = 1, static gating)
    fusion    full online fusion mixture

Example (WSL, where the fold directories live):
    .venv/bin/python scripts/fusion_mixture_replay.py \
        --run-root outputs/runs/lovo_manoeuvre_benchmark_flax
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from load_forecasting.scaling import inverse_scale_targets  # noqa: E402
from compute_peak_metrics import default_rating_kw, peak_metrics  # noqa: E402

GRID = {"eta": (0.5, 2.0, 8.0, 32.0, 128.0), "alpha": (0.005, 0.02, 0.1, 0.3), "kappa": (0.0, 2.0, 8.0, 32.0)}
POWER = 2.0  # loss exponent, set from --loss-power
GAMMA = 1.0  # extra weight on under-forecasts (missing reserve), set from --gamma
EXCLUDE = ()  # prediction stems to skip, set from --exclude
METRICS = ("MAPE_all", "RMSE_all", "tail_rmse_pu", "tail_rmse_pu_leg", "energy_err_mean_kwh",
           "mape_ramp_steep_%", "tail_under_pu", "tail_over_pu")
# --ems plant, in PU of one generator rating: six generators, zero to rated in 30 s, no start time (user,
# 25 Oct 2026); unit limits, BESS x* = (0.57 PU, 0.46 PU h), efficiencies and SoC window from the APEN BESS paper
EMS = {"n_units": 6, "ramp_s": 30.0, "p_min": 0.20, "p_max": 0.95, "p_bess": 0.57, "e_bess": 0.46 * 3600,
       "eta_ch": 0.94, "eta_dis": 0.97, "soc_min": 0.20, "soc_max": 0.80, "soc0": 0.50}
EMS_METRICS = ("unserved_pus", "surplus_pus", "bess_throughput_pus", "bess_peak_pu", "unit_hours", "starts", "fuel_kg")
# SFOC (g/kWh) against unit load (%), pchip as in the APEN DP (Real-data_Load-sharing main_opt.m, calculate_dp.m)
SFOC = interpolate.PchipInterpolator([0, 20, 40, 60, 75, 85, 95, 100], [300, 250, 230, 215, 211, 208, 210, 213])


def load_split(fold, split, extra_roots=()):
    """Experts (K, N, H) with persistence first, targets (N, H), contexts (N,) inputs."""
    w = np.load(fold / "windows.npz")
    meta = pickle.load(open(fold / "scalers_and_meta.pkl", "rb"))
    inv = lambda z: inverse_scale_targets(z, meta["targetNames"], meta["MU"], meta["SIG"])[..., 0]  # noqa: E731
    y = inv(w[f"Y{split}"])
    x = inv(w[f"X{split}"][:, :, :1])  # (N, L) observed active power
    names, preds = ["Persistence"], [np.repeat(x[:, -1:], y.shape[1], axis=1)]
    extra = [f for r in extra_roots for f in sorted((r / fold.name / "predictions").glob("*.npz"))]
    for f in sorted((fold / "predictions").glob("*.npz")) + extra:
        if f.stem in EXCLUDE and f not in extra:  # experts asked for with --extra-root are never excluded
            continue
        d = np.load(f)
        if "yhat_val_phys" not in d.files:  # experts without validation predictions cannot be weighted
            continue
        names.append(str(d["model_label"]))
        preds.append(d[f"yhat_{'test' if split == 'te' else 'val'}_phys"][..., 0])
    return names, np.stack(preds), y, x[:, -1], np.abs(x[:, -1] - x[:, -6])


def contexts(level, ramp, edges):
    return np.digitize(level, edges["level"]) * 2 + (ramp > edges["ramp"]).astype(int)  # 6 cells


def replay(P, Y, ctx, prior, eta, alpha, omega_fn, scale, stride, weights_out=None):
    """Causal online aggregation. P (K, N, H); returns fused (N, H)."""
    K, N, H = P.shape
    delay = np.ceil(np.arange(1, H + 1) / stride).astype(int)
    w = prior[:, :, ctx[0]].copy()  # (K, H)
    fused = np.empty((N, H))
    hh = np.arange(H)
    for n in range(N):
        if weights_out is not None:
            weights_out[n] = w.mean(1)
        fused[n] = (w * P[:, n, :]).sum(0)
        if n + 1 == N:
            break
        m = n + 1 - delay  # window whose lead-h target is observed at t_{n+1}
        ok = m >= 0
        if ok.any():
            mh, h_ok = m[ok], hh[ok]
            yv = Y[mh, h_ok]
            e = P[:, mh, h_ok] - yv
            loss = omega_fn(yv) * np.where(e < 0, GAMMA, 1.0) * np.abs(e) ** POWER / scale[h_ok]
            w[:, h_ok] *= np.exp(-eta * (loss - loss.min(0)))
            w[:, h_ok] /= w[:, h_ok].sum(0)
        w = (1 - alpha) * w + alpha * prior[:, :, ctx[n + 1]]
    return fused


def fit_prior(P, Y, ctx, kappa, omega_fn, scale, n_ctx=6):
    """pi_h(c) proportional to exp(-kappa * mean normalised loss of expert k in cell c)."""
    K, _, H = P.shape
    e = P - Y[None]
    loss = omega_fn(Y)[None] * np.where(e < 0, GAMMA, 1.0) * np.abs(e) ** POWER / scale[None, None]
    glob = loss.mean(1)  # (K, H) fallback for empty cells
    prior = np.empty((K, H, n_ctx))
    for c in range(n_ctx):
        sel = ctx == c
        lc = loss[:, sel].mean(1) if sel.sum() >= 20 else glob
        z = np.exp(-kappa * (lc - lc.min(0)))
        prior[:, :, c] = z / z.sum(0)
    return prior


def metrics(y_true, y_hat, y_train, rating):
    e = y_hat - y_true
    out = {"MAPE_all": 100 * np.mean(np.abs(e) / np.maximum(np.abs(y_true), 1e-6)),
           "RMSE_all": np.sqrt(np.mean(e ** 2))}
    pm = peak_metrics(y_true, y_hat, y_train, rating, 0.95, [0.5, 0.9])
    pm_leg = peak_metrics(y_true, y_hat, y_true, rating, 0.95, [0.5, 0.9])
    out.update({k: pm[k] for k in ("tail_rmse_pu", "energy_err_mean_kwh", "mape_ramp_steep_%")})
    tail = y_true > np.quantile(y_train, 0.95)
    out["tail_under_pu"] = float(np.maximum(-e[tail], 0).mean()) if tail.any() else float("nan")
    out["tail_over_pu"] = float(np.maximum(e[tail], 0).mean()) if tail.any() else float("nan")
    out["tail_rmse_pu_leg"] = pm_leg["tail_rmse_pu"]
    return out


def ems_replay(s, F, stride, rating, p=EMS, trace=None):
    """Rule-based generator loading with BESS on the 1 s load s, driven by the forecasts F (N, H) issued every
    stride s. Generators move at N/ramp_s PU/s towards the lowest output that still meets every forecast step
    within the ramp limit; a unit starts above p_max and stops below p_min of the online rating; the BESS covers
    the difference with the measured load within its power and SoC limits. Energies in PU s; fuel of equally
    loaded units, corrected for the final SoC at the minimum SFOC through the discharge path."""
    H = F.shape[1]
    N = int(min(p["n_units"], max(1, np.ceil(s[0] / p["p_max"]))))
    pg, soc, e_b = float(s[0]), p["soc0"], p["e_bess"]
    acc = dict.fromkeys(("unserved_pus", "surplus_pus", "bess_throughput_pus", "bess_peak_pu", "unit_s", "fuel_kg"), 0.0)
    starts, soc_lo = 0, soc
    for t in range(stride * (len(F) - 1) + 1):  # seconds covered by a forecast origin
        pb = float(np.clip(s[t] - pg, -p["p_bess"], p["p_bess"]))
        pb = min(pb, (soc - p["soc_min"]) * e_b * p["eta_dis"]) if pb > 0 else max(pb, -(p["soc_max"] - soc) * e_b / p["eta_ch"])
        soc -= (pb / p["eta_dis"] if pb > 0 else pb * p["eta_ch"]) / e_b
        u = s[t] - pg - pb
        if trace is not None:  # generator and BESS power at second t, for plotting
            trace.append((pg, pb))
        acc["unserved_pus"] += max(u, 0.0)
        acc["surplus_pus"] += max(-u, 0.0)
        acc["bess_throughput_pus"] += abs(pb)
        acc["bess_peak_pu"] = max(acc["bess_peak_pu"], abs(pb))
        acc["unit_s"] += N
        acc["fuel_kg"] += pg * rating * float(SFOC(100 * pg / N)) / 3.6e6  # 1 s at pg PU, g/kWh -> kg
        soc_lo = min(soc_lo, soc)
        n, j = divmod(t, stride)
        lead = np.arange(j + 1, H + 1)  # leads of origin n still ahead of t
        r = max(s[t], float((F[n, lead - 1] - N / p["ramp_s"] * (lead - j)).max()))
        while N < p["n_units"] and r > p["p_max"] * N:
            N, starts = N + 1, starts + 1
        while N > 1 and r < p["p_min"] * N:
            N -= 1
        rate = N / p["ramp_s"]
        pg = float(np.clip(np.clip(r, p["p_min"] * N, p["p_max"] * N), pg - rate, pg + rate))
    acc["unit_hours"] = acc.pop("unit_s") / 3600
    acc["fuel_kg"] -= (soc - p["soc0"]) * e_b * p["eta_dis"] * rating * float(SFOC(85)) / 3.6e6
    return {**acc, "starts": starts, "soc_min": soc_lo, "soc_end": soc}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--lam", type=float, default=4.0, help="Tail weight: omega(y) = 1 + lam * 1[y > q95_train].")
    ap.add_argument("--stride", type=int, default=5, help="Seconds between consecutive windows.")
    ap.add_argument("--loss-power", type=float, default=2.0,
                    help="p in omega(y) * |y_hat - y|^p: 2 targets RMSE-type error, 1 targets MAE/MAPE-type error.")
    ap.add_argument("--tag", default="", help="Suffix of the output folder <run-root>/fusion<tag>.")
    ap.add_argument("--dump-weights", type=int, default=-1, help="Save the expert weights of this fold for plotting.")
    ap.add_argument("--bench", action="store_true", help="Time the per-tick combination and update, then exit.")
    ap.add_argument("--bench-ticks", type=int, default=3600, help="Timed ticks for --bench.")
    ap.add_argument("--bench-warmup", type=int, default=100, help="Untimed warm-up ticks for --bench.")
    ap.add_argument("--bench-experts", type=int, default=8, help="Number of experts K timed by --bench.")
    ap.add_argument("--bench-threads", choices=("default", "1core"), default="default",
                    help="1core: pin --bench to the performance core used by measure_inference_latency.py.")
    ap.add_argument("--exclude", default="dlinear", help="Comma-separated prediction stems to leave out of the mixture.")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="Run root with the same folds whose fold*/predictions are added as experts (repeatable).")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Asymmetry: under-forecasts (missing reserve) weigh gamma times over-forecasts.")
    ap.add_argument("--fixed", default="", help="Skip the validation search and use these 'eta,alpha,kappa' instead.")
    ap.add_argument("--grid", default="",
                    help="Override the search grid, e.g. 'eta=0.03,0.5;alpha=0.3,1;kappa=0,32' (keys left out keep GRID).")
    ap.add_argument("--ems", action="store_true",
                    help="Also replay the rule-based generator/BESS loading (EMS) driven by persistence, fusion and the measured future.")
    a = ap.parse_args()
    global POWER, EXCLUDE, GAMMA
    for part in filter(None, (p.strip() for p in a.grid.split(";"))):
        key, values = part.split("=")
        GRID[key.strip()] = tuple(float(v) for v in values.split(","))
    POWER = a.loss_power
    GAMMA = a.gamma
    EXCLUDE = tuple(s.strip() for s in a.exclude.split(",") if s.strip())
    root, rating, rows, chosen, ems_rows = Path(a.run_root), default_rating_kw(), [], [], []

    if a.bench:  # cost of one 1 Hz tick of the mixture itself, experts excluded
        # Same step as replay(): combination, delayed tail-weighted scoring normalised by the
        # per-lead scale, exponential update, normalisation, fixed share towards the context prior.
        import time
        if a.bench_threads == "1core":
            import psutil
            psutil.Process().cpu_affinity([2])  # performance core used by measure_inference_latency.py
        K, H, eta, alpha, q95 = a.bench_experts, 25, 0.5, 0.3, 1.9
        rng = np.random.default_rng(0)
        N = a.bench_warmup + a.bench_ticks + 1
        P, Y = rng.random((K, N, H)) + 1.0, rng.random((N, H)) + 1.0
        ctx, scale = rng.integers(0, 6, N), rng.random(H) + 0.5
        prior = rng.random((K, H, 6))
        prior /= prior.sum(0)
        omega_fn = lambda y: 1.0 + a.lam * (y > q95)  # noqa: E731
        delay = np.ceil(np.arange(1, H + 1) / a.stride).astype(int)
        hh = np.arange(H)
        w = prior[:, :, ctx[0]].copy()
        t = []
        for n in range(N - 1):
            t0 = time.perf_counter_ns()
            _ = (w * P[:, n, :]).sum(0)
            m = n + 1 - delay
            ok = m >= 0
            if ok.any():
                mh, h_ok = m[ok], hh[ok]
                yv = Y[mh, h_ok]
                e = P[:, mh, h_ok] - yv
                loss = omega_fn(yv) * np.where(e < 0, GAMMA, 1.0) * np.abs(e) ** POWER / scale[h_ok]
                w[:, h_ok] *= np.exp(-eta * (loss - loss.min(0)))
                w[:, h_ok] /= w[:, h_ok].sum(0)
            w = (1 - alpha) * w + alpha * prior[:, :, ctx[n + 1]]
            t.append(time.perf_counter_ns() - t0)
        t = np.array(t[a.bench_warmup:]) / 1e3
        print(f"fusion tick, K={K} experts, H={H} leads, p={POWER:g}, threads={a.bench_threads}: mean {t.mean():.1f} us, "
              f"p95 {np.percentile(t, 95):.1f} us, p99 {np.percentile(t, 99):.1f} us, max {t.max():.1f} us over {len(t)} ticks")
        return

    folds = sorted(root.glob("fold*"), key=lambda p: int(p.name[4:])) or [root]  # single archive: one "fold"
    for fold in folds:
        fid = int(fold.name[4:]) if fold.name.startswith('fold') else 0
        extra = [Path(r) for r in a.extra_root]
        names, Pva, Yva, lva, rva = load_split(fold, "va", extra)
        _, Pte, Yte, lte, rte = load_split(fold, "te", extra)
        w = np.load(fold / "windows.npz")
        meta = pickle.load(open(fold / "scalers_and_meta.pkl", "rb"))
        y_train = inverse_scale_targets(w["Ytr"], meta["targetNames"], meta["MU"], meta["SIG"])[..., 0]
        q95 = np.quantile(y_train, 0.95)
        omega_fn = lambda y: 1.0 + a.lam * (y > q95)  # noqa: E731
        x_tr = inverse_scale_targets(w["Xtr"][:, :, :1], meta["targetNames"], meta["MU"], meta["SIG"])[..., 0]
        edges = {"level": np.quantile(x_tr[:, -1], [0.5, 0.9]), "ramp": np.quantile(np.abs(x_tr[:, -1] - x_tr[:, -6]), 0.9)}
        cva, cte = contexts(lva, rva, edges), contexts(lte, rte, edges)
        scale = (omega_fn(Yva) * np.abs(Pva[0] - Yva) ** POWER).mean(0) + 1e-12  # symmetric reference scale  # persistence validation loss per lead

        best, best_loss = None, np.inf  # hyperparameters on the validation legs only
        combos = [tuple(float(v) for v in a.fixed.split(","))] if a.fixed else \
            list(itertools.product(GRID["eta"], GRID["alpha"], GRID["kappa"]))
        for eta, alpha, kappa in combos:
            prior = fit_prior(Pva, Yva, cva, kappa, omega_fn, scale)
            f = replay(Pva, Yva, cva, prior, eta, alpha, omega_fn, scale, a.stride)
            ef = f - Yva
            L = np.mean(omega_fn(Yva) * np.where(ef < 0, GAMMA, 1.0) * np.abs(ef) ** POWER)
            if L < best_loss:
                best, best_loss = (eta, alpha, kappa), L
        eta, alpha, kappa = best
        prior = fit_prior(Pva, Yva, cva, kappa, omega_fn, scale)
        wdump = np.empty((Pte.shape[1], Pte.shape[0])) if fid == a.dump_weights else None
        variants = {
            "Persistence": Pte[0],
            "mean": Pte.mean(0),
            "prior": (prior[:, :, cte] * Pte.transpose(0, 2, 1)).sum(0).T,
            "fusion": replay(Pte, Yte, cte, prior, eta, alpha, omega_fn, scale, a.stride, wdump),
        }
        if wdump is not None:
            np.savez(root / f"fusion{a.tag}_weights_fold{a.dump_weights}.npz", weights=wdump,
                     names=np.array(names), load=lte, target=Yte)
        for k, name in enumerate(names[1:], start=1):
            variants[name] = Pte[k]
        res = {v: metrics(Yte, yh, y_train, rating) for v, yh in variants.items()}
        for v, r in res.items():
            rows.append({"fold": fid, "variant": v, **r,
                         **{f"S_{m}": (np.nan if not res["Persistence"][m] else 1 - r[m] / res["Persistence"][m])
                            for m in METRICS}})
        chosen.append({"fold": fid, "eta": eta, "alpha": alpha, "kappa": kappa})
        if a.ems:  # 1 s load of the test leg rebuilt from the overlapping target windows
            H = Yte.shape[1]
            if not (np.allclose(Yte[1:, :H - a.stride], Yte[:-1, a.stride:]) and np.allclose(lte[1:], Yte[:-1, a.stride - 1])):
                raise ValueError(f"{fold.name}: test windows are not contiguous at stride {a.stride}")
            s = np.concatenate([lte[:1], Yte[:, :a.stride].ravel(), Yte[-1, a.stride:]])
            traces = {}
            for v, F in (("Persistence", variants["Persistence"]), ("fusion", variants["fusion"]), ("oracle", Yte)):
                tr = [] if fid == a.dump_weights else None
                ems_rows.append({"fold": fid, "variant": v, **ems_replay(s, F, a.stride, rating, trace=tr)})
                if tr is not None:
                    traces[v] = np.array(tr)
            if traces:
                np.savez(root / f"fusion{a.tag}_ems_fold{fid}.npz", load=s, **traces)
        print(f"{fold.name}: eta={eta} alpha={alpha} kappa={kappa}  S_MAPE fusion={rows[-len(variants) + 3]['S_MAPE_all']:+.3f}")

    df = pd.DataFrame(rows)
    out = root / f"fusion{a.tag}"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "fusion_by_fold.csv", index=False)
    pd.DataFrame(chosen).to_csv(out / "fusion_hyperparameters.csv", index=False)
    summ = []
    for v, g in df[df.variant != "Persistence"].groupby("variant", sort=False):
        r = {"variant": v}
        for m in METRICS:
            x = g[f"S_{m}"].dropna().to_numpy()
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            r[m] = f"{x.mean():+.3f} [{x.mean() - h:+.3f},{x.mean() + h:+.3f}] {int((x > 0).sum())}/{len(x)}"
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "fusion_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nPersistence-relative skill, mean [95% t-CI] positive folds:")
    print(summ.to_string(index=False))
    if a.ems:
        de = pd.DataFrame(ems_rows)
        de.to_csv(out / "ems_by_fold.csv", index=False)
        base = de[de.variant == "Persistence"].set_index("fold")
        es = []
        for v, g in de.groupby("variant", sort=False):
            g = g.set_index("fold")
            r = {"variant": v}
            for m in EMS_METRICS:
                x = (1 - g[m] / base[m].replace(0, np.nan)).dropna().to_numpy()
                h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                r[m] = f"{g[m].mean():.3f} | {x.mean():+.3f} [{x.mean() - h:+.3f},{x.mean() + h:+.3f}] {int((x > 0).sum())}/{len(x)}"
            es.append(r)
        es = pd.DataFrame(es)
        es.to_csv(out / "ems_summary.csv", index=False)
        print("\nEMS replay: mean over folds | persistence-relative reduction, mean [95% t-CI] positive folds:")
        print(es.to_string(index=False))


if __name__ == "__main__":
    main()
