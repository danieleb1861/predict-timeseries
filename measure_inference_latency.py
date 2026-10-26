"""
measure_inference_latency.py
============================
Measures single-sample inference latency for saved Keras or Flax models
(--backend). Keras entries are LABEL:MODEL_FILE (.keras); Flax entries are
LABEL:ARTIFACT_DIR (model_<key>_flax, legacy params.msgpack or Orbax
deployment artifacts). Each model is timed with a framework-overhead call
(keras predict() / non-JIT module.apply) and a compiled call (tf.function /
jax.jit). Flax models also report parameter counts and FLOPs from artifact
metadata for the model-complexity table.

Examples
--------
python measure_inference_latency.py
python measure_inference_latency.py --backend flax --cpu \
    --archive outputs/test_results/<date>_ForecastModels_Manoeuvre_L45_H25_flax7
python measure_inference_latency.py --family both --backend flax \
    --archive outputs/test_results/<date>_ForecastModels_Manoeuvre_L60_H25_flax8_legsafe \
    --device cuda --out-fig outputs/latency/manoeuvre.png

Outputs
-------
- latency figure -> OUT_FIG (.png and .eps)
- latency summary -> OUT_FIG.with_suffix(".csv")
- raw per-call latencies -> OUT_FIG stem + "_calls.csv"
- execution metadata -> OUT_FIG stem + "_metadata.json"
- Console: mean, median, p95 and max latency per model
"""

import argparse
import csv
import json
import math
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------ Style ------------------
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
    'grid.alpha':       0.25,
    'xtick.direction':  'out',
    'ytick.direction':  'out',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    'ps.fonttype':      3,
    'pdf.fonttype':     42,
    'figure.constrained_layout.use': False,
    'figure.dpi':       200,
    'savefig.dpi':      200,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.02,
})

# ------------------ CONFIG ------------------
ARCHIVE = Path(r"outputs\test_results\20260422_7Models_Manoeuvre_L45_H25")
N_WARMUP = 20
N_CALLS  = 200
LIMIT_MS = 100
PREDICT_METHOD = "keras predict()"
DIRECT_METHOD = "tf.function direct call"
FLAX_NON_JIT_METHOD = "module.apply (non-JIT)"
FLAX_JIT_METHOD = "jax.jit direct call"
OUT_FIG  = Path(r"outputs\runs\comprehensive_benchmark\img\inference_latency.png")
DPI      = 200
DEFAULT_MODELS = "CNN:model_cnn.keras,TCN:model_tcn.keras,ConvTCN-H:model_convtcn_hybrid.keras"
DEFAULT_FLAX_MODELS = (
    "CNN:model_cnn_flax,TCN:model_tcn_flax,Transformer:model_tr_flax,"
    "iTransformer:model_itransformer_flax,DualStream-FiLM:model_dual_stream_film_flax,"
    "ConvTCN-H:model_convtcn_hybrid_flax,ConvLSTMTCN-H:model_convlstmtcn_hybrid_flax"
)
DEFAULT_FOUNDATION_MODELS = "chronos2,timesfm,tirex2_univariate"
FOUNDATION_KEYS = {"chronos2", "timesfm", "tirex2_univariate"}
REPO_ROOT = Path(__file__).resolve().parent
PIN_CPU = 2  # logical CPU on a performance core of the i9-14900HX (P-cores are 0-15)
CPU_TIMES = {}  # (label, method) -> mean process CPU ms per call, filled with --cpu-time
N_PARAMS = {}  # label -> parameter count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure single-sample inference latency for tuned or foundation models."
    )
    parser.add_argument(
        "--family", choices=("tuned", "foundation", "both"), default="tuned",
        help="Measure tuned Keras/Flax models, foundation adapters, or both.",
    )
    parser.add_argument("--backend", choices=("keras", "flax"), default="keras", help="Model backend to load and time.")
    parser.add_argument("--archive", default=str(ARCHIVE), help="Run/archive directory containing model files and scalers_and_meta.pkl.")
    parser.add_argument("--out-fig", default=str(OUT_FIG), help="Output PNG path for the latency figure.")
    parser.add_argument("--models", default=None, help="Comma-separated LABEL:MODEL_FILE (keras) or LABEL:ARTIFACT_DIR (flax) entries relative to --archive.")
    parser.add_argument("--foundation-models", default=DEFAULT_FOUNDATION_MODELS, help="Comma-separated foundation adapter keys.")
    parser.add_argument("--model-specs", default=str(REPO_ROOT / "config" / "model_specs.json"), help="Model specs JSON used to rebuild legacy Flax MsgPack artifacts.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (hides GPUs from JAX/TensorFlow).")
    parser.add_argument("--device", default="cuda", help="Foundation-model device, e.g. cuda or cpu.")
    parser.add_argument(
        "--disable-foundation-compile", action="store_true",
        help="Disable optional torch.compile wrappers for CPU foundation-model timing.",
    )
    parser.add_argument("--window-index", type=int, default=0, help="Test window used for foundation-model timing.")
    parser.add_argument("--out-csv", default=None, help="Latency CSV path; defaults beside --out-fig.")
    parser.add_argument("--check-only", action="store_true", help="Validate archive inputs and selected model paths without loading models.")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP, help="Warm-up calls to discard per model.")
    parser.add_argument("--n-calls", type=int, default=N_CALLS, help="Timed calls per model.")
    parser.add_argument("--limit-ms", type=float, default=LIMIT_MS, help="Real-time latency limit drawn on plots.")
    parser.add_argument("--threads", choices=("default", "1core"), default="default",
                        help="1core: pin the process to one performance core with one intra-op thread (XLA, PyTorch, BLAS).")
    parser.add_argument("--end-to-end", action="store_true",
                        help="Time one 1 Hz tick: real test window in host memory -> forecast in PU in host memory.")
    parser.add_argument("--cpu-time", action="store_true", help="Also record mean process CPU time per call.")
    parser.add_argument("--eager-chunk", type=int, default=600,
                        help="Non-JIT Flax calls per chunk; JAX caches are cleared and re-warmed between chunks "
                             "because eager dispatch grows memory by ~2 MB per call.")
    parser.add_argument("--isolate", action="store_true",
                        help="Time every model in a fresh process and merge the CSVs (no figure).")
    parser.add_argument("--dpi", type=int, default=DPI, help="Figure DPI.")
    args = parser.parse_args()
    if args.models is None:
        args.models = DEFAULT_FLAX_MODELS if args.backend == "flax" else DEFAULT_MODELS
    args.foundation_models = [
        key.strip() for key in args.foundation_models.split(",") if key.strip()
    ]
    if args.family in ("foundation", "both") and not args.foundation_models:
        raise ValueError("--foundation-models must contain at least one model key.")
    unknown_foundation = set(args.foundation_models) - FOUNDATION_KEYS
    if unknown_foundation:
        raise ValueError(f"Unknown foundation model key(s): {sorted(unknown_foundation)}")
    return args


def parse_model_specs(raw_models, archive):
    specs = []
    for item in raw_models.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Model spec must be LABEL:MODEL_FILE, got: {item}")
        label, model_file = item.split(":", 1)
        specs.append((label.strip(), archive / model_file.strip()))
    if not specs:
        raise ValueError("--models must contain at least one LABEL:MODEL_FILE entry.")
    return specs


def build_infer_fn(model, tf_module):
    """
    Compile a low-overhead single-batch inference callable.

    Keras predict() includes overhead from data checks, batch handling, and
    graph tracing. A tf.function can be faster for repeated fixed-shape calls.
    """
    @tf_module.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)

    return infer


def measure_calls(call_fn, n_calls):
    """Measure n_calls calls to call_fn, returning latencies in ms."""
    latencies = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        call_fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    return np.array(latencies)


def measure_calls_chunked(call_fn, n_calls, chunk, n_rewarm):
    """measure_calls for eager JAX, whose dispatch cache grows by ~2 MB per call.

    After every `chunk` timed calls the JAX caches are cleared and the function is
    re-warmed with `n_rewarm` untimed calls, so memory stays bounded and no
    recompilation falls inside a timed call.
    """
    import gc

    import jax

    latencies = []
    while len(latencies) < n_calls:
        latencies.extend(measure_calls(call_fn, min(chunk, n_calls - len(latencies))))
        if len(latencies) < n_calls:
            jax.clear_caches()
            gc.collect()
            measure_warmup(call_fn, n_rewarm)
    return np.array(latencies)


def measure_cpu(call_fn, n_calls, min_seconds=2.0, min_calls=20):
    """Mean process CPU time per call in ms, over at least min_seconds and min_calls calls.

    The Windows process clock advances in 15.6 ms ticks, so CPU time is resolved
    only over long loops; 2 s keeps its relative error below 1 %. n_calls is kept
    for a uniform call signature and does not lengthen the loop.
    """
    k, c0, t_end = 0, time.process_time_ns(), time.perf_counter() + min_seconds
    while k < min_calls or time.perf_counter() < t_end:
        call_fn()
        k += 1
    return (time.process_time_ns() - c0) / 1e6 / k


def measure_warmup(call_fn, n_warmup):
    start = time.perf_counter()
    for _ in range(n_warmup):
        call_fn()
    return (time.perf_counter() - start) * 1000


def print_method_stats(method_results):
    for method, latencies in method_results.items():
        print(f"  {method:<26} "
              f"mean={np.mean(latencies):.3f} ms  "
              f"max={np.max(latencies):.3f} ms  "
              f"p95={np.percentile(latencies, 95):.3f} ms")


def measure_keras_models(models, X_dummy, args):
    import keras
    import tensorflow as tf
    from tcn import TCN

    X_dummy_tf = tf.convert_to_tensor(X_dummy)
    results = {}
    load_times = {}
    warmup_times = {}
    compile_times = {}
    for label, mpath in models:
        print(f"\nLoading {label} ...")
        load_start = time.perf_counter()
        # safe_mode=False: archives are trusted local artifacts (see README) and
        # legacy CNN models contain Lambda layers that safe mode refuses to load.
        model = keras.models.load_model(
            mpath,
            compile=False,
            custom_objects={"TCN": TCN},
            safe_mode=False,
        )
        load_times[label] = (time.perf_counter() - load_start) * 1000
        infer = build_infer_fn(model, tf)

        predict_call = lambda model=model: model.predict(X_dummy, verbose=0)
        direct_call = lambda infer=infer: infer(X_dummy_tf).numpy()
        compile_start = time.perf_counter()
        direct_call()
        compile_times[(label, DIRECT_METHOD)] = (
            time.perf_counter() - compile_start
        ) * 1000
        warmup_times[(label, PREDICT_METHOD)] = measure_warmup(
            predict_call, args.n_warmup,
        )
        warmup_times[(label, DIRECT_METHOD)] = measure_warmup(
            direct_call, args.n_warmup,
        )

        results[label] = {
            PREDICT_METHOD: measure_calls(predict_call, args.n_calls),
            DIRECT_METHOD: measure_calls(direct_call, args.n_calls),
        }
        print_method_stats(results[label])
        del infer, model
        keras.backend.clear_session()
    return results, load_times, warmup_times, compile_times


def load_flax_model(artifact_dir, meta, specs_file):
    """Keep the benchmark tuple interface while sharing artifact reconstruction."""
    from load_forecasting.deployment_flax import load_flax_run_model

    # Golden inference stays outside loading so warmup/compile timings remain separate.
    loaded = load_flax_run_model(artifact_dir, meta, specs_file, verify_golden=False)
    return loaded.module, loaded.params, loaded.metadata


def measure_flax_models(models, meta, X_dummy, args):
    import jax
    import jax.numpy as jnp

    from load_forecasting.training_flax import count_params

    print(f"JAX devices: {jax.devices()} | default backend: {jax.default_backend()}")
    x_dev = jnp.asarray(X_dummy)
    # --end-to-end: one 1 Hz tick, host window in -> forecast in PU out (transfers and inverse scaling included).
    mu_t = meta["MU"][meta["targetNames"]].to_numpy(dtype="float32")
    sig_t = meta["SIG"][meta["targetNames"]].to_numpy(dtype="float32")
    end_to_end = args.end_to_end
    results = {}
    load_times = {}
    warmup_times = {}
    compile_times = {}
    for label, artifact_dir in models:
        print(f"\nLoading {label} ...")
        load_start = time.perf_counter()
        module, params, metadata = load_flax_model(artifact_dir, meta, args.model_specs)
        load_times[label] = (time.perf_counter() - load_start) * 1000

        def non_jit_call(module=module, params=params):
            if end_to_end:
                output = module.apply({"params": params}, jnp.asarray(X_dummy), deterministic=True)
                return np.asarray(jax.block_until_ready(output)) * sig_t + mu_t
            output = module.apply({"params": params}, x_dev, deterministic=True)
            return jax.block_until_ready(output)

        jitted = jax.jit(
            lambda x, module=module, params=params: module.apply(
                {"params": params}, x, deterministic=True
            )
        )

        def jit_call():
            if end_to_end:
                return np.asarray(jax.block_until_ready(jitted(jnp.asarray(X_dummy)))) * sig_t + mu_t
            return jax.block_until_ready(jitted(x_dev))

        compile_start = time.perf_counter()
        jit_call()
        compile_times[(label, FLAX_JIT_METHOD)] = (
            time.perf_counter() - compile_start
        ) * 1000
        warmup_times[(label, FLAX_NON_JIT_METHOD)] = measure_warmup(
            non_jit_call, args.n_warmup,
        )
        warmup_times[(label, FLAX_JIT_METHOD)] = measure_warmup(
            jit_call, args.n_warmup,
        )

        # JIT first: the eager chunks clear the JAX caches, which also drops the compiled executable.
        jit_latencies = measure_calls(jit_call, args.n_calls)
        if args.cpu_time:
            CPU_TIMES[(label, FLAX_JIT_METHOD)] = measure_cpu(jit_call, args.n_calls)
        results[label] = {
            FLAX_NON_JIT_METHOD: measure_calls_chunked(non_jit_call, args.n_calls, args.eager_chunk, args.n_warmup),
            FLAX_JIT_METHOD: jit_latencies,
        }
        print_method_stats(results[label])
        if args.cpu_time:
            CPU_TIMES[(label, FLAX_NON_JIT_METHOD)] = measure_cpu(non_jit_call, args.n_calls)

        n_params = count_params(params)
        N_PARAMS[label] = int(n_params)
        flops = metadata.get("flops_per_window")
        flops_txt = f"{flops:,.0f}" if isinstance(flops, (int, float)) else "n/a"
        print(f"  parameters={n_params:,}  FLOPs/window (est.)={flops_txt}")
    return results, load_times, warmup_times, compile_times


def load_foundation_inputs(archive, meta, window_index):
    windows = np.load(archive / "windows.npz")
    Xte = np.asarray(windows["Xte"])
    horizon = int(meta["Horizon forecasting window H"])
    if not 0 <= window_index < len(Xte):
        raise IndexError(f"--window-index {window_index} outside [0, {len(Xte)})")
    target_names = list(meta["targetNames"])
    feature_names = list(meta["featureNames"])
    if len(target_names) != 1:
        raise ValueError(f"Expected one target for foundation latency, got {target_names}")
    target_index = feature_names.index(target_names[0])

    from load_forecasting.scaling import inverse_scale_targets

    history_phys = inverse_scale_targets(
        Xte[window_index : window_index + 1, :, [target_index]],
        target_names,
        meta["MU"],
        meta["SIG"],
    )[:, :, 0].astype(np.float32)
    if history_phys.shape[1] != int(meta["Lookback window L"]):
        raise ValueError(
            f"Foundation input lookback mismatch: {history_phys.shape[1]} vs "
            f"{meta['Lookback window L']}"
        )
    return history_phys, horizon


def build_foundation_adapter(model_key, args, lookback, horizon):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from foundation_adapters import ADAPTER_BUILDERS

    if model_key not in ADAPTER_BUILDERS:
        raise ValueError(
            f"Unknown foundation model {model_key!r}; "
            f"available: {sorted(ADAPTER_BUILDERS)}"
        )
    device = "cpu" if args.cpu else args.device
    if model_key == "chronos2":
        return ADAPTER_BUILDERS[model_key](device=device)
    if model_key == "timesfm":
        max_context = math.ceil(lookback / 32) * 32
        max_horizon = math.ceil(horizon / 128) * 128
        return ADAPTER_BUILDERS[model_key](
            device=device,
            per_core_batch_size=1,
            max_context=max_context,
            max_horizon=max_horizon,
            force_flip_invariance=False,
        )
    if model_key == "tirex2_univariate":
        return ADAPTER_BUILDERS[model_key](
            device=device,
            internal_batch_size=1,
            tta_sign_flip=False,
            tta_diff=False,
        )
    raise ValueError(f"Foundation latency is not configured for {model_key!r}")


def measure_foundation_models(archive, meta, args):
    history_phys, horizon = load_foundation_inputs(
        archive, meta, args.window_index
    )
    lookback = history_phys.shape[1]
    results = {}
    load_times = {}
    warmup_times = {}
    compile_times = {}
    for model_key in args.foundation_models:
        print(f"\nLoading foundation model {model_key} ...")
        load_start = time.perf_counter()
        adapter = build_foundation_adapter(model_key, args, lookback, horizon)
        load_times[adapter.label] = (time.perf_counter() - load_start) * 1000

        call = lambda adapter=adapter: adapter.predict_batch(
            history_phys, horizon, None
        )
        if args.end_to_end:
            import torch

            def call(adapter=adapter, torch=torch):
                with torch.inference_mode():
                    return adapter.predict_batch(history_phys, horizon, None)
        method = "foundation predict_batch"
        warmup_times[(adapter.label, method)] = measure_warmup(call, args.n_warmup)
        results[adapter.label] = {
            method: measure_calls(call, args.n_calls)
        }
        print_method_stats(results[adapter.label])
        if args.cpu_time:
            CPU_TIMES[(adapter.label, method)] = measure_cpu(call, args.n_calls)
        import gc
        import torch
        N_PARAMS[adapter.label] = max(
            (sum(p.numel() for p in o.parameters()) for o in gc.get_objects() if isinstance(o, torch.nn.Module)),
            default=0,
        )
    return results, load_times, warmup_times, compile_times


def write_latency_csv(path, results, load_times, warmup_times, compile_times, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model", "method", "load_ms", "compile_ms", "warmup_ms", "mean_ms", "median_ms",
        "p95_ms", "max_ms", "within_limit_pct", "n_warmup", "n_calls",
        "p99_ms", "cpu_mean_ms", "threads", "n_params",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, method_results in results.items():
            for method, latencies in method_results.items():
                writer.writerow({
                    "model": label,
                    "method": method,
                    "load_ms": f"{load_times.get(label, float('nan')):.6f}",
                    "compile_ms": f"{compile_times.get((label, method), float('nan')):.6f}",
                    "warmup_ms": f"{warmup_times.get((label, method), float('nan')):.6f}",
                    "mean_ms": f"{np.mean(latencies):.6f}",
                    "median_ms": f"{np.median(latencies):.6f}",
                    "p95_ms": f"{np.percentile(latencies, 95):.6f}",
                    "max_ms": f"{np.max(latencies):.6f}",
                    "within_limit_pct": f"{np.mean(latencies < args.limit_ms) * 100:.6f}",
                    "n_warmup": args.n_warmup,
                    "n_calls": args.n_calls,
                    "p99_ms": f"{np.percentile(latencies, 99):.6f}",
                    "cpu_mean_ms": f"{CPU_TIMES.get((label, method), float('nan')):.6f}",
                    "threads": args.threads,
                    "n_params": N_PARAMS.get(label, ""),
                })
    print(f"CSV saved: {path}")


def write_raw_latency_csv(path, results):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "method", "call", "latency_ms"))
        for label, method_results in results.items():
            for method, latencies in method_results.items():
                for call_index, latency_ms in enumerate(latencies, start=1):
                    writer.writerow((label, method, call_index, f"{latency_ms:.9f}"))
    print(f"Raw-call CSV saved: {path}")


def write_latency_metadata(path, args, meta):
    metadata = {
        "archive": str(Path(args.archive).resolve()),
        "family": args.family,
        "backend": args.backend,
        "requested_device": "cpu" if args.cpu else args.device,
        "batch_size": 1,
        "lookback_steps": int(meta["Lookback window L"]),
        "horizon_steps": int(meta["Horizon forecasting window H"]),
        "num_features": len(meta["featureNames"]),
        "dtype": "float32",
        "n_warmup": args.n_warmup,
        "n_calls": args.n_calls,
        "threads": args.threads,
        "pinned_cpu": PIN_CPU if args.threads == "1core" else None,
        "end_to_end": args.end_to_end,
        "window_index": args.window_index,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    if args.backend == "flax" and args.family in ("tuned", "both"):
        import jax

        metadata.update({
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        })
    if args.family in ("foundation", "both"):
        import torch

        metadata.update({
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_compile_disabled": args.disable_foundation_compile,
        })
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata saved: {path}")


def run_isolated(args):
    """Re-invoke this script once per model in a fresh process and merge the CSV outputs."""
    import subprocess

    import pandas as pd

    drop = {"--isolate", "--models", "--family", "--foundation-models", "--out-csv", "--out-fig"}
    passthrough, skip = [], False
    for tok in sys.argv[1:]:
        if skip:
            skip = False
            continue
        key = tok.split("=", 1)[0]
        if key in drop:
            skip = key != "--isolate" and "=" not in tok
            continue
        passthrough.append(tok)
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.out_fig).with_suffix(".csv")
    tmp = out_csv.parent / f"_{out_csv.stem}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.family in ("tuned", "both"):
        jobs += [["--family", "tuned", "--models", item.strip()] for item in args.models.split(",") if item.strip()]
    if args.family in ("foundation", "both"):
        jobs += [["--family", "foundation", "--foundation-models", key] for key in args.foundation_models]
    summaries, calls = [], []
    for i, job in enumerate(jobs):
        part = tmp / f"part{i:02d}.csv"
        cmd = [sys.executable, str(Path(__file__).resolve()), *passthrough, *job,
               "--out-csv", str(part), "--out-fig", str(tmp / f"part{i:02d}.png")]
        print(f"\n[isolate] {' '.join(job)}", flush=True)
        subprocess.run(cmd, check=True)
        summaries.append(pd.read_csv(part))
        calls.append(pd.read_csv(part.with_name(f"{part.stem}_calls.csv")))
    pd.concat(summaries).to_csv(out_csv, index=False)
    pd.concat(calls).to_csv(out_csv.with_name(f"{out_csv.stem}_calls.csv"), index=False)
    merged = {}
    for i in range(len(jobs)):
        merged.update(json.loads((tmp / f"part{i:02d}_metadata.json").read_text(encoding="utf-8")))
    merged.update({"family": args.family, "isolated_processes": len(jobs)})
    out_csv.with_name(f"{out_csv.stem}_metadata.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nMerged {len(jobs)} isolated runs into {out_csv}")


def main():
    args = parse_args()

    if args.threads == "1core":
        # Before any framework import: one intra-op thread everywhere, process pinned to one P-core.
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                          XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        import psutil

        psutil.Process().cpu_affinity([PIN_CPU])
    if args.isolate:
        run_isolated(args)
        return
    if args.threads == "1core" and args.family in ("foundation", "both"):
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.disable_foundation_compile:
        import torch

        # TiRex-2's optional compile wrappers require a local C++ compiler on CPU.
        torch.compile = lambda fn, *compile_args, **compile_kwargs: fn

    archive = Path(args.archive)
    meta_path = archive / "scalers_and_meta.pkl"
    out_fig = Path(args.out_fig)
    models = (
        parse_model_specs(args.models, archive)
        if args.family in ("tuned", "both")
        else []
    )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    LookbackSteps = int(meta["Lookback window L"])
    HorizonSteps = int(meta["Horizon forecasting window H"])
    featureNames = len(meta["featureNames"])
    print(
        f"Config: family={args.family}, backend={args.backend}, "
        f"L={LookbackSteps}, H={HorizonSteps}, F={featureNames}"
    )

    if args.check_only:
        if args.family in ("tuned", "both"):
            missing = [str(path) for _, path in models if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Missing tuned model files:\n" + "\n".join(missing)
                )
        if args.family in ("foundation", "both"):
            load_foundation_inputs(archive, meta, args.window_index)
        print("Latency archive and model-input validation complete.")
        return

    results = {}
    load_times = {}
    warmup_times = {}
    compile_times = {}
    if args.family in ("tuned", "both"):
        X_dummy = np.random.randn(
            1, LookbackSteps, featureNames
        ).astype(np.float32)
        if args.end_to_end:  # the same real test window the foundation models receive
            X_dummy = np.ascontiguousarray(
                np.load(archive / "windows.npz")["Xte"][args.window_index:args.window_index + 1],
                dtype=np.float32,
            )
        if args.backend == "flax":
            tuned_results, tuned_load, tuned_warmup, tuned_compile = measure_flax_models(
                models, meta, X_dummy, args
            )
        else:
            tuned_results, tuned_load, tuned_warmup, tuned_compile = measure_keras_models(
                models, X_dummy, args
            )
        results.update(tuned_results)
        load_times.update(tuned_load)
        warmup_times.update(tuned_warmup)
        compile_times.update(tuned_compile)
    if args.family in ("foundation", "both"):
        foundation_results, foundation_load, foundation_warmup, foundation_compile = (
            measure_foundation_models(archive, meta, args)
        )
        results.update(foundation_results)
        load_times.update(foundation_load)
        warmup_times.update(foundation_warmup)
        compile_times.update(foundation_compile)

    plt.rcParams.update({
        "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "lines.linewidth": 1.0, "axes.linewidth": 0.8,
        "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "figure.dpi": args.dpi, "savefig.dpi": args.dpi,
    })

    ncols = 2
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 3 * nrows), sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, (label, method_results) in zip(axes, results.items()):
        methods = list(method_results)
        method_labels = {
            FLAX_NON_JIT_METHOD: "Non-JIT module.apply",
            FLAX_JIT_METHOD: "JIT-compiled",
            "foundation predict_batch": "predict_batch",
        }
        colors = ("#2ca02c", "#1f77b4", "#ff7f0e")
        boxplot = ax.boxplot(
            [method_results[method] for method in methods],
            positions=np.arange(1, len(methods) + 1),
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 3},
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.axhline(args.limit_ms, color="#d62728", lw=0.9, ls=":",
                   label=f"{args.limit_ms:g} ms design budget")

        ax.set_title(label)
        ax.set_ylabel("Latency (ms)")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels([method_labels.get(method, method) for method in methods])
        ax.grid(True, which="both")

    for ax in axes[len(results):]:
        ax.remove()
    fig.tight_layout()

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_fig.with_suffix(".eps"), format="eps", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_fig}")
    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else out_fig.with_suffix(".csv")
    )
    write_latency_csv(
        out_csv, results, load_times, warmup_times, compile_times, args
    )
    write_raw_latency_csv(
        out_csv.with_name(f"{out_csv.stem}_calls.csv"), results
    )
    write_latency_metadata(
        out_csv.with_name(f"{out_csv.stem}_metadata.json"), args, meta
    )

    print("\n=== LATENCY SUMMARY (ms) ===")
    print(f"{'Model':<18}  {'Method':<26}  {'Mean':>9}  {'Median':>9}  {'p95':>9}  {'Max':>9}  {f'<{args.limit_ms:g}ms':>8}")
    for label, method_results in results.items():
        for method, lats in method_results.items():
            pct_ok = np.mean(lats < args.limit_ms) * 100
            print(f"{label:<18}  {method:<26}  {np.mean(lats):>9.3f}  "
                  f"{np.median(lats):>9.3f}  "
                  f"{np.percentile(lats,95):>9.3f}  "
                  f"{np.max(lats):>9.3f}  "
                  f"{pct_ok:>7.1f}%")


if __name__ == "__main__":
    main()
