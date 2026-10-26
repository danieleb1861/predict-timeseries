"""Adapters for zero-shot time-series foundation models (torch-based).

Each adapter loads its pretrained checkpoint once and exposes a
`predict_batch(history_phys, horizon, covariates_phys=None) -> np.ndarray`
closure returning the median forecast, shape (batch, horizon), in the same
physical units as the input history.

Run only from the isolated `.venv-foundation` (see scripts/run_foundation_models.py)
-- never imported by the main JAX/TensorFlow pipeline, so kept outside
load_forecasting/ on purpose.

Chronos-2's predict_df() quantile-column naming is resolved defensively below
rather than guessed silently; the adapter raises a clear, actionable error on
the first mismatch instead of silently producing wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class FoundationAdapter:
    key: str
    label: str
    covariate_aware: bool
    predict_batch: Callable[[np.ndarray, int, Optional[np.ndarray]], np.ndarray]


def build_chronos2_adapter(device: str = "cuda") -> FoundationAdapter:
    import pandas as pd
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    # device_map= depends on `accelerate` to actually dispatch weights; if it's
    # missing (or silently no-ops for another reason), force placement here so a
    # CPU fallback is visible immediately instead of showing up as "just slow".
    inner_model = getattr(pipeline, "model", None)
    if inner_model is not None and hasattr(inner_model, "to"):
        actual_device = next(inner_model.parameters()).device
        if actual_device.type != device.split(":")[0]:
            print(f"  [chronos2] model was on {actual_device}, forcing .to({device!r})")
            inner_model.to(device)
        print(f"  [chronos2] model device: {next(inner_model.parameters()).device}")

    def predict_batch(history_phys: np.ndarray, horizon: int,
                       covariates_phys: np.ndarray | None = None) -> np.ndarray:
        if covariates_phys is not None:
            raise ValueError("chronos2 adapter is run univariate-only; got covariates")
        batch, context_len = history_phys.shape
        item_id = np.repeat(np.arange(batch), context_len)
        timestamp = np.tile(
            pd.date_range("2000-01-01", periods=context_len, freq="1s"), batch
        )
        context_df = pd.DataFrame({
            "item_id": item_id,
            "timestamp": timestamp,
            "target": history_phys.reshape(-1),
        })
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=horizon,
            quantile_levels=[0.5],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )
        quantile_col = next(
            (c for c in pred_df.columns if str(c).strip() in ("0.5", "q0.5", "median")),
            None,
        )
        if quantile_col is None:
            raise KeyError(
                "Could not find the 0.5-quantile column in Chronos-2's predict_df "
                f"output; available columns: {list(pred_df.columns)}. Update "
                "quantile_col lookup in build_chronos2_adapter accordingly."
            )
        ordered = pred_df.sort_values(["item_id", "timestamp"])
        median = ordered[quantile_col].to_numpy(dtype=np.float32)
        if median.size != batch * horizon:
            raise ValueError(
                f"Expected {batch * horizon} forecast rows, got {median.size}; "
                "predict_df's output shape assumption is wrong for this version."
            )
        return median.reshape(batch, horizon)

    return FoundationAdapter(
        key="chronos2", label="Chronos-2 (0-shot)",
        covariate_aware=False, predict_batch=predict_batch,
    )


def build_timesfm_adapter(
    device: str = "cuda",
    *,
    per_core_batch_size: int = 128,
    max_context: int = 64,
    max_horizon: int = 128,
    force_flip_invariance: bool = False,
) -> FoundationAdapter:
    import torch
    import timesfm

    if per_core_batch_size < 1:
        raise ValueError("per_core_batch_size must be positive")
    if max_context < 32 or max_context % 32 != 0:
        raise ValueError("max_context must be a positive multiple of TimesFM's 32-step patch")
    if max_horizon < 128 or max_horizon % 128 != 0:
        raise ValueError("max_horizon must be a positive multiple of TimesFM's 128-step output patch")

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    actual_device = getattr(model.model, "device", None)
    if actual_device is None:
        actual_device = next(model.model.parameters()).device
    requested_device = torch.device(device)
    if actual_device.type != requested_device.type:
        raise RuntimeError(
            f"TimesFM requested device {requested_device}, but loaded on {actual_device}. "
            "Check the PyTorch CUDA installation or use --device cpu explicitly."
        )
    print(
        f"  [timesfm] model device={actual_device}  "
        f"per_core_batch_size={per_core_batch_size}  "
        f"max_context={max_context}  max_horizon={max_horizon}  "
        f"force_flip_invariance={force_flip_invariance}"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        per_core_batch_size=per_core_batch_size,
        normalize_inputs=True,
        force_flip_invariance=force_flip_invariance,
    ))

    def predict_batch(history_phys: np.ndarray, horizon: int,
                       covariates_phys: np.ndarray | None = None) -> np.ndarray:
        if covariates_phys is not None:
            raise ValueError("timesfm adapter is run univariate-only; got covariates")
        inputs = [row.astype(np.float32) for row in history_phys]
        point_forecast, _ = model.forecast(horizon=horizon, inputs=inputs)
        point_forecast = np.asarray(point_forecast, dtype=np.float32)
        if point_forecast.shape != (history_phys.shape[0], horizon):
            raise ValueError(
                f"Expected point_forecast shape {(history_phys.shape[0], horizon)}, "
                f"got {point_forecast.shape}."
            )
        return point_forecast

    return FoundationAdapter(
        key="timesfm", label="TimesFM (0-shot)",
        covariate_aware=False, predict_batch=predict_batch,
    )


def build_tirex2_adapter(device: str = "cuda", *, use_covariates: bool = True,
                         internal_batch_size: int | None = None,
                         tta_sign_flip: bool = False,
                         tta_diff: bool = False) -> FoundationAdapter:
    import torch
    from tirex2 import TimeseriesType, load_model

    if internal_batch_size is None:
        # TiRex-2's dense grouped-attention path scales poorly with the packed
        # number of variates. Keep covariate-aware groups small; singleton
        # univariate groups bypass the variate mixer and can use larger batches.
        internal_batch_size = 8 if use_covariates else 512
    if internal_batch_size < 1:
        raise ValueError("TiRex-2 internal_batch_size must be >= 1")

    torch.set_float32_matmul_precision("high")
    model = load_model("NX-AI/TiRex-2", device=device)
    mode = "covariate-aware" if use_covariates else "univariate"
    print(
        f"  [tirex2] input_mode={mode}  internal_batch_size={internal_batch_size}  "
        f"context_len={getattr(model, 'context_len', '?')}  "
        f"future_len={getattr(model, 'future_len', '?')}  "
        f"tta_sign_flip={tta_sign_flip}  "
        f"tta_diff={tta_diff} "
        f"(checkpoint default={getattr(model, 'tta_diff', '?')})"
    )
    MEDIAN_QUANTILE_INDEX = 4  # 9 evenly-spaced quantiles [0.1..0.9]; verify on first run

    def predict_batch(history_phys: np.ndarray, horizon: int,
                       covariates_phys: np.ndarray | None = None) -> np.ndarray:
        if use_covariates and covariates_phys is None:
            raise ValueError("covariate-aware TiRex-2 adapter requires covariates")
        if not use_covariates and covariates_phys is not None:
            raise ValueError("univariate TiRex-2 adapter got unexpected covariates")
        batch = history_phys.shape[0]
        series_list = []
        for i in range(batch):
            target = torch.as_tensor(history_phys[i], dtype=torch.float32).unsqueeze(0)
            past_cov = None
            if covariates_phys is not None:
                # TiRex-2 documents past covariates as (n_covariates, context_length).
                past_cov = torch.as_tensor(
                    covariates_phys[i].T, dtype=torch.float32
                )
            series_list.append(TimeseriesType(
                target=target, past_covariates=past_cov, future_covariates=None,
            ))
        raw = model.forecast(
            series_list,
            prediction_length=horizon,
            output_type="numpy",
            batch_size=internal_batch_size,
            tta_sign_flip=tta_sign_flip,
            tta_diff=tta_diff,
        )
        results = []
        for item in raw:
            item = np.asarray(item)
            if item.ndim != 3 or item.shape[0] != 1 or item.shape[1] != 9:
                raise ValueError(
                    f"Expected per-series forecast shape (1, 9, {horizon}), got "
                    f"{item.shape}. Check MEDIAN_QUANTILE_INDEX / output layout "
                    "against the installed tirex2 version."
                )
            results.append(item[0, MEDIAN_QUANTILE_INDEX, :])
        return np.stack(results, axis=0).astype(np.float32)

    return FoundationAdapter(
        key="tirex2" if use_covariates else "tirex2_univariate",
        label=("TiRex-2 (0-shot, covariate-aware)" if use_covariates
               else "TiRex-2 (0-shot)"),
        covariate_aware=use_covariates, predict_batch=predict_batch,
    )


def build_tirex2_univariate_adapter(device: str = "cuda", *,
                                     internal_batch_size: int | None = None,
                                     tta_sign_flip: bool = False,
                                     tta_diff: bool = False) -> FoundationAdapter:
    return build_tirex2_adapter(
        device=device,
        use_covariates=False,
        internal_batch_size=internal_batch_size,
        tta_sign_flip=tta_sign_flip,
        tta_diff=tta_diff,
    )


ADAPTER_BUILDERS = {
    "chronos2": build_chronos2_adapter,
    "timesfm": build_timesfm_adapter,
    "tirex2": build_tirex2_adapter,
    "tirex2_univariate": build_tirex2_univariate_adapter,
}
