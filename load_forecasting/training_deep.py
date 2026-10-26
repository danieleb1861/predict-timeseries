"""Shared deep-learning training utilities for forecasting backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np


@dataclass(frozen=True)
class DeepTrainConfig:
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 1e-3
    predict_batch_size: int = 256
    shuffle: bool = False
    loss: str = "mse"
    patience: int = 10
    min_delta: float = 1e-5
    min_delta_rel: float = 0.01
    restore_best: bool = True
    seed: int = 0
    lr_factor: float = 0.5
    lr_patience: int = 4
    min_lr: float = 1e-6


@dataclass
class DeepTrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    yhat_phys: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_dir: Path | None = None


def append_history(history: dict[str, list[float]], **values: float) -> None:
    for key, value in values.items():
        history.setdefault(key, []).append(float(value))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_predictions(run_dir, model_key, model_label, yhat_test_phys, yhat_val_phys=None):
    """Persist per-window predictions in physical units for post-hoc analysis.

    Written to <run_dir>/predictions/<model_key>.npz. Together with the split
    ground truth already stored in windows.npz, this enables test-window
    bootstrap CIs and validation-calibrated conformal prediction without
    retraining.
    """
    pred_dir = Path(run_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "model_label": np.asarray(model_label),
        "yhat_test_phys": np.asarray(yhat_test_phys, dtype=np.float32),
    }
    if yhat_val_phys is not None:
        arrays["yhat_val_phys"] = np.asarray(yhat_val_phys, dtype=np.float32)
    np.savez_compressed(pred_dir / f"{model_key}.npz", **arrays)


def iter_batches(X, Y, batch_size: int, shuffle: bool = False, rng=None):
    """Yield aligned batches; unshuffled batches are views into the input arrays."""
    n = X.shape[0]
    indices = np.arange(n) if shuffle else None
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        # Sequential slices avoid both the index array and a copy of each batch.
        batch_idx = indices[start : start + batch_size] if shuffle else slice(start, start + batch_size)
        yield X[batch_idx], Y[batch_idx]
