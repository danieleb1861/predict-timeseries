"""Fine-tuning of saved Flax forecasting artifacts on data from another domain.

A deployment artifact (``save_flax_deployment_artifact``) carries its module, its
parameters and its own preprocessing contract (feature and target scalers). Fine
tuning keeps that contract: data standardised with any other scaler (another
operating mode, vessel or period) are mapped back to physical units and
re-standardised with the artifact's scalers, the pretrained parameters are the
starting point of the usual training loop (``fit_flax_model``), and the result
is saved as a new artifact with the same builder, scalers and provenance.

Typical use (navigation model adapted to manoeuvre windows):

    art = load_source(nav_run / "model_tcn_flax")
    X = to_artifact_space(Xtr_man, man_scaler, art)   # also Y with target scalers
    result = fine_tune(art, Xtr, Ytr, Xva, Yva, config, frozen=[r"^Conv_0/"])
    save_fine_tuned(art, result.params, out_dir, metadata={...})

Layer selection uses regular expressions on the ``/``-joined parameter path, so
it works for any architecture without architecture-specific code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from load_forecasting.deployment_flax import load_flax_artifact, save_flax_deployment_artifact
from load_forecasting.training_deep import DeepTrainConfig
from load_forecasting.training_flax import count_params, fit_flax_model


@dataclass
class Scaler:
    """Per-channel affine standardisation z = (x - mean) / scale."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_meta(cls, meta, columns: Sequence[str]) -> "Scaler":
        """Scaler of the named columns from a run's ``scalers_and_meta.pkl`` (MU/SIG Series)."""
        return cls(meta["MU"].loc[list(columns)].to_numpy(np.float32),
                   meta["SIG"].loc[list(columns)].to_numpy(np.float32))


@dataclass
class FineTuneResult:
    params: Any
    history: dict[str, list[float]]
    best_val: float
    n_params: int
    n_trainable: int
    config: dict[str, Any] = field(default_factory=dict)


def load_source(artifact_dir, *, verify_golden: bool = True):
    """Load the pretrained deployment artifact (module, params, scalers, manifest)."""
    return load_flax_artifact(Path(artifact_dir), verify_golden=verify_golden)


def artifact_scalers(art) -> tuple[Scaler, Scaler]:
    """Feature and target scalers stored in the artifact."""
    return Scaler(art.feature_mean, art.feature_scale), Scaler(art.target_mean, art.target_scale)


def restandardize(z, source: Scaler, target: Scaler) -> np.ndarray:
    """Map data standardised with ``source`` to the standardisation of ``target`` (last axis = channels)."""
    physical = np.asarray(z, dtype=np.float32) * source.scale + source.mean
    return ((physical - target.mean) / target.scale).astype(np.float32)


def to_artifact_space(X, Y, data_features: Scaler, data_targets: Scaler, art):
    """Windows standardised with the data scalers, re-standardised for the artifact."""
    feat, targ = artifact_scalers(art)
    Xa = restandardize(X, data_features, feat)
    Ya = None if Y is None else restandardize(Y, data_targets, targ)
    return Xa, Ya


def _path_names(params):
    paths = jax.tree_util.tree_flatten_with_path(params)[0]
    return ["/".join(str(getattr(k, "key", k)) for k in path) for path, _ in paths]


def trainable_mask(params, trainable: Sequence[str] | None = None, frozen: Sequence[str] | None = None):
    """Boolean pytree over the params: True where the leaf is updated.

    trainable : regexes; if given, only matching leaves are trainable.
    frozen    : regexes; matching leaves are frozen (applied after ``trainable``).
    Returns None when every leaf is trainable (plain fine-tuning).
    """
    if not trainable and not frozen:
        return None
    names = _path_names(params)
    keep = [bool(not trainable or any(re.search(p, n) for p in trainable)) for n in names]
    keep = [k and not any(re.search(p, n) for p in (frozen or ())) for k, n in zip(keep, names)]
    if not any(keep):
        raise ValueError("the trainable/frozen patterns leave no parameter to train")
    treedef = jax.tree_util.tree_structure(params)
    return jax.tree_util.tree_unflatten(treedef, keep)


def fine_tune(art, Xtr, Ytr, Xva, Yva, config: DeepTrainConfig, *, trainable=None, frozen=None) -> FineTuneResult:
    """Continue training the artifact's parameters on windows already in the artifact's scale."""
    mask = trainable_mask(art.params, trainable, frozen)
    state, history, best_val = fit_flax_model(
        art.module, Xtr, Ytr, Xva, Yva, config, init_params=art.params, trainable_mask=mask)
    n_train = count_params(art.params) if mask is None else int(sum(
        leaf.size for leaf, keep in zip(jax.tree_util.tree_leaves(art.params), jax.tree_util.tree_leaves(mask)) if keep))
    return FineTuneResult(state.params, history, float(best_val), count_params(state.params), n_train, dict(config.__dict__))


def predict_physical(art, params, X, batch_size: int = 512) -> np.ndarray:
    """Forecast in physical units from windows in the artifact's standardised scale."""
    apply = jax.jit(lambda x: art.module.apply({"params": params}, x, deterministic=True))
    out = [np.asarray(apply(jnp.asarray(X[i:i + batch_size]))) for i in range(0, len(X), batch_size)]
    return np.concatenate(out) * art.target_scale.reshape(1, 1, -1) + art.target_mean.reshape(1, 1, -1)


def save_fine_tuned(art, params, artifact_dir, metadata: dict[str, Any] | None = None):
    """Save the fine-tuned parameters as a deployment artifact with the source contract and provenance."""
    model, data = art.manifest["model"], art.manifest["data"]
    golden = np.linspace(-0.5, 0.5, num=int(np.prod(model["input_shape"])), dtype=np.float32)
    return save_flax_deployment_artifact(
        art.module, params, artifact_dir,
        builder_key=model["builder_key"], builder_args=model["builder_args"],
        input_shape=model["input_shape"], output_shape=model["output_shape"],
        feature_names=data["feature_names"], target_names=data["target_names"],
        feature_mean=art.feature_mean, feature_scale=art.feature_scale,
        target_mean=art.target_mean, target_scale=art.target_scale,
        sampling_rate=data.get("sampling_rate"),
        golden_input_standardized=golden.reshape((1, *model["input_shape"])),
        metadata={**art.manifest.get("metadata", {}), "fine_tuned_from": str(art.artifact_dir), **(metadata or {})},
    )
