"""Flax training and evaluation helpers for forecasting models."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from time import perf_counter, time as tmer

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training import train_state
import optax

from load_forecasting.deployment_flax import save_flax_deployment_artifact
from load_forecasting.metrics import compute_metrics
from load_forecasting.training_deep import (
    DeepTrainConfig,
    DeepTrainResult,
    append_history,
    iter_batches,
    save_json,
    save_predictions,
)


class FlaxTrainState(train_state.TrainState):
    """A simple extension of flax.training.train_state.TrainState to include dropout RNG (Random Number Generation)."""
    dropout_rng: jax.Array

def _loss_fn(pred, target, loss: str):
    """Compute the loss between predictions and targets based on the specified loss type."""
    if loss == "mse":
        return jnp.mean((pred - target) ** 2)
    if loss == "huber":
        return optax.huber_loss(pred, target, delta=1.0).mean()
    raise ValueError(f"Unsupported Flax loss: {loss!r}")

def create_train_state(module, input_shape, config: DeepTrainConfig, init_params=None, trainable_mask=None):
    """Create a FlaxTrainState with initialized parameters and optimizer.

    init_params : optional parameter pytree to start from (fine-tuning); random
        initialisation from ``config.seed`` when None.
    trainable_mask : optional boolean pytree with the structure of the params;
        leaves marked False receive a zero update (frozen layers).
    """
    rng = jax.random.PRNGKey(config.seed)
    params_rng, dropout_rng = jax.random.split(rng)
    if init_params is None:
        sample = jnp.ones((1, *input_shape), dtype=jnp.float32)
        params = module.init(
            {"params": params_rng, "dropout": dropout_rng},
            sample,
            deterministic=False,
        )["params"]
    else:
        params = init_params
    if trainable_mask is None:
        tx = optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate)
    else:
        frozen = jax.tree_util.tree_map(lambda trainable: not trainable, trainable_mask)
        # Frozen gradients are zeroed before Adam, so their update is exactly zero; the
        # learning rate stays an injected hyperparameter for set_learning_rate.
        tx = optax.inject_hyperparams(
            lambda learning_rate: optax.chain(optax.masked(optax.set_to_zero(), frozen), optax.adam(learning_rate))
        )(learning_rate=config.learning_rate)
    return FlaxTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        dropout_rng=dropout_rng,
    )

def set_learning_rate(state: FlaxTrainState, learning_rate: float):
    """Update the injected Adam learning rate (epoch-level plateau reduction)."""
    hyperparams = dict(state.opt_state.hyperparams)
    hyperparams["learning_rate"] = jnp.asarray(learning_rate, dtype=jnp.float32)
    return state.replace(opt_state=state.opt_state._replace(hyperparams=hyperparams))

@partial(jax.jit, static_argnames=("loss_name",))
def _train_step(state: FlaxTrainState, x, y, loss_name: str):
    """Perform a single training step, including forward pass, loss computation, backpropagation, and parameter update."""
    dropout_rng, next_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_with_params(params):
        pred = state.apply_fn(
            {"params": params},
            x,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        return _loss_fn(pred, y, loss_name)

    loss, grads = jax.value_and_grad(loss_with_params)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=next_dropout_rng)
    return state, loss

@partial(jax.jit, static_argnames=("loss_name",))
def _eval_step(state: FlaxTrainState, x, y, loss_name: str):
    """Perform a single evaluation step, including forward pass and loss computation."""
    pred = state.apply_fn({"params": state.params}, x, deterministic=True)
    return _loss_fn(pred, y, loss_name)

def count_params(params) -> int:
    """Total number of trainable parameters in a Flax params pytree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

def estimate_inference_flops(state: FlaxTrainState, input_shape) -> float | None:
    """Estimate FLOPs of a single-window forward pass via XLA cost analysis.

    Returns None when the backend does not expose a flops estimate.
    """
    try:
        x = jnp.ones((1, *input_shape), dtype=jnp.float32)
        fn = jax.jit(lambda p, xb: state.apply_fn({"params": p}, xb, deterministic=True))
        cost = fn.lower(state.params, x).compile().cost_analysis()
        if isinstance(cost, (list, tuple)):
            cost = cost[0] if cost else None
        if not cost:
            return None
        flops = cost.get("flops")
        return float(flops) if flops is not None else None
    except Exception as exc:  # pragma: no cover - backend-specific
        print(f"FLOPs estimate unavailable: {exc}")
        return None

@partial(jax.jit, static_argnames=("apply_fn",))
def _predict_step(apply_fn, params, x):
    """Reuse compiled inference across splits without capturing model parameters."""
    return apply_fn({"params": params}, x, deterministic=True)


def _predict_batches(state: FlaxTrainState, X, batch_size: int):
    """Predict in batches to manage memory usage during inference."""
    preds = []
    for start in range(0, X.shape[0], batch_size):
        xb = jnp.asarray(X[start : start + batch_size], dtype=jnp.float32)
        pred = _predict_step(state.apply_fn, state.params, xb)
        preds.append(np.asarray(pred))
    return np.concatenate(preds, axis=0)

def save_flax_artifact(module, state: FlaxTrainState, artifact_dir, metadata):
    """Save the trained Flax model parameters and associated metadata to the specified artifact directory."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    save_json(
        artifact_dir / "metadata.json",
            {"backend": "flax",
            "module": module.__class__.__name__,
            **metadata,
            },
    )

def fit_flax_model(module, Xtr, Ytr, Xva, Yva, config: DeepTrainConfig | None = None, epoch_callback=None,
                   init_params=None, trainable_mask=None):
    """Run the Flax training loop (early stopping + plateau LR) and return the fitted state.

    Inputs
    ----------
    epoch_callback : callable or None
        Optional ``epoch_callback(epoch, val_loss) -> bool`` invoked after each
        epoch; returning True stops training early (used by hyperparameter
        search for trial pruning).
    init_params, trainable_mask : optional
        Starting parameters and trainable-leaf mask, passed to
        ``create_train_state`` (see load_forecasting/fine_tuning.py).

    Returns
    -------
    tuple
        ``(state, history, best_val)`` where ``state`` holds the best
        parameters when ``config.restore_best`` is set.
    """
    config = config or DeepTrainConfig()

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Ytr = np.asarray(Ytr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)
    Yva = np.asarray(Yva, dtype=np.float32)

    state = create_train_state(module, Xtr.shape[1:], config, init_params=init_params, trainable_mask=trainable_mask)
    history: dict[str, list[float]] = {}
    rng = np.random.default_rng(config.seed)
    best_params = state.params
    best_val = np.inf
    stale_epochs = 0
    lr_stale_epochs = 0
    current_lr = float(config.learning_rate)

    for epoch in range(config.epochs):
        train_losses = []
        for xb, yb in iter_batches(
            Xtr,
            Ytr,
            config.batch_size,
            shuffle=config.shuffle,
            rng=rng,
        ):
            state, loss = _train_step(
                state,
                jnp.asarray(xb, dtype=jnp.float32),
                jnp.asarray(yb, dtype=jnp.float32),
                config.loss,
            )
            train_losses.append(float(loss))

        val_losses = []
        for xb, yb in iter_batches(Xva, Yva, config.batch_size, shuffle=False):
            loss = _eval_step(
                state,
                jnp.asarray(xb, dtype=jnp.float32),
                jnp.asarray(yb, dtype=jnp.float32),
                config.loss,
            )
            val_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        append_history(history, loss=train_loss, val_loss=val_loss)
        print(
            f"epoch {epoch + 1:03d}/{config.epochs} "
            f"loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        # Improvement threshold: relative to the current best, with an absolute
        # floor for near-zero losses (wiki evaluation protocol).
        if np.isinf(best_val):
            improvement_threshold = 0.0
        else:
            improvement_threshold = max(config.min_delta, config.min_delta_rel * abs(best_val))
        if val_loss < best_val - improvement_threshold:
            best_val = val_loss
            best_params = state.params
            stale_epochs = 0
            lr_stale_epochs = 0
        else:
            stale_epochs += 1
            lr_stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break
            if lr_stale_epochs >= config.lr_patience and current_lr > config.min_lr:
                current_lr = max(current_lr * config.lr_factor, config.min_lr)
                state = set_learning_rate(state, current_lr)
                lr_stale_epochs = 0
                print(f"Reducing learning rate to {current_lr:.2e}.")

        if epoch_callback is not None and epoch_callback(epoch, val_loss):
            print(f"Stopping at epoch {epoch + 1} (requested by callback).")
            break

    if config.restore_best:
        state = state.replace(params=best_params)

    return state, history, float(best_val)


def train_eval_flax_model(
    model_key,
    model_label,
    module,
    Xtr,
    Ytr,
    Xva,
    Yva,
    Xte,
    Yte,
    run_dir,
    inv_scale,
    y_true_phys,
    d_plot,
    log_time,
    config: DeepTrainConfig | None = None,
    artifact_key=None,
    artifact_spec=None,
    ):
    """Train and evaluate a Flax forecasting model, returning training history, predictions, and metrics."""
    del Yte
    config = config or DeepTrainConfig()
    stage_start = tmer()
    print()
    print(f"Training {model_label} (Flax)...")
    print()

    Xte = np.asarray(Xte, dtype=np.float32)

    state, history, _best_val = fit_flax_model(module, Xtr, Ytr, Xva, Yva, config)

    num_params = count_params(state.params)
    flops_per_window = estimate_inference_flops(state, Xtr.shape[1:])
    print(f"{model_label} parameters: {num_params:,}")
    if flops_per_window is not None:
        print(f"{model_label} FLOPs per window (est.): {flops_per_window:,.0f}")

    artifact_key = artifact_key or model_key
    artifact_dir = Path(run_dir) / f"model_{artifact_key}_flax"
    artifact_metadata = {
        "model_key": model_key,
        "model_label": model_label,
        "config": config.__dict__,
        "num_params": num_params,
        "flops_per_window": flops_per_window,
    }
    if artifact_spec is None:
        save_flax_artifact(module, state, artifact_dir, artifact_metadata)
    else:
        golden_input = np.linspace(
            -0.5,
            0.5,
            num=int(np.prod(Xtr.shape[1:])),
            dtype=np.float32,
        ).reshape((1, *Xtr.shape[1:]))
        save_flax_deployment_artifact(
            module,
            state.params,
            artifact_dir,
            golden_input_standardized=golden_input,
            metadata=artifact_metadata,
            **artifact_spec,
        )

    # Warm each batch shape; use a precise clock for short prediction timings.
    setup_start = perf_counter()
    warmed_shapes = set()
    for inputs in (Xte, Xva):
        for size in {min(len(inputs), config.predict_batch_size), len(inputs) % config.predict_batch_size} - {0}:
            shape = (size, *inputs.shape[1:])
            if shape not in warmed_shapes:
                _predict_step(state.apply_fn, state.params,
                              jnp.asarray(inputs[:size], dtype=jnp.float32)).block_until_ready()
                warmed_shapes.add(shape)
    prediction_setup_elapsed = perf_counter() - setup_start

    start_time = perf_counter()
    yhat = _predict_batches(state, Xte, config.predict_batch_size)
    prediction_elapsed = perf_counter() - start_time
    yhat_phys = inv_scale(yhat)
    yhat_val_phys = inv_scale(_predict_batches(state, Xva, config.predict_batch_size))
    save_predictions(run_dir, model_key, model_label, yhat_phys, yhat_val_phys)
    # Keep bootstrap/scoring cost separate from test inference.
    scoring_start = perf_counter()
    metrics = compute_metrics(y_true_phys, yhat_phys, d_plot, include_confidence=True)
    scoring_elapsed = perf_counter() - scoring_start

    print(f"{model_label} RMSE:", metrics["RMSE_horizon"])
    print(f"{model_label} RMSE all (PU):", metrics["RMSE_all"])
    print(f"{model_label} MAE all (PU):", metrics["MAE_all"])
    print(f"{model_label} MAPE all (%):", metrics["MAPE_all"])
    print()
    print(f"{model_label} prediction setup time (s):", prediction_setup_elapsed)
    print(f"{model_label} test prediction time (s):", prediction_elapsed)
    print(f"{model_label} scoring time (s):", scoring_elapsed)
    print()
    log_time(f"model_{model_key}_flax_train_eval", stage_start)

    return DeepTrainResult(
        history=history,
        yhat_phys=yhat_phys,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )
