import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

import jax.numpy as jnp  # noqa: E402

from load_forecasting.models.dlinear_flax import build_dlinear_model  # noqa: E402
from load_forecasting.training_deep import DeepTrainConfig  # noqa: E402
from load_forecasting.training_flax import (  # noqa: E402
    _train_step,
    create_train_state,
    set_learning_rate,
)


def test_injected_lr_is_present_and_mutable():
    config = DeepTrainConfig(learning_rate=1e-3)
    model = build_dlinear_model((20, 3), (5, 1))
    state = create_train_state(model, (20, 3), config)

    assert float(state.opt_state.hyperparams["learning_rate"]) == pytest.approx(1e-3)

    state = set_learning_rate(state, 5e-4)
    assert float(state.opt_state.hyperparams["learning_rate"]) == pytest.approx(5e-4)


def test_train_step_runs_after_lr_change():
    config = DeepTrainConfig(learning_rate=1e-3)
    model = build_dlinear_model((20, 3), (5, 1))
    state = create_train_state(model, (20, 3), config)

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(8, 20, 3)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(8, 5, 1)), dtype=jnp.float32)

    state, loss1 = _train_step(state, x, y, "mse")
    state = set_learning_rate(state, 5e-4)
    state, loss2 = _train_step(state, x, y, "mse")

    assert np.isfinite(float(loss1))
    assert np.isfinite(float(loss2))


@pytest.mark.parametrize("model_key", ["mlp", "cnn", "dlinear", "itransformer"])
def test_compiled_predictions_match_eager_full_and_partial_batches(model_key):
    """Preserve predictions for regular, remainder, and shorter-than-batch splits."""
    from load_forecasting.models.registry_flax import FLAX_MODEL_BUILDERS
    from load_forecasting.training_flax import _predict_batches

    model = FLAX_MODEL_BUILDERS[model_key]((8, 3), (2, 1))
    state = create_train_state(model, (8, 3), DeepTrainConfig())
    # Strided inputs also exercise the conversion at the host/device boundary.
    inputs = np.random.default_rng(7).normal(size=(14, 8, 3)).astype(np.float32)[::2]
    for count in (6, 7, 2):
        expected = np.concatenate([
            np.asarray(state.apply_fn({"params": state.params},
                                      jnp.asarray(inputs[start:min(start + 3, count)]), deterministic=True))
            for start in range(0, count, 3)
        ])
        actual = _predict_batches(state, inputs[:count], 3)
        assert actual.shape == (count, 2, 1) and actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_prediction_compilation_reused_with_updated_parameters():
    """Trace once per batch shape, without retaining the first parameter values."""
    from types import SimpleNamespace
    from load_forecasting.training_flax import _predict_batches

    traces = []

    def apply(variables, values, deterministic):
        assert deterministic is True
        traces.append(values.shape)
        return values[:, :2, :1] * variables["params"]["scale"]

    state = SimpleNamespace(apply_fn=apply, params={"scale": jnp.asarray(2.0)})
    inputs = np.ones((7, 4, 2), dtype=np.float32)
    np.testing.assert_array_equal(_predict_batches(state, inputs, 3), np.full((7, 2, 1), 2.0))
    assert traces == [(3, 4, 2), (1, 4, 2)]
    state.params = {"scale": jnp.asarray(5.0)}
    np.testing.assert_array_equal(_predict_batches(state, inputs[:4], 3), np.full((4, 2, 1), 5.0))
    assert traces == [(3, 4, 2), (1, 4, 2)]


def test_flax_prediction_timer_excludes_setup_and_validation(tmp_path, monkeypatch, capsys):
    """Warm every split's batch shape before timing test inference alone."""
    from load_forecasting import training_flax as training

    config = DeepTrainConfig(predict_batch_size=3)
    model = build_dlinear_model((8, 2), (2, 1))
    state = create_train_state(model, (8, 2), config)
    inputs = np.random.default_rng(7).normal(size=(7, 8, 2)).astype(np.float32)
    truth = np.ones((7, 2, 1), dtype=np.float32)
    clock, compiled = [0.0], set()
    original_predict = training._predict_step
    original_save = training.save_predictions

    def predict(apply_fn, params, values):
        # Simulate compile cost separately from each synchronized forward pass.
        if values.shape not in compiled:
            clock[0] += 10
            compiled.add(values.shape)
        clock[0] += 2
        return original_predict(apply_fn, params, values)

    def inverse(values):
        clock[0] += 3
        return values * 2

    def save(*args):
        clock[0] += 11
        original_save(*args)

    def score(y_true, y_hat, d_plot, **kwargs):
        assert y_true is truth and kwargs == {"include_confidence": True}
        clock[0] += 17
        return {"RMSE_horizon": np.ones(2), "RMSE_all": 1, "MAE_all": 1, "MAPE_all": 1}

    monkeypatch.setattr(training, "tmer", lambda: clock[0])
    monkeypatch.setattr(training, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(training, "fit_flax_model", lambda *args: (state, {"loss": [1.]}, 1.))
    monkeypatch.setattr(training, "estimate_inference_flops", lambda *args: None)
    monkeypatch.setattr(training, "_predict_step", predict)
    monkeypatch.setattr(training, "save_predictions", save)
    monkeypatch.setattr(training, "compute_metrics", score)
    result = training.train_eval_flax_model(
        "dlinear", "DLinear", model, inputs, truth, inputs[:5], truth[:5], inputs, truth,
        tmp_path, inverse, truth, 0, lambda *args: None, config=config,
    )
    assert compiled == {(3, 8, 2), (1, 8, 2), (2, 8, 2)}
    with np.load(tmp_path / "predictions/dlinear.npz") as saved:
        np.testing.assert_array_equal(saved["yhat_test_phys"], result.yhat_phys)
        expected_val = np.asarray(state.apply_fn({"params": state.params},
                                                 jnp.asarray(inputs[:5]), deterministic=True)) * 2
        np.testing.assert_allclose(saved["yhat_val_phys"], expected_val, rtol=1e-5, atol=1e-6)
    output = capsys.readouterr().out
    assert "DLinear prediction setup time (s): 36.0" in output
    assert "DLinear test prediction time (s): 6.0" in output
    assert "DLinear scoring time (s): 17.0" in output
