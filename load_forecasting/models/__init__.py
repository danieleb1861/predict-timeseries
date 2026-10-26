"""Lazy model-builder registry.

Importing this package must not import TensorFlow/Keras. Backend-specific
builders are loaded only when the corresponding public attribute is requested.
"""

_EXPORTS = {
    "build_ridge_baseline": "load_forecasting.models.baselines",
    "build_xgb_baseline": "load_forecasting.models.baselines",
    "build_cnn_model": "load_forecasting.models.deep",
    "build_gru_model": "load_forecasting.models.deep",
    "build_mlp_model": "load_forecasting.models.deep",
    "build_rnn_model": "load_forecasting.models.deep",
    "build_tcn_model": "load_forecasting.models.deep",
    "build_transformer_forecaster": "load_forecasting.models.deep",
    "receptive_field": "load_forecasting.models.deep",
    "build_conv_lstm_hybrid": "load_forecasting.models.hybrid",
    "build_conv_lstm_tcn_parallel": "load_forecasting.models.hybrid",
    "build_conv_tcn_parallel": "load_forecasting.models.hybrid",
    "build_tcn_lstm_hybrid": "load_forecasting.models.hybrid",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
