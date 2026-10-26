"""Explicit Flax model-builder registry."""

from load_forecasting.models.deep_flax import (
    build_cnn_model,
    build_gru_model,
    build_mlp_model,
    build_rnn_model,
    build_tcn_model,
    build_transformer_forecaster,
    receptive_field,
)
from load_forecasting.models.dlinear_flax import build_dlinear_model
from load_forecasting.models.dual_stream_flax import build_dual_stream_film_model
from load_forecasting.models.itransformer_flax import build_itransformer_model
from load_forecasting.models.deep_hybrid_flax import (
    build_conv_lstm_hybrid,
    build_conv_lstm_tcn_parallel,
    build_conv_tcn_parallel,
    build_tcn_lstm_hybrid,
)


FLAX_DEEP_BUILDERS = {
    "mlp": build_mlp_model,
    "gru": build_gru_model,
    "rnn": build_rnn_model,
    "cnn": build_cnn_model,
    "tcn": build_tcn_model,
    "transformer": build_transformer_forecaster,
    "dlinear": build_dlinear_model,
    "itransformer": build_itransformer_model,
    "dual_stream_film": build_dual_stream_film_model,
}

FLAX_HYBRID_BUILDERS = {
    "convlstm_hybrid": build_conv_lstm_hybrid,
    "tcnlstm_hybrid": build_tcn_lstm_hybrid,
    "convtcn_hybrid": build_conv_tcn_parallel,
    "convlstmtcn_hybrid": build_conv_lstm_tcn_parallel,
}

FLAX_MODEL_BUILDERS = {
    **FLAX_DEEP_BUILDERS,
    **FLAX_HYBRID_BUILDERS,
}

__all__ = [
    "FLAX_DEEP_BUILDERS",
    "FLAX_HYBRID_BUILDERS",
    "FLAX_MODEL_BUILDERS",
    "receptive_field",
]
