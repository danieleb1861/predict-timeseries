"""Compatibility exports for Flax hybrid forecasting models."""

from load_forecasting.models.hybrid_flax import (
    ConvLSTMHybrid,
    ConvLSTMTCNParallel,
    ConvTCNParallel,
    TCNLSTMHybrid,
    build_conv_lstm_hybrid,
    build_conv_lstm_tcn_parallel,
    build_conv_tcn_parallel,
    build_tcn_lstm_hybrid,
)

__all__ = [
    "ConvLSTMHybrid",
    "ConvLSTMTCNParallel",
    "ConvTCNParallel",
    "TCNLSTMHybrid",
    "build_conv_lstm_hybrid",
    "build_conv_lstm_tcn_parallel",
    "build_conv_tcn_parallel",
    "build_tcn_lstm_hybrid",
]
