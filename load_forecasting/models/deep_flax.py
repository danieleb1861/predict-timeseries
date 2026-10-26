"""Flax deep-learning model builders for load forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def _output_size(output_shape: tuple[int, int]) -> int:
    return int(np.prod(output_shape))


def _last_step(x):
    return x[:, -1, :]


class MLPForecaster(nn.Module):
    output_shape: tuple[int, int]

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dropout(0.2, name="drop1")(x, deterministic=deterministic)
        x = nn.Dense(32, name="fc2")(x)
        x = nn.relu(x)
        x = nn.Dense(16, name="fc3")(x)
        x = nn.relu(x)
        x = nn.Dense(_output_size(self.output_shape), name="out_flat")(x)
        return x.reshape((x.shape[0], *self.output_shape))


class GRUCell(nn.Module):
    """Single GRU step; scanned over time so XLA traces the cell once regardless of lookback length."""
    hidden_size: int

    @nn.compact
    def __call__(self, h, xt):
        z = nn.sigmoid(nn.Dense(self.hidden_size, name="gru_z_x")(xt) + nn.Dense(self.hidden_size, use_bias=False, name="gru_z_h")(h))
        r = nn.sigmoid(nn.Dense(self.hidden_size, name="gru_r_x")(xt) + nn.Dense(self.hidden_size, use_bias=False, name="gru_r_h")(h))
        n = jnp.tanh(nn.Dense(self.hidden_size, name="gru_n_x")(xt) + nn.Dense(self.hidden_size, use_bias=False, name="gru_n_h")(r * h))
        h_new = (1.0 - z) * n + z * h
        return h_new, h_new


class GRUForecaster(nn.Module):
    output_shape: tuple[int, int]
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        batch = x.shape[0]
        h0 = jnp.zeros((batch, self.hidden_size), dtype=x.dtype)
        scan_cell = nn.scan(
            GRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(hidden_size=self.hidden_size, name="gru_cell")
        h, _ = scan_cell(h0, x)

        y = nn.Dropout(0.2, name="drop1")(h, deterministic=deterministic)
        y = nn.Dense(32, name="fc1")(y)
        y = nn.relu(y)
        y = nn.Dense(_output_size(self.output_shape), name="out_flat")(y)
        return y.reshape((batch, *self.output_shape))


class LSTMCell(nn.Module):
    """Single LSTM step; scanned over time so XLA traces the cell once regardless of lookback length."""
    hidden_size: int

    @nn.compact
    def __call__(self, carry, xt):
        h, c = carry
        gates = nn.Dense(4 * self.hidden_size, name="lstm_gates_x")(xt) + nn.Dense(4 * self.hidden_size, use_bias=False, name="lstm_gates_h")(h)
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = jnp.tanh(g)
        o = nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class LSTMForecaster(nn.Module):
    output_shape: tuple[int, int]
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        batch = x.shape[0]
        h0 = jnp.zeros((batch, self.hidden_size), dtype=x.dtype)
        c0 = jnp.zeros((batch, self.hidden_size), dtype=x.dtype)
        scan_cell = nn.scan(
            LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(hidden_size=self.hidden_size, name="lstm_cell")
        (h, _c), _ = scan_cell((h0, c0), x)

        y = nn.Dropout(0.2, name="drop1")(h, deterministic=deterministic)
        y = nn.Dense(32, name="fc1")(y)
        y = nn.relu(y)
        y = nn.Dense(_output_size(self.output_shape), name="out_flat")(y)
        return y.reshape((batch, *self.output_shape))


class CausalConv1D(nn.Module):
    features: int
    kernel_size: int
    dilation_rate: int = 1

    @nn.compact
    def __call__(self, x):
        pad = (self.kernel_size - 1) * self.dilation_rate
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
        return nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation_rate,),
            padding="VALID",
            name="conv",
        )(x)


class CNNForecaster(nn.Module):
    output_shape: tuple[int, int]
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for index, dilation in enumerate(self.dilations, start=1):
            x = CausalConv1D(
                64,
                kernel_size=self.kernel_size,
                dilation_rate=dilation,
                name=f"conv{index}",
            )(x)
            x = nn.relu(x)
        x = _last_step(x)
        x = nn.Dense(96)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.15, name="drop2")(x, deterministic=deterministic)
        x = nn.Dense(_output_size(self.output_shape), name="out_flat")(x)
        return x.reshape((x.shape[0], *self.output_shape))


class ResidualTCNBlock(nn.Module):
    features: int
    kernel_size: int
    dilation_rate: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        residual = x
        y = CausalConv1D(
            self.features,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            name="conv1",
        )(x)
        y = nn.relu(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        y = CausalConv1D(
            self.features,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            name="conv2",
        )(y)
        y = nn.relu(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        if residual.shape[-1] != self.features:
            residual = nn.Dense(self.features, name="residual_projection")(residual)
        return nn.relu(y + residual)


class TCNForecaster(nn.Module):
    output_shape: tuple[int, int]
    nb_filters: int = 96
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for dilation in self.dilations:
            x = ResidualTCNBlock(
                self.nb_filters,
                self.kernel_size,
                dilation,
                self.dropout_rate,
                name=f"tcn_d{dilation}",
            )(x, deterministic=deterministic)
        x = _last_step(x)
        x = nn.Dense(64, name="fc_tcn")(x)
        x = nn.relu(x)
        x = nn.Dense(_output_size(self.output_shape), name="out_flat")(x)
        return x.reshape((x.shape[0], *self.output_shape))


class PositionalEmbedding(nn.Module):
    seq_len: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        pos_emb = nn.Embed(
            num_embeddings=self.seq_len,
            features=self.d_model,
            name="pos_emb",
        )
        positions = jnp.arange(self.seq_len)
        return x + pos_emb(positions)


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x_norm = nn.LayerNorm(epsilon=1e-6)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            name="mha",
        )(x_norm, x_norm, deterministic=deterministic)
        attn = nn.Dropout(self.dropout)(attn, deterministic=deterministic)
        x = x + attn

        x_norm = nn.LayerNorm(epsilon=1e-6)(x)
        ff = nn.Dense(self.ff_dim)(x_norm)
        ff = nn.gelu(ff)
        ff = nn.Dropout(self.dropout)(ff, deterministic=deterministic)
        ff = nn.Dense(self.d_model)(ff)
        ff = nn.Dropout(self.dropout)(ff, deterministic=deterministic)
        return x + ff


class TransformerForecaster(nn.Module):
    output_shape: tuple[int, int]
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_transformer_blocks: int = 3
    dropout: float = 0.1
    mlp_units: tuple[int, ...] = (64,)
    mlp_dropout: float = 0.1
    use_conv_stem: bool = True

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        seq_len = x.shape[1]
        if self.use_conv_stem:
            x = CausalConv1D(
                self.d_model,
                kernel_size=3,
                name="conv_stem",
            )(x)
            x = nn.relu(x)
        else:
            x = nn.Dense(self.d_model, name="input_projection")(x)

        x = PositionalEmbedding(
            seq_len=seq_len,
            d_model=self.d_model,
            name="pos_embedding",
        )(x)

        for index in range(self.num_transformer_blocks):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.ff_dim,
                self.dropout,
                name=f"transformer_block_{index}",
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = _last_step(x)
        for index, dim in enumerate(self.mlp_units):
            x = nn.Dense(dim, name=f"mlp_{index}")(x)
            x = nn.gelu(x)
            x = nn.Dropout(self.mlp_dropout)(x, deterministic=deterministic)
        x = nn.Dense(_output_size(self.output_shape), name="out_flat")(x)
        return x.reshape((x.shape[0], *self.output_shape))


@dataclass(frozen=True)
class FlaxModelSpec:
    key: str
    label: str
    module: nn.Module


def receptive_field(kernel_sizes: Sequence[int], dilations: Sequence[int]) -> int:
    rf = 1
    for k, d in zip(kernel_sizes, dilations):
        rf += (k - 1) * d
    return rf


def dilations_for_lookback(lookback_steps: int, kernel_size: int = 3) -> tuple[int, ...]:
    """Return the smallest power-of-two dilation stack covering the lookback."""
    if lookback_steps < 1:
        raise ValueError("lookback_steps must be positive")
    if kernel_size < 2:
        raise ValueError("kernel_size must be at least 2")

    dilations = []
    rf = 1
    dilation = 1
    while rf < lookback_steps:
        dilations.append(dilation)
        rf += (kernel_size - 1) * dilation
        dilation *= 2
    return tuple(dilations or (1,))


def build_mlp_model(input_shape, output_shape):
    del input_shape
    return MLPForecaster(tuple(output_shape))


def build_gru_model(input_shape, output_shape, hidden_size=64):
    del input_shape
    return GRUForecaster(tuple(output_shape), hidden_size=hidden_size)


def build_rnn_model(input_shape, output_shape, hidden_size=64):
    del input_shape
    return LSTMForecaster(tuple(output_shape), hidden_size=hidden_size)


def build_cnn_model(input_shape, output_shape, kernel_size=3, dilations=None):
    lookback_steps, _num_features = input_shape
    if dilations is None:
        dilations = dilations_for_lookback(int(lookback_steps), int(kernel_size))
    return CNNForecaster(
        tuple(output_shape),
        dilations=tuple(int(d) for d in dilations),
        kernel_size=int(kernel_size),
    )


def build_tcn_model(input_shape, output_shape, nb_filters=96, kernel_size=3, dilations=(1, 2, 4, 8), dropout_rate=0.1):
    del input_shape
    return TCNForecaster(
        tuple(output_shape),
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=tuple(dilations),
        dropout_rate=dropout_rate,
    )


def build_transformer_forecaster(
    input_shape,
    output_shape,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=3,
    dropout=0.1,
    mlp_units=(64,),
    mlp_dropout=0.1,
    use_conv_stem=True,
):
    del input_shape
    return TransformerForecaster(
        output_shape=tuple(output_shape),
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout=dropout,
        mlp_units=tuple(mlp_units),
        mlp_dropout=mlp_dropout,
        use_conv_stem=use_conv_stem,
    )


def init_model(module: nn.Module, rng, input_shape):
    sample = jnp.ones((1, *input_shape), dtype=jnp.float32)
    return module.init({"params": rng, "dropout": rng}, sample, deterministic=True)
