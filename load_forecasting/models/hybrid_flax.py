"""Flax hybrid model builders for load forecasting."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from load_forecasting.models.deep_flax import (
    CausalConv1D,
    LSTMCell,
    ResidualTCNBlock,
    TCNForecaster,  # noqa: F401  # compatibility re-export
    _last_step,
    _output_size,
)


# ConvLSTMHybrid and TCNLSTMHybrid route the entire input through a single LSTM
# (return_sequences=False) as the sole path to the output head. At L up to 60 s
# this branch's gradients vanish early in training (aggravated by lr=1e-3, the
# highest in the roster) and the head collapses to a near-constant prediction
# close to the unconditional target mean -- confirmed on the Keras benchmark
# (outputs/runs/comprehensive_benchmark/img/models_comparison_full_test.png)
# and consistent with both models being bottom-ranked in all 18 grid cells.
# Excluded from the active benchmark roster (--models) for this reason; the
# classes/builders stay registered for reference, just not selected for training.
class ConvLSTMHybrid(nn.Module):
    output_shape: tuple[int, int]

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = CausalConv1D(64, kernel_size=3, dilation_rate=1, name="conv1")(x)
        x = nn.relu(x)
        x = CausalConv1D(64, kernel_size=3, dilation_rate=2, name="conv2")(x)
        x = nn.relu(x)

        batch = x.shape[0]
        hidden_size = 64
        h = jnp.zeros((batch, hidden_size), dtype=x.dtype)
        c = jnp.zeros((batch, hidden_size), dtype=x.dtype)
        gates_x = nn.Dense(4 * hidden_size, name="lstm_gates_x")
        gates_h = nn.Dense(4 * hidden_size, use_bias=False, name="lstm_gates_h")

        for t in range(x.shape[1]):
            gates = gates_x(x[:, t, :]) + gates_h(h)
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = nn.sigmoid(i)
            f = nn.sigmoid(f)
            g = jnp.tanh(g)
            o = nn.sigmoid(o)
            c = f * c + i * g
            h = o * jnp.tanh(c)

        x = nn.Dropout(0.1)(h, deterministic=deterministic)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(_output_size(self.output_shape))(x)
        return x.reshape((batch, *self.output_shape))


class TCNLSTMHybrid(nn.Module):
    output_shape: tuple[int, int]

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for dilation in (1, 2, 4):
            x = ResidualTCNBlock(
                48,
                kernel_size=3,
                dilation_rate=dilation,
                dropout_rate=0.1,
                name=f"tcn_d{dilation}",
            )(x, deterministic=deterministic)

        batch = x.shape[0]
        hidden_size = 48
        h = jnp.zeros((batch, hidden_size), dtype=x.dtype)
        c = jnp.zeros((batch, hidden_size), dtype=x.dtype)
        gates_x = nn.Dense(4 * hidden_size, name="lstm_gates_x")
        gates_h = nn.Dense(4 * hidden_size, use_bias=False, name="lstm_gates_h")

        for t in range(x.shape[1]):
            gates = gates_x(x[:, t, :]) + gates_h(h)
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = nn.sigmoid(i)
            f = nn.sigmoid(f)
            g = jnp.tanh(g)
            o = nn.sigmoid(o)
            c = f * c + i * g
            h = o * jnp.tanh(c)

        x = nn.Dropout(0.1)(h, deterministic=deterministic)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(_output_size(self.output_shape))(x)
        return x.reshape((batch, *self.output_shape))


class ConvTCNParallel(nn.Module):
    output_shape: tuple[int, int]

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        branch_conv = CausalConv1D(48, kernel_size=3, name="conv_branch_1")(x)
        branch_conv = nn.relu(branch_conv)
        branch_conv = CausalConv1D(
            48,
            kernel_size=3,
            dilation_rate=2,
            name="conv_branch_2",
        )(branch_conv)
        branch_conv = nn.relu(branch_conv)
        branch_conv = jnp.mean(branch_conv, axis=1)

        branch_tcn = x
        for dilation in (1, 2, 4, 8):
            branch_tcn = ResidualTCNBlock(
                48,
                kernel_size=3,
                dilation_rate=dilation,
                dropout_rate=0.1,
                name=f"tcn_d{dilation}",
            )(branch_tcn, deterministic=deterministic)
        branch_tcn = _last_step(branch_tcn)

        y = jnp.concatenate([branch_conv, branch_tcn], axis=-1)
        y = nn.Dense(80)(y)
        y = nn.relu(y)
        y = nn.Dropout(0.15)(y, deterministic=deterministic)
        y = nn.Dense(_output_size(self.output_shape))(y)
        return y.reshape((x.shape[0], *self.output_shape))


class ConvLSTMTCNParallel(nn.Module):
    output_shape: tuple[int, int]

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        branch_conv = CausalConv1D(48, kernel_size=3, name="conv_branch_1")(x)
        branch_conv = nn.relu(branch_conv)
        branch_conv = CausalConv1D(
            48,
            kernel_size=3,
            dilation_rate=2,
            name="conv_branch_2",
        )(branch_conv)
        branch_conv = nn.relu(branch_conv)
        branch_conv = jnp.mean(branch_conv, axis=1)

        batch = x.shape[0]
        hidden_size = 40
        h0 = jnp.zeros((batch, hidden_size), dtype=x.dtype)
        c0 = jnp.zeros((batch, hidden_size), dtype=x.dtype)
        scan_cell = nn.scan(
            LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(hidden_size=hidden_size, name="lstm_cell")
        (branch_lstm, _c), _ = scan_cell((h0, c0), x)

        branch_tcn = x
        for dilation in (1, 2, 4, 8):
            branch_tcn = ResidualTCNBlock(
                48,
                kernel_size=3,
                dilation_rate=dilation,
                dropout_rate=0.1,
                name=f"tcn_d{dilation}",
            )(branch_tcn, deterministic=deterministic)
        branch_tcn = _last_step(branch_tcn)

        y = jnp.concatenate([branch_conv, branch_lstm, branch_tcn], axis=-1)
        y = nn.Dense(96)(y)
        y = nn.relu(y)
        y = nn.Dropout(0.15)(y, deterministic=deterministic)
        y = nn.Dense(_output_size(self.output_shape))(y)
        return y.reshape((batch, *self.output_shape))


def build_conv_lstm_hybrid(input_shape, output_shape):
    del input_shape
    return ConvLSTMHybrid(tuple(output_shape))


def build_tcn_lstm_hybrid(input_shape, output_shape):
    del input_shape
    return TCNLSTMHybrid(tuple(output_shape))


def build_conv_tcn_parallel(input_shape, output_shape):
    del input_shape
    return ConvTCNParallel(tuple(output_shape))


def build_conv_lstm_tcn_parallel(input_shape, output_shape):
    del input_shape
    return ConvLSTMTCNParallel(tuple(output_shape))
