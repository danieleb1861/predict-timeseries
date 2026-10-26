"""Flax DLinear baseline (Zeng et al., AAAI 2023, arXiv:2205.13504).

Trend/seasonal decomposition + per-channel Linear(L -> H) maps, followed by a
pointwise linear channel projection F -> output_dim because the forecast
targets are a subset of the input channels rather than all of them. The model
is linear end-to-end. See docs/dlinear_spec.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


def moving_average(x, kernel_size: int):
    """Replicate-padded moving average over the time axis of (N, L, F)."""
    front = jnp.repeat(x[:, :1, :], (kernel_size - 1) // 2, axis=1)
    back = jnp.repeat(x[:, -1:, :], kernel_size // 2, axis=1)
    padded = jnp.concatenate([front, x, back], axis=1)
    return nn.avg_pool(padded, window_shape=(kernel_size,), strides=(1,), padding="VALID")


class DLinearForecaster(nn.Module):
    output_shape: tuple[int, int]  # (HorizonSteps, output_dim)
    num_channels: int
    lookback_steps: int
    moving_avg_kernel: int = 25

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        del deterministic  # linear model, no dropout
        horizon, output_dim = self.output_shape
        num_ch = self.num_channels
        lookback = self.lookback_steps
        kernel = min(self.moving_avg_kernel, lookback)

        trend = moving_average(x, kernel)
        seasonal = x - trend

        # Per-channel ("individual") temporal linear maps: (F, L, H) weights.
        w_trend = self.param("w_trend", nn.initializers.lecun_normal(), (num_ch, lookback, horizon))
        b_trend = self.param("b_trend", nn.initializers.zeros, (num_ch, horizon))
        w_seasonal = self.param("w_seasonal", nn.initializers.lecun_normal(), (num_ch, lookback, horizon))
        b_seasonal = self.param("b_seasonal", nn.initializers.zeros, (num_ch, horizon))

        trend_out = jnp.einsum("nlf,flh->nfh", trend, w_trend) + b_trend
        seasonal_out = jnp.einsum("nlf,flh->nfh", seasonal, w_seasonal) + b_seasonal
        combined = trend_out + seasonal_out  # (N, F, H)

        # Pointwise channel projection F -> output_dim at each horizon step.
        w_proj = self.param("w_proj", nn.initializers.lecun_normal(), (num_ch, output_dim))
        b_proj = self.param("b_proj", nn.initializers.zeros, (output_dim,))
        y = jnp.einsum("nfh,fo->nho", combined, w_proj) + b_proj
        return y  # (N, H, output_dim)


def build_dlinear_model(input_shape, output_shape, moving_avg_kernel=25):
    lookback_steps, num_channels = input_shape
    return DLinearForecaster(
        output_shape=tuple(output_shape),
        num_channels=int(num_channels),
        lookback_steps=int(lookback_steps),
        moving_avg_kernel=int(moving_avg_kernel),
    )
