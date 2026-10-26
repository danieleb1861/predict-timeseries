"""Flax iTransformer (Liu et al., ICLR 2024, arXiv:2310.06625).

Inverted Transformer: each input channel's full lookback series becomes one
token (variate token), self-attention runs across the F variates instead of
across time. The forecast head is a shared Dense(d_model -> H) applied per
token; the targets are a subset of the input channels, so their tokens are
selected directly (the paper's own setting). See docs/dlinear_spec.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class ITransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float

    @nn.compact
    def __call__(self, tokens, deterministic: bool = True):
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            name="attn",
        )(tokens, tokens, deterministic=deterministic)
        attn = nn.Dropout(self.dropout, name="attn_drop")(attn, deterministic=deterministic)
        tokens = nn.LayerNorm(name="norm1")(tokens + attn)

        ff = nn.Dense(self.ff_dim, name="ff1")(tokens)
        ff = nn.gelu(ff)
        ff = nn.Dropout(self.dropout, name="ff_drop")(ff, deterministic=deterministic)
        ff = nn.Dense(self.d_model, name="ff2")(ff)
        return nn.LayerNorm(name="norm2")(tokens + ff)


class ITransformerForecaster(nn.Module):
    output_shape: tuple[int, int]  # (HorizonSteps, output_dim)
    target_indices: tuple[int, ...]
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_blocks: int = 2
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        horizon, output_dim = self.output_shape
        if len(self.target_indices) != output_dim:
            raise ValueError(
                f"target_indices {self.target_indices} does not match output_dim {output_dim}"
            )

        # (N, L, F) -> (N, F, L): one token per variate.
        tokens = jnp.swapaxes(x, 1, 2)
        tokens = nn.Dense(self.d_model, name="embed")(tokens)
        tokens = nn.Dropout(self.dropout, name="embed_drop")(tokens, deterministic=deterministic)

        for i in range(self.num_blocks):
            tokens = ITransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f"block{i}",
            )(tokens, deterministic=deterministic)

        # Shared per-token head, then select the target variates' tokens.
        y = nn.Dense(horizon, name="head")(tokens)  # (N, F, H)
        y = y[:, jnp.asarray(self.target_indices), :]  # (N, output_dim, H)
        return jnp.swapaxes(y, 1, 2)  # (N, H, output_dim)


def build_itransformer_model(
    input_shape,
    output_shape,
    target_indices=(0,),
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.1,
):
    del input_shape
    return ITransformerForecaster(
        output_shape=tuple(output_shape),
        target_indices=tuple(int(i) for i in target_indices),
        d_model=int(d_model),
        num_heads=int(num_heads),
        ff_dim=int(ff_dim),
        num_blocks=int(num_blocks),
        dropout=float(dropout),
    )
