"""Dual-stream FiLM forecaster for short-term active-power prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import jax.numpy as jnp
from flax import linen as nn

from load_forecasting.models.deep_flax import (
    CausalConv1D,
    GRUCell,
    dilations_for_lookback,
    receptive_field,
)


FEATURE_GROUP_NAMES = ("endogenous", "propulsion", "environment")
HEAD_MODES = ("flat_direct", "horizon_residual")


def _normalize_indices(values: Sequence[int], name: str) -> tuple[int, ...]:
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{name} must contain only integer indices")
    return tuple(int(value) for value in values)


def validate_feature_group_indices(
    num_features: int,
    output_dim: int,
    endogenous_indices: Sequence[int],
    propulsion_indices: Sequence[int],
    environment_indices: Sequence[int],
    target_indices: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate a complete, disjoint routing of input channels."""
    if num_features < 1 or output_dim < 1:
        raise ValueError("num_features and output_dim must be positive")

    endogenous = _normalize_indices(endogenous_indices, "endogenous_indices")
    propulsion = _normalize_indices(propulsion_indices, "propulsion_indices")
    environment = _normalize_indices(environment_indices, "environment_indices")
    targets = _normalize_indices(target_indices, "target_indices")
    groups = endogenous + propulsion + environment

    if any(index < 0 or index >= num_features for index in groups + targets):
        raise ValueError(f"feature indices must be between 0 and {num_features - 1}")
    if len(set(groups)) != len(groups):
        raise ValueError("feature groups must be disjoint and contain no duplicates")
    if set(groups) != set(range(num_features)):
        raise ValueError("feature groups must cover every input feature exactly once")
    if len(targets) != output_dim:
        raise ValueError(
            f"target_indices has length {len(targets)}, expected output dimension {output_dim}"
        )
    if not set(targets).issubset(endogenous):
        raise ValueError("target_indices must belong to the endogenous feature group")
    return endogenous, propulsion, environment, targets


def resolve_feature_group_indices(
    feature_names: Sequence[str],
    target_names: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    """Resolve configured feature names to the immutable indices used by Flax."""
    feature_names = tuple(feature_names)
    target_names = tuple(target_names)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must not contain duplicates")
    if not isinstance(feature_groups, Mapping):
        raise ValueError("feature_groups must be an object")

    keys = set(feature_groups)
    expected_keys = set(FEATURE_GROUP_NAMES)
    if keys != expected_keys:
        raise ValueError(
            f"feature_groups must contain exactly {FEATURE_GROUP_NAMES}; "
            f"missing={sorted(expected_keys - keys)}, unknown={sorted(keys - expected_keys)}"
        )

    lookup = {name: index for index, name in enumerate(feature_names)}
    resolved = {}
    for group_name in FEATURE_GROUP_NAMES:
        names = feature_groups[group_name]
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError(f"feature_groups.{group_name} must be a non-empty list")
        unknown = [name for name in names if name not in lookup]
        if unknown:
            raise ValueError(f"feature_groups.{group_name} contains unknown features: {unknown}")
        resolved[f"{group_name}_indices"] = tuple(lookup[name] for name in names)

    unknown_targets = [name for name in target_names if name not in lookup]
    if unknown_targets:
        raise ValueError(f"target_names contains unknown features: {unknown_targets}")
    resolved["target_indices"] = tuple(lookup[name] for name in target_names)

    validated = validate_feature_group_indices(
        len(feature_names),
        len(target_names),
        resolved["endogenous_indices"],
        resolved["propulsion_indices"],
        resolved["environment_indices"],
        resolved["target_indices"],
    )
    return dict(zip(
        (
            "endogenous_indices",
            "propulsion_indices",
            "environment_indices",
            "target_indices",
        ),
        validated,
    ))


class PreNormCausalResidualBlock(nn.Module):
    features: int
    kernel_size: int
    dilation_rate: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        residual = x
        x = nn.LayerNorm(name="norm")(x)
        x = CausalConv1D(
            self.features,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            name="causal_conv",
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, name="dropout")(
            x,
            deterministic=deterministic,
        )
        return residual + x


class BoundedFiLM(nn.Module):
    features: int
    scale: float = 0.5

    @nn.compact
    def __call__(self, dynamics, context):
        modulation = nn.Dense(
            2 * self.features,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="projection",
        )(context)
        gamma_raw, beta_raw = jnp.split(modulation, 2, axis=-1)
        gamma = self.scale * jnp.tanh(gamma_raw)
        beta = self.scale * jnp.tanh(beta_raw)
        return dynamics * (1.0 + gamma) + beta


class FlatDirectHead(nn.Module):
    output_shape: tuple[int, int]
    hidden_width: int
    dropout_rate: float

    @nn.compact
    def __call__(self, state, deterministic: bool = True):
        horizon, output_dim = self.output_shape
        x = nn.Dense(self.hidden_width, name="hidden")(state)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, name="dropout")(
            x,
            deterministic=deterministic,
        )
        x = nn.Dense(horizon * output_dim, name="output")(x)
        return x.reshape((x.shape[0], horizon, output_dim))


class HorizonResidualHead(nn.Module):
    output_shape: tuple[int, int]
    horizon_embed_dim: int
    hidden_width: int
    dropout_rate: float

    @nn.compact
    def __call__(self, state, anchor, deterministic: bool = True):
        horizon, output_dim = self.output_shape
        horizon_ids = jnp.arange(horizon)
        embeddings = nn.Embed(
            num_embeddings=horizon,
            features=self.horizon_embed_dim,
            name="horizon_embedding",
        )(horizon_ids)
        embeddings = jnp.broadcast_to(
            embeddings[None, :, :],
            (state.shape[0], horizon, self.horizon_embed_dim),
        )
        state = jnp.broadcast_to(
            state[:, None, :],
            (state.shape[0], horizon, state.shape[-1]),
        )
        lead_time = ((horizon_ids + 1) / float(horizon))[None, :, None]
        lead_time = jnp.broadcast_to(lead_time, (state.shape[0], horizon, 1))

        query = jnp.concatenate([state, embeddings, lead_time], axis=-1)
        query = nn.Dense(self.hidden_width, name="hidden")(query)
        query = nn.gelu(query)
        query = nn.Dropout(self.dropout_rate, name="dropout")(
            query,
            deterministic=deterministic,
        )
        delta = nn.Dense(
            output_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="output",
        )(query)
        return anchor[:, None, :] + delta


class DualStreamFiLMForecaster(nn.Module):
    output_shape: tuple[int, int]
    endogenous_indices: tuple[int, ...]
    propulsion_indices: tuple[int, ...]
    environment_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    dynamics_width: int = 48
    command_hidden: int = 16
    environment_hidden: int = 8
    context_width: int = 24
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout_rate: float = 0.1
    film_scale: float = 0.5
    head_mode: str = "flat_direct"
    horizon_embed_dim: int = 16
    decoder_width: int = 64

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        if x.ndim != 3:
            raise ValueError(f"expected input shape (batch, lookback, features), got {x.shape}")
        validate_feature_group_indices(
            int(x.shape[-1]),
            self.output_shape[1],
            self.endogenous_indices,
            self.propulsion_indices,
            self.environment_indices,
            self.target_indices,
        )

        endogenous = jnp.take(x, jnp.asarray(self.endogenous_indices), axis=-1)
        dynamics = nn.Dense(self.dynamics_width, name="dynamics_projection")(endogenous)
        dynamics = nn.gelu(dynamics)
        for dilation in self.dilations:
            dynamics = PreNormCausalResidualBlock(
                features=self.dynamics_width,
                kernel_size=self.kernel_size,
                dilation_rate=dilation,
                dropout_rate=self.dropout_rate,
                name=f"dynamics_d{dilation}",
            )(dynamics, deterministic=deterministic)
        dynamics = nn.LayerNorm(name="dynamics_output_norm")(dynamics[:, -1, :])

        propulsion = jnp.take(x, jnp.asarray(self.propulsion_indices), axis=-1)
        command_state = jnp.zeros(
            (x.shape[0], self.command_hidden),
            dtype=x.dtype,
        )
        command_encoder = nn.scan(
            GRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(
            hidden_size=self.command_hidden,
            name="propulsion_encoder",
        )
        command_state, _ = command_encoder(command_state, propulsion)

        environment = jnp.take(x, jnp.asarray(self.environment_indices), axis=-1)
        environment_slope = (
            environment[:, -1, :] - environment[:, 0, :]
        ) / float(max(int(x.shape[1]) - 1, 1))
        environment_summary = jnp.concatenate(
            [
                environment[:, -1, :],
                jnp.mean(environment, axis=1),
                environment_slope,
            ],
            axis=-1,
        )
        environment_state = nn.Dense(
            self.environment_hidden,
            name="environment_encoder",
        )(environment_summary)
        environment_state = nn.gelu(environment_state)

        context = jnp.concatenate([command_state, environment_state], axis=-1)
        context = nn.Dense(self.context_width, name="context_projection")(context)
        context = nn.gelu(context)
        fused = BoundedFiLM(
            features=self.dynamics_width,
            scale=self.film_scale,
            name="film",
        )(dynamics, context)

        anchor = jnp.take(x[:, -1, :], jnp.asarray(self.target_indices), axis=-1)
        if self.head_mode == "flat_direct":
            return FlatDirectHead(
                output_shape=self.output_shape,
                hidden_width=self.decoder_width,
                dropout_rate=self.dropout_rate,
                name="flat_head",
            )(fused, deterministic=deterministic)
        if self.head_mode == "horizon_residual":
            return HorizonResidualHead(
                output_shape=self.output_shape,
                horizon_embed_dim=self.horizon_embed_dim,
                hidden_width=self.decoder_width,
                dropout_rate=self.dropout_rate,
                name="horizon_head",
            )(fused, anchor, deterministic=deterministic)
        raise ValueError(f"head_mode must be one of {HEAD_MODES}, got {self.head_mode!r}")


def build_dual_stream_film_model(
    input_shape,
    output_shape,
    endogenous_indices,
    propulsion_indices,
    environment_indices,
    target_indices=(0,),
    dynamics_width=48,
    command_hidden=16,
    environment_hidden=8,
    context_width=24,
    kernel_size=3,
    dilations=None,
    dropout_rate=0.1,
    film_scale=0.5,
    head_mode="flat_direct",
    horizon_embed_dim=16,
    decoder_width=64,
):
    lookback, num_features = (int(value) for value in input_shape)
    horizon, output_dim = (int(value) for value in output_shape)
    endogenous, propulsion, environment, targets = validate_feature_group_indices(
        num_features,
        output_dim,
        endogenous_indices,
        propulsion_indices,
        environment_indices,
        target_indices,
    )

    integer_widths = {
        "dynamics_width": dynamics_width,
        "command_hidden": command_hidden,
        "environment_hidden": environment_hidden,
        "context_width": context_width,
        "horizon_embed_dim": horizon_embed_dim,
        "decoder_width": decoder_width,
    }
    if any(int(value) < 1 for value in integer_widths.values()):
        raise ValueError(f"model widths must be positive: {integer_widths}")
    if int(kernel_size) < 2:
        raise ValueError("kernel_size must be at least 2")
    if not 0.0 <= float(dropout_rate) < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    if float(film_scale) <= 0.0:
        raise ValueError("film_scale must be positive")
    if head_mode not in HEAD_MODES:
        raise ValueError(f"head_mode must be one of {HEAD_MODES}, got {head_mode!r}")

    if dilations is None:
        dilations = dilations_for_lookback(lookback, int(kernel_size))
    dilations = tuple(int(value) for value in dilations)
    if not dilations or any(value < 1 for value in dilations):
        raise ValueError("dilations must contain positive integers")
    if receptive_field([int(kernel_size)] * len(dilations), dilations) < lookback:
        raise ValueError("dilations do not cover the configured lookback")

    return DualStreamFiLMForecaster(
        output_shape=(horizon, output_dim),
        endogenous_indices=endogenous,
        propulsion_indices=propulsion,
        environment_indices=environment,
        target_indices=targets,
        dynamics_width=int(dynamics_width),
        command_hidden=int(command_hidden),
        environment_hidden=int(environment_hidden),
        context_width=int(context_width),
        kernel_size=int(kernel_size),
        dilations=dilations,
        dropout_rate=float(dropout_rate),
        film_scale=float(film_scale),
        head_mode=str(head_mode),
        horizon_embed_dim=int(horizon_embed_dim),
        decoder_width=int(decoder_width),
    )
