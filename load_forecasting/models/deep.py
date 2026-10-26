"""Deep learning model builders for load forecasting."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tcn import TCN


def build_mlp_model(input_shape, output_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape, name="in"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", name="fc1"),
        layers.Dropout(0.2, name="drop1"),
        layers.Dense(32, activation="relu", name="fc2"),
        layers.Dense(16, activation="relu", name="fc3"),
        layers.Dense(np.prod(output_shape), name="out_flat"),
        layers.Reshape(output_shape, name="out")
    ])


def build_gru_model(input_shape, output_shape):
    return models.Sequential([
        layers.Input(shape=input_shape, name="in"),
        layers.GRU(64, return_sequences=False, name="gru1"),
        layers.Dropout(0.2, name="drop1"),
        layers.Dense(32, activation="relu", name="fc1"),
        layers.Dense(np.prod(output_shape), name="out_flat"),
        layers.Reshape(output_shape, name="out")
    ])


def build_rnn_model(input_shape, output_shape):
    return models.Sequential([
        layers.Input(shape=input_shape, name="in"),
        layers.LSTM(64, return_sequences=False, name="lstm1"),
        layers.Dropout(0.2, name="drop1"),
        layers.Dense(32, activation="relu", name="fc1"),
        layers.Dense(np.prod(output_shape), name="out_flat"),
        layers.Reshape(output_shape, name="out")
    ])


def receptive_field(kernel_sizes, dilations):
    rf = 1
    for k, d in zip(kernel_sizes, dilations):
        rf += (k - 1) * d
    return rf


def build_cnn_model(input_shape, output_shape):
    return models.Sequential([
        layers.Input(shape=input_shape, name="in"),
        layers.Conv1D(64, kernel_size=3, padding="causal", dilation_rate=1, activation="relu", name="conv1"),
        layers.Conv1D(64, kernel_size=3, padding="causal", dilation_rate=2, activation="relu", name="conv2"),
        layers.Conv1D(64, kernel_size=3, padding="causal", dilation_rate=4, activation="relu", name="conv3"),
        layers.Conv1D(64, kernel_size=3, padding="causal", dilation_rate=8, activation="relu", name="conv4"),
        layers.Lambda(lambda x: x[:, -1, :], name="lambda_layer"),
        layers.Dense(96, activation="relu"),
        layers.Dropout(0.15, name="drop2"),
        layers.Dense(np.prod(output_shape), name="out_flat"),
        layers.Reshape(output_shape, name="out")
    ])


def build_tcn_model(input_shape, output_shape):
    return models.Sequential([
        layers.Input(shape=input_shape, name="in"),
        TCN(nb_filters=96,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            padding='causal',
            dropout_rate=0.1,
            use_skip_connections=True,
            return_sequences=False,
            name="tcn"),
        layers.Dense(64, activation="relu", name="fc_tcn"),
        layers.Dense(np.prod(output_shape), name="out_flat"),
        layers.Reshape(output_shape, name="out")
    ])


class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        # x: (batch, seq_len, d_model)
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos = self.pos_emb(positions)  # (seq_len, d_model)
        return x + pos


def transformer_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    # Pre-norm attention
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x_norm, x_norm)
    attn = layers.Dropout(dropout)(attn)
    x = x + attn

    # Pre-norm feed-forward
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ff_dim, activation="gelu")(x_norm)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = x + ff

    return x


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
    """
    Encoder-only Transformer for direct multi-step forecasting.
    input_shape = (LookbackSteps, num_features)
    output_shape = (HorizonSteps, output_dim)
    """
    seq_len, num_features = input_shape
    inputs = keras.Input(shape=input_shape, name="in")

    x = inputs

    # Optional causal/local feature extractor before attention
    if use_conv_stem:
        x = layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            padding="causal",
            activation="relu",
            name="conv_stem"
        )(x)
    else:
        x = layers.Dense(d_model, name="input_projection")(x)

    x = PositionalEmbedding(seq_len=seq_len, d_model=d_model, name="pos_embedding")(x)

    for i in range(num_transformer_blocks):
        x = transformer_block(
            x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

    # Better than GlobalAveragePooling1D for forecasting
    # Keep recency by reading the last token
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, -1, :]   # last timestep representation

    for i, dim in enumerate(mlp_units):
        x = layers.Dense(dim, activation="gelu", name=f"mlp_{i}")(x)
        x = layers.Dropout(mlp_dropout)(x)

    x = layers.Dense(np.prod(output_shape), name="out_flat")(x)
    outputs = layers.Reshape(output_shape, name="out")(x)

    return keras.Model(inputs, outputs)
