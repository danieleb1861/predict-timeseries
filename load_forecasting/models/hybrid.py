"""Hybrid model builders for load forecasting."""

import numpy as np
from tensorflow import keras
from keras import layers
from tcn import TCN


def build_conv_lstm_hybrid(input_shape, output_shape):
    """
    Build a hybrid forecasting model that combines convolutional layers (for local feature extraction) with an LSTM layer (for temporal dependencies).
    """
    inputs = keras.Input(shape=input_shape, name="in")
    concatenateNN = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inputs)
    concatenateNN = layers.Conv1D(64, kernel_size=3, padding="causal", dilation_rate=2, activation="relu")(concatenateNN)
    concatenateNN = layers.LSTM(64, return_sequences=False)(concatenateNN)
    concatenateNN = layers.Dropout(0.1)(concatenateNN)
    concatenateNN = layers.Dense(64, activation="relu")(concatenateNN)
    concatenateNN = layers.Dense(np.prod(output_shape))(concatenateNN)
    outputs = layers.Reshape(output_shape)(concatenateNN)
    return keras.Model(inputs, outputs, name="ConvLSTM")


def build_tcn_lstm_hybrid(input_shape, output_shape):
    """
    Build a hybrid forecasting model that combines Temporal Convolutional Network (TCN) layers (for capturing long-range temporal dependencies with dilated convolutions) with an LSTM layer (for sequential modeling).
    """
    inputs = keras.Input(shape=input_shape, name="in")
    concatenateNN = TCN(nb_filters=48, kernel_size=3, dilations=(1, 2, 4), padding='causal',
            dropout_rate=0.1, use_skip_connections=True, return_sequences=True)(inputs)
    concatenateNN = layers.LSTM(48, return_sequences=False)(concatenateNN)
    concatenateNN = layers.Dropout(0.1)(concatenateNN)
    concatenateNN = layers.Dense(64, activation="relu")(concatenateNN)
    concatenateNN = layers.Dense(np.prod(output_shape))(concatenateNN)
    outputs = layers.Reshape(output_shape)(concatenateNN)
    return keras.Model(inputs, outputs, name="TCNLSTM")


def build_conv_tcn_parallel(input_shape, output_shape):
    """
    Build a parallel hybrid forecasting model that processes the input sequence through two separate branches: one with convolutional layers (for local feature extraction) and another with Temporal Convolutional Network (TCN) layers (for capturing long-range temporal dependencies). The outputs of both branches are then concatenated and passed through fully connected layers to produce the final forecast.
    Leverage both local patterns and long-term dependencies in the data.
    """
    inputs = keras.Input(shape=input_shape, name="in")
    branch_conv = layers.Conv1D(48, kernel_size=3, padding="causal", activation="relu")(inputs)
    branch_conv = layers.Conv1D(48, kernel_size=3, padding="causal", dilation_rate=2, activation="relu")(branch_conv)
    branch_conv = layers.GlobalAveragePooling1D()(branch_conv)
    branch_tcn = TCN(nb_filters=48, kernel_size=3, dilations=(1, 2, 4, 8), padding='causal',
             dropout_rate=0.1, use_skip_connections=True, return_sequences=False)(inputs)
    concatenateNN = layers.Concatenate()([branch_conv, branch_tcn])
    concatenateNN = layers.Dense(80, activation="relu")(concatenateNN)
    concatenateNN = layers.Dropout(0.15)(concatenateNN)
    concatenateNN = layers.Dense(np.prod(output_shape))(concatenateNN)
    outputs = layers.Reshape(output_shape)(concatenateNN)
    return keras.Model(inputs, outputs, name="ConvTCN")


def build_conv_lstm_tcn_parallel(input_shape, output_shape):
    """
    Build a parallel hybrid forecasting model that processes the input sequence through three separate branches: one with convolutional layers (for local feature extraction), another with an LSTM layer (for sequential modeling), and a third branch with Temporal Convolutional Network (TCN) layers (for capturing long-range temporal dependencies). The outputs of all three branches are then concatenated and passed through fully connected layers to produce the final forecast.
    """
    inputs = keras.Input(shape=input_shape, name="in")
    branch_conv = layers.Conv1D(48, kernel_size=3, padding="causal", activation="relu")(inputs)
    branch_conv = layers.Conv1D(48, kernel_size=3, padding="causal", dilation_rate=2, activation="relu")(branch_conv)
    branch_conv = layers.GlobalAveragePooling1D()(branch_conv)
    branch_lstm = layers.LSTM(40, return_sequences=False)(inputs)
    branch_tcn = TCN(nb_filters=48, kernel_size=3, dilations=(1, 2, 4, 8), padding='causal',
             dropout_rate=0.1, use_skip_connections=True, return_sequences=False)(inputs)
    concatenateNN = layers.Concatenate()([branch_conv, branch_lstm, branch_tcn])
    concatenateNN = layers.Dense(96, activation="relu")(concatenateNN)
    concatenateNN = layers.Dropout(0.15)(concatenateNN)
    concatenateNN = layers.Dense(np.prod(output_shape))(concatenateNN)
    outputs = layers.Reshape(output_shape)(concatenateNN)
    return keras.Model(inputs, outputs, name="ConvLSTMTCN")
