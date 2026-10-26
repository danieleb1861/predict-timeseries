"""Train-only scaling helpers for forecasting datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_train_scaler(train_frame, columns):
    """
    Fit a StandardScaler on training rows only.

    Inputs
    ----------
    train_frame : pandas.DataFrame
        Training split containing the continuous columns to standardize.
    columns : list[str]
        Column names used to fit the scaler.

    Returns
    -------
    tuple
        Tuple ``(scaler, mu, sig)`` where ``scaler`` is the fitted
        StandardScaler, ``mu`` is a pandas.Series of training means, and
        ``sig`` is a pandas.Series of training standard deviations.
    """
    scaler = StandardScaler()
    scaler.fit(train_frame[columns])

    mu = pd.Series(scaler.mean_, index=columns).astype("float32")
    sig = pd.Series(np.sqrt(scaler.var_), index=columns).astype("float32")
    sig[sig == 0] = 1.0

    return scaler, mu, sig


def transform_with_scaler(data_frame, columns, scaler):
    """
    Apply an already-fitted scaler and return a DataFrame aligned to input rows.

    Inputs
    ----------
    data_frame : pandas.DataFrame
        DataFrame containing the columns to transform.
    columns : list[str]
        Column names to transform.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used for transformation.

    Returns
    -------
    pandas.DataFrame
        Standardized data with the same index as ``data_frame``.
    """
    values = scaler.transform(data_frame[columns])
    return pd.DataFrame(values, columns=columns, index=data_frame.index)


def scale_train_val_test(train_frame, val_frame, test_frame, columns):
    """
    Scale train/validation/test splits using train-only statistics.

    Inputs
    ----------
    train_frame : pandas.DataFrame
        Training split used to fit the scaler.
    val_frame : pandas.DataFrame
        Validation split transformed with training statistics.
    test_frame : pandas.DataFrame
        Test split transformed with training statistics.
    columns : list[str]
        Continuous column names to standardize.

    Returns
    -------
    tuple
        Tuple ``(train_z, val_z, test_z, mu, sig)`` containing standardized
        DataFrames and train-fitted scaling metadata.
    """
    scaler, mu, sig = fit_train_scaler(train_frame, columns)
    train_z = transform_with_scaler(train_frame, columns, scaler)
    val_z = transform_with_scaler(val_frame, columns, scaler)
    test_z = transform_with_scaler(test_frame, columns, scaler)

    return train_z, val_z, test_z, mu, sig


def scale_full_dataset_for_diagnostics(data_frame, columns):
    """
    Fit and transform the full dataset for diagnostics-only workflows.

    Inputs
    ----------
    data_frame : pandas.DataFrame
        Full dataset to normalize for diagnostic branches.
    columns : list[str]
        Continuous column names to standardize.

    Returns
    -------
    pandas.DataFrame
        Full-dataset standardized DataFrame preserving the historical ``TTZ``
        behavior used by PCA/dPCA diagnostics.
    """
    scaler = StandardScaler()
    scaler.fit(data_frame[columns])
    return transform_with_scaler(data_frame, columns, scaler)


def inverse_scale_targets(z, target_names, mu, sig):
    """
    Restore normalized target predictions to physical/per-unit scale.

    Inputs
    ----------
    z : array-like
        Normalized target predictions. Typical shape is
        ``(num_windows, HorizonSteps, output_dim)``.
    target_names : list[str]
        Target column names corresponding to the last dimension of ``z``.
    mu : pandas.Series
        Training means indexed by continuous variable name.
    sig : pandas.Series
        Training standard deviations indexed by continuous variable name.

    Returns
    -------
    numpy.ndarray
        Predictions restored to physical/per-unit scale with the same shape as
        ``z``.
    """
    z = np.asarray(z, dtype=np.float32)
    mu_t = mu[target_names].to_numpy(dtype="float32")
    sig_t = sig[target_names].to_numpy(dtype="float32")

    shape = (1,) * (z.ndim - 1) + (-1,)
    return z * sig_t.reshape(shape) + mu_t.reshape(shape)
