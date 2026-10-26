"""Training and evaluation helpers for forecasting models."""

import os
from time import time as tmer

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from load_forecasting.metrics import compute_metrics
from load_forecasting.training_deep import save_predictions
from load_forecasting.windowing import flatten_XY, unflatten_Y


def train_eval_flat_model(
    model_key,
    model_label,
    model,
    Xtr,
    Ytr,
    Xte,
    Yte,
    output_shape,
    run_dir,
    inv_scale,
    y_true_phys,
    d_plot,
    log_time,
    Xva=None,
    ):
    """
    Train a scikit-learn model on flattened data (Xtr,Ytr), then evaluate on the test set (Xte,Yte). This function:
    - saves the trained model to disk,
    - runs inference on Xte,
    - converts predictions back to physical units via inv_scale(...),
    - computes regression metrics via compute_metrics(...).

    Inputs
    ------
    model : sklearn.base.BaseEstimator
        Scikit-learn model that maps:
          X : (num_windows, LookbackSteps * num_features)
        to
          Y : (num_windows, HorizonSteps * output_dim)
    model_key : str
        Identifier used for saved filenames.
    model_label : str
        Human-readable label for printing/logging.
    Xtr, Ytr : np.ndarray
        Training windows and targets.
    Xte, Yte : np.ndarray
        Test windows and targets (Yte is not used directly here; metrics use y_true_phys).
    output_shape : tuple
        Shape of the output (HorizonSteps, output_dim) for unflattening predictions.
    run_dir : str
        Directory where the trained model will be saved.
    inv_scale : callable
        Function to convert predictions back to physical units.
    y_true_phys : np.ndarray
        True target values in physical units for computing metrics.
    d_plot : dict
        Dictionary of plotting parameters for compute_metrics(...).
    log_time : callable
        Function to log timing information.
    
    Returns
    -------
    yhat_phys : np.ndarray
        Predicted values in physical units.
    metrics : dict
        Computed regression metrics.
    """
    stage_start = tmer()
    print()
    print(f"Training {model_label}...")
    print()

    Xtr2d, Ytr2d = flatten_XY(Xtr, Ytr)
    Xte2d, _ = flatten_XY(Xte, Yte)

    model.fit(Xtr2d, Ytr2d)
    joblib.dump(model, os.path.join(run_dir, f"model_{model_key}.joblib"))

    start_time = tmer()
    yhat2d = model.predict(Xte2d)
    prediction_elapsed = tmer() - start_time
    yhat = unflatten_Y(yhat2d, output_shape)
    yhat_phys = inv_scale(yhat)
    yhat_val_phys = None
    if Xva is not None:
        Xva2d = Xva.reshape(Xva.shape[0], -1)
        yhat_val_phys = inv_scale(unflatten_Y(model.predict(Xva2d), output_shape))
    save_predictions(run_dir, model_key, model_label, yhat_phys, yhat_val_phys)
    # Keep bootstrap/scoring cost separate from test inference.
    scoring_start = tmer()
    metrics = compute_metrics(y_true_phys, yhat_phys, d_plot, include_confidence=True)
    scoring_elapsed = tmer() - scoring_start

    print(f"{model_label} RMSE:", metrics["RMSE_horizon"])
    print(f"{model_label} RMSE all (PU):", metrics["RMSE_all"])
    print(f"{model_label} MAE all (PU):", metrics["MAE_all"])
    print(f"{model_label} MAPE all (%):", metrics["MAPE_all"])
    print()
    print(f"{model_label} test prediction time (s):", prediction_elapsed)
    print(f"{model_label} scoring time (s):", scoring_elapsed)
    print()
    log_time(f"model_{model_key}_train_eval", stage_start)
    return yhat_phys, metrics


def train_eval_keras_model(
    model_key,
    model_label,
    model,
    Xtr,
    Ytr,
    Xva,
    Yva,
    Xte,
    Yte,
    run_dir,
    inv_scale,
    y_true_phys,
    d_plot,
    callbacks,
    predict_batch_size,
    imgs_dir,
    save_plot,
    log_time,
    batch_size=128,
    epochs=40,
    shuffle=False,
    lr=1e-3,
    loss="mse",
    artifact_key=None,
    plot_key=None,
    log_key=None,
    ):
    """
    Train a Keras forecasting model on windowed data (Xtr,Ytr), monitor validation loss on (Xva,Yva), then evaluate on the test set (Xte,Yte).
    Additionally, this function:
    - saves the trained model to disk,
    - runs inference on Xte,
    - converts predictions back to physical units via inv_scale(...),
    - computes regression metrics via compute_metrics(...),
    - plots and saves the training/validation loss curves.

    Inputs
    ------
    model : keras.Model
        Compiled/uncompiled Keras model that maps:
          X : (num_windows, LookbackSteps, num_features)
        to
          Y : (num_windows, HorizonSteps, output_dim)
    model_key : str
        Identifier used for saved filenames and plot titles.
    model_label : str
        Human-readable label for printing/logging.
    Xtr, Ytr : np.ndarray
        Training windows and targets.
    Xva, Yva : np.ndarray
        Validation windows and targets.
    Xte, Yte : np.ndarray
        Test windows and targets (Yte is not used directly here; metrics use y_true_phys).
    batch_size : int, default=128
        Mini-batch size for training.
    epochs : int, default=40
        Maximum number of training epochs (callbacks may stop earlier).
    shuffle : bool, default=False
        Whether to shuffle training data at each epoch. For time series forecasting, it's common to set this to False to preserve temporal order.
    lr : float, default=1e-3
        Adam learning rate.
    loss : str | callable, default="mse"
        Loss passed to model.compile(). Can be a string identifier or a Keras loss instance.
    callbacks : list
        List of Keras callbacks to use during training (e.g., EarlyStopping, ReduceLROnPlateau).
    predict_batch_size : int
        Batch size to use during prediction on Xte.
    imgs_dir : str
        Directory where training/validation loss plot will be saved.
    save_plot : callable
        Function to save a matplotlib figure, signature: save_plot(fig, filename).
    inv_scale : callable
        Function to convert predictions back to physical units.
    y_true_phys : np.ndarray
        True target values in physical units for computing metrics.
    d_plot : dict
        Dictionary of plotting parameters for compute_metrics(...).
    log_time : callable
        Function to log timing information.
    artifact_key : str | None
        Identifier for saved artifacts. If None, defaults to model_key.
    plot_key : str | None
        Identifier for saved plots. If None, defaults to model_key.
    log_key : str | None
        Identifier for logged metrics. If None, defaults to model_key.
    
    Returns
    -------
    yhat_phys : np.ndarray
        Test predictions in physical units, shape (num_test_windows, HorizonSteps, output_dim).
    metrics : dict
        Output of compute_metrics(y_true_phys, yhat_phys, d_plot). Contains horizon-wise and aggregate error metrics.
    """
    # Preserve the standard wrapper's two-value return contract.
    _, yhat_phys, metrics = _train_eval_keras(
        model, model_label, Xtr, Ytr, Xva, Yva, Xte, run_dir,
        inv_scale, y_true_phys, d_plot, callbacks, predict_batch_size,
        imgs_dir, save_plot, log_time, batch_size, epochs, shuffle, lr, loss,
        artifact_key=artifact_key or model_key,
        plot_key=plot_key or model_key, log_key=log_key or model_key,
    )
    return yhat_phys, metrics


def _train_eval_keras(
    model, model_label, Xtr, Ytr, Xva, Yva, Xte, run_dir,
    inv_scale, y_true_phys, d_plot, callbacks, predict_batch_size,
    imgs_dir, save_plot, log_time, batch_size, epochs, shuffle, lr, loss,
    *, artifact_key, plot_key, log_key,
):
    """Share fitting, saving, scoring, and plotting across Keras wrappers."""
    from tensorflow import keras

    stage_start = tmer()
    print()
    print(f"Training {model_label}...")
    print()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
    model.summary()
    history = model.fit(
        Xtr, Ytr,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_data=(Xva, Yva),
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(run_dir, f"model_{artifact_key}.keras"))

    start_time = tmer()
    yhat = model.predict(Xte, batch_size=predict_batch_size)
    prediction_elapsed = tmer() - start_time
    yhat_phys = inv_scale(yhat)
    # Keep bootstrap/scoring cost separate from test inference.
    scoring_start = tmer()
    metrics = compute_metrics(y_true_phys, yhat_phys, d_plot, include_confidence=True)
    scoring_elapsed = tmer() - scoring_start

    print(f"{model_label} RMSE:", metrics["RMSE_horizon"])
    print(f"{model_label} RMSE all (PU):", metrics["RMSE_all"])
    print(f"{model_label} MAE all (PU):", metrics["MAE_all"])
    print(f"{model_label} MAPE all (%):", metrics["MAPE_all"])
    print()
    print(f"{model_label} test prediction time (s):", prediction_elapsed)
    print(f"{model_label} scoring time (s):", scoring_elapsed)
    print()

    pd.DataFrame(history.history).plot(figsize=(6.5, 3.0))
    plt.title(f"{model_label} - Training & Validation Loss")
    plt.grid(True)
    save_plot(imgs_dir, plot_key)
    log_time(f"model_{log_key}_train_eval", stage_start)
    return history, yhat_phys, metrics


def train_eval_hybrid(
    model,
    model_name,
    Xtr,
    Ytr,
    Xva,
    Yva,
    Xte,
    Yte,
    run_dir,
    inv_scale,
    y_true_phys,
    d_plot,
    callbacks,
    predict_batch_size,
    imgs_dir,
    save_plot,
    log_time,
    batch_size=128,
    epochs=40,
    shuffle=False,
    lr=7e-4,
    loss=None,
):
    """
    Train a Keras forecasting model on windowed data (Xtr,Ytr), monitor validation loss on (Xva,Yva), then evaluate on the test set (Xte,Yte). Built to clean the code with multiple hybrid architectures.
    Additionally, this function:
    - saves the trained model to disk,
    - runs inference on Xte,
    - converts predictions back to physical units via inv_scale(...),
    - computes regression metrics via compute_metrics(...),
    - plots and saves the training/validation loss curves.

    Inputs
    ------
    model : keras.Model
        Compiled/uncompiled Keras model that maps:
          X : (num_windows, LookbackSteps, num_features)
        to
          Y : (num_windows, HorizonSteps, output_dim)
    model_name : str
        Identifier used for saved filenames and plot titles.
    Xtr, Ytr : np.ndarray
        Training windows and targets.
    Xva, Yva : np.ndarray
        Validation windows and targets.
    Xte, Yte : np.ndarray
        Test windows and targets (Yte is not used directly here; metrics use y_true_phys).
    batch_size : int, default=128
        Mini-batch size for training.
    epochs : int, default=40
        Maximum number of training epochs (callbacks may stop earlier).
    shuffle : bool, default=False
        Whether to shuffle training data at each epoch. For time series forecasting, it's common to set this to False to preserve temporal order.
    lr : float, default=7e-4
        Adam learning rate.
    loss : str | callable | None, default=None
        Loss passed to model.compile(). If None, uses Huber(delta=1.0).

    Returns
    -------
    history : keras.callbacks.History
        Training history returned by model.fit().
    yhat_phys : np.ndarray
        Test predictions in physical units, shape (num_test_windows, HorizonSteps, output_dim).
    metrics : dict
        Output of compute_metrics(y_true_phys, yhat_phys, d_plot). Contains horizon-wise and aggregate error metrics.
    """
    # Keep the hybrid default loss and its three-value return contract.
    if loss is None:
        from tensorflow import keras

        loss = keras.losses.Huber(delta=1.0)
    return _train_eval_keras(
        model, model_name, Xtr, Ytr, Xva, Yva, Xte, run_dir,
        inv_scale, y_true_phys, d_plot, callbacks, predict_batch_size,
        imgs_dir, save_plot, log_time, batch_size, epochs, shuffle, lr, loss,
        artifact_key=model_name, plot_key=model_name.lower(), log_key=model_name,
    )
