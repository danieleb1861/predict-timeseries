"""Post-processing helpers for forecast outputs."""

import numpy as np


def stitch_by_overlap(y_pred, anchors, test_time_length, HorizonSteps, mode="horizon_weighted", horizon_weights=None):
    """
    Aggregate overlapping window forecasts (num_windows, HorizonSteps, output_dim) into a single full timeline (test_time_length, output_dim).
    Intended for plotting/diagnostics (not needed for training).
    """
    num_windows, H2, output_dim = y_pred.shape
    assert H2 == HorizonSteps

    if horizon_weights is None:
        # default weights based on mode
        if mode == "mean":
            # Equal weights for all steps in the horizon (w_k = 1)
            horizon_weights = np.ones(HorizonSteps, dtype=np.float64)
        elif mode == "horizon_weighted":
            # Weights inversely proportional to horizon step (w_k = 1/(k+1)) to give more importance to near-term predictions
            horizon_weights = 1.0 / (np.arange(HorizonSteps, dtype=np.float64) + 1.0)
        else:
            raise ValueError(f"Unsupported stitching mode: {mode}")
    else:
        # custom weights provided by user
        horizon_weights = np.asarray(horizon_weights, dtype=np.float64)
        if horizon_weights.shape[0] != HorizonSteps:
            raise ValueError(f"Expected {HorizonSteps} horizon weights, got {horizon_weights.shape[0]}")

    num = np.zeros((test_time_length, output_dim), dtype=np.float64)
    den = np.zeros((test_time_length, 1), dtype=np.float64)
    boundaries = []

    for num_anchor, time_index in enumerate(anchors[:num_windows]):
        # The prediction value y_pred[n, k] corresponds to time index = anchors[num_anchor] + 1 + k
        start = int(time_index) + 1
        if start >= test_time_length:
            continue
        boundaries.append(start)
        horizon_window = min(HorizonSteps, test_time_length - start) # adjust horizon window if it exceeds timeline length
        weights = horizon_weights[:horizon_window].reshape(-1, 1)
        # num, den for weighted average:
        num[start:start+horizon_window, :] += y_pred[num_anchor, :horizon_window, :] * weights
        den[start:start+horizon_window, :] += weights

    y_stitch = np.full((test_time_length, output_dim), np.nan, dtype=np.float64)
    valid = den[:, 0] > 0
    y_stitch[valid, :] = num[valid, :] / den[valid, :]
    return y_stitch, np.array(boundaries, dtype=np.int64)


def stitch_prediction_dict(models_phys, anchors, time_length_test, HorizonSteps, d_plot, mode="horizon_weighted"):
    """
    Build full-length (time_length_test,) prediction series for multiple models by stitching and overlapping
    multi-horizon window forecasts. Meant to be used for plotting the forecast series against the measured time series.
    """
    preds_full = {}
    boundaries = None
    for name, y_pred in models_phys.items():
        yhat_full, b = stitch_by_overlap(y_pred, anchors, time_length_test, HorizonSteps, mode=mode)
        preds_full[name] = yhat_full[:, d_plot]
        if b is not None:
            boundaries = b
    return preds_full, boundaries
