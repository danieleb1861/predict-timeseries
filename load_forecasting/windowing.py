"""Window construction and flattening helpers for multi-step forecasting."""

import numpy as np


def make_windows(data_frame, featureNames, targetNames, LookbackSteps, HorizonSteps,
                 WindowSlidingCoeff=2, valid_col=None, group_col=None):
    """
    Build supervised learning windows from a time-ordered DataFrame.

    Inputs
    ----------
    data_frame : pandas.DataFrame
        DataFrame containing normalized features and targets, ordered by time.
    featureNames : list[str]
        Feature column names.
    targetNames : list[str]
        Target column names.
    LookbackSteps : int
        Number of past time steps used as input (L).
    HorizonSteps : int
        Number of future time steps to predict (H).
    WindowSlidingCoeff : int, default=2
        Stride between consecutive windows.
    valid_col : str | None, default=None
        Optional column name with 1=valid / 0=invalid samples. If provided, any
        window that includes at least one invalid sample in
        [t-L+1 : t+H] is skipped.
    group_col : str | None, default=None
        Optional group identifier. If provided, windows spanning a change in
        group (for example, a voyage-leg boundary) are skipped.

    Returns
    -------
    X_tensor : np.ndarray
        Features Tensor of dim(X_tensor) = [num_windows, LookbackSteps, num_features]
    Y_tensor : np.ndarray
        Targets Tensor of dim(Y_tensor) = [num_windows, HorizonSteps, output_dim]

    Note: regularly spaced surviving windows are read-only strided views over
    the base series, including after validity/group filtering. Irregular windows
    are copied; use .copy() explicitly when writable arrays are needed.
    """

    """
    ASCII diagram:
        DataFrame
        |
        +--> featureNames --> Xall -------------------+
        |                                             |
        +--> targetNames  --> Yall ----------------+  |
        |                                          |  |
        +--> valid_col?   --> valid (optional)     |  |
                                                   |  |
        anchors ---------------------------------> for each t
                                                   |
                                        valid check on [t-L+1 : t+H]
                                                   |
                                        +----------+----------+
                                        |                     |
                                        fail                  pass
                                        |                     |
                                        skip          index strided views over base arrays
                                                              |
                                                        return X, Y
    """
    Xall_time = data_frame[featureNames].to_numpy(dtype=np.float32)
    Yall_time = data_frame[targetNames].to_numpy(dtype=np.float32)

    T = len(data_frame)
    anchors = np.arange(LookbackSteps - 1, # first anchor with full lookback available
                        T - HorizonSteps,  # last anchor so that t+H <= T-1
                        WindowSlidingCoeff # step size between windows (e.g. HorizonSteps/5 means 80% overlap)
                    )  # array of time indices for windowing
    if group_col is not None and anchors.size:
        groups = data_frame[group_col].to_numpy()
        group_change = np.zeros(T, dtype=np.int32)
        group_change[1:] = groups[1:] != groups[:-1]
        change_cumsum = np.concatenate(([0], np.cumsum(group_change)))
        starts = anchors - LookbackSteps + 1
        stops = anchors + HorizonSteps + 1
        change_counts = change_cumsum[stops] - change_cumsum[starts + 1]
        anchors = anchors[change_counts == 0]

    if valid_col is not None and anchors.size:
        valid = data_frame[valid_col].to_numpy(dtype=np.int8)
        invalid = (valid == 0).astype(np.int32)
        invalid_cumsum = np.concatenate(([0], np.cumsum(invalid)))
        starts = anchors - LookbackSteps + 1
        stops = anchors + HorizonSteps + 1
        invalid_counts = invalid_cumsum[stops] - invalid_cumsum[starts]
        anchors = anchors[invalid_counts == 0]

    num_windows = len(anchors)
    if num_windows == 0:
        X_tensor = np.zeros((0, LookbackSteps, len(featureNames)), dtype=np.float32)
        Y_tensor = np.zeros((0, HorizonSteps, len(targetNames)), dtype=np.float32)
    else:
        # Zero-copy windowing: windows are read-only strided views over the base
        # series, so memory stays O(T*F) instead of O(num_windows*L*F).
        # sliding_window_view appends the window axis last -> transpose to (N, L, F).
        X_view = np.lib.stride_tricks.sliding_window_view(
            Xall_time, LookbackSteps, axis=0).transpose(0, 2, 1)
        Y_view = np.lib.stride_tricks.sliding_window_view(
            Yall_time, HorizonSteps, axis=0).transpose(0, 2, 1)
        x_starts = anchors - LookbackSteps + 1
        y_starts = anchors + 1
        step = int(anchors[1] - anchors[0]) if num_windows > 1 else WindowSlidingCoeff
        if (valid_col is None and group_col is None) or np.all(np.diff(anchors) == step):
            # Filtering may leave regular anchors with a larger step than requested.
            X_tensor = X_view[x_starts[0]:x_starts[-1] + 1:step]
            Y_tensor = Y_view[y_starts[0]:y_starts[-1] + 1:step]
        else:
            # Irregular anchors still require copying only the surviving windows.
            X_tensor = X_view[x_starts]
            Y_tensor = Y_view[y_starts]

    print(f"Made {X_tensor.shape[0]} windows of size LookbackSteps={LookbackSteps}, HorizonSteps={HorizonSteps} from {T} samples.")
    print(f"  X_tensor shape: {X_tensor.shape}, Y_tensor shape: {Y_tensor.shape}")
    return X_tensor, Y_tensor


def flatten_XY(X, Y):
    """
    Docstring for flatten_XY

    :param X: training dataset of shape (num_windows, LookbackSteps, num_features)
    :param Y: output targets of shape (num_windows, HorizonSteps, output_dim)

    In the function:
    - (N, L, F) = (num_windows, LookbackSteps, num_features)
    - (N, H, D) = (num_windows, HorizonSteps, output_dim)

    :return: flattened versions of X and Y
    """
    N, L_, F_ = X.shape
    N2, H_, D_ = Y.shape
    assert N == N2
    return X.reshape(N, L_*F_), Y.reshape(N, H_*D_)


def unflatten_Y(y2d, output_shape):
    """
    Inverses the flattening operation to restore original shape.

    :param y2d: 2D array of shape (num_windows, HorizonSteps * output_dim)
    :param output_shape: tuple (HorizonSteps, output_dim) for reshaping

    In the function:
    - (N, H*D) = (num_windows, HorizonSteps * output_dim)
    - output_shape = (H, D) = (HorizonSteps, output_dim)

    :return: 3D array of shape (num_windows, HorizonSteps, output_dim)
    """
    H, D = output_shape
    return y2d.reshape(-1, H, D)
