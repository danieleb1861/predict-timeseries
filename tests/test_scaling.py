import numpy as np
import pandas as pd

from load_forecasting.scaling import (
    inverse_scale_targets,
    scale_full_dataset_for_diagnostics,
    scale_train_val_test,
)


def test_scale_train_val_test_uses_train_statistics_for_all_splits():
    """
    Verify that validation and test data are transformed with train statistics.

    Inputs
    ----------
    None
        The test builds synthetic train, validation, and test DataFrames where
        validation/test values are intentionally far from the training range.

    Returns
    -------
    None
        Assertions confirm that scaling metadata comes only from training data
        and that validation/test transforms use those train-fitted values.
    """
    train = pd.DataFrame({"target": [0.0, 2.0], "feature": [10.0, 14.0]})
    val = pd.DataFrame({"target": [100.0], "feature": [200.0]})
    test = pd.DataFrame({"target": [300.0], "feature": [400.0]})

    train_z, val_z, test_z, mu, sig = scale_train_val_test(
        train,
        val,
        test,
        ["target", "feature"],
    )

    assert mu["target"] == 1.0
    assert sig["target"] == 1.0
    np.testing.assert_allclose(train_z["target"].to_numpy(), [-1.0, 1.0])
    np.testing.assert_allclose(val_z["target"].to_numpy(), [99.0])
    np.testing.assert_allclose(test_z["target"].to_numpy(), [299.0])


def test_scale_full_dataset_for_diagnostics_preserves_full_fit_behavior():
    """
    Verify the diagnostics-only full-dataset scaling path.

    Inputs
    ----------
    None
        The test builds a synthetic full DataFrame with one target column.

    Returns
    -------
    None
        Assertions confirm that the full-dataset diagnostic transform has zero
        mean and unit population standard deviation.
    """
    data_frame = pd.DataFrame({"target": [0.0, 2.0, 100.0, 300.0]})

    full_z = scale_full_dataset_for_diagnostics(data_frame, ["target"])

    assert abs(full_z["target"].mean()) < 1e-6
    np.testing.assert_allclose(full_z["target"].std(ddof=0), 1.0)


def test_inverse_scale_targets_restores_physical_values():
    """
    Verify inverse scaling for multi-horizon target tensors.

    Inputs
    ----------
    None
        The test builds synthetic normalized predictions and train-fitted
        target metadata.

    Returns
    -------
    None
        Assertions confirm that normalized values are restored to the expected
        physical/per-unit target scale.
    """
    mu = pd.Series({"target": 10.0, "other": 99.0}, dtype="float32")
    sig = pd.Series({"target": 2.0, "other": 1.0}, dtype="float32")
    z = np.array([[[-1.0], [0.0], [1.0]]], dtype=np.float32)

    y = inverse_scale_targets(z, ["target"], mu, sig)

    np.testing.assert_allclose(y, [[[8.0], [10.0], [12.0]]])
