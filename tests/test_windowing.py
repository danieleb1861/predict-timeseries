import numpy as np
import pandas as pd
import pytest

from load_forecasting.training_deep import iter_batches
from load_forecasting.windowing import flatten_XY, make_windows, unflatten_Y


def test_make_windows_uses_expected_shapes_and_future_targets():
    data_frame = pd.DataFrame(
        {
            "feature": np.arange(8, dtype=np.float32),
            "target": np.arange(100, 108, dtype=np.float32),
        }
    )

    X, Y = make_windows(
        data_frame,
        ["feature"],
        ["target"],
        LookbackSteps=3,
        HorizonSteps=2,
        WindowSlidingCoeff=2,
    )

    assert X.shape == (2, 3, 1)
    assert Y.shape == (2, 2, 1)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(Y[0, :, 0], [103, 104])
    np.testing.assert_array_equal(X[1, :, 0], [2, 3, 4])
    np.testing.assert_array_equal(Y[1, :, 0], [105, 106])


def test_make_windows_skips_invalid_valid_col_ranges():
    data_frame = pd.DataFrame(
        {
            "feature": np.arange(8, dtype=np.float32),
            "target": np.arange(100, 108, dtype=np.float32),
            "valid": [1, 1, 1, 1, 1, 0, 1, 1],
        }
    )

    X, Y = make_windows(
        data_frame,
        ["feature"],
        ["target"],
        LookbackSteps=3,
        HorizonSteps=2,
        WindowSlidingCoeff=1,
        valid_col="valid",
    )

    assert X.shape == (1, 3, 1)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(Y[0, :, 0], [103, 104])


def test_make_windows_does_not_cross_group_boundaries():
    data_frame = pd.DataFrame(
        {
            "feature": np.arange(8, dtype=np.float32),
            "target": np.arange(100, 108, dtype=np.float32),
            "leg_id": ["a"] * 4 + ["b"] * 4,
        }
    )

    X, Y = make_windows(
        data_frame,
        ["feature"],
        ["target"],
        LookbackSteps=2,
        HorizonSteps=1,
        WindowSlidingCoeff=1,
        group_col="leg_id",
    )

    assert X.shape == (4, 2, 1)
    np.testing.assert_array_equal(X[:, :, 0], [[0, 1], [1, 2], [4, 5], [5, 6]])
    np.testing.assert_array_equal(Y[:, :, 0], [[102], [103], [106], [107]])


def test_flatten_and_unflatten_y_round_trip():
    X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    Y = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    X_flat, Y_flat = flatten_XY(X, Y)
    Y_round_trip = unflatten_Y(Y_flat, output_shape=(3, 2))

    assert X_flat.shape == (2, 12)
    assert Y_flat.shape == (2, 6)
    np.testing.assert_array_equal(Y_round_trip, Y)


@pytest.mark.parametrize("groups, valid, stride, starts, uses_view", [
    ([0] * 12, [1] * 12, 2, [0, 2, 4, 6, 8], True),
    ([0] * 3 + [1] * 3 + [2] * 3, None, 1, [0, 3, 6], True),
    (None, [0, 0, 1, 1, 1, 1, 1, 1, 0], 1, [2, 3, 4, 5], True),
    (None, [0, 0, 1, 1, 1], 1, [2], True),
    ([0] * 4 + [1] * 4, None, 1, [0, 1, 4, 5], False),
    ([0] * 6 + [1] * 6, [1, 0] + [1] * 10, 1, [2, 3, 6, 7, 8, 9], False),
    (None, [0] * 5, 1, [], False),
    ([0] * 2, None, 1, [], False),
])
def test_filtered_windows_preserve_values_and_memory_policy(monkeypatch, groups, valid, stride, starts, uses_view):
    # Capture the actual base arrays to distinguish views from materialized windows.
    bases = []
    original_view = np.lib.stride_tricks.sliding_window_view

    def capture_view(array, *args, **kwargs):
        bases.append(array)
        return original_view(array, *args, **kwargs)

    monkeypatch.setattr(np.lib.stride_tricks, "sliding_window_view", capture_view)
    count = len(groups if groups is not None else valid)
    frame = pd.DataFrame({"feature": np.arange(count, dtype=np.float32),
                          "target": np.arange(count, dtype=np.float32) + 100})
    if groups is not None:
        frame["leg"] = groups
    if valid is not None:
        frame["valid"] = valid
    X, Y = make_windows(frame, ["feature"], ["target"], 2, 1, stride,
                        valid_col="valid" if valid is not None else None,
                        group_col="leg" if groups is not None else None)
    expected = np.asarray(starts, dtype=int)
    np.testing.assert_array_equal(X[:, :, 0], expected[:, None] + [0, 1])
    np.testing.assert_array_equal(Y[:, :, 0], expected[:, None] + [102])
    assert X.dtype == Y.dtype == np.float32
    if starts:
        assert np.shares_memory(X, bases[0]) == uses_view
        assert np.shares_memory(Y, bases[1]) == uses_view
        assert X.flags.writeable == Y.flags.writeable == (not uses_view)
    else:
        assert X.shape == (0, 2, 1) and Y.shape == (0, 1, 1)


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("batch_size", [3, 20])
def test_unshuffled_batches_share_strided_inputs(readonly, batch_size):
    # Non-contiguous and read-only inputs are common outputs of window construction.
    X = np.arange(84, dtype=np.float32).reshape(14, 3, 2)[::2]
    Y = np.arange(28, dtype=np.float32).reshape(14, 2, 1)[::2]
    if readonly:
        X.setflags(write=False)
        Y.setflags(write=False)
    batches = list(iter_batches(X, Y, batch_size))
    assert [len(xb) for xb, _ in batches] == ([3, 3, 1] if batch_size == 3 else [7])
    np.testing.assert_array_equal(np.concatenate([xb for xb, _ in batches]), X)
    np.testing.assert_array_equal(np.concatenate([yb for _, yb in batches]), Y)
    for xb, yb in batches:
        assert np.shares_memory(xb, X) and np.shares_memory(yb, Y)
        assert xb.flags.writeable == yb.flags.writeable == (not readonly)


@pytest.mark.parametrize("seed", [None, 17])
def test_shuffled_batches_preserve_seeded_order_and_pairing(seed):
    X = np.arange(7, dtype=np.float32).reshape(7, 1, 1)
    Y = X + 100
    rng = np.random.default_rng(seed) if seed is not None else None
    expected_rng = np.random.default_rng(seed) if seed is not None else None
    for _ in range(2):
        # An explicit generator advances across epochs; the default seed resets.
        order = np.arange(7)
        (expected_rng if expected_rng is not None else np.random.default_rng(0)).shuffle(order)
        batches = list(iter_batches(X, Y, 3, shuffle=True, rng=rng))
        np.testing.assert_array_equal(np.concatenate([xb for xb, _ in batches]), X[order])
        np.testing.assert_array_equal(np.concatenate([yb for _, yb in batches]), Y[order])
        assert [len(xb) for xb, _ in batches] == [3, 3, 1]
        assert all(not np.shares_memory(xb, X) and not np.shares_memory(yb, Y) for xb, yb in batches)


@pytest.mark.parametrize("shuffle", [False, True])
def test_empty_inputs_produce_no_batches(shuffle):
    assert list(iter_batches(np.empty((0, 3, 2)), np.empty((0, 2, 1)), 3, shuffle=shuffle)) == []
