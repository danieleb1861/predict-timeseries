import numpy as np
import pytest

from load_forecasting.metrics import bootstrap_mean_ci, moving_block_resample
from scripts.compute_leg_bootstrap_ci import clean_leg_window_masks, combined_leg_bootstrap


def test_clean_leg_window_masks_supports_boundary_filtered_archives():
    masks = clean_leg_window_masks(
        T=8, leg_sizes=[4, 4], lookback=2, horizon=1, stride=1, n_windows=4,
    )

    np.testing.assert_array_equal(masks[0], [True, True, False, False])
    np.testing.assert_array_equal(masks[1], [False, False, True, True])


def test_combined_leg_bootstrap_equal_weights_three_legs():
    result = combined_leg_bootstrap(
        [np.ones(5), np.full(20, 3.0), np.full(50, 5.0)],
        n_boot=20,
    )

    assert result == {"mean": 3.0, "low": 3.0, "high": 3.0}


def ar1_series(size):
    rng = np.random.default_rng(0)
    series = np.zeros(size)
    for t in range(1, size):
        series[t] = 0.9 * series[t - 1] + rng.normal()
    return series


def test_block_length_one_keeps_the_ordinary_bootstrap():
    values = np.arange(30.0)
    rng = np.random.default_rng(42)
    legacy = [rng.choice(values, size=values.size, replace=True).mean() for _ in range(200)]

    result = bootstrap_mean_ci(values, n_boot=200)

    assert result["mean"] == pytest.approx(np.mean(legacy))
    assert result["low"] == pytest.approx(np.percentile(legacy, 2.5))
    assert result["high"] == pytest.approx(np.percentile(legacy, 97.5))


def test_moving_blocks_are_contiguous_and_keep_the_sample_size():
    sample = moving_block_resample(np.arange(20.0), np.random.default_rng(1), block_len=6)

    assert sample.size == 20
    for start in range(0, 18, 6):
        np.testing.assert_array_equal(np.diff(sample[start:start + 6]), np.ones(5))


@pytest.mark.parametrize("block_len", [0, -1])
def test_block_length_must_be_positive(block_len):
    with pytest.raises(ValueError, match="block_len"):
        moving_block_resample(np.arange(5.0), np.random.default_rng(0), block_len)


def test_block_length_above_sample_size_returns_the_whole_series():
    values = np.arange(5.0)

    np.testing.assert_array_equal(moving_block_resample(values, np.random.default_rng(0), 50), values)


def test_blocks_widen_the_interval_on_autocorrelated_values():
    series = ar1_series(2000)

    iid = bootstrap_mean_ci(series, n_boot=400)
    blocked = bootstrap_mean_ci(series, n_boot=400, block_len=50)

    assert blocked["high"] - blocked["low"] > 2.0 * (iid["high"] - iid["low"])


def test_combined_leg_bootstrap_uses_the_block_length():
    series = ar1_series(1000)

    iid = combined_leg_bootstrap([series[:500], series[500:]], n_boot=300)
    blocked = combined_leg_bootstrap([series[:500], series[500:]], n_boot=300, block_len=40)

    assert blocked["high"] - blocked["low"] > iid["high"] - iid["low"]
