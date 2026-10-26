import numpy as np

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
