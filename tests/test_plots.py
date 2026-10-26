import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import pytest

from load_forecasting.plots import (
    _auto_zoom_bounds,
    compress_time_gaps,
    plot_full_test,
    save_plot,
)


def test_plot_full_test_compresses_and_breaks_time_gaps():
    test_data = pd.DataFrame(
        {
            "dtime": np.array(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:06",
                    "2026-01-02 00:00:00",
                    "2026-01-02 00:00:01",
                    "2025-12-31 00:00:00",
                    "2025-12-31 00:00:01",
                ],
                dtype="datetime64[us]",
            )
        }
    )
    measured = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    predictions = {"Model": measured + 0.01}

    plot_full_test(
        test_data,
        measured,
        predictions,
        gap="auto",
        zoom="auto",
        zoom_loc="auto",
    )
    figure = plt.gcf()
    figure.canvas.draw()
    main_axis, inset_axis = figure.axes

    plotted_time = pd.to_datetime(main_axis.lines[0].get_xdata())
    expected_steps = np.array([1, 5, 5, 1, 5, 1], dtype="timedelta64[s]")
    np.testing.assert_array_equal(np.diff(plotted_time), expected_steps)
    assert plotted_time[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert np.isnan(main_axis.lines[0].get_ydata()[3])
    assert np.isnan(main_axis.lines[0].get_ydata()[5])
    assert np.isfinite(main_axis.lines[0].get_ydata()[[0, 1, 2, 4, 6]]).all()
    assert main_axis.get_xlabel() == "Compressed timeline"
    legend = main_axis.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Measured",
        "Model",
        "Voyage-leg boundary",
    ]
    assert legend.get_frame().get_alpha() == 1.0
    leg_boundary = next(
        collection
        for collection in main_axis.collections
        if collection.get_label() == "Voyage-leg boundary"
    )
    np.testing.assert_allclose(
        leg_boundary.get_colors()[0],
        mcolors.to_rgba("#c62828"),
    )
    assert leg_boundary.get_linestyles()[0][1] is not None
    main_position = main_axis.get_position()
    inset_position = inset_axis.get_position()
    assert inset_position.x0 > (main_position.x0 + main_position.x1) / 2
    assert (inset_position.y0 + inset_position.y1) / 2 < (
        main_position.y0 + main_position.y1
    ) / 2
    assert inset_axis.xaxis.get_major_formatter().fmt == "%H:%M"
    assert np.isfinite(inset_axis.lines[0].get_ydata()).all()
    renderer = figure.canvas.get_renderer()
    main_boundary = main_axis.get_window_extent(renderer).y0
    label_boundaries = [
        label.get_window_extent(renderer).y0
        for label in inset_axis.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    assert min(label_boundaries) >= main_boundary
    assert min(label_boundaries) - main_boundary <= 20

    plt.close(figure)


def test_auto_gap_scales_with_sampling_cadence():
    timestamps = np.array(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:10",
            "2026-01-01 00:00:20",
            "2026-01-01 01:00:00",
            "2026-01-01 01:00:10",
        ],
        dtype="datetime64[us]",
    )

    compressed = compress_time_gaps(timestamps, gap="auto")

    expected_steps = np.array([10, 10, 50, 10], dtype="timedelta64[s]")
    np.testing.assert_array_equal(np.diff(compressed), expected_steps)


def test_auto_zoom_scales_and_stays_inside_navigation_leg():
    manoeuvre_zoom = _auto_zoom_bounds(20_999, np.array([8_000, 15_000]))
    navigation_zoom = _auto_zoom_bounds(781_101, np.array([535_676]))

    assert manoeuvre_zoom[1] - manoeuvre_zoom[0] == 420
    assert navigation_zoom[1] - navigation_zoom[0] == 1200
    assert navigation_zoom[1] < 535_676


def test_save_plot_keeps_eps_beside_requested_image_directory(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    plt.figure()
    plt.plot([0, 1], [0, 1])

    save_plot(images_dir, "example", dpi=20)

    assert (images_dir / "example.png").is_file()
    assert (images_dir / "eps" / "example.eps").is_file()
    assert not (unrelated_cwd / "outputs").exists()


@pytest.mark.parametrize("dpca,pca", [(False, False), (True, False), (False, True)])
def test_pca_diagnostics_only_fit_requested_projections(tmp_path, monkeypatch, dpca, pca):
    """Keep feature order and enabled plots while avoiding unused PCA fits."""
    from sklearn.decomposition import PCA
    from load_forecasting.pca_dpca import run_pca_dpca_analysis

    columns = [
        "act_pow_pu", "react_pow_pu", "air_temp", "sea_temp",
        "wind_intensity", "sea_state", "lever_diff", "pfactor",
        "spd_port", "spd_stbd", "dP", "moving_avg_5", "moving_avg_10",
        "moving_avg_20", "moving_std_5", "moving_std_10", "moving_std_20",
    ]
    frame = pd.DataFrame(np.random.default_rng(7).normal(size=(140, 17)), columns=columns)
    frame["dtime"] = pd.date_range("2000-01-01", periods=len(frame), freq="s")
    train = frame.iloc[:120]
    fitted, transformed, saved = [], [], []

    class TrackedPCA(PCA):
        # Record the real estimator inputs to detect extra fits or data leakage.
        def fit_transform(self, values, *args, **kwargs):
            fitted.append(values.copy())
            return super().fit_transform(values, *args, **kwargs)

        def transform(self, values):
            transformed.append(values.copy())
            return super().transform(values)

    def record_plot(directory, name):
        saved.append(name)
        plt.close()

    monkeypatch.setattr("sklearn.decomposition.PCA", TrackedPCA)
    monkeypatch.setattr("load_forecasting.plots.save_plot", record_plot)
    monkeypatch.setattr("load_forecasting.plots.set_pub_style", lambda **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda: plt.close())
    result = run_pca_dpca_analysis(
        frame, train if dpca or pca else None, None, None,
        frame if pca else None, tmp_path, dpca, pca,
    )

    assert result == columns
    assert len(fitted) == int(dpca or pca)
    assert len(transformed) == int(pca)
    if pca:
        np.testing.assert_array_equal(fitted[0], train[columns].to_numpy(np.float32))
        np.testing.assert_array_equal(transformed[0], frame[columns].to_numpy(np.float32))
        assert saved == ["PCA_cumsum_variance", "pca12_biplot", "pca34_biplot", "data_trajectory"]
    elif dpca:
        assert fitted[0].shape == (21, 1700)
        assert saved == []
