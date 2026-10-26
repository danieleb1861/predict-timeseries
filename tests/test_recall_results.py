"""Compatibility checks without executing the legacy script's data/model loading."""

import ast
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # Compatibility checks render in memory without opening windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def recall():
    """Load only helper definitions and package imports; skip top-level execution."""
    path = Path(__file__).resolve().parents[1] / "recall_results.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("load_forecasting.")
    )]
    namespace = {"np": np, "pd": pd, "mpl": mpl, "plt": plt}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["_module_tree"] = tree
    return namespace


@pytest.mark.parametrize("valid_col, starts", [(None, [0, 2, 4, 6]), ("valid", [4, 6])])
def test_legacy_windows_keep_stride_order_and_boundary_behavior(recall, valid_col, starts):
    # A leg boundary at row 4 must not silently change historical window alignment.
    frame = pd.DataFrame({"a": np.arange(12), "b": np.arange(12) * 10,
                          "leg_id": [0] * 4 + [1] * 8, "valid": [1, 1, 0] + [1] * 9})
    X, Y = recall["make_windows"](frame, ["b", "a"], ["a"], 3, 2, valid_col=valid_col)
    np.testing.assert_array_equal(X[:, :, 1], np.asarray(starts)[:, None] + [0, 1, 2])
    np.testing.assert_array_equal(X[:, :, 0], X[:, :, 1] * 10)
    np.testing.assert_array_equal(Y[:, :, 0], np.asarray(starts)[:, None] + [3, 4])
    assert X.dtype == Y.dtype == np.float32
    assert X.flags.writeable and Y.flags.writeable


def test_legacy_windows_keep_explicit_stride_and_empty_shapes(recall):
    frame = pd.DataFrame({"a": np.arange(9)})
    X, Y = recall["make_windows"](frame, ["a"], ["a"], 3, 2, stride=3)
    np.testing.assert_array_equal(X[:, :, 0], [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(Y[:, :, 0], [[3, 4], [6, 7]])
    X, Y = recall["make_windows"](frame.iloc[:3], ["a"], ["a"], 3, 2)
    assert X.shape == (0, 3, 1) and Y.shape == (0, 2, 1)


@pytest.mark.parametrize("fill_limit", [None, 1])
def test_legacy_preprocessing_preserves_units_resampling_and_duplicates(recall, monkeypatch, fill_limit):
    # Supply irregular source rows in memory; never read a production dataset.
    keys = ["date", "time", "act_pow", "react_pow", "man_port", "man_stbd",
            "spd_port", "spd_stbd", "air_temp", "wind_int", "sea_temp", "sea_state"]
    mapping = dict(zip(keys, keys))
    source = pd.DataFrame({key: [2, 1, 99, 3] for key in keys})
    source["date"] = "2026-01-01"
    source["time"] = ["00:00:02", "00:00:00", "00:00:02", "00:00:04"]
    source["act_pow"] = [200, 100, 999, 300]
    source["react_pow"] = [20, 10, 99, 30]
    source["man_port"] = [1, 0, 1, 1]
    source["man_stbd"] = [0, 0, 0, 1]
    types = {key: "float32" for key in keys if key not in ("date", "time")}
    types.update(man_port="Int8", man_stbd="Int8")
    recall.update(rating_kW=100, varsPrev=["man_port", "man_stbd", "spd_port", "spd_stbd"],
                  varsLin=["act_pow_pu", "react_pow_pu", "air_temp", "wind_intensity", "sea_temp", "sea_state"])
    monkeypatch.setattr(pd, "read_parquet", lambda path: source.copy())
    result = recall["preprocess_dataset"]("fixture.parquet", 17, mapping, types, max_fill_seconds=fill_limit)
    np.testing.assert_allclose(result["act_pow_pu"], [1, 1.5, 2, 2.5, 3])
    np.testing.assert_allclose(result["react_pow_pu"], [0.1, 0.15, 0.2, 0.25, 0.3])
    np.testing.assert_array_equal(result["spd_port"], [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(result["man_port"], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result["time_leg"], np.arange(5))
    np.testing.assert_array_equal(result["t_seconds"], np.arange(5))
    assert (result["leg_id"] == 17).all()
    assert result["act_pow_pu"].dtype == np.float32
    assert result["dtime"].tolist() == list(pd.date_range("2026-01-01", periods=5, freq="s"))


def test_engineered_features_reset_at_leg_boundaries(recall):
    frame = pd.DataFrame({"act_pow_pu": [1., 2., 10., 12.], "react_pow_pu": [0., 0., 0., 0.],
                          "spd_port": [2., 3., 4., 5.], "spd_stbd": [1., 1., 1., 1.], "leg_id": [0, 0, 1, 1]})
    result = recall["add_engineered"](frame)
    np.testing.assert_array_equal(result["dP"], [0, 1, 0, 2])
    np.testing.assert_array_equal(result["moving_avg_5"], [1, 1.5, 10, 11])
    np.testing.assert_allclose(result["moving_std_5"], [0, np.sqrt(0.5), 0, np.sqrt(2)])
    assert "dP" not in frame  # Feature engineering must not mutate its input.


def test_legacy_inverse_scaling_preserves_target_order_and_dtype(recall):
    recall.update(MU=pd.Series({"b": 10., "a": 3.}), SIG=pd.Series({"b": 2., "a": 4.}),
                  targetNames=["a", "b"])
    result = recall["inv_scale"](np.array([[[1., 2.], [0., -1.]]]))
    np.testing.assert_array_equal(result, [[[7., 14.], [3., 8.]]])
    assert result.dtype == np.float32


def test_recall_scaling_block_keeps_training_statistics(recall):
    # Execute the actual scaling block, skipping data loading and model evaluation.
    nodes = recall["_module_tree"].body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Tuple)
                 and isinstance(node.targets[0].elts[0], ast.Name)
                 and node.targets[0].elts[0].id == "scaler")
    stop = next(i for i, node in enumerate(nodes) if i > start and isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "TTZ")
    train = pd.DataFrame({"p": [0., 2., 4.], "constant": [1., 1., 1.]}, dtype=np.float32)
    val = pd.DataFrame({"p": [6.], "constant": [1.]}, index=[10], dtype=np.float32)
    test = pd.DataFrame({"p": [8.], "constant": [1.]}, index=[20], dtype=np.float32)
    recall.update(TTtrain=train, TTval=val, TTtest=test, TT=pd.concat([train, val, test]),
                  varsCont=["p", "constant"])
    exec(compile(ast.Module(body=nodes[start:stop + 1], type_ignores=[]), "recall_scaling", "exec"), recall)
    np.testing.assert_array_equal(recall["MU"], [2., 1.])
    np.testing.assert_allclose(recall["SIG"], [np.sqrt(8 / 3), 1.])
    np.testing.assert_allclose(recall["TTvalZ"]["p"], [4 / np.sqrt(8 / 3)])
    np.testing.assert_allclose(recall["TTtestZ"]["p"], [6 / np.sqrt(8 / 3)])
    assert recall["TTvalZ"].index.tolist() == [10]
    assert recall["TTtestZ"].index.tolist() == [20]
    assert recall["TTtrainZ"]["constant"].eq(0).all()
    assert abs(recall["TTZ"]["p"].mean()) < 1e-6  # Full-fit diagnostics stay separate.


def test_legacy_metrics_preserve_values_schema_and_skip_bootstrap(recall, monkeypatch):
    from load_forecasting import metrics

    def unexpected_bootstrap(*args, **kwargs):
        raise AssertionError("Legacy recall must not add bootstrap work")

    monkeypatch.setattr(metrics, "bootstrap_mean_ci", unexpected_bootstrap)
    truth = np.array([[[2.], [4.]], [[4.], [8.]]])
    prediction = np.array([[[3.], [2.]], [[2.], [12.]]])
    # Unrelated first-target errors must not leak into the selected second target.
    truth = np.concatenate([truth * 100, truth], axis=2)
    prediction = np.concatenate([prediction * -100, prediction], axis=2)
    result = recall["compute_metrics"](truth, prediction, 1)
    assert list(result) == ["RMSE_horizon", "MAE_horizon", "MAPE_horizon", "MAPE_targetax",
                            "RMSE_all", "MAE_all", "MAPE_all"]
    np.testing.assert_allclose(result["RMSE_horizon"], np.sqrt([2.5, 10.]))
    np.testing.assert_array_equal(result["MAE_horizon"], [1.5, 3.])
    assert result["MAPE_horizon"] == result["MAPE_targetax"] == [50., 50.]
    assert result["RMSE_all"] == 2.5 and result["MAE_all"] == 2.25 and result["MAPE_all"] == 50.
    with pytest.raises(ValueError, match="Expected"):
        recall["compute_metrics"](truth, prediction[:1], 0)


def test_legacy_scatter_keeps_zero_based_horizon(recall):
    truth = np.arange(6).reshape(2, 3, 1)
    recall["plot_scatter_grid"](truth, {"Model": truth + 10}, k=1)
    figure = plt.gcf()
    np.testing.assert_array_equal(figure.axes[0].collections[0].get_offsets(), [[1, 11], [4, 14]])
    assert figure.axes[0].get_title() == "Model @ t+2"
    plt.close(figure)


def test_legacy_style_preserves_dpi_and_horizon_keyword(recall):
    with mpl.rc_context({"figure.dpi": 123, "savefig.dpi": 234}):
        recall["set_pub_style"](font_family="DejaVu Sans", fontsize=11)
        assert mpl.rcParams["figure.dpi"] == 123 and mpl.rcParams["savefig.dpi"] == 234
        assert mpl.rcParams["font.size"] == 11
    frame = pd.DataFrame({"MAE_horizon": [[1, 2], [3, 4]]}, index=["A", "B"])
    models, matrix = recall["get_horizon_matrix"](df_results=frame, metric_key="MAE_horizon", model_order=["B", "A"])
    assert models == ["B", "A"]
    np.testing.assert_array_equal(matrix, [[3, 4], [1, 2]])
