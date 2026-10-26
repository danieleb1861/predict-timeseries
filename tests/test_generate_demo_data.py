import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from load_forecasting.config import load_system_config, validate_inputs
from load_forecasting.data_prep import add_engineered, preprocess_dataset
from scripts.generate_demo_data import generate_demo_frame, write_demo_dataset


EXPECTED_COLUMNS = [
    "date",
    "time",
    "act_pow_kW",
    "react_pow_kVAR",
    "man_selector_port",
    "man_selector_starboard",
    "lever_port",
    "lever_starboard",
    "air_temp",
    "wind_intensity",
    "sea_temp",
    "sea_state",
]


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_engineered_features_preserve_leg_windows_and_column_order(interleaved, dtype):
    """Check both signals against independent trailing-window calculations."""
    rng = np.random.default_rng(7)
    legs = np.tile([9, 2], 26) if interleaved else np.repeat([9, 2], 26)
    frame = pd.DataFrame({
        "leg_id": np.append(legs, 7),  # A singleton leg must have zero std/difference.
        "act_pow_pu": rng.integers(1, 20, 53).astype(dtype),
        "react_pow_pu": rng.integers(-10, 10, 53).astype(dtype),
        "spd_port": np.ones(53, dtype=dtype),
        "spd_stbd": np.zeros(53, dtype=dtype),
    }, index=np.arange(53)[::-1] * 3 + 100)
    frame.iloc[5, frame.columns.get_loc("act_pow_pu")] = np.nan
    frame.iloc[8, frame.columns.get_loc("react_pow_pu")] = np.nan
    original = frame.copy(deep=True)
    result = add_engineered(frame)
    expected_columns = [*frame.columns, "lever_diff", "pfactor"]

    # Index labels are deliberately nonconsecutive and opposite to row order.
    for column, suffix, derivative in (("act_pow_pu", "", "dP"), ("react_pow_pu", "_Q", "dQ")):
        expected_columns.append(derivative)
        for w in (5, 10, 20):
            expected_columns.extend([f"moving_avg_{w}{suffix}", f"moving_std_{w}{suffix}"])
        for leg in frame["leg_id"].unique():
            rows = frame.loc[frame["leg_id"] == leg]
            values = rows[column].to_numpy()
            expected_diff = np.nan_to_num(np.diff(values, prepend=values[0]), nan=0.0)
            np.testing.assert_array_equal(result.loc[rows.index, derivative], expected_diff)
            for w in (5, 10, 20):
                means, stds = [], []
                for i in range(len(values)):
                    window = values[max(0, i - w + 1):i + 1].astype(np.float64)
                    valid = window[~np.isnan(window)]
                    means.append(valid.mean() if len(valid) else np.nan)
                    stds.append(valid.std(ddof=1) if len(valid) > 1 else 0.0)
                np.testing.assert_allclose(result.loc[rows.index, f"moving_avg_{w}{suffix}"],
                                           np.asarray(means, dtype=np.float32), rtol=1e-6, equal_nan=True)
                np.testing.assert_allclose(result.loc[rows.index, f"moving_std_{w}{suffix}"],
                                           np.asarray(stds, dtype=np.float32), rtol=1e-6)

    assert list(result.columns) == expected_columns
    assert result[expected_columns[len(frame.columns) + 2:]].dtypes.eq(np.float32).all()
    pd.testing.assert_frame_equal(frame, original)
    # Removing future samples must not change any earlier engineered features.
    pd.testing.assert_frame_equal(result.iloc[:19], add_engineered(frame.iloc[:19]), check_exact=True)


def test_generate_demo_frame_is_deterministic_and_matches_input_schema():
    first = generate_demo_frame("navigation", num_rows=512, seed=7)
    second = generate_demo_frame("navigation", num_rows=512, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == EXPECTED_COLUMNS
    assert first["date"].iloc[0] == "2000-01-01"
    assert first["time"].iloc[0] == "00:00:00"
    assert not first.isna().any().any()
    assert first["man_selector_port"].eq(0).all()
    assert first["man_selector_starboard"].eq(0).all()


def test_manoeuvre_demo_contains_bounded_commands_and_active_selectors():
    frame = generate_demo_frame("manoeuvre", num_rows=2_000, seed=7)
    rating_kw = load_system_config("config/demo_config.json")["physical_system"]["rating_kW"]

    assert frame["lever_port"].between(-1.0, 1.0).all()
    assert frame["lever_starboard"].between(-1.0, 1.0).all()
    assert frame[["man_selector_port", "man_selector_starboard"]].to_numpy().sum() > 0
    assert frame["act_pow_kW"].between(0.03 * rating_kw, 1.20 * rating_kw).all()


def test_generate_demo_frame_reads_rating_from_config(tmp_path):
    config = load_system_config("config/demo_config.json")
    default_rating = config["physical_system"]["rating_kW"]
    config["physical_system"]["rating_kW"] = default_rating / 2.0
    config_path = tmp_path / "demo_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    default_frame = generate_demo_frame("navigation", num_rows=128, seed=7)
    scaled_frame = generate_demo_frame("navigation", num_rows=128, seed=7, config_path=config_path)

    np.testing.assert_allclose(
        default_frame["act_pow_kW"],
        2.0 * scaled_frame["act_pow_kW"],
    )


def test_demo_parquet_round_trip_matches_preprocessing_contract(tmp_path, capsys):
    path, expected = write_demo_dataset("navigation", tmp_path, num_rows=512, seed=7)
    actual = pd.read_parquet(path)
    pd.testing.assert_frame_equal(actual, expected)

    config = load_system_config("config/demo_config.json")
    dataset = config["dataset"]
    columns = dataset["columns"]
    dtypes = {columns[key]: value for key, value in dataset["dtypes"].items()}
    validate_inputs([path.name], tmp_path, columns)
    processed = preprocess_dataset(
        path,
        leg_id=0,
        colMap=columns,
        data_types=dtypes,
        rating_kW=config["physical_system"]["rating_kW"],
        varsPrev=config["physical_system"]["signals"]["step_hold"],
        varsLin=config["physical_system"]["signals"]["resample_continuous"],
        Ts=dataset["sampling_rate"],
        max_fill_seconds=dataset["max_fill_seconds"],
    )

    assert len(processed) == len(expected)
    assert not processed.isna().any().any()
    assert processed["dtime"].is_monotonic_increasing
    capsys.readouterr()


def test_ridge_demo_skips_disabled_diagnostics_and_deep_backends(tmp_path):
    """Train real Ridge models with optional imports and full-data scaling blocked."""
    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    path, _ = write_demo_dataset("navigation", data_dir, num_rows=256, seed=7)
    config = load_system_config(root / "config/demo_config.json")
    config["paths"]["data_dir"] = str(data_dir)
    config["dataset"]["selectors"]["navigation"] = [path.name]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # A fresh interpreter catches optional imports hidden by pytest's module cache.
    program = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'jax', 'jaxlib', 'flax', 'optax', 'tensorflow', 'keras', 'tcn'}:
        raise AssertionError('Unused backend imported: ' + name)
    if name in {'load_forecasting.bess', 'load_forecasting.eda', 'load_forecasting.load_dynamics'}:
        raise AssertionError('Disabled diagnostics imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import train_load_forecasters as training
def reject_diagnostic_scaling(*args, **kwargs):
    raise AssertionError('Disabled diagnostics normalized the full dataset')
training.scale_full_dataset_for_diagnostics = reject_diagnostic_scaling
training.main()
"""
    predictions = []
    for backend in ("keras", "flax"):
        run_dir = tmp_path / backend
        result = subprocess.run(
            [sys.executable, "-B", "-c", program, "--config", str(config_path),
             "--selector", "navigation", "--lookback", "8", "--horizon", "5",
             "--models", "ridge", "--backend", backend, "--skip-eda", "--skip-plots",
             "--run-dir", str(run_dir)],
            cwd=root, capture_output=True, text=True, timeout=90,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        runtime = summary["provenance"]["runtime"]
        assert runtime["backend"] == backend
        assert runtime["default_backend"] is None
        assert runtime["devices"] == []
        with np.load(run_dir / "predictions/ridge.npz") as saved:
            predictions.append(saved["yhat_test_phys"])
        assert predictions[-1].shape[1:] == (5, 1)
        assert np.isfinite(predictions[-1]).all()

    # Backend selection must not alter baseline predictions.
    np.testing.assert_array_equal(*predictions)
