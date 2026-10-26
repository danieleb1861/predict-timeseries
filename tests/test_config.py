import json
from types import SimpleNamespace

import pytest

from load_forecasting.config import cfg_or, parse_cli_config, parse_selected_models, resolve_project_path
from train_load_forecasters import resolve_optional_branch_flags


def test_parse_selected_models_accepts_all_labels():
    labels = ["ridge", "xgb", "mlp"]

    assert parse_selected_models("all", labels) == set(labels)


def test_parse_selected_models_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown --models values"):
        parse_selected_models("ridge,unknown", ["ridge", "xgb"])


def test_cfg_or_keeps_explicit_falsey_values():
    assert cfg_or(0, 10) == 0
    assert cfg_or(False, True) is False
    assert cfg_or(None, 10) == 10


def test_resolve_project_path_joins_relative_path(tmp_path):
    resolved = resolve_project_path("data/input.parquet", tmp_path)

    assert resolved == str(tmp_path / "data" / "input.parquet")


def test_parse_cli_config_accepts_backend_override(tmp_path, monkeypatch):
    preset_file = tmp_path / "presets.json"
    preset_file.write_text(
        json.dumps({
            "presets": {
                "fast": {
                    "selector": "manoeuvre",
                    "backend": "keras",
                    "models": "mlp",
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_load_forecasters.py",
            "--preset",
            "fast",
            "--preset-file",
            str(preset_file),
            "--backend",
            "flax",
        ],
    )

    args = parse_cli_config(
        default_config_file="config/default_config.json",
        default_preset_file=str(preset_file),
        model_labels={"mlp": "MLP"},
        preset_keys={"selector", "backend", "models"},
        cli_defaults={
            "selector": "manoeuvre",
            "backend": "keras",
            "models": "all",
        },
    )

    assert args.backend == "flax"


def test_parse_cli_config_model_specs_override_and_default(tmp_path, monkeypatch):
    preset_file = tmp_path / "presets.json"
    preset_file.write_text(
        json.dumps({"presets": {"fast": {"selector": "manoeuvre"}}}),
        encoding="utf-8",
    )
    common_kwargs = dict(
        default_config_file="config/default_config.json",
        default_preset_file=str(preset_file),
        model_labels={"mlp": "MLP"},
        preset_keys={"selector", "model_specs"},
        cli_defaults={"selector": "manoeuvre", "model_specs": None},
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_load_forecasters.py",
            "--preset",
            "fast",
            "--preset-file",
            str(preset_file),
            "--model-specs",
            "config/model_specs_tuned.json",
        ],
    )
    args = parse_cli_config(**common_kwargs)
    assert args.model_specs == "config/model_specs_tuned.json"

    monkeypatch.setattr(
        "sys.argv",
        ["train_load_forecasters.py", "--preset", "fast", "--preset-file", str(preset_file)],
    )
    args = parse_cli_config(**common_kwargs)
    assert args.model_specs is None


def test_parse_cli_config_preserves_skip_flag_semantics(tmp_path, monkeypatch):
    preset_file = tmp_path / "presets.json"
    preset_file.write_text(
        json.dumps({"presets": {"fast": {"skip_eda": True, "skip_plots": False}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["train_load_forecasters.py", "--preset", "fast", "--preset-file", str(preset_file)],
    )

    args = parse_cli_config(
        default_config_file="config/default_config.json",
        default_preset_file=str(preset_file),
        model_labels={"mlp": "MLP"},
        preset_keys={"skip_eda", "skip_plots"},
        cli_defaults={"skip_eda": False, "skip_plots": True},
    )

    assert args.skip_eda is True
    assert args.skip_plots is False


def test_optional_branch_flags_invert_skip_flags():
    flags = resolve_optional_branch_flags(
        SimpleNamespace(skip_eda=True, skip_plots=True, load_dynamics=False)
    )
    assert flags == {"EDA": False, "PostTraining": False, "LoadDynamics": False}

    flags = resolve_optional_branch_flags(
        SimpleNamespace(skip_eda=False, skip_plots=False, load_dynamics=True)
    )
    assert flags == {"EDA": True, "PostTraining": True, "LoadDynamics": True}


@pytest.mark.parametrize("column_kind", ["column", "missing", "index"])
def test_validate_inputs_uses_schema_without_loading_rows(tmp_path, monkeypatch, column_kind):
    """Validate actual columns without mistaking saved index fields for data."""
    import pandas as pd
    import pyarrow.parquet as parquet
    from load_forecasting.config import validate_inputs

    frame = pd.DataFrame({"power": [1.0, 2.0]})
    if column_kind == "missing":
        frame = frame.rename(columns={"power": "other"})
    elif column_kind == "index":
        frame = frame.set_index("power")
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path)

    def reject_rows(*args, **kwargs):
        raise AssertionError("Validation must not load Parquet rows")

    monkeypatch.setattr(pd, "read_parquet", reject_rows)
    monkeypatch.setattr(parquet, "read_table", reject_rows)
    if column_kind == "column":
        validate_inputs([path.name], tmp_path, {"power": "power"})
    else:
        with pytest.raises(ValueError, match="Missing required columns.*power"):
            validate_inputs([path.name], tmp_path, {"power": "power"})
