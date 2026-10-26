"""Configuration, path, and input validation helpers."""

import argparse
import json
from pathlib import Path


def resolve_project_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path)


def resolve_default_config(config_dir):
    """Return my_system_config.json when present, otherwise default_config.json.

    The local override is git-ignored, so a fresh clone resolves to the bundled
    synthetic demo while a working checkout keeps using its own dataset.
    """
    config_dir = Path(config_dir)
    local_config = config_dir / "my_system_config.json"

    return str(local_config if local_config.exists() else config_dir / "default_config.json")


def load_model_specs(model_specs_file):
    """Read the shared deep/hybrid model catalog without importing training scripts."""
    with open(model_specs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    deep_specs = data.get("deep_models", {})
    hybrid_specs = data.get("hybrid_models", {})
    if not isinstance(deep_specs, dict):
        raise ValueError(f"Model specs file must contain a 'deep_models' object: {model_specs_file}")
    if not isinstance(hybrid_specs, dict):
        raise ValueError(f"Model specs file must contain a 'hybrid_models' object: {model_specs_file}")
    return deep_specs, hybrid_specs


def load_system_config(config_file):
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    for key in ("paths", "dataset", "physical_system"):
        if key not in config:
            raise ValueError(f"Config file missing required section: {key}")
    return config


def validate_inputs(files, data_dir, colMap):
    if not files:
        raise ValueError("No input parquet files configured.")

    data_dir = Path(data_dir)
    missing = [f for f in files if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input parquet files: {missing}")

    sample_path = data_dir / files[0]
    from pyarrow.parquet import read_schema

    # Read only the footer; stored pandas index fields are not DataFrame columns.
    schema = read_schema(sample_path)
    index_fields = (schema.pandas_metadata or {}).get("index_columns", [])
    columns = [name for name in schema.names if name not in index_fields]
    missing_cols = [v for v in colMap.values() if v not in columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {sample_path}: {missing_cols}")


def load_preset_values(preset_file, preset_name, preset_keys):
    if preset_name is None:
        return {}

    preset_path = Path(preset_file)
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    with preset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    presets = data.get("presets", data)
    if not isinstance(presets, dict):
        raise ValueError(f"Preset file must contain a JSON object of presets: {preset_path}")
    if preset_name not in presets:
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {sorted(presets)}")

    values = presets[preset_name]
    if not isinstance(values, dict):
        raise ValueError(f"Preset '{preset_name}' must be a JSON object.")

    unknown = set(values) - preset_keys
    if unknown:
        raise ValueError(f"Preset '{preset_name}' has unsupported keys: {sorted(unknown)}")

    return values


def merge_cli_with_preset(args, cli_defaults, preset_keys):
    preset_values = load_preset_values(args.preset_file, args.preset, preset_keys)

    for key, default in cli_defaults.items():
        cli_value = getattr(args, key)
        setattr(args, key, preset_values.get(key, default) if cli_value is None else cli_value)

    return args


def parse_cli_config(default_config_file, default_preset_file, model_labels, preset_keys, cli_defaults):
    """
    Parse command-line arguments for model training and evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with attributes:
        - selector: str, one of "navigation", "manoeuvre", or "all"
        - lovo_fold: int or None, leave-one-voyage-out fold index
        - config: str, system/dataset config file
        - preset: str or None, named experiment preset loaded from --preset-file
        - preset_file: str, JSON file containing named presets
        - lookback: int, lookback window in seconds
        - horizon: int, forecast horizon in seconds
        - run_dir: str, output directory for models, metrics, windows, and figures
        - backend: str, deep-learning backend, either "keras" or "flax"
        - skip_eda: bool, whether to disable optional exploratory-data-analysis outputs
        - skip_plots: bool, whether to disable post-training comparison plots
        - models: str, comma-separated model keys or "all"
        - epochs: int or None, override epochs for every selected model
        - batch_size: int or None, override training batch size for every selected model
        - learning_rate: float or None, override learning rate for every selected model
        - predict_batch_size: int, batch size used for test prediction
        - model_specs: str or None, JSON file with per-model training specs
        - resume: bool, whether to skip models with matching saved predictions
    """
    parser = argparse.ArgumentParser(
        description="Train/evaluate load forecasting models for one selector and window configuration."
    )
    parser.add_argument("--config", default=default_config_file, help="JSON file containing dataset and physical-system configuration.")
    parser.add_argument("--preset", default="manoeuvre", help="Named experiment preset loaded from --preset-file.")
    parser.add_argument("--preset-file", default=default_preset_file, help="JSON file containing named run presets.")
    parser.add_argument("--selector", choices=("navigation", "manoeuvre", "all"), default=None)
    parser.add_argument("--lovo-fold", type=int, default=None, metavar="K",
                        help="Leave-one-voyage-out fold: hold out leg #K (0-based, dataset order) as the "
                             "test split and train on the rest. Default: the configured train/val/test split.")
    parser.add_argument("--lookback", type=int, default=None, help="Lookback window in seconds.")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in seconds.")
    parser.add_argument("--run-dir", default=None, help="Output directory for models, metrics, windows, and figures.")
    parser.add_argument("--backend", choices=("keras", "flax"), default=None, help="Deep-learning backend.")
    parser.add_argument("--skip-eda", default=None, action="store_true", help="Disable optional exploratory data analysis outputs.")
    parser.add_argument("--skip-plots", default=None, action="store_true", help="Disable post-training comparison plots.")
    parser.add_argument("--load-dynamics", default=None, action="store_true", help="Enable per-leg/per-split load-dynamics diagnostics tables.")
    parser.add_argument("--models", default=None, help=f"Comma-separated model keys or 'all'. Available: {', '.join(model_labels)}")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for every selected model.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size for every selected model.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate for every selected model.")
    parser.add_argument("--predict-batch-size", type=int, default=None, help="Batch size used for test prediction.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for weight init, dropout, batch shuffling, and tree baselines.")
    parser.add_argument("--model-specs", default=None, help="JSON file with per-model training specs (default: config/model_specs.json).")
    parser.add_argument("--resume", default=None, action="store_true", help="Skip models whose saved predictions in <run-dir>/predictions match the current window configuration.")
    return merge_cli_with_preset(parser.parse_args(), cli_defaults, preset_keys)


def parse_selected_models(raw_models, model_labels):
    raw_models = raw_models.strip().lower()
    if raw_models == "all":
        return set(model_labels)
    selected = {item.strip() for item in raw_models.split(",") if item.strip()}
    unknown = selected - set(model_labels)
    if unknown:
        raise ValueError(f"Unknown --models values: {sorted(unknown)}. Valid: {sorted(model_labels)} or 'all'.")
    if not selected:
        raise ValueError("--models must be 'all' or a comma-separated list of model keys.")
    return selected


def cfg_or(value, default):
    return default if value is None else value
