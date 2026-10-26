# Load Forecasting TUI

This repo includes a lightweight Typer + Rich command dashboard in
`load_forecasting/cli.py`. The installed command is `predict-timeseries`;
root-level `tui.py` remains as a compatibility wrapper.
It wraps existing scripts; it does not change the forecasting pipeline.

## Install

Install the package with ML and TUI dependencies:

```bash
pip install -e ".[ml]"
```

For development and tests:

```bash
pip install -e ".[ml,dev]"
```

## Commands

Show repo status:

```bash
predict-timeseries status
```

The root command also defaults to status:

```bash
predict-timeseries
predict-timeseries --status
```

List presets:

```bash
predict-timeseries presets
predict-timeseries --presets
```

Preview a training command without running it:

```bash
predict-timeseries train --preset navigation --models xgb,mlp --epochs 5 --dry-run
```

Run training:

```bash
predict-timeseries train --preset navigation --models xgb,mlp --epochs 5
```

Run the benchmark wrapper:

```bash
predict-timeseries benchmark --selector manoeuvre
```

Show recent result metrics:

```bash
predict-timeseries results
predict-timeseries --results
```

Run tests:

```bash
predict-timeseries test
```

Check that the TUI is still aligned with script parsers, presets, and pytest config:

```bash
predict-timeseries check
```

Run post-processing figures:

```bash
predict-timeseries postprocess --kind all
```

The command uses the current directory as the repository workspace. From
another directory, point it to a checkout explicitly:

```bash
# Linux / macOS
predict-timeseries --workspace /path/to/predict-timeseries status
```

```powershell
# Windows (PowerShell)
predict-timeseries --workspace D:\path\to\predict-timeseries status
```

`python tui.py ...` remains supported and automatically uses the checkout that
contains `tui.py` unless `--workspace` is supplied explicitly.

## Testing the TUI

Basic checks:

```bash
python -m py_compile load_forecasting/cli.py tui.py
predict-timeseries --help
predict-timeseries status
predict-timeseries presets
predict-timeseries check
predict-timeseries train --preset manoeuvre_fast --models cnn --epochs 1 --dry-run
predict-timeseries test
```

Use `--dry-run` before expensive training or benchmark commands.
