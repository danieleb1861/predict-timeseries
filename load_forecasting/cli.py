"""Lightweight command dashboard for the load forecasting repository."""

from __future__ import annotations

import csv
import ast
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _default_config(config_dir: Path) -> Path:
    """Mirror load_forecasting.config.resolve_default_config.

    Duplicated on purpose: importing that module would pull pandas into the
    dependency path of every `predict-timeseries --help`.
    """
    local_config = config_dir / "my_system_config.json"

    return local_config if local_config.exists() else config_dir / "default_config.json"


ROOT = Path.cwd().resolve()
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "outputs" / "runs"
DATA_DIR = ROOT / "data"
DEFAULT_CONFIG = _default_config(CONFIG_DIR)
DEFAULT_PRESET_FILE = CONFIG_DIR / "presets.json"
PYPROJECT = ROOT / "pyproject.toml"

SCRIPT_PATHS = {
    "train": ROOT / "train_load_forecasters.py",
    "train_parser": ROOT / "load_forecasting" / "config.py",
    "benchmark": ROOT / "run_benchmark.py",
    "postprocess": ROOT / "postprocess_figures.py",
}

app = typer.Typer(
    add_completion=False,
    help="Operate the load forecasting repo without memorizing every command.",
)
console = Console()


def _configure_workspace(workspace: Path) -> None:
    """Resolve and validate the repository checkout used by TUI commands."""
    root = workspace.expanduser().resolve()
    required_paths = (
        root / "pyproject.toml",
        root / "train_load_forecasters.py",
        root / "config",
        root / "load_forecasting",
    )
    missing = [path.name for path in required_paths if not path.exists()]

    if missing:
        raise typer.BadParameter(
            "not a repository checkout; missing: " + ", ".join(missing),
            param_hint="--workspace",
        )

    global ROOT, CONFIG_DIR, RUNS_DIR, DATA_DIR
    global DEFAULT_CONFIG, DEFAULT_PRESET_FILE, PYPROJECT, SCRIPT_PATHS
    ROOT = root
    CONFIG_DIR = ROOT / "config"
    RUNS_DIR = ROOT / "outputs" / "runs"
    DATA_DIR = ROOT / "data"
    DEFAULT_CONFIG = _default_config(CONFIG_DIR)
    DEFAULT_PRESET_FILE = CONFIG_DIR / "presets.json"
    PYPROJECT = ROOT / "pyproject.toml"
    SCRIPT_PATHS = {
        "train": ROOT / "train_load_forecasters.py",
        "train_parser": ROOT / "load_forecasting" / "config.py",
        "benchmark": ROOT / "run_benchmark.py",
        "postprocess": ROOT / "postprocess_figures.py",
    }


def _raise_csv_field_limit() -> None:
    """
    Increase the maximum CSV field size limit.

    Inputs
    ----------
    None

    Returns
    -------
    None
        Updates the global CSV parser field size limit.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _run_command(command: list[str], dry_run: bool = False) -> None:
    """
    Run a command from the repository root.

    Inputs
    ----------
    command : list[str]
        Command and arguments to execute.
    dry_run : bool, optional
        If True, print the command without executing it.

    Returns
    -------
    None
        Executes the command unless dry_run is enabled.
    """
    console.print("[bold]Command:[/bold] " + " ".join(command))
    if dry_run:
        console.print("[yellow]Dry run only. Nothing executed.[/yellow]")
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def _relative(path: Path) -> str:
    """
    Convert a path to a repository-relative string when possible.

    Inputs
    ----------
    path : pathlib.Path
        Path to format.

    Returns
    -------
    str
        Repository-relative path if the path is inside ROOT, otherwise the
        original path converted to string.
    """
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _exists_status(path: Path) -> Text:
    """
    Return a Rich status label based on path existence.

    Inputs
    ----------
    path : pathlib.Path
        Path to check.

    Returns
    -------
    rich.text.Text
        Green OK label if the path exists, red missing label otherwise.
    """
    return Text("OK", style="green") if path.exists() else Text("missing", style="red")


def _load_presets(preset_file: Optional[Path] = None) -> dict:
    """
    Load experiment presets from a JSON file.

    Inputs
    ----------
    preset_file : pathlib.Path, optional
        Preset JSON file to read.

    Returns
    -------
    dict
        Dictionary of available presets. Returns an empty dictionary if the file
        is missing or does not contain a valid preset mapping.
    """
    preset_file = preset_file or DEFAULT_PRESET_FILE

    if not preset_file.exists():
        return {}

    with preset_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    presets = data.get("presets", data)
    return presets if isinstance(presets, dict) else {}


def _load_pyproject() -> dict:
    """
    Load pyproject.toml from the repository root.

    Inputs
    ----------
    None

    Returns
    -------
    dict
        Parsed pyproject.toml content. Returns an empty dictionary if the file
        does not exist.
    """
    if not PYPROJECT.exists():
        return {}

    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _pytest_command() -> list[str]:
    """
    Build the pytest command from pyproject.toml configuration.

    Inputs
    ----------
    None

    Returns
    -------
    list[str]
        Python pytest command, including configured test paths when available.
    """
    pyproject = _load_pyproject()
    pytest_options = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})
    testpaths = pytest_options.get("testpaths", [])

    return [sys.executable, "-m", "pytest", *testpaths]


def _script_options(source_path: Path) -> set[str]:
    """
    Extract supported CLI options from argparse add_argument calls.

    Inputs
    ----------
    source_path : pathlib.Path
        Python source file to inspect.

    Returns
    -------
    set[str]
        Set of option strings defined by the script, such as "--selector".
    """
    if not source_path.exists():
        return set()

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    options = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("-")
            ):
                options.add(arg.value)

    return options


def _script_choices(source_path: Path, option_name: str) -> tuple[str, ...]:
    """
    Extract argparse choices for a specific CLI option.

    Inputs
    ----------
    source_path : pathlib.Path
        Python source file to inspect.
    option_name : str
        CLI option whose choices should be extracted.

    Returns
    -------
    tuple[str, ...]
        Tuple of allowed choice values. Returns an empty tuple if no choices are
        found.
    """
    if not source_path.exists():
        return ()

    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        has_option = any(
            isinstance(arg, ast.Constant) and arg.value == option_name
            for arg in node.args
        )
        if not has_option:
            continue

        for keyword in node.keywords:
            if keyword.arg != "choices":
                continue

            if isinstance(keyword.value, (ast.Tuple, ast.List)):
                return tuple(
                    elt.value
                    for elt in keyword.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )

    return ()


def _validate_script_flags(command_name: str, flags: list[str]) -> None:
    """
    Validate that command flags are supported by the target script.

    Inputs
    ----------
    command_name : str
        Key identifying the target script in SCRIPT_PATHS.
    flags : list[str]
        CLI flags passed by the TUI wrapper.

    Returns
    -------
    None
        Raises typer.BadParameter if unsupported flags are detected.
    """
    source_path = SCRIPT_PATHS[command_name]
    options = _script_options(source_path)
    unsupported = sorted({flag for flag in flags if flag.startswith("--")} - options)

    if unsupported:
        raise typer.BadParameter(
            f"{_relative(source_path)} does not define these options: {unsupported}"
        )


def _validate_train_flags(flags: list[str]) -> None:
    """
    Validate training flags against the training parser source.

    Inputs
    ----------
    flags : list[str]
        CLI flags passed to the training command.

    Returns
    -------
    None
        Raises typer.BadParameter if unsupported training flags are detected.
    """
    source_path = SCRIPT_PATHS["train_parser"]
    options = _script_options(source_path)
    unsupported = sorted({flag for flag in flags if flag.startswith("--")} - options)

    if unsupported:
        raise typer.BadParameter(
            f"{_relative(source_path)} does not define these training options: {unsupported}"
        )


def _latest_run_dir() -> Optional[Path]:
    """
    Find the most recently modified run directory.

    Inputs
    ----------
    None

    Returns
    -------
    pathlib.Path or None
        Latest run directory, or None if no run directory is available.
    """
    if not RUNS_DIR.exists():
        return None

    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _metrics_files() -> list[Path]:
    """
    Find available metrics CSV files under the runs directory.

    Inputs
    ----------
    None

    Returns
    -------
    list[pathlib.Path]
        Metrics CSV files sorted by modification time, newest first.
    """
    if not RUNS_DIR.exists():
        return []

    return sorted(
        RUNS_DIR.rglob("results_metrics*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _add_optional(command: list[str], flag: str, value: object | None) -> None:
    """
    Append an optional CLI argument to a command list.

    Inputs
    ----------
    command : list[str]
        Command list to update in place.
    flag : str
        CLI flag to append.
    value : object or None
        Value associated with the flag. If None, nothing is appended.

    Returns
    -------
    None
        Updates command in place.
    """
    if value is not None:
        command.extend([flag, str(value)])


def _command_flags(command: list[str]) -> list[str]:
    """
    Extract CLI flags from a command list.

    Inputs
    ----------
    command : list[str]
        Command and arguments.

    Returns
    -------
    list[str]
        Items starting with "--".
    """
    return [item for item in command if item.startswith("--")]


def _preview_cell(value: str, max_chars: int = 48) -> str:
    """
    Shorten a string for compact table display.

    Inputs
    ----------
    value : str
        Cell value to display.
    max_chars : int, optional
        Maximum number of characters to keep.

    Returns
    -------
    str
        Original value if short enough, otherwise truncated value with ellipsis.
    """
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


def _render_status() -> None:
    """
    Render a compact repository status table.

    Inputs
    ----------
    None

    Returns
    -------
    None
        Prints repository status information to the console.
    """
    table = Table(title="Load Forecasting Repo Status")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Path / Value")

    package_ok = importlib.util.find_spec("load_forecasting") is not None
    package_status = (
        Text("OK", style="green")
        if package_ok
        else Text("not importable", style="red")
    )

    table.add_row("Python", Text("OK", style="green"), sys.version.split()[0])
    table.add_row("Package", package_status, "load_forecasting")
    table.add_row(
        "Default config",
        _exists_status(DEFAULT_CONFIG),
        str(DEFAULT_CONFIG.relative_to(ROOT)),
    )
    table.add_row(
        "Preset file",
        _exists_status(DEFAULT_PRESET_FILE),
        str(DEFAULT_PRESET_FILE.relative_to(ROOT)),
    )
    table.add_row(
        "Data folder",
        _exists_status(DATA_DIR),
        str(DATA_DIR.relative_to(ROOT)),
    )
    table.add_row(
        "Runs folder",
        _exists_status(RUNS_DIR),
        str(RUNS_DIR.relative_to(ROOT)),
    )

    latest_run = _latest_run_dir()
    latest_text = str(latest_run.relative_to(ROOT)) if latest_run else "none"
    latest_status = Text("OK", style="green") if latest_run else Text("none", style="yellow")
    table.add_row("Latest run dir", latest_status, latest_text)

    metrics_count = len(_metrics_files())
    table.add_row(
        "Metrics CSVs",
        Text(str(metrics_count), style="cyan"),
        "outputs/runs/**/results_metrics*.csv",
    )

    pytest_cmd = " ".join(
        Path(part).name if index == 0 else part
        for index, part in enumerate(_pytest_command())
    )
    table.add_row("Test command", Text("OK", style="green"), pytest_cmd)

    console.print(table)


def _render_presets(preset_file: Optional[Path] = None) -> None:
    """
    Render available experiment presets.

    Inputs
    ----------
    preset_file : pathlib.Path, optional
        Preset JSON file to read.

    Returns
    -------
    None
        Prints a preset summary table to the console.
    """
    preset_file = preset_file or DEFAULT_PRESET_FILE
    preset_data = _load_presets(preset_file)

    if not preset_data:
        console.print(Panel(f"No presets found in {preset_file}", style="yellow"))
        raise typer.Exit(code=1)

    table = Table(
        title=(
            f"Presets from "
            f"{preset_file.relative_to(ROOT) if preset_file.is_relative_to(ROOT) else preset_file}"
        )
    )
    table.add_column("Name")
    table.add_column("Selector")
    table.add_column("Lookback")
    table.add_column("Horizon")
    table.add_column("Models")

    for name, values in sorted(preset_data.items()):
        table.add_row(
            name,
            str(values.get("selector", "")),
            str(values.get("lookback", "")),
            str(values.get("horizon", "")),
            str(values.get("models", "")),
        )

    console.print(table)


def _render_results(limit: int = 5, rows: int = 8) -> None:
    """
    Render compact previews of recent metrics CSV files.

    Inputs
    ----------
    limit : int, optional
        Maximum number of metrics files to show.
    rows : int, optional
        Maximum number of rows to show for each metrics file.

    Returns
    -------
    None
        Prints result preview tables to the console.
    """
    _raise_csv_field_limit()
    files = _metrics_files()[:limit]

    if not files:
        console.print(Panel("No results_metrics*.csv files found under outputs/runs/.", style="yellow"))
        return

    for csv_path in files:
        console.print(Panel(str(csv_path.relative_to(ROOT)), style="cyan"))

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)

            try:
                header = next(reader)
            except StopIteration:
                console.print("[yellow]Empty CSV.[/yellow]")
                continue

            table = Table(show_header=True)

            for column in header[:6]:
                table.add_column(column or "model")

            for index, row in enumerate(reader):
                if index >= rows:
                    break

                table.add_row(*[_preview_cell(value) for value in row[:6]])

            console.print(table)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        help="Repository checkout root (default: current directory).",
        file_okay=False,
        dir_okay=True,
    ),
    show_status: bool = typer.Option(
        False,
        "--status",
        help="Show repository status and exit.",
    ),
    show_presets: bool = typer.Option(
        False,
        "--presets",
        help="List configured presets and exit.",
    ),
    show_results: bool = typer.Option(
        False,
        "--results",
        help="Show recent results and exit.",
    ),
) -> None:
    """
    Run the default TUI entry point.

    Inputs
    ----------
    ctx : typer.Context
        Typer execution context.
    workspace : pathlib.Path or None, optional
        Repository checkout root. Defaults to the current directory.
    show_status : bool, optional
        If True, show repository status and exit.
    show_presets : bool, optional
        If True, list configured presets and exit.
    show_results : bool, optional
        If True, show recent results and exit.

    Returns
    -------
    None
        Prints the requested dashboard view and exits.
    """
    _configure_workspace(workspace or Path.cwd())

    if ctx.invoked_subcommand is not None:
        return

    if show_presets:
        _render_presets()
        raise typer.Exit()

    if show_results:
        _render_results()
        raise typer.Exit()

    _render_status()
    raise typer.Exit()


@app.command()
def status() -> None:
    """
    Show repository health, important folders, and latest outputs.

    Inputs
    ----------
    None

    Returns
    -------
    None
        Prints repository status information to the console.
    """
    _render_status()


@app.command()
def presets(
    preset_file: Optional[Path] = typer.Option(
        None,
        "--preset-file",
        help="Preset JSON file (default: <workspace>/config/presets.json).",
    ),
) -> None:
    """
    List configured experiment presets.

    Inputs
    ----------
    preset_file : pathlib.Path or None
        Preset JSON file to read.

    Returns
    -------
    None
        Prints configured presets to the console.
    """
    _render_presets(preset_file)


@app.command()
def train(
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Preset passed to train_load_forecasters.py.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="System/dataset config JSON.",
    ),
    preset_file: Optional[Path] = typer.Option(
        None,
        "--preset-file",
        help="Preset JSON file.",
    ),
    selector: Optional[str] = typer.Option(
        None,
        "--selector",
        help="Override selector: navigation, manoeuvre, all.",
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        help="Comma-separated models or all.",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Deep-learning backend: keras or flax.",
    ),
    lookback: Optional[int] = typer.Option(
        None,
        "--lookback",
        help="Lookback window in seconds.",
    ),
    horizon: Optional[int] = typer.Option(
        None,
        "--horizon",
        help="Forecast horizon in seconds.",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override training epochs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override training batch size.",
    ),
    run_dir: Optional[Path] = typer.Option(
        None,
        "--run-dir",
        help="Output directory.",
    ),
    skip_eda: bool = typer.Option(
        False,
        "--skip-eda",
        help="Disable optional EDA outputs.",
    ),
    skip_plots: bool = typer.Option(
        False,
        "--skip-plots",
        help="Disable post-training plots.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print command without running it.",
    ),
) -> None:
    """
    Run the main forecasting training and evaluation script.

    Inputs
    ----------
    preset : str or None
        Preset name passed to the training script.
    config : pathlib.Path or None
        System or dataset configuration file.
    preset_file : pathlib.Path or None
        Preset JSON file.
    selector : str or None
        Dataset selector override.
    models : str or None
        Comma-separated model list or "all".
    lookback : int or None
        Lookback window in seconds.
    horizon : int or None
        Forecast horizon in seconds.
    epochs : int or None
        Training epoch override.
    batch_size : int or None
        Training batch-size override.
    run_dir : pathlib.Path or None
        Output directory.
    skip_eda : bool
        If True, disable optional EDA outputs.
    skip_plots : bool
        If True, disable post-training plots.
    dry_run : bool
        If True, print the command without executing it.

    Returns
    -------
    None
        Builds, validates, and optionally executes the training command.
    """
    command = [
        sys.executable,
        str(SCRIPT_PATHS["train"]),
    ]

    _add_optional(command, "--config", config)
    _add_optional(command, "--preset", preset)
    _add_optional(command, "--preset-file", preset_file)
    _add_optional(command, "--selector", selector)
    _add_optional(command, "--models", models)
    _add_optional(command, "--backend", backend)
    _add_optional(command, "--lookback", lookback)
    _add_optional(command, "--horizon", horizon)
    _add_optional(command, "--epochs", epochs)
    _add_optional(command, "--batch-size", batch_size)
    _add_optional(command, "--run-dir", run_dir)

    if skip_eda:
        command.append("--skip-eda")

    if skip_plots:
        command.append("--skip-plots")

    _validate_train_flags(_command_flags(command))
    _run_command(command, dry_run=dry_run)


@app.command()
def benchmark(
    selector: Optional[str] = typer.Option(
        None,
        "--selector",
        help="Benchmark selector passed to run_benchmark.py.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print command without running it.",
    ),
) -> None:
    """
    Run the benchmark grid wrapper.

    Inputs
    ----------
    selector : str or None
        Benchmark selector passed to the benchmark script.
    dry_run : bool
        If True, print the command without executing it.

    Returns
    -------
    None
        Builds, validates, and optionally executes the benchmark command.
    """
    command = [sys.executable, str(SCRIPT_PATHS["benchmark"])]

    _add_optional(command, "--selector", selector)
    _validate_script_flags("benchmark", _command_flags(command))
    _run_command(command, dry_run=dry_run)


@app.command()
def postprocess(
    kind: str = typer.Option(
        "all",
        "--kind",
        help="Postprocess kind: figures_4papers, inference, all.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print command without running it.",
    ),
) -> None:
    """
    Run optional figure post-processing workflows.

    Inputs
    ----------
    kind : str
        Post-processing workflow to run.
    dry_run : bool
        If True, print the command without executing it.

    Returns
    -------
    None
        Builds, validates, and optionally executes the post-processing command.
    """
    choices = _script_choices(SCRIPT_PATHS["postprocess"], "--kind")

    if choices and kind not in choices:
        raise typer.BadParameter(f"--kind must be one of {choices}")

    command = [
        sys.executable,
        str(SCRIPT_PATHS["postprocess"]),
        "--kind",
        kind,
    ]

    _validate_script_flags("postprocess", _command_flags(command))
    _run_command(command, dry_run=dry_run)


@app.command("test")
def run_tests(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print command without running it.",
    ),
) -> None:
    """
    Run the repository pytest suite.

    Inputs
    ----------
    dry_run : bool
        If True, print the command without executing it.

    Returns
    -------
    None
        Builds and optionally executes the pytest command.
    """
    command = _pytest_command()
    _run_command(command, dry_run=dry_run)


@app.command()
def check() -> None:
    """
    Check alignment between TUI wrappers, script parsers, and pytest config.

    Inputs
    ----------
    None

    Returns
    -------
    None
        Prints an alignment check table to the console.
    """
    table = Table(title="TUI Alignment Check")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Details")

    train_options = _script_options(SCRIPT_PATHS["train_parser"])
    benchmark_options = _script_options(SCRIPT_PATHS["benchmark"])
    postprocess_kinds = _script_choices(SCRIPT_PATHS["postprocess"], "--kind")
    presets_data = _load_presets(DEFAULT_PRESET_FILE)
    pytest_cmd = _pytest_command()

    table.add_row(
        _relative(SCRIPT_PATHS["train_parser"]),
        Text("OK", style="green"),
        f"{len(train_options)} train CLI options",
    )
    table.add_row(
        _relative(DEFAULT_PRESET_FILE),
        Text("OK", style="green") if presets_data else Text("missing", style="red"),
        f"{len(presets_data)} presets",
    )
    table.add_row(
        _relative(SCRIPT_PATHS["benchmark"]),
        Text("OK", style="green"),
        ", ".join(sorted(benchmark_options)) or "no options",
    )
    table.add_row(
        _relative(SCRIPT_PATHS["postprocess"]),
        Text("OK", style="green"),
        f"--kind choices: {', '.join(postprocess_kinds)}",
    )
    table.add_row(
        _relative(PYPROJECT),
        Text("OK", style="green"),
        " ".join(pytest_cmd),
    )

    console.print(table)


@app.command()
def results(
    limit: int = typer.Option(
        5,
        "--limit",
        min=1,
        help="Maximum metrics files to show.",
    ),
    rows: int = typer.Option(
        8,
        "--rows",
        min=1,
        help="Maximum rows per metrics file.",
    ),
) -> None:
    """
    Show recent metrics CSV files and compact metric previews.

    Inputs
    ----------
    limit : int
        Maximum number of metrics files to show.
    rows : int
        Maximum number of rows to show per metrics file.

    Returns
    -------
    None
        Prints compact previews of recent metrics files.
    """
    _render_results(limit, rows)


if __name__ == "__main__":
    app()
