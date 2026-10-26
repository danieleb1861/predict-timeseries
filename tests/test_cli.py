import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from load_forecasting.cli import app


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_cli_uses_current_directory_as_default_workspace(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT)

    result = runner.invoke(app, ["train", "--models", "ridge", "--dry-run"])

    assert result.exit_code == 0, result.output
    compact_output = "".join(result.output.split())
    assert "train_load_forecasters.py" in compact_output
    assert "--modelsridge" in compact_output


def test_cli_accepts_explicit_workspace_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "--workspace",
            str(REPOSITORY_ROOT),
            "train",
            "--models",
            "ridge",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "train_load_forecasters.py" in result.output.replace("\n", "")


def test_cli_rejects_workspace_without_repository_markers(tmp_path):
    result = runner.invoke(app, ["--workspace", str(tmp_path), "status"])

    assert result.exit_code == 2
    assert "not a repository checkout" in result.output


def test_root_wrapper_uses_its_checkout_from_another_directory(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tui.py"),
            "train",
            "--models",
            "ridge",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "train_load_forecasters.py" in completed.stdout.replace("\n", "")


def test_root_wrapper_preserves_typer_exit_code(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tui.py"), "unknown-command"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
