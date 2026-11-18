import io
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



class _EncodedBuffer(io.StringIO):
    """In-memory console file that reports a chosen terminal encoding."""

    def __init__(self, encoding: str) -> None:
        super().__init__()
        self._reported_encoding = encoding

    @property
    def encoding(self) -> str:
        return self._reported_encoding


def _banner_text(width: int, encoding: str, **kwargs) -> str:
    from rich.console import Console

    from load_forecasting.brand import render_banner

    console = Console(file=_EncodedBuffer(encoding), width=width, record=True, color_system=None)
    console.print(render_banner(console, **kwargs))
    return console.export_text()


def test_banner_wordmark_rows_have_equal_width():
    from load_forecasting.brand import WORDMARK, render_wordmark

    lines = render_wordmark(WORDMARK).plain.splitlines()

    assert len(lines) == 2
    assert len({len(line) for line in lines}) == 1


def test_banner_uses_full_layout_on_wide_unicode_console():
    from load_forecasting.brand import TAGLINE

    text = _banner_text(140, "utf-8", version="0.1.0", git_ref="main@abc1234")

    assert "█" in text and "●" in text
    assert TAGLINE in text and "v0.1.0" in text and "main@abc1234" in text


def test_banner_falls_back_to_one_line_on_narrow_console():
    from load_forecasting.brand import WORDMARK

    text = _banner_text(80, "utf-8", version="0.1.0").strip()

    assert WORDMARK in text
    assert "\n" not in text
    assert "█" not in text


def test_banner_is_ascii_only_on_non_unicode_console():
    from load_forecasting.brand import WORDMARK

    text = _banner_text(140, "cp1252", version="0.1.0", git_ref="main@abc1234").strip()

    assert text.startswith(WORDMARK)
    assert text.isascii()


def test_root_command_skips_banner_when_output_is_captured(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT)

    result = runner.invoke(app, ["--status"])

    assert result.exit_code == 0, result.output
    assert "█" not in result.output
