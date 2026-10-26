"""Backward-compatible launcher for the package-native TUI."""

from __future__ import annotations

import sys
from pathlib import Path

from load_forecasting.cli import app


if __name__ == "__main__":
    if not any(
        argument == "--workspace" or argument.startswith("--workspace=")
        for argument in sys.argv[1:]
    ):
        sys.argv[1:1] = ["--workspace", str(Path(__file__).resolve().parent)]
    app()
