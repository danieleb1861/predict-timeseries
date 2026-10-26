import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Probed in a subprocess so the assertion cannot be weakened by a TensorFlow
# import performed earlier in the session by another test.
BACKEND_LAZY_IMPORT_PROBE = """
import sys

import load_forecasting.models  # noqa: F401

print("tensorflow" in sys.modules, "keras" in sys.modules)
"""


def test_model_registry_import_is_backend_lazy():
    completed = subprocess.run(
        [sys.executable, "-c", BACKEND_LAZY_IMPORT_PROBE],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["False", "False"]
