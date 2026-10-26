"""One-off smoke test for scripts/foundation_adapters.py on synthetic data.

Run from the isolated venv:
    source .venv-foundation/bin/activate
    python scripts/smoke_test_foundation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from foundation_adapters import ADAPTER_BUILDERS  # noqa: E402


def main() -> int:
    batch, context_len, horizon, n_covariates = 4, 45, 25, 16
    rng = np.random.default_rng(0)
    history = np.sin(np.linspace(0, 8 * np.pi, context_len))[None, :] + \
        0.05 * rng.standard_normal((batch, context_len))
    history = history.astype(np.float32)
    covariates = rng.standard_normal((batch, context_len, n_covariates)).astype(np.float32)

    for key, builder in ADAPTER_BUILDERS.items():
        print(f"\n=== {key} ===")
        try:
            adapter = builder(device="cuda")
        except Exception:
            print(f"  falling back to CPU for {key}")
            adapter = builder(device="cpu")
        cov = covariates if adapter.covariate_aware else None
        out = adapter.predict_batch(history, horizon, cov)
        print(f"  label={adapter.label}  output shape={out.shape}  "
              f"finite={np.isfinite(out).all()}  sample={out[0, :5]}")
        assert out.shape == (batch, horizon), f"{key}: unexpected shape {out.shape}"
        assert np.isfinite(out).all(), f"{key}: non-finite values in output"
    print("\nAll adapters produced finite, correctly-shaped output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
