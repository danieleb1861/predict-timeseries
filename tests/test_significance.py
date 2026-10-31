import numpy as np
import pytest

from scripts.run_significance import newey_west_variance


def test_newey_west_divides_autocovariances_by_full_sample_size():
    d = np.array([1.0, -2.0, 3.0, -1.0, 0.5, 2.0])
    centred = d - d.mean()
    n = d.size
    gamma0 = np.dot(centred, centred) / n
    gamma1 = np.dot(centred[1:], centred[:-1]) / n
    gamma2 = np.dot(centred[2:], centred[:-2]) / n

    expected = (gamma0 + 2.0 * (2 / 3) * gamma1 + 2.0 * (1 / 3) * gamma2) / n

    assert newey_west_variance(d, 2) == pytest.approx(expected)
