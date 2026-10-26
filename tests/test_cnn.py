import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

from load_forecasting.models.deep_flax import (  # noqa: E402
    build_cnn_model,
    dilations_for_lookback,
    receptive_field,
)


@pytest.mark.parametrize(
    ("lookback", "expected"),
    [
        (20, (1, 2, 4, 8)),
        (50, (1, 2, 4, 8, 16)),
    ],
)
def test_cnn_dilations_cover_lookback(lookback, expected):
    dilations = dilations_for_lookback(lookback, kernel_size=3)

    assert dilations == expected
    assert receptive_field([3] * len(dilations), dilations) >= lookback


def test_cnn_builder_uses_lookback_specific_dilations():
    short = build_cnn_model((20, 17), (5, 1))
    long = build_cnn_model((50, 17), (25, 1))

    assert short.dilations == (1, 2, 4, 8)
    assert long.dilations == (1, 2, 4, 8, 16)
