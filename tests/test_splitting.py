import pandas as pd

from load_forecasting.splitting import split_train_val_test


def test_split_train_val_test_keeps_legs_disjoint():
    """
    Verify that leg-based splitting does not mix legs across splits.

    Inputs
    ----------
    None
        The test builds a synthetic DataFrame with 20 ordered legs and two
        samples per leg.

    Returns
    -------
    None
        Assertions confirm that train, validation, and test leg sets are
        mutually disjoint and preserve the expected 70/15/15 allocation.
    """
    data_frame = pd.DataFrame(
        {
            "leg_id": [f"leg_{i}" for i in range(20) for _ in range(2)],
            "value": range(40),
        }
    )

    train, val, test = split_train_val_test(data_frame)

    train_legs = set(train["leg_id"])
    val_legs = set(val["leg_id"])
    test_legs = set(test["leg_id"])

    assert train_legs.isdisjoint(val_legs)
    assert train_legs.isdisjoint(test_legs)
    assert val_legs.isdisjoint(test_legs)
    assert len(train_legs) == 14
    assert len(val_legs) == 3
    assert len(test_legs) == 3


def test_split_train_val_test_falls_back_to_chronological_single_leg_split():
    """
    Verify chronological fallback when only one leg is available.

    Inputs
    ----------
    None
        The test builds a synthetic single-leg DataFrame with 20 ordered
        samples.

    Returns
    -------
    None
        Assertions confirm that train, validation, and test are contiguous
        chronological ranges.
    """
    data_frame = pd.DataFrame({"leg_id": ["only"] * 20, "value": range(20)})

    train, val, test = split_train_val_test(data_frame)

    assert train["value"].tolist() == list(range(14))
    assert val["value"].tolist() == [14, 15, 16]
    assert test["value"].tolist() == [17, 18, 19]
