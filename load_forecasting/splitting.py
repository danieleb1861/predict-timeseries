"""Train/validation/test split helpers."""

from __future__ import annotations

import itertools

import numpy as np


def split_train_val_test(data_frame, leg_col="leg_id", train_frac=0.70, val_frac=0.15,
                          mode="ordered", min_legs_per_split=1):
    """
    Split a time-ordered DataFrame into train, validation, and test sets.

    Inputs
    ----------
    data_frame : pandas.DataFrame
        Time-ordered dataset containing one row per timestamp.
    leg_col : str, default="leg_id"
        Column identifying voyage/operating legs. If more than one leg is
        available, whole legs are assigned to each split.
    train_frac : float, default=0.70
        Target fraction of legs (mode="ordered") or samples (mode="balanced")
        assigned to training.
    val_frac : float, default=0.15
        Target fraction of legs (mode="ordered") or samples (mode="balanced")
        assigned to validation.
    mode : {"ordered", "balanced"}, default="ordered"
        "ordered": chronological cut applying train_frac/val_frac to the
        *leg count* (legs stay in dataset order; a run's split is fully
        determined by how many legs the dataset has).
        "balanced": whole-leg assignment chosen, from the legs actually
        present, to match train_frac/val_frac by *sample count* as closely
        as possible. Still leakage-safe (whole legs, never split mid-leg),
        but the chronological order across splits is not preserved, and the
        resulting legs-per-split depend on per-leg sizes -- not comparable
        to an "ordered" run on the same data, and the exact assignment can
        change if legs are added, removed, or resized.
    min_legs_per_split : int, default=1
        Minimum number of legs required in each split for mode="balanced".

    Returns
    -------
    tuple
        Tuple ``(train, validation, test)`` of pandas.DataFrame objects. If
        only one leg is available, the split falls back to chronological
        sample ranges regardless of ``mode``.
    """
    if leg_col in data_frame.columns:
        leg_ids = data_frame[leg_col].unique()
    else:
        leg_ids = np.array([])

    num_legs = len(leg_ids)
    num_samples = len(data_frame)

    if num_legs > 1:
        if mode == "ordered":
            train_stop = int(train_frac * num_legs)
            val_stop = int((train_frac + val_frac) * num_legs)
            tr_legs = leg_ids[:train_stop]
            va_legs = leg_ids[train_stop:val_stop]
            te_legs = leg_ids[val_stop:]
        elif mode == "balanced":
            leg_sizes = data_frame.groupby(leg_col).size().reindex(leg_ids).to_numpy()
            tr_legs, va_legs, te_legs = _balanced_leg_split(
                leg_ids, leg_sizes, train_frac, val_frac, min_legs_per_split
            )
        else:
            raise ValueError(f"Unknown split mode: {mode!r}. Use 'ordered' or 'balanced'.")

        train = data_frame[data_frame[leg_col].isin(tr_legs)].reset_index(drop=True)
        val = data_frame[data_frame[leg_col].isin(va_legs)].reset_index(drop=True)
        test = data_frame[data_frame[leg_col].isin(te_legs)].reset_index(drop=True)
    else:
        train_end = int(np.floor(train_frac * num_samples))
        val_end = int(np.floor((train_frac + val_frac) * num_samples))

        train = data_frame.iloc[:train_end].reset_index(drop=True)
        val = data_frame.iloc[train_end:val_end].reset_index(drop=True)
        test = data_frame.iloc[val_end:].reset_index(drop=True)

    return train, val, test


def _split_cost(train_n, val_n, test_n, total, train_frac, val_frac):
    """Squared deviation of achieved sample fractions from the targets."""
    test_frac = 1.0 - train_frac - val_frac
    return (
        (train_n / total - train_frac) ** 2
        + (val_n / total - val_frac) ** 2
        + (test_n / total - test_frac) ** 2
    )


def _exhaustive_balanced(leg_sizes, train_frac, val_frac, min_legs_per_split):
    """Vectorized search over every 3-way whole-leg assignment (small num_legs only)."""
    num_legs = len(leg_sizes)
    total = leg_sizes.sum()
    n_combos = 3 ** num_legs

    digits = np.zeros((n_combos, num_legs), dtype=np.int8)
    idx = np.arange(n_combos)
    for k in range(num_legs):
        digits[:, k] = (idx // (3 ** k)) % 3

    train_n = (digits == 0) @ leg_sizes
    val_n = (digits == 1) @ leg_sizes
    test_n = total - train_n - val_n

    counts_ok = (
        (digits == 0).sum(axis=1) >= min_legs_per_split
    ) & (
        (digits == 1).sum(axis=1) >= min_legs_per_split
    ) & (
        (digits == 2).sum(axis=1) >= min_legs_per_split
    )

    cost = _split_cost(train_n, val_n, test_n, total, train_frac, val_frac)
    cost = np.where(counts_ok, cost, np.inf)
    return digits[int(np.argmin(cost))]


def _greedy_balanced(leg_sizes, train_frac, val_frac, min_legs_per_split, max_passes=200):
    """Deterministic LPT greedy construction + pairwise-swap hill-climbing (larger num_legs)."""
    num_legs = len(leg_sizes)
    total = leg_sizes.sum()
    targets = np.array([train_frac, val_frac, 1.0 - train_frac - val_frac]) * total

    # Largest legs first, each into whichever split is currently furthest
    # below its target sample share.
    order = np.argsort(-leg_sizes)
    labels = np.full(num_legs, -1, dtype=int)
    running = np.zeros(3)
    for i in order:
        labels[i] = int(np.argmax(targets - running))
        running[labels[i]] += leg_sizes[i]

    # Repair any split starved below min_legs_per_split by donating the
    # smallest leg from whichever split currently has the most legs.
    for split_id in range(3):
        while np.sum(labels == split_id) < min_legs_per_split:
            donor_split = int(np.argmax(np.bincount(labels, minlength=3)))
            donor_idxs = np.flatnonzero(labels == donor_split)
            move = donor_idxs[np.argmin(leg_sizes[donor_idxs])]
            labels[move] = split_id

    def valid(labels):
        counts = np.bincount(labels, minlength=3)
        return bool((counts >= min_legs_per_split).all())

    def cost_of(labels):
        train_n = leg_sizes[labels == 0].sum()
        val_n = leg_sizes[labels == 1].sum()
        test_n = total - train_n - val_n
        return _split_cost(train_n, val_n, test_n, total, train_frac, val_frac)

    current_cost = cost_of(labels)
    for _ in range(max_passes):
        improved = False
        for i, j in itertools.combinations(range(num_legs), 2):
            if labels[i] == labels[j]:
                continue
            labels[i], labels[j] = labels[j], labels[i]
            new_cost = cost_of(labels) if valid(labels) else np.inf
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                labels[i], labels[j] = labels[j], labels[i]  # revert
        if not improved:
            break

    return labels


def _balanced_leg_split(leg_ids, leg_sizes, train_frac, val_frac, min_legs_per_split,
                         exhaustive_limit=12):
    """
    Assign whole legs to train/val/test minimizing squared deviation from
    the target sample fractions (see split_train_val_test mode="balanced").

    Exhaustive search over all 3**num_legs assignments when num_legs is
    small enough to be tractable (default up to 12 legs, ~531k
    combinations); otherwise a deterministic greedy construction refined by
    pairwise-swap hill-climbing.
    """
    num_legs = len(leg_ids)
    if num_legs <= exhaustive_limit:
        labels = _exhaustive_balanced(leg_sizes, train_frac, val_frac, min_legs_per_split)
    else:
        labels = _greedy_balanced(leg_sizes, train_frac, val_frac, min_legs_per_split)

    return leg_ids[labels == 0], leg_ids[labels == 1], leg_ids[labels == 2]


def lovo_split(data_frame, fold, leg_col="leg_id", val_legs=2):
    """
    Leave-one-voyage-out split: hold out one whole leg as the test set.

    Fold ``fold`` assigns leg ``leg_ids[fold]`` (dataset order) to test. Of the
    remaining legs, the last ``val_legs`` in dataset order become validation and
    the rest become training, so the training/validation boundary stays as late
    in the record as it does under mode="ordered" of split_train_val_test().
    Leakage safety is unchanged: legs are never split mid-leg, and both the
    normalisation statistics and the upper-tail threshold are still fitted on
    the per-fold training set only.

    Inputs
    ----------
    data_frame : pandas.DataFrame
        Time-ordered dataset containing one row per timestamp.
    fold : int
        Zero-based index into the legs present in ``data_frame``.
    leg_col : str, default="leg_id"
        Column identifying voyage/operating legs.
    val_legs : int, default=2
        Number of legs assigned to validation, matching the validation leg count
        of the reported manoeuvre benchmark split.

    Returns
    -------
    tuple
        Tuple ``(train, validation, test)`` of pandas.DataFrame objects.
    """
    if leg_col not in data_frame.columns:
        raise ValueError(f"Leave-one-voyage-out split requires a '{leg_col}' column.")

    leg_ids = list(data_frame[leg_col].unique())
    num_legs = len(leg_ids)
    if num_legs < val_legs + 2:
        raise ValueError(
            f"Leave-one-voyage-out needs at least {val_legs + 2} legs, found {num_legs}."
        )
    if not 0 <= fold < num_legs:
        raise ValueError(f"LOVO fold must be in [0, {num_legs}), got {fold}.")

    held_out = leg_ids[fold]
    remaining = [leg for leg in leg_ids if leg != held_out]
    va_legs = remaining[-val_legs:]
    tr_legs = remaining[:-val_legs]

    def subset(legs):
        return data_frame[data_frame[leg_col].isin(legs)].reset_index(drop=True)

    print(f"LOVO fold {fold} of {num_legs}: test leg={int(held_out)}, "
          f"val legs={[int(v) for v in va_legs]}, "
          f"{len(tr_legs)} training legs")

    return subset(tr_legs), subset(va_legs), subset([held_out])
