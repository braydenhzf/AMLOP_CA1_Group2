"""Preserve historical evaluation membership across pipeline runs."""

import pandas as pd

from src.features import chronological_split


KEYS = ["country_code", "order_id"]
SPLIT_NAMES = ("train", "validation", "test")


def _check_keys(frame, name):
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing key columns: {missing}")

    if frame[KEYS].isna().any().any():
        raise ValueError(f"{name}: missing order keys.")

    if frame.duplicated(subset=KEYS).any():
        raise ValueError(f"{name}: duplicate order keys.")


def make_reference_splits(metadata):
    """Create reference membership from historical base data only."""
    _check_keys(metadata, "Historical metadata")

    development_index, test_index = chronological_split(
        metadata,
        evaluation_fraction=0.20,
    )
    train_index, validation_index = chronological_split(
        metadata.loc[development_index],
        evaluation_fraction=0.25,
    )

    reference = metadata[KEYS].copy()
    reference["split"] = "train"
    reference.loc[validation_index, "split"] = "validation"
    reference.loc[test_index, "split"] = "test"

    return reference.reset_index(drop=True)


def apply_reference_splits(metadata, reference):
    """Keep reference membership and assign unseen order keys to training."""
    _check_keys(metadata, "Current metadata")
    _check_keys(reference, "Reference splits")

    if not metadata.index.is_unique:
        raise ValueError("Current metadata must have unique row indexes.")

    if "split" not in reference.columns:
        raise ValueError("Reference splits: missing split column.")

    if not reference["split"].isin(SPLIT_NAMES).all():
        raise ValueError("Reference splits contain invalid split names.")

    if set(reference["split"]) != set(SPLIT_NAMES):
        raise ValueError("Reference must contain train, validation and test.")

    membership = metadata[KEYS].merge(
        reference[KEYS + ["split"]],
        on=KEYS,
        how="left",
        sort=False,
    )
    # Unique keys on both sides make matched-row count a complete coverage check.
    if int(membership["split"].notna().sum()) != len(reference):
        raise ValueError("Historical reference orders are missing from current data.")

    assignments = membership["split"].fillna("train").to_numpy()

    return {
        name: metadata.index[assignments == name]
        for name in SPLIT_NAMES
    }
