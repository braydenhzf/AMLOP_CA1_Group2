"""Combine cleaned historical and incoming records without silent overwrites."""

import pandas as pd
from pathlib import Path

from src.data_cleaning import (
    clean_customers_data,
    clean_orders_data,
    clean_labels_data,
)

def combine_accepted_data(history, incoming, keys):
    """Return combined records, conflicting keys and processing counts.

    Both inputs must already use the team's cleaned schema and data types.
    Conflicting incoming records do not replace historical records.
    """
    if set(history.columns) != set(incoming.columns):
        raise ValueError("Historical and incoming columns must match.")

    for name, frame in (("history", history), ("incoming", incoming)):
        if frame[keys].isna().any().any():
            raise ValueError(f"{name}: missing record keys.")
        if frame.duplicated(subset=keys).any():
            raise ValueError(f"{name}: duplicate record keys.")

    existing = history.set_index(keys)
    arriving = incoming.set_index(keys)
    arriving = arriving[existing.columns]

    already_exists = arriving.index.isin(existing.index)
    overlapping = arriving.loc[already_exists]
    previous = existing.reindex(overlapping.index)

    same_values = (
        previous.eq(overlapping)
        | (previous.isna() & overlapping.isna())
    ).fillna(False).all(axis=1)

    conflicts = overlapping.loc[~same_values].reset_index()[keys].copy()
    conflicts["reason"] = "Unverified conflict; historical record retained."

    new_records = arriving.loc[~already_exists]
    combined = pd.concat([existing, new_records]).reset_index()

    summary = {
        "historical_rows": len(history),
        "incoming_rows": len(incoming),
        "new_records": len(new_records),
        "identical_records": int(same_values.sum()),
        "conflicting_records": len(conflicts),
        "combined_rows": len(combined),
    }

    return combined, conflicts, summary

def reconcile_local_history(incoming_tables, history_dir, report_dir):
    """Combine incoming tables with locally stored historical base files."""
    history_dir = Path(history_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cleaners = {
        "customers": clean_customers_data,
        "orders": clean_orders_data,
        "labels": clean_labels_data,
    }

    # Customer IDs are globally unique in the team's cleaning rules.
    keys = {
        "customers": ["customer_id"],
        "orders": ["country_code", "order_id"],
        "labels": ["country_code", "order_id"],
    }

    combined_tables = {}
    summaries = []

    for name, cleaner in cleaners.items():
        print(f"Reconciling historical and incoming {name}...", flush=True)

        raw_history = pd.read_csv(
            history_dir / f"{name}_base.csv",
            dtype="string",
        )
        history, history_issues = cleaner(raw_history)
        del raw_history

        combined, conflicts, counts = combine_accepted_data(
            history,
            incoming_tables[name],
            keys[name],
        )

        combined_tables[name] = combined
        summaries.append({"dataset": name, **counts})

        history_issues.to_csv(
            report_dir / f"{name}_history_issues.csv",
            index=False,
        )
        conflicts.to_csv(
            report_dir / f"{name}_history_conflicts.csv",
            index=False,
        )

    reconciliation_summary = pd.DataFrame(summaries)
    reconciliation_summary.to_csv(
        report_dir / "reconciliation_summary.csv",
        index=False,
    )

    return combined_tables, reconciliation_summary