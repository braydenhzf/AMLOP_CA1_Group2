"""Prepare a reproducible offline rehearsal batch and expected outputs for PBI 12.

Rows use historical examples with synthetic IDs and later order dates. They
exercise the workflow; they are not new ground truth for model-quality claims.
Generated CSVs and receipts belong under ignored data/raw, not in Git.
"""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from src.data_cleaning import (
    CUSTOMER_COLUMNS, ORDER_COLUMNS, LABEL_COLUMNS,
    clean_customers_data, clean_orders_data, clean_labels_data,
)
from src.features import join_training_data, prepare_training_data


COLUMNS = {
    "customers": CUSTOMER_COLUMNS,
    "orders": ORDER_COLUMNS,
    "labels": LABEL_COLUMNS,
}
CLEANERS = {
    "customers": clean_customers_data,
    "orders": clean_orders_data,
    "labels": clean_labels_data,
}
MODEL_NAMES = [
    "brayden_lr_v1", "brayden_lr_v2", "vivian_rf_v1", "vivian_rf_v2",
    "isabelle_xgb_v1", "isabelle_xgb_v2",
]


def _history_and_templates(history_dir):
    history = {}
    for name, cleaner in CLEANERS.items():
        raw = pd.read_csv(history_dir / f"{name}_base.csv", dtype="string")
        history[name], _ = cleaner(raw)
        print(f"Historical {name}: {len(history[name]):,} accepted rows", flush=True)
    joined, audit = join_training_data(
        history["customers"], history["orders"], history["labels"]
    )
    eligible = joined.loc[
        joined["label_match"].eq("both")
        & joined["customer_match"].eq("both")
        & joined["is_fraud"].isin([0, 1])
        & joined["order_date"].notna()
    ].copy()
    fields = list(dict.fromkeys(column for cols in COLUMNS.values() for column in cols))
    usable = eligible[fields].notna().all(axis=1)
    usable &= eligible["first_order_datetime"].dt.normalize().le(eligible["order_date"])
    usable &= eligible["refund_value"].le(eligible["order_value"])
    for column in ("num_cancelled_orders_last_50days", "num_refund_orders_last_50days"):
        usable &= eligible[column].le(eligible["num_orders_last_50days"])
    pool = eligible.loc[usable, fields].copy()
    if set(pool["is_fraud"].unique()) != {0, 1}:
        raise ValueError("Historical templates must contain both fraud classes.")
    return history, eligible, pool, audit


def _split_expectations(path, eligible, new_orders):
    if path is None:
        return None
    reference = pd.read_csv(path, dtype="string")
    keys = ["country_code", "order_id"]
    if not set(keys + ["split"]).issubset(reference.columns):
        raise ValueError("Split manifest requires country_code, order_id and split.")
    if reference[keys + ["split"]].isna().any().any():
        raise ValueError("Split manifest has missing values.")
    if reference.duplicated(keys).any():
        raise ValueError("Split manifest has duplicate order keys.")
    if set(reference["split"]) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation and test only.")
    actual = pd.MultiIndex.from_frame(eligible[keys])
    fixed = pd.MultiIndex.from_frame(reference[keys])
    if len(actual) != len(fixed) or not actual.isin(fixed).all():
        raise ValueError("Split manifest does not match the historical eligible orders.")
    baseline = {name: int(reference["split"].eq(name).sum())
                for name in ("train", "validation", "test")}
    after = dict(baseline)
    after["train"] += new_orders
    return {"historical": baseline, "after_first_run_and_repeat": after,
            "policy": "New keys join training; historical memberships stay fixed."}


def prepare_batch(history_dir, output_dir, *, batch_id="practice-001",
                  new_orders=1000, repeat_orders=20, seed=42, split_manifest=None):
    """Write three input CSVs, expected counts and a checksum manifest atomically."""
    history_dir, output_dir = Path(history_dir), Path(output_dir)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", batch_id):
        raise ValueError("batch_id must use 1-32 lowercase letters, digits or hyphens.")
    if type(new_orders) is not int or new_orders < 2:
        raise ValueError("new_orders must be an integer of at least 2.")
    if type(repeat_orders) is not int or repeat_orders < 1:
        raise ValueError("repeat_orders must be a positive integer.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 3:
        raise ValueError("seed must be an integer between 0 and 4294967293.")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new folder: {output_dir}")

    history, eligible, pool, audit = _history_and_templates(history_dir)
    if repeat_orders > len(pool):
        raise ValueError("Not enough complete historical orders for the repeat sample.")
    fraud_count = max(1, min(new_orders - 1, round(new_orders * pool["is_fraud"].mean())))
    samples = [
        pool.loc[pool["is_fraud"].eq(label)].sample(
            n=count, replace=True, random_state=seed + label
        )
        for label, count in ((0, new_orders - fraud_count), (1, fraud_count))
    ]
    new = pd.concat(samples, ignore_index=True).sample(
        frac=1, random_state=seed + 2
    ).reset_index(drop=True)
    repeated = pool.sample(n=repeat_orders, random_state=seed).reset_index(drop=True)

    source_customers = list(new["customer_id"].drop_duplicates())
    customer_ids = {old: f"pbi12-{batch_id}-c-{i:06d}"
                    for i, old in enumerate(source_customers)}
    new["customer_id"] = new["customer_id"].map(customer_ids)
    new["order_id"] = [f"pbi12-{batch_id}-o-{i:06d}" for i in range(new_orders)]
    new["order_date"] = history["orders"]["order_date"].max().normalize() + pd.to_timedelta(
        [1 + i % 7 for i in range(new_orders)], unit="D"
    )
    if new["customer_id"].isin(history["customers"]["customer_id"]).any():
        raise ValueError("Generated customer IDs already occur in history; change batch_id.")
    old_orders = pd.MultiIndex.from_frame(history["orders"][["country_code", "order_id"]])
    new_keys = pd.MultiIndex.from_frame(new[["country_code", "order_id"]])
    if new_keys.isin(old_orders).any():
        raise ValueError("Generated order IDs already occur in history; change batch_id.")

    inputs, counts = {}, []
    for name, columns in COLUMNS.items():
        fresh = new.loc[:, list(columns)].copy()
        old = repeated.loc[:, list(columns)].copy()
        if name == "customers":
            fresh = fresh.drop_duplicates("customer_id")
            old = old.drop_duplicates("customer_id")
        inputs[name] = pd.concat([fresh, old], ignore_index=True)
        counts.append({
            "dataset": name, "input_rows": len(inputs[name]),
            "expected_accepted_rows": len(inputs[name]),
            "first_run_new_records": len(fresh),
            "first_run_identical_records": len(old),
            "first_run_conflicting_records": 0,
            "first_run_combined_rows": len(history[name]) + len(fresh),
            "repeat_run_new_records": 0,
            "repeat_run_identical_records": len(inputs[name]),
        })

    expected = {
        "batch_id": batch_id, "seed": seed, "synthetic": True,
        "purpose": "Workflow rehearsal using historical patterns, synthetic IDs and later dates.",
        "baseline_assumption": "RDS contains these accepted base records and none of the generated IDs.",
        "new_orders": new_orders, "existing_orders_resubmitted": repeat_orders,
        "new_fraud_orders": fraud_count, "new_legitimate_orders": new_orders - fraud_count,
        "historical_eligible_orders": int(audit["eligible_training_orders"]),
        "combined_eligible_orders_after_first_run": int(audit["eligible_training_orders"]) + new_orders,
        "repeat_run_additional_eligible_orders": 0,
        "expected_model_names": MODEL_NAMES, "expected_model_versions": len(MODEL_NAMES),
        "required_complete_pipeline_seconds": 180,
        "timing_scope": "Download, validation, cleaning, RDS read/write, features, training, shared logging and checks.",
        "metric_expectation": "Finite validation metrics; do not assert exact model scores.",
        "splits": _split_expectations(split_manifest, eligible, new_orders),
        "datasets": counts,
        "coverage": {
            column: sorted(str(value) for value in new[column].unique())
            for column in ("country_code", "collect_type", "payment_method")
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbi12-", dir=output_dir.parent) as folder:
        staging = Path(folder)
        hashes, cleaned = {}, {}
        for name, frame in inputs.items():
            filename = f"{name}_practice.csv"
            path = staging / filename
            frame.to_csv(path, index=False)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            cleaned[name], issues = CLEANERS[name](pd.read_csv(path, dtype="string"))
            if len(cleaned[name]) != len(frame) or len(issues):
                raise ValueError(f"Generated {name} did not pass clean-batch checks.")
        joined, _ = join_training_data(cleaned["customers"], cleaned["orders"], cleaned["labels"])
        X, y, _ = prepare_training_data(joined)
        if len(X) != new_orders + repeat_orders or set(y.unique()) != {0, 1}:
            raise ValueError("Generated batch failed relationship or class checks.")
        expected["practice_only_eligible_orders"] = len(X)
        expected["predictor_columns"] = list(X.columns)
        expected["sha256"] = hashes
        pd.DataFrame(counts).to_csv(staging / "expected_counts.csv", index=False)
        (staging / "expected_outputs.json").write_text(
            json.dumps(expected, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return expected


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--batch-id", default="practice-001")
    parser.add_argument("--new-orders", type=int, default=1000)
    parser.add_argument("--repeat-orders", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = args.output_dir or root / "data" / "raw" / "incoming" / "pbi12" / args.batch_id
    result = prepare_batch(
        args.history_dir, destination, batch_id=args.batch_id,
        new_orders=args.new_orders, repeat_orders=args.repeat_orders,
        seed=args.seed, split_manifest=args.split_manifest,
    )
    print(pd.DataFrame(result["datasets"]).to_string(index=False))
    print(f"Practice-only eligible orders: {result['practice_only_eligible_orders']:,}")
    print(f"Batch and expected outputs saved to: {destination.resolve()}")
    print("Offline preparation completed. No models were trained or cloud services contacted.")


if __name__ == "__main__":
    main()
