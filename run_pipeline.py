"""Automated batch-processing workflow for PBI 10."""

import argparse
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd

from download_batch import download_batch
from src.data_cleaning import (
    clean_customers_data,
    clean_orders_data,
    clean_labels_data,
)
from src.features import (
    chronological_split,
    join_training_data,
    prepare_training_data,
)
from src.pipeline_training import train_available_models
from src.pipeline_data import reconcile_local_history


def prepare_incoming_batch(
    customers_url,
    orders_url,
    labels_url,
    output_dir,
):
    """Download, validate and clean the three incoming datasets."""
    paths = download_batch(
        customers_url,
        orders_url,
        labels_url,
        output_dir,
    )

    cleaners = {
        "customers": clean_customers_data,
        "orders": clean_orders_data,
        "labels": clean_labels_data,
    }

    cleaned = {}
    issue_reports = {}
    summary = []

    for name, cleaner in cleaners.items():
        print(f"Cleaning {name}...", flush=True)

        raw = pd.read_csv(paths[name], dtype="string")
        clean, issues = cleaner(raw)

        cleaned[name] = clean
        issue_reports[name] = issues

        summary.append({
            "dataset": name,
            "input_rows": len(raw),
            "accepted_rows": len(clean),
            "rows_not_accepted": len(raw) - len(clean),
            "issue_records": len(issues),
        })

        del raw

    return cleaned, issue_reports, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an incoming batch and optionally train locally."
    )
    parser.add_argument("--customers-url", required=True)
    parser.add_argument("--orders-url", required=True)
    parser.add_argument("--labels-url", required=True)
    parser.add_argument(
        "--train-local",
        action="store_true",
        help="Train available models and evaluate on validation data only.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Explicit classification threshold for local training.",
    )
    parser.add_argument(
        "--rf-jobs",
        type=int,
        default=4,
        help="Random Forest worker count; -1 uses all available CPUs.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        help="Folder containing historical base CSVs for local rehearsal.",
    )
    args = parser.parse_args()

    if args.train_local:
        if args.threshold is None or not 0 <= args.threshold <= 1:
            parser.error("--train-local requires --threshold between 0 and 1.")

        if args.rf_jobs == 0 or args.rf_jobs < -1:
            parser.error("--rf-jobs must be -1 or a positive integer.")

    project_root = Path(__file__).resolve().parent
    batch_id = f"batch-{uuid4().hex[:12]}"

    download_dir = project_root / "data" / "raw" / "incoming" / batch_id
    report_dir = project_root / "reports" / "pipeline" / batch_id
    report_dir.mkdir(parents=True, exist_ok=True)

    stage = "local preparation"
    batch_start = perf_counter()
    print(f"Batch ID: {batch_id}", flush=True)

    try:
        cleaned, issue_reports, summary = prepare_incoming_batch(
            args.customers_url,
            args.orders_url,
            args.labels_url,
            download_dir,
        )

        if args.history_dir is not None:
            stage = "local history reconciliation"

            cleaned, reconciliation_summary = reconcile_local_history(
                cleaned,
                args.history_dir,
                report_dir,
            )

            print(reconciliation_summary.to_string(index=False))

        stage = "joining and feature preparation"

        joined, join_audit = join_training_data(
            cleaned["customers"],
            cleaned["orders"],
            cleaned["labels"],
        )
        X, y, metadata = prepare_training_data(joined)

        pd.Series(join_audit, name="rows").to_csv(
            report_dir / "join_audit.csv"
        )
        summary.to_csv(
            report_dir / "cleaning_summary.csv",
            index=False,
        )

        for name, issues in issue_reports.items():
            issues.to_csv(
                report_dir / f"{name}_issues.csv",
                index=False,
            )

        print(summary.to_string(index=False))
        print(f"Eligible labelled orders: {len(X):,}")
        print(f"Predictor columns: {X.shape[1]}")

        if args.train_local:
            stage = "local model training"

            development_index, test_index = chronological_split(
                metadata,
                evaluation_fraction=0.20,
            )
            train_index, validation_index = chronological_split(
                metadata.loc[development_index],
                evaluation_fraction=0.25,
            )

            splits = {
                "train": train_index,
                "validation": validation_index,
                "test": test_index,
            }

            split_summary = pd.DataFrame([
                {
                    "split": name,
                    "rows": len(rows),
                    "first_date": metadata.loc[
                        rows, "order_date"
                    ].min().date(),
                    "last_date": metadata.loc[
                        rows, "order_date"
                    ].max().date(),
                }
                for name, rows in splits.items()
            ])
            split_summary.to_csv(
                report_dir / "split_summary.csv",
                index=False,
            )
            print(split_summary.to_string(index=False))

            models, model_results = train_available_models(
                X.loc[train_index],
                y.loc[train_index],
                X.loc[validation_index],
                y.loc[validation_index],
                report_dir / "models",
                threshold=args.threshold,
                rf_n_jobs=args.rf_jobs,
            )

            print(model_results.to_string(index=False))

    except Exception as exc:
        message = (
            f"Batch: {batch_id}\n"
            f"Stage: {stage}\n"
            f"Error: {type(exc).__name__}: {exc}\n"
        )
        (report_dir / "failure.txt").write_text(
            message,
            encoding="utf-8",
        )
        parser.exit(1, message)

    elapsed = perf_counter() - batch_start
    print(f"Local preparation completed for {batch_id}.")

    if args.train_local:
        print("Available-model training and validation completed.")
        print("The third member's model is not integrated yet.")
    else:
        print("Model training was not requested.")

    print(f"Reports saved to: {report_dir}")
    print(f"Total local execution time: {elapsed:.2f} seconds")
    print("RDS loading and MLflow logging have not run.")


if __name__ == "__main__":
    main()