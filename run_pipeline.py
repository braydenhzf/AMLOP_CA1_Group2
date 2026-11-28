"""Automated batch-processing workflow for PBI 10."""

import argparse
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import os

import pandas as pd

from config import get_mlflow_tracking_uri
from src.pipeline_tracking import create_tracking_client

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
from src.pipeline_splits import apply_reference_splits
from src.pipeline_rds import reconcile_rds_history
from src.pipeline_timing import StageTimings


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
        "--xgb-jobs",
        type=int,
        default=4,
        help="XGBoost worker count; -1 uses all available CPUs.",
    )
    history_source = parser.add_mutually_exclusive_group()

    history_source.add_argument(
        "--history-dir",
        type=Path,
        help="Folder containing historical base CSVs for local rehearsal.",
    )
    history_source.add_argument(
        "--use-rds",
        action="store_true",
        help="Read RDS history and reconcile it with the incoming batch.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="CSV containing fixed historical train/validation/test membership.",
    )
    logging_options = parser.add_mutually_exclusive_group()

    logging_options.add_argument(
        "--log-mlflow-local",
        action="store_true",
        help="Save MLflow runs to a local SQLite store under reports.",
    )
    logging_options.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log runs to the shared MLflow URI configured in .env.",
    )

    parser.add_argument(
        "--write-rds",
        action="store_true",
        help="Insert only new records into RDS; requires --use-rds.",
    )

    parser.add_argument("--model-workers", type=int, default=2,
                        help="Number of model fits allowed concurrently (1-6).")
    parser.add_argument("--rds-batch-size", type=int, default=100000,
                        help="Historical rows per database query; all rows are read.")
    parser.add_argument("--rds-transfer", choices=("rows", "compressed"), default="rows",
                        help="Compressed transfer uses MySQL 8 JSON/COMPRESS SELECTs.")
    args = parser.parse_args()

    if args.write_rds and not args.use_rds:
        parser.error("--write-rds requires --use-rds.")

    if (args.log_mlflow_local or args.log_mlflow) and not args.train_local:
        parser.error("MLflow logging requires --train-local.")

    if args.train_local and args.split_manifest is None:
        parser.error("--train-local requires --split-manifest.")

    if args.rds_batch_size < 1:
        parser.error("--rds-batch-size must be positive.")
    if not 1 <= args.model_workers <= 6:
        parser.error("--model-workers must be between 1 and 6.")

    if args.train_local:
        if args.threshold is None or not 0 <= args.threshold <= 1:
            parser.error("--train-local requires --threshold between 0 and 1.")
        if args.rf_jobs == 0 or args.rf_jobs < -1:
            parser.error("--rf-jobs must be -1 or a positive integer.")

        if args.xgb_jobs == 0 or args.xgb_jobs < -1:
            parser.error("--xgb-jobs must be -1 or a positive integer.")

    project_root = Path(__file__).resolve().parent
    batch_id = f"batch-{uuid4().hex[:12]}"

    download_dir = project_root / "data" / "raw" / "incoming" / batch_id
    report_dir = project_root / "reports" / "pipeline" / batch_id
    report_dir.mkdir(parents=True, exist_ok=True)

    stage = "local preparation"
    batch_start = perf_counter()
    timings = StageTimings(report_dir, started=batch_start)
    print(f"Batch ID: {batch_id}", flush=True)

    tracking_client = None
    experiment_id = None

    try:
        if args.log_mlflow_local or args.log_mlflow:
            stage = "MLflow setup"
            artifact_location = None

            if args.log_mlflow_local:
                tracking_dir = project_root / "reports" / "mlflow_local"
                tracking_dir.mkdir(parents=True, exist_ok=True)

                tracking_uri = (
                    f"sqlite:///{(tracking_dir / 'tracking.db').as_posix()}"
                )
                artifact_location = (tracking_dir / "artifacts").as_uri()

                print(
                    "MLflow destination: local tracking store.",
                    flush=True,
                )
            else:
                os.environ.setdefault(
                    "MLFLOW_HTTP_REQUEST_TIMEOUT",
                    "10",
                )
                os.environ.setdefault(
                    "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
                    "0",
                )

                tracking_uri = get_mlflow_tracking_uri()

                print(
                    "MLflow destination: shared server.",
                    flush=True,
                )

            tracking_client, experiment_id = timings.call(
                "mlflow_setup", create_tracking_client,
                tracking_uri,
                artifact_location=artifact_location,
            )

        stage = "local preparation"

        cleaned, issue_reports, summary = timings.call(
                "incoming_preparation", prepare_incoming_batch,
            args.customers_url,
            args.orders_url,
            args.labels_url,
            download_dir,
        )

        if args.use_rds:
            stage = (
                "RDS reconciliation and insertion"
                if args.write_rds
                else "RDS history reconciliation"
            )

            cleaned, reconciliation_summary = timings.call(
                "rds_reconciliation", reconcile_rds_history,
                cleaned,
                report_dir,
                write_new=args.write_rds,
                batch_size=args.rds_batch_size,
                transfer=args.rds_transfer,
                timings=timings,
            )

            print(reconciliation_summary.to_string(index=False))

        elif args.history_dir is not None:
            stage = "local history reconciliation"

            cleaned, reconciliation_summary = timings.call(
                "local_reconciliation", reconcile_local_history,
                cleaned,
                args.history_dir,
                report_dir,
            )

            print(reconciliation_summary.to_string(index=False))

        stage = "joining and feature preparation"

        joined, join_audit = timings.call(
                "joining", join_training_data,
            cleaned["customers"],
            cleaned["orders"],
            cleaned["labels"],
        )
        X, y, metadata = timings.call("feature_preparation", prepare_training_data, joined)

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

        if args.split_manifest is not None:
            stage = "reference split assignment"

            reference = timings.call(
                "reference_manifest_read", pd.read_csv,
                args.split_manifest,
                dtype="string",
            )
            splits = timings.call("reference_split_assignment", apply_reference_splits, metadata, reference)

            train_index = splits["train"]
            validation_index = splits["validation"]
            test_index = splits["test"]

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

            if args.train_local:
                stage = "local model training"

                models, model_results = timings.call(
                "training_evaluation_and_logging", train_available_models,
                    X.loc[train_index],
                    y.loc[train_index],
                    X.loc[validation_index],
                    y.loc[validation_index],
                    report_dir / "models",
                    threshold=args.threshold,
                    rf_n_jobs=args.rf_jobs,
                    xgb_n_jobs=args.xgb_jobs,
                    model_workers=args.model_workers,
                    tracking_client=tracking_client,
                    experiment_id=experiment_id,
                    batch_id=batch_id,
                )

                print(model_results.to_string(index=False))

    except Exception as exc:
        try:
            timings.finish(success=False)
        except Exception:
            pass
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
    performance = timings.finish(
        success=True, elapsed=elapsed,
        complete_workflow=args.use_rds and args.write_rds and args.train_local and args.log_mlflow,
    )
    print(f"Local preparation completed for {batch_id}.")

    if args.train_local:
        print("Available-model training and validation completed.")
        print(f"Completed {len(model_results)} model versions.")
    else:
        print("Model training was not requested.")

    print(f"Reports saved to: {report_dir}")
    print(f"Total local execution time: {elapsed:.2f} seconds")
    print("Stage timings saved to stage_timings.csv and performance_summary.json.")
    if performance["complete_cloud_workflow"]:
        print("PBI 12 runtime requirement:", "MET" if performance["pbi12_runtime_check_passed"] else "NOT MET")
    if args.write_rds:
        print("RDS insertion step completed; see rds_insert_summary.csv.")
    else:
        print("RDS writes were not requested.")

    if args.log_mlflow_local:
        print("MLflow runs saved in the local tracking store.")
    elif args.log_mlflow:
        print("MLflow runs logged to the configured shared server.")
    else:
        print("MLflow logging was not requested.")


if __name__ == "__main__":
    main()
