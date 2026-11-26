"""Train and evaluate the PBI 7 Random Forest versions."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data_cleaning import clean_customers_data, clean_labels_data, clean_orders_data
from src.features import (
    build_preprocessor,
    chronological_split,
    join_training_data,
    prepare_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANERS = {
    "customers": clean_customers_data,
    "orders": clean_orders_data,
    "labels": clean_labels_data,
}
CLASS_WEIGHTS = {"v1": None, "v2": "balanced_subsample"}


def load_historical_data(data_dir):
    """Apply the shared cleaners and prepare eligible labelled orders."""
    cleaned = {}
    for name, clean in CLEANERS.items():
        file = data_dir / f"{name}_base.csv"
        cleaned[name], _ = clean(pd.read_csv(file, dtype="string"))

    joined, audit = join_training_data(
        cleaned["customers"], cleaned["orders"], cleaned["labels"]
    )
    features, labels, metadata = prepare_training_data(joined)
    return features, labels, metadata, audit


def train_random_forest(features, labels, version, n_jobs=-1):
    """Fit a pipeline on already prepared training rows."""
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=n_jobs,
        class_weight=CLASS_WEIGHTS[version],
    )
    model = Pipeline(
        [("preprocessor", build_preprocessor()), ("classifier", classifier)]
    )
    return model.fit(features, labels)


def decision_metrics(labels, fraud_scores, threshold):
    flagged = fraud_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, flagged, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tn + tp) / (tn + fp + fn + tp)),
        "average_precision": float(average_precision_score(labels, fraud_scores)),
        "fraud_precision": float(precision_score(labels, flagged, zero_division=0)),
        "fraud_recall": float(recall_score(labels, flagged, zero_division=0)),
        "fraud_f1": float(f1_score(labels, flagged, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def split_summary(metadata, labels, rows):
    dates = metadata.loc[rows, "order_date"]
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "rows": len(rows),
        "fraud_rows": int(labels.loc[rows].sum()),
    }


def log_run(model, report, sample_features, sample_probabilities, run_name):
    """Log one fitted model to the configured MLflow experiment."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    threshold = report["threshold"]
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                **report["parameters"],
                "model_type": "RandomForestClassifier",
                "model_version": report["model_version"],
                "scored_split": report["scored_split"],
                "threshold": threshold,
                "train_start": report["train"]["start"],
                "train_end": report["train"]["end"],
                "train_rows": report["train"]["rows"],
                "scored_start": report["scored"]["start"],
                "scored_end": report["scored"]["end"],
                "scored_rows": report["scored"]["rows"],
            }
        )
        mlflow.log_metrics(report["metrics"])
        mlflow.log_dict(report, "evaluation.json")
        # The shared preprocessor contains a NumPy log transform.
        logged = mlflow.sklearn.log_model(
            model,
            name="random-forest",
            serialization_format="cloudpickle",
            signature=infer_signature(sample_features, sample_probabilities),
            pyfunc_predict_fn="predict_proba",
            metadata={"classification_threshold": threshold, "fraud_class": 1},
        )
        return {"run_id": run.info.run_id, "model_uri": logged.model_uri}


def run_training(args):
    if args.log_mlflow:
        from config import configure_mlflow

        configure_mlflow()

    features, labels, metadata, audit = load_historical_data(args.data_dir)
    train_rows, scored_rows = chronological_split(metadata)
    if args.split == "validation":
        # Reserve the latest portion of the training period for model iteration.
        train_rows, scored_rows = chronological_split(metadata.loc[train_rows])

    for name, rows in (("Training", train_rows), ("Scored", scored_rows)):
        if labels.loc[rows].nunique() != 2:
            raise ValueError(f"{name} data must contain both fraud and non-fraud orders.")

    train_summary = split_summary(metadata, labels, train_rows)
    scored_summary = split_summary(metadata, labels, scored_rows)
    print(
        f"Fitting {args.version.upper()} on {train_summary['rows']:,} orders. "
        f"{args.split.capitalize()} has {scored_summary['rows']:,} orders.",
        flush=True,
    )
    started = perf_counter()
    model = train_random_forest(
        features.loc[train_rows], labels.loc[train_rows], args.version, args.n_jobs
    )
    fit_seconds = perf_counter() - started
    scored_features = features.loc[scored_rows]
    probabilities = model.predict_proba(scored_features)
    metrics = decision_metrics(labels.loc[scored_rows], probabilities[:, 1], args.threshold)
    metrics["fit_seconds"] = round(fit_seconds, 2)
    report = {
        "model_version": args.version,
        "threshold": args.threshold,
        "scored_split": args.split,
        "train": train_summary,
        "scored": scored_summary,
        "metrics": metrics,
        "parameters": model.named_steps["classifier"].get_params(),
        "features": features.columns.tolist(),
        "join_audit": audit,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{args.split}-{args.version}.json"
    if args.split == "evaluation":
        model_path = report_path.with_suffix(".joblib")
        dump(model, model_path, compress=3)
        report["model_file"] = model_path.name
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Report saved to {report_path}", flush=True)

    if args.log_mlflow:
        import mlflow

        mlflow.set_experiment(args.experiment)
        run_name = args.run_name or f"random-forest-{args.version}-{args.split}"
        report["mlflow"] = log_run(
            model, report, scored_features.iloc[:5], probabilities[:5], run_name
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"MLflow run ID {report['mlflow']['run_id']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=CLASS_WEIGHTS, required=True)
    parser.add_argument("--split", choices=("validation", "evaluation"), default="validation")
    parser.add_argument("--threshold", type=float, required=True, help="Fraud review threshold.")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.environ.get("SPBUY_DATA_DIR") or PROJECT_ROOT / "data" / "raw"),
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "pbi7-random-forest"
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for the forest.")
    parser.add_argument("--log-mlflow", action="store_true", help="Log to the team MLflow service.")
    parser.add_argument("--experiment", help="Name of the shared MLflow experiment.")
    parser.add_argument("--run-name", help="Run name to use in MLflow.")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.n_jobs == 0:
        parser.error("--n-jobs must be nonzero.")
    if args.log_mlflow and not args.experiment:
        parser.error("--log-mlflow requires --experiment.")
    run_training(args)


if __name__ == "__main__":
    main()
