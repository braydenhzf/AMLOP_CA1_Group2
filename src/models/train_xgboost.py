
"""Train and evaluate Isabelle's PBI 7 XGBoost fraud models."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from src.features import (
    build_preprocessor,
    chronological_split,
)

from src.models.train_random_forest import (
    load_historical_data,
    decision_metrics,
    split_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_xgboost(features, labels, version="v1", n_jobs=-1):
    """Fit preprocessing and XGBoost using prepared training data."""

    if version not in ("v1", "v2"):
        raise ValueError("Version must be v1 or v2.")

    if labels.nunique() != 2 or not labels.isin([0, 1]).all():
        raise ValueError(
            "Training data must contain valid binary fraud labels."
        )

    # V1: Baseline without class weighting.
    # V2: Address class imbalance using training labels only.
    if version == "v1":
        scale_pos_weight = 1.0
    else:
        negative_count = int((labels == 0).sum())
        positive_count = int((labels == 1).sum())

        scale_pos_weight = negative_count / positive_count

    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=n_jobs,
    )

    preprocessor = build_preprocessor()

    # Preserve numerical zeros for XGBoost.
    preprocessor.set_params(sparse_threshold=0.0)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return model.fit(features, labels)


def log_run(
    model,
    report,
    sample_features,
    sample_probabilities,
    run_name,
):
    """Log the fitted pipeline and evaluation results to MLflow."""

    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    started = perf_counter()

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.set_tags({
            "author": "Isabelle",
            "pbi": "7",
            "algorithm": "XGBoost",
        })

        mlflow.log_params({
            **report["parameters"],
            "model_type": "XGBClassifier",
            "model_version": report["model_version"],
            "scored_split": report["scored_split"],
            "threshold": report["threshold"],
            "train_start": report["train"]["start"],
            "train_end": report["train"]["end"],
            "train_rows": report["train"]["rows"],
            "scored_start": report["scored"]["start"],
            "scored_end": report["scored"]["end"],
            "scored_rows": report["scored"]["rows"],
        })

        mlflow.log_metrics(report["metrics"])

        print("Uploading model to AWS MLflow...", flush=True)

        upload_started = perf_counter()

        logged = mlflow.sklearn.log_model(
            model,
            name="xgboost",
            serialization_format="cloudpickle",
            signature=infer_signature(
                sample_features,
                sample_probabilities,
            ),
            pyfunc_predict_fn="predict_proba",
            metadata={
                "classification_threshold": report["threshold"],
                "fraud_class": 1,
                "model_version": report["model_version"],
            },
        )

        upload_seconds = round(
            perf_counter() - upload_started, 2
        )

        logging_seconds = round(
            perf_counter() - started, 2
        )

        report["metrics"]["model_logging_seconds"] = (
            upload_seconds
        )

        report["metrics"]["mlflow_logging_seconds"] = (
            logging_seconds
        )

        mlflow.log_metrics({
            "model_logging_seconds": upload_seconds,
            "mlflow_logging_seconds": logging_seconds,
        })

        mlflow.log_dict(report, "evaluation.json")

        return {
            "run_id": run.info.run_id,
            "model_uri": logged.model_uri,
            "model_logging_seconds": upload_seconds,
            "mlflow_logging_seconds": logging_seconds,
        }


def run_training(args):
    """Load historical data, train and evaluate XGBoost."""

    if args.log_mlflow:
        from config import configure_mlflow

        configure_mlflow()

    print(
        "Loading and preparing historical data...",
        flush=True,
    )

    started = perf_counter()

    features, labels, metadata, audit = load_historical_data(
        args.data_dir
    )

    preparation_seconds = perf_counter() - started

    print("\nData preparation completed.")
    print(f"Eligible orders: {len(features):,}")
    print(f"Features: {features.shape[1]}")
    print(f"Fraud orders: {int(labels.sum()):,}")
    print(f"Fraud rate: {labels.mean():.2%}")

    # Reserve the newest 20% for final evaluation.
    development_rows, evaluation_rows = chronological_split(
        metadata
    )

    if args.split == "validation":

        # Split historical development data for model iteration.
        train_rows, scored_rows = chronological_split(
            metadata.loc[development_rows]
        )

    else:

        # Train on the complete development period.
        # Evaluate on the untouched final evaluation period.
        train_rows = development_rows
        scored_rows = evaluation_rows

    X_train = features.loc[train_rows]
    y_train = labels.loc[train_rows]

    X_scored = features.loc[scored_rows]
    y_scored = labels.loc[scored_rows]

    if y_train.nunique() != 2:
        raise ValueError(
            "Training data must contain both fraud and non-fraud orders."
        )

    if y_scored.nunique() != 2:
        raise ValueError(
            "Scored data must contain both fraud and non-fraud orders."
        )

    train_summary = split_summary(
        metadata, labels, train_rows
    )

    scored_summary = split_summary(
        metadata, labels, scored_rows
    )

    print("\nTraining split:")
    print(json.dumps(train_summary, indent=2))

    print(f"\n{args.split.capitalize()} split:")
    print(json.dumps(scored_summary, indent=2))

    print(
        f"\nTraining XGBoost {args.version.upper()}...",
        flush=True,
    )

    started = perf_counter()

    model = train_xgboost(
        X_train,
        y_train,
        version=args.version,
        n_jobs=args.n_jobs,
    )

    fit_seconds = perf_counter() - started

    classifier = model.named_steps["classifier"]

    print("Training completed.", flush=True)
    print(f"Training time: {fit_seconds:.2f} seconds")
    print(
        f"Scale positive weight: "
        f"{classifier.get_params()['scale_pos_weight']:.4f}"
    )

    # Score the selected held-out dataset.
    started = perf_counter()

    probabilities = model.predict_proba(X_scored)

    fraud_scores = probabilities[:, 1]

    predictions = (
        fraud_scores >= args.threshold
    ).astype(int)

    metrics = decision_metrics(
        y_scored,
        fraud_scores,
        args.threshold,
    )

    metrics["accuracy"] = float(
        accuracy_score(y_scored, predictions)
    )

    metrics["roc_auc"] = float(
        roc_auc_score(y_scored, fraud_scores)
    )

    metrics["fit_seconds"] = round(fit_seconds, 2)

    metrics["evaluation_seconds"] = round(
        perf_counter() - started, 2
    )

    metrics["preparation_seconds"] = round(
        preparation_seconds, 2
    )

    parameter_names = [
        "objective",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "scale_pos_weight",
        "subsample",
        "colsample_bytree",
        "tree_method",
        "eval_metric",
        "random_state",
        "n_jobs",
    ]

    classifier_params = classifier.get_params()

    parameters = {
        name: classifier_params[name]
        for name in parameter_names
    }

    report = {
        "model_type": "XGBClassifier",
        "model_version": args.version,
        "threshold": args.threshold,
        "scored_split": args.split,
        "train": train_summary,
        "scored": scored_summary,
        "metrics": metrics,
        "parameters": parameters,
        "features": features.columns.tolist(),
        "join_audit": audit,
    }

    if args.split == "validation":
        report["reserved_evaluation"] = split_summary(
            metadata, labels, evaluation_rows
        )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_prefix = f"{args.split}-{args.version}"

    model_path = (
        args.report_dir / f"{file_prefix}.joblib"
    )

    dump(model, model_path, compress=3)

    metrics["model_size_mb"] = round(
        model_path.stat().st_size / (1024 ** 2), 3
    )

    report["model_file"] = model_path.name

    report_path = (
        args.report_dir / f"{file_prefix}.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n========== XGBOOST "
        f"{args.version.upper()} "
        f"{args.split.upper()} RESULTS =========="
    )

    print(json.dumps(metrics, indent=2))

    print("\nConfusion Matrix:")
    print([
        [
            metrics["true_negatives"],
            metrics["false_positives"],
        ],
        [
            metrics["false_negatives"],
            metrics["true_positives"],
        ],
    ])

    print(f"\nModel saved to: {model_path}")
    print(f"Report saved to: {report_path}")

    # Optional MLflow logging.
    if args.log_mlflow:

        import mlflow

        mlflow.set_experiment(args.experiment)

        run_name = (
            args.run_name
            or f"isabelle-xgboost-{args.version}-{args.split}"
        )

        report["mlflow"] = log_run(
            model,
            report,
            X_scored.iloc[:5],
            probabilities[:5],
            run_name,
        )

        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\nMLflow logging completed successfully.")

        print(
            f"MLflow run ID: "
            f"{report['mlflow']['run_id']}"
        )

        print(
            f"Model URI: "
            f"{report['mlflow']['model_uri']}"
        )

        print(
            f"Model logging time: "
            f"{report['mlflow']['model_logging_seconds']} seconds"
        )

        print(
            f"Total MLflow logging time: "
            f"{report['mlflow']['mlflow_logging_seconds']} seconds"
        )

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        default="v1",
        help="XGBoost model version.",
    )

    parser.add_argument(
        "--split",
        choices=("validation", "evaluation"),
        default="validation",
        help="Dataset used for scoring.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Fraud review threshold.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            os.environ.get("SPBUY_DATA_DIR")
            or PROJECT_ROOT / "data" / "raw"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "pbi7-xgboost",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for XGBoost.",
    )

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log to the shared AWS MLflow service.",
    )

    parser.add_argument(
        "--experiment",
        help="Name of the shared MLflow experiment.",
    )

    parser.add_argument(
        "--run-name",
        help="Custom MLflow run name.",
    )

    args = parser.parse_args(argv)

    if not 0 <= args.threshold <= 1:
        parser.error(
            "--threshold must be between 0 and 1."
        )

    if args.n_jobs == 0:
        parser.error("--n-jobs must be nonzero.")

    if args.log_mlflow and not args.experiment:
        parser.error(
            "--log-mlflow requires --experiment."
        )

    run_training(args)


if __name__ == "__main__":
    main()
