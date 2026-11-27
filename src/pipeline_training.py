"""Train available models and optionally record their MLflow runs."""

from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import pandas as pd
from joblib import dump

from src.models.brayden_logistic_regression import train_logistic_regression
from src.models.train_random_forest import decision_metrics, train_random_forest
from src.pipeline_tracking import log_model_outputs, tracked_model_run
from src.models.train_xgboost import train_xgboost


def train_available_models(
    X_train,
    y_train,
    X_validation,
    y_validation,
    report_dir,
    threshold,
    rf_n_jobs=4,
    *,
    xgb_n_jobs=4,
    tracking_client=None,
    experiment_id=None,
    batch_id=None,
):
    """Train on training rows and score validation rows only."""
    if rf_n_jobs == 0 or rf_n_jobs < -1:
        raise ValueError("rf_n_jobs must be -1 or a positive integer.")

    if xgb_n_jobs == 0 or xgb_n_jobs < -1:
        raise ValueError("xgb_n_jobs must be -1 or a positive integer.")

    if not 0 <= threshold <= 1:
        raise ValueError(
            "The classification threshold must be between 0 and 1."
        )

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training data must contain both fraud classes.")

    if set(y_validation.unique()) != {0, 1}:
        raise ValueError("Validation data must contain both fraud classes.")

    if (tracking_client is None) != (experiment_id is None):
        raise ValueError(
            "Supply both tracking_client and experiment_id."
        )

    if tracking_client is not None and not batch_id:
        raise ValueError("MLflow logging requires a batch_id.")

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    training_tasks = [
        (
            "brayden_lr_v1",
            lambda: train_logistic_regression(
                X_train,
                y_train,
                class_weight=None,
            )[0],
        ),
        (
            "brayden_lr_v2",
            lambda: train_logistic_regression(
                X_train,
                y_train,
                class_weight="balanced",
            )[0],
        ),
        (
            "vivian_rf_v1",
            lambda: train_random_forest(
                X_train,
                y_train,
                version="v1",
                n_jobs=rf_n_jobs,
            ),
        ),
        (
            "vivian_rf_v2",
            lambda: train_random_forest(
                X_train,
                y_train,
                version="v2",
                n_jobs=rf_n_jobs,
            ),
        ),
        (
            "isabelle_xgb_v1",
            lambda: train_xgboost(
                X_train,
                y_train,
                version="v1",
                n_jobs=xgb_n_jobs,
            ),
        ),
        (
            "isabelle_xgb_v2",
            lambda: train_xgboost(
                X_train,
                y_train,
                version="v2",
                n_jobs=xgb_n_jobs,
            ),
        ),
    ]

    models = {}
    results = []

    for model_name, fit_model in training_tasks:
        print(f"Training {model_name}...", flush=True)

        run_context = (
            tracked_model_run(
                tracking_client,
                experiment_id,
                model_name,
                batch_id,
            )
            if tracking_client is not None
            else nullcontext(None)
        )

        with run_context as run_id:
            start = perf_counter()
            model = fit_model()
            training_seconds = perf_counter() - start

            fraud_column = list(model.classes_).index(1)
            fraud_scores = model.predict_proba(
                X_validation
            )[:, fraud_column]

            metrics = decision_metrics(
                y_validation,
                fraud_scores,
                threshold,
            )

            metrics["accuracy"] = (
                metrics["true_negatives"] + metrics["true_positives"]
            ) / len(y_validation)

            model_path = report_dir / f"{model_name}.joblib"
            dump(model, model_path)

            result = {
                "model": model_name,
                "threshold": threshold,
                "training_seconds": training_seconds,
                **metrics,
            }

            if run_id is not None:
                result["mlflow_run_id"] = run_id

            models[model_name] = model
            results.append(result)

            # Keep local results even if uploading to MLflow later fails.
            pd.DataFrame(results).to_csv(
                report_dir / "model_results.csv",
                index=False,
            )

            if run_id is not None:
                classifier = model.named_steps["classifier"]

                parameters = {
                    **classifier.get_params(deep=False),
                    "model_type": type(classifier).__name__,
                    "model_name": model_name,
                    "threshold": threshold,
                    "training_rows": len(y_train),
                    "validation_rows": len(y_validation),
                    "feature_count": X_train.shape[1],
                    "scored_split": "validation",
                }

                log_model_outputs(
                    tracking_client,
                    run_id,
                    parameters=parameters,
                    metrics={
                        **metrics,
                        "training_seconds": training_seconds,
                    },
                    model_path=model_path,
                )

        print(
            f"{model_name} fitted in {training_seconds:.2f} seconds; "
            f"validation AP={metrics['average_precision']:.4f}; "
            f"accuracy={metrics['accuracy']:.2%}",
            flush=True,
        )

        if run_id is not None:
            print(f"MLflow run recorded: {run_id}", flush=True)

    return models, pd.DataFrame(results)