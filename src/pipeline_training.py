"""Local model-training integration for PBI 10."""

from time import perf_counter

import pandas as pd
from joblib import dump

from src.models.brayden_logistic_regression import train_logistic_regression
from src.models.train_random_forest import (
    decision_metrics,
    train_random_forest,
)


def train_available_models(
    X_train,
    y_train,
    X_validation,
    y_validation,
    report_dir,
    threshold,
    rf_n_jobs=4,
):
    """Train available models and score validation data only."""
    if rf_n_jobs == 0 or rf_n_jobs < -1:
        raise ValueError("rf_n_jobs must be -1 or a positive integer.")

    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training data must contain both fraud classes.")

    if set(y_validation.unique()) != {0, 1}:
        raise ValueError("Validation data must contain both fraud classes.")

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
    ]

    models = {}
    results = []

    for model_name, fit_model in training_tasks:
        print(f"Training {model_name}...", flush=True)

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

        models[model_name] = model
        results.append({
            "model": model_name,
            "threshold": threshold,
            "training_seconds": training_seconds,
            **metrics,
        })

        dump(model, report_dir / f"{model_name}.joblib")

        # Preserve completed results if a later model fails.
        pd.DataFrame(results).to_csv(
            report_dir / "model_results.csv",
            index=False,
        )

        print(
            f"{model_name} fitted in {training_seconds:.2f} seconds; "
            f"validation AP={metrics['average_precision']:.4f}; "
            f"accuracy={metrics['accuracy']:.2%}",
            flush=True,
        )

    return models, pd.DataFrame(results)