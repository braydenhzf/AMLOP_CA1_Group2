"""Fit all six model versions with bounded concurrency and shared logging."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, nullcontext
from pathlib import Path
from time import perf_counter

import pandas as pd
from joblib import dump

from src.models.brayden_logistic_regression import train_logistic_regression
from src.models.train_random_forest import decision_metrics, train_random_forest
from src.models.train_xgboost import train_xgboost
from src.pipeline_tracking import log_model_outputs, tracked_model_run


def _fit_timed(name, fit_model):
    print(f"Training {name}...", flush=True)
    start = perf_counter()
    model = fit_model()
    return model, perf_counter() - start, start


def train_available_models(
    X_train, y_train, X_validation, y_validation, report_dir, threshold,
    rf_n_jobs=4, *, xgb_n_jobs=4, model_workers=1,
    tracking_client=None, experiment_id=None, batch_id=None,
):
    """Fit training rows only; score fixed validation rows; preserve settings.

    Workers only fit models. MLflow access and report writes stay on the
    calling thread. Already completed results are saved if a later task fails.
    """
    for name, value in (("rf_n_jobs", rf_n_jobs), ("xgb_n_jobs", xgb_n_jobs)):
        if type(value) is not int or value == 0 or value < -1:
            raise ValueError(f"{name} must be -1 or a positive integer.")
    if type(model_workers) is not int or not 1 <= model_workers <= 6:
        raise ValueError("model_workers must be an integer between 1 and 6.")
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")
    if set(y_train.unique()) != {0, 1} or set(y_validation.unique()) != {0, 1}:
        raise ValueError("Training and validation data must each contain both fraud classes.")
    if (tracking_client is None) != (experiment_id is None):
        raise ValueError("Supply both tracking_client and experiment_id.")
    if tracking_client is not None and not batch_id:
        raise ValueError("MLflow logging requires a batch_id.")

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("brayden_lr_v1", lambda: train_logistic_regression(X_train, y_train, class_weight=None)[0]),
        ("brayden_lr_v2", lambda: train_logistic_regression(X_train, y_train, class_weight="balanced")[0]),
        ("vivian_rf_v1", lambda: train_random_forest(X_train, y_train, version="v1", n_jobs=rf_n_jobs)),
        ("vivian_rf_v2", lambda: train_random_forest(X_train, y_train, version="v2", n_jobs=rf_n_jobs)),
        ("isabelle_xgb_v1", lambda: train_xgboost(X_train, y_train, version="v1", n_jobs=xgb_n_jobs)),
        ("isabelle_xgb_v2", lambda: train_xgboost(X_train, y_train, version="v2", n_jobs=xgb_n_jobs)),
    ]
    ranks = {name: i for i, (name, _) in enumerate(tasks)}
    models, results, pending = {}, [], {}

    def save_results():
        frame = pd.DataFrame(results)
        if not frame.empty:
            frame = frame.sort_values("model", key=lambda values: values.map(ranks)).reset_index(drop=True)
        frame.to_csv(report_dir / "model_results.csv", index=False)
        return frame

    executor = ThreadPoolExecutor(max_workers=model_workers)
    try:
        for name, fit_model in tasks:
            stack = ExitStack()
            try:
                context = (
                    tracked_model_run(tracking_client, experiment_id, name, batch_id)
                    if tracking_client is not None else nullcontext(None)
                )
                run_id = stack.enter_context(context)
                future = executor.submit(_fit_timed, name, fit_model)
            except BaseException:
                stack.__exit__(*sys.exc_info())
                raise
            pending[future] = (name, stack, run_id)

        for future in as_completed(list(pending)):
            name, stack, run_id = pending.pop(future)
            with stack:
                model, training_seconds, fit_start = future.result()
                started = perf_counter()
                fraud_column = list(model.classes_).index(1)
                scores = model.predict_proba(X_validation)[:, fraud_column]
                metrics = decision_metrics(y_validation, scores, threshold)
                metrics["accuracy"] = (
                    metrics["true_negatives"] + metrics["true_positives"]
                ) / len(y_validation)
                evaluation_seconds = perf_counter() - started

                started = perf_counter()
                model_path = report_dir / f"{name}.joblib"
                dump(model, model_path, compress=("lzma", 3))
                save_seconds = perf_counter() - started
                result = {
                    "model": name, "threshold": threshold,
                    "training_seconds": training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "save_seconds": save_seconds,
                    "model_bytes": model_path.stat().st_size,
                    **metrics,
                }
                if run_id is not None:
                    result["mlflow_run_id"] = run_id
                models[name] = model
                results.append(result)
                save_results()

                started = perf_counter()
                if run_id is not None:
                    classifier = model.named_steps["classifier"]
                    parameters = {
                        **classifier.get_params(deep=False),
                        "model_type": type(classifier).__name__,
                        "model_name": name, "threshold": threshold,
                        "training_rows": len(y_train), "validation_rows": len(y_validation),
                        "feature_count": X_train.shape[1], "scored_split": "validation",
                        "model_workers": model_workers, "artifact_compression": "lzma:3",
                    }
                    log_model_outputs(
                        tracking_client, run_id, parameters=parameters,
                        metrics={**metrics, "training_seconds": training_seconds,
                                 "evaluation_seconds": evaluation_seconds, "save_seconds": save_seconds},
                        model_path=model_path,
                    )
                result["logging_seconds"] = perf_counter() - started
                result["total_model_seconds"] = perf_counter() - fit_start
                save_results()
            print(
                f"{name} fitted in {training_seconds:.2f}s; "
                f"validation AP={metrics['average_precision']:.4f}; "
                f"accuracy={metrics['accuracy']:.2%}; "
                f"artifact={result['model_bytes']/1024/1024:.2f} MiB",
                flush=True,
            )
            if run_id is not None:
                print(f"MLflow run recorded: {run_id}", flush=True)
    except BaseException:
        error = sys.exc_info()
        for future, (_, stack, _) in pending.items():
            future.cancel()
            try:
                stack.__exit__(*error)
            except BaseException:
                pass  # Preserve the original failure when closing other runs.
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return models, save_results()
