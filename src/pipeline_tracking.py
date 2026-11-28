"""MLflow experiment logging for the automated pipeline."""

from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from time import time

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient


def create_tracking_client(
    tracking_uri,
    experiment_name="group2-pbi10",
    artifact_location=None,
):
    """Find or create an experiment in the supplied tracking store."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    else:
        if experiment.lifecycle_stage != "active":
            raise ValueError("The selected MLflow experiment is deleted.")

        experiment_id = experiment.experiment_id

    return client, experiment_id


@contextmanager
def tracked_model_run(client, experiment_id, model_name, batch_id):
    """Record whether the work inside this run succeeds or fails."""
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": f"{batch_id}-{model_name}",
            "batch_id": str(batch_id),
            "model_name": str(model_name),
            "pbi": "10",
        },
    )
    run_id = run.info.run_id

    try:
        yield run_id
    except BaseException:
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception:
            # Keep the original error if tracking also fails.
            pass
        raise
    else:
        client.set_terminated(run_id, status="FINISHED")


def log_model_outputs(
    client,
    run_id,
    *,
    parameters,
    metrics,
    model_path,
):
    """Record settings, numeric metrics and an existing model file."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    numeric_metrics = {}

    for name, value in metrics.items():
        number = float(value)

        if not isfinite(number):
            raise ValueError(f"MLflow metric must be finite: {name}")

        numeric_metrics[str(name)] = number

    timestamp = int(time() * 1000)

    client.log_batch(
        run_id,
        params=[
            Param(str(name), str(value))
            for name, value in parameters.items()
        ],
        metrics=[
            Metric(name, value, timestamp, 0)
            for name, value in numeric_metrics.items()
        ],
        synchronous=True,
    )

    client.log_dict(
        run_id,
        numeric_metrics,
        "validation_metrics.json",
    )

    client.log_artifact(
        run_id,
        str(model_path),
        artifact_path="models",
    )