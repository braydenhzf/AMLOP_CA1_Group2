"""Time an existing model upload in a separate MLflow diagnostic experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="batch-c21e7568e6dd")
    parser.add_argument("--label", default="current-connection")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model_name = "vivian_rf_v2"
    source = root / "reports/pipeline" / args.batch_id
    model_path = source / "models" / f"{model_name}.joblib"
    if not model_path.is_file():
        parser.error(f"Saved model is missing: {model_path}")
    with (source / "models/model_results.csv").open(encoding="utf-8", newline="") as stream:
        result = next(row for row in csv.DictReader(stream) if row["model"] == model_name)
    with (source / "split_summary.csv").open(encoding="utf-8", newline="") as stream:
        splits = {row["split"]: int(row["rows"]) for row in csv.DictReader(stream)}

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    import joblib
    from mlflow.entities import Metric, Param
    from time import time
    from config import get_mlflow_tracking_uri
    from src.pipeline_tracking import create_tracking_client

    model = joblib.load(model_path)
    parameters = {
        **model.named_steps["classifier"].get_params(deep=False),
        "model_type": "RandomForestClassifier", "model_name": model_name,
        "threshold": result["threshold"], "training_rows": splits["train"],
        "validation_rows": splits["validation"], "feature_count": 9,
        "scored_split": "validation", "model_workers": 1, "artifact_compression": "lzma:3",
    }
    names = ("accuracy", "average_precision", "fraud_precision", "fraud_recall", "fraud_f1",
             "true_negatives", "false_positives", "false_negatives", "true_positives",
             "training_seconds", "evaluation_seconds", "save_seconds")
    metrics = {name: float(result[name]) for name in names}
    probe_id = uuid4().hex[:12]
    evidence = {
        "label": args.label, "source_batch": args.batch_id, "model_name": model_name,
        "artifact_bytes": model_path.stat().st_size, "diagnostic_only": True,
        "model_was_retrained": False, "status": "STARTED", "timings": {},
    }

    def timed(name, action):
        started = perf_counter()
        try:
            return action()
        finally:
            duration = perf_counter() - started
            evidence["timings"][name] = duration
            print(f"{name}: {duration:.2f} seconds", flush=True)

    print(f"Existing RF V2 file: {model_path.stat().st_size / 1048576:.2f} MiB", flush=True)
    print("Separate diagnostic experiment; no training or RDS access.", flush=True)
    run_id = None
    client = None
    try:
        client, experiment = timed("tracking_setup", lambda: create_tracking_client(
            get_mlflow_tracking_uri(), experiment_name="group2-pbi12-upload-diagnostics"))
        run = timed("create_diagnostic_run", lambda: client.create_run(experiment, tags={
            "mlflow.runName": f"upload-{args.label}-{probe_id}", "diagnostic": "artifact-upload",
            "source_batch": args.batch_id, "retrained": "false",
        }))
        run_id = run.info.run_id
        evidence["mlflow_run_id"] = run_id
        timestamp = int(time() * 1000)
        timed("parameters_and_metrics", lambda: client.log_batch(
            run_id, params=[Param(str(k), str(v)) for k, v in parameters.items()],
            metrics=[Metric(k, v, timestamp, 0) for k, v in metrics.items()], synchronous=True))
        timed("metrics_json", lambda: client.log_dict(run_id, metrics, "validation_metrics.json"))
        timed("model_file_upload", lambda: client.log_artifact(run_id, str(model_path), artifact_path="models"))
        timed("finish_run", lambda: client.set_terminated(run_id, status="FINISHED"))
        seconds = evidence["timings"]["model_file_upload"]
        evidence["effective_upload_mib_per_second"] = evidence["artifact_bytes"] / 1048576 / seconds
        evidence["status"] = "FINISHED"
        print(f"Effective artifact transfer: {evidence['effective_upload_mib_per_second']:.3f} MiB/s", flush=True)
        print("PASS: existing model uploaded to the diagnostic run.", flush=True)
    except Exception as error:
        evidence["status"] = "FAILED"
        evidence["error_type"] = type(error).__name__
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        raise
    finally:
        reports = root / "reports/pipeline/upload-diagnostics"
        reports.mkdir(parents=True, exist_ok=True)
        receipt = reports / f"{probe_id}.json"
        receipt.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"Receipt: {receipt}", flush=True)


if __name__ == "__main__":
    main()
