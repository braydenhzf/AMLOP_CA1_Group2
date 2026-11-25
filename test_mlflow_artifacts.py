
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(
    "http://group2-mlflow-isabelle.us-east-1.elasticbeanstalk.com"
)

mlflow.set_experiment("PBI11_Persistence_Test")

with mlflow.start_run(run_name="S3_Artifact_Test") as run:
    mlflow.log_param("storage", "AWS S3")
    mlflow.log_metric("test_metric", 1.0)

    mlflow.log_text(
        "PBI 11 persistent artifact storage test",
        "persistence_test.txt"
    )

    run_id = run.info.run_id
    print("Run ID:", run_id)
    print("Artifact URI:", mlflow.get_artifact_uri())

client = MlflowClient()

downloaded = client.download_artifacts(
    run_id, "persistence_test.txt"
)

assert Path(downloaded).read_text(
    encoding="utf-8"
) == "PBI 11 persistent artifact storage test"

print("SUCCESS: Artifact uploaded and retrieved!")
