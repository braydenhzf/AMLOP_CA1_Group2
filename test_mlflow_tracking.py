
import mlflow

mlflow.set_tracking_uri(
    "http://group2-mlflow-isabelle.us-east-1.elasticbeanstalk.com"
)

mlflow.set_experiment("Group2_AWS_MLflow_Test")

with mlflow.start_run(run_name="Isabelle_AWS_Test"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("f1_score", 0.89)

print("SUCCESS: Experiment logged to AWS MLflow!")
