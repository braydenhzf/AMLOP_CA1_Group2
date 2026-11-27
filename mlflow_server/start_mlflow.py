
import os
import sys
from sqlalchemy.engine import URL

required = [
    "MLFLOW_DB_HOST",
    "MLFLOW_DB_USER",
    "MLFLOW_DB_PASSWORD",
]

missing = [key for key in required if not os.getenv(key)]

if missing:
    sys.exit(f"Missing environment variables: {', '.join(missing)}")

db_url = URL.create(
    drivername="mysql+pymysql",
    username=os.environ["MLFLOW_DB_USER"],
    password=os.environ["MLFLOW_DB_PASSWORD"],
    host=os.environ["MLFLOW_DB_HOST"],
    port=3306,
    database="mlflow_tracking",
    query={
        "ssl_ca": "/app/global-bundle.pem",
        "ssl_check_hostname": "true",
    },
)

os.environ["MLFLOW_BACKEND_STORE_URI"] = (
    db_url.render_as_string(hide_password=False)
)

artifact_location = (
    "s3://spbuy-group2-mlflow-artifacts-064045000380"
)

command = [
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--workers", "1",
    "--serve-artifacts",
    "--artifacts-destination", artifact_location,
]

os.execvp(command[0], command)
