"""Exercise the six-model local pipeline with a small new-data batch."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "brayden_lr_v1",
    "brayden_lr_v2",
    "vivian_rf_v1",
    "vivian_rf_v2",
    "isabelle_xgb_v1",
    "isabelle_xgb_v2",
}


def make_tables(numbers):
    from src.data_cleaning import (
        CUSTOMER_COLUMNS,
        ORDER_COLUMNS,
        LABEL_COLUMNS,
    )

    rows = {"customers": [], "orders": [], "labels": []}

    for number in numbers:
        customer = f"c{number:03}"
        order = f"o{number:03}"
        date = (
            pd.Timestamp("2023-01-01") + pd.Timedelta(days=number)
        ).strftime("%Y/%m/%d")

        rows["customers"].append([
            "SG", customer, "true", 2, 0, 0,
            float(number + 10), 0, "2022-01-01",
        ])
        rows["orders"].append([
            "SG", order, "delivery", "card",
            float(number + 10), 2, 0.0, date,
        ])
        rows["labels"].append([
            "SG", order, customer, int(number % 4 == 0),
        ])

    columns = {
        "customers": CUSTOMER_COLUMNS,
        "orders": ORDER_COLUMNS,
        "labels": LABEL_COLUMNS,
    }
    return {
        name: pd.DataFrame(values, columns=columns[name])
        for name, values in rows.items()
    }


def run_worker(project_root, scratch):
    sys.path.insert(0, str(project_root))

    import run_pipeline
    from joblib import load
    from mlflow.tracking import MlflowClient
    from src.models.train_random_forest import load_historical_data
    from src.pipeline_splits import make_reference_splits

    history_dir = scratch / "history"
    incoming_dir = scratch / "incoming"
    history_dir.mkdir()
    incoming_dir.mkdir()

    history = make_tables(range(80))
    incoming = make_tables([0, *range(80, 88)])

    for name in history:
        history[name].to_csv(
            history_dir / f"{name}_base.csv",
            index=False,
        )
        incoming[name].to_csv(
            incoming_dir / f"{name}.csv",
            index=False,
        )

    reference_features, _, metadata, _ = load_historical_data(history_dir)
    manifest = scratch / "reference.csv"
    make_reference_splits(metadata).to_csv(manifest, index=False)

    arguments = ["run_pipeline.py"]
    for name in history:
        arguments.extend([
            f"--{name}-url",
            (incoming_dir / f"{name}.csv").as_uri(),
        ])

    arguments.extend([
        "--history-dir", str(history_dir),
        "--split-manifest", str(manifest),
        "--train-local",
        "--threshold", "0.5",
        "--rf-jobs", "1",
        "--xgb-jobs", "1",
        "--log-mlflow-local",
    ])

    with (
        patch.object(sys, "argv", arguments),
        patch.object(
            run_pipeline,
            "__file__",
            str(scratch / "run_pipeline.py"),
        ),
    ):
        run_pipeline.main()

    report_dirs = list((scratch / "reports" / "pipeline").glob("batch-*"))
    assert len(report_dirs) == 1
    report_dir = report_dirs[0]

    reconciliation = pd.read_csv(
        report_dir / "reconciliation_summary.csv"
    )
    expected_counts = {
        "historical_rows": 80,
        "incoming_rows": 9,
        "new_records": 8,
        "identical_records": 1,
        "conflicting_records": 0,
        "combined_rows": 88,
    }
    for column, expected in expected_counts.items():
        assert reconciliation[column].tolist() == [expected] * 3, column

    splits = pd.read_csv(
        report_dir / "split_summary.csv"
    ).set_index("split")
    assert splits["rows"].to_dict() == {
        "train": 56,
        "validation": 16,
        "test": 16,
    }

    results = pd.read_csv(report_dir / "models" / "model_results.csv")
    assert len(results) == 6
    assert set(results["model"]) == EXPECTED_MODELS

    database = scratch / "reports" / "mlflow_local" / "tracking.db"
    client = MlflowClient(tracking_uri=f"sqlite:///{database.as_posix()}")
    experiment = client.get_experiment_by_name("group2-pbi10")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 6
    assert {run.info.run_id for run in runs} == set(results["mlflow_run_id"])

    download_dir = scratch / "downloaded_models"
    download_dir.mkdir()

    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.params["training_rows"] == "56"
        assert run.data.params["validation_rows"] == "16"
        assert run.data.params["scored_split"] == "validation"

        artifacts = client.list_artifacts(run.info.run_id, "models")
        assert len(artifacts) == 1

        downloaded = client.download_artifacts(
            run.info.run_id,
            artifacts[0].path,
            str(download_dir),
        )
        restored = load(downloaded)
        assert restored.predict_proba(reference_features.iloc[:2]).shape == (2, 2)

    print("OFFLINE_FLOW_OK")


class PipelineLocalFlowTests(unittest.TestCase):
    def test_new_batch_trains_and_logs_all_six_models(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(PROJECT_ROOT),
                    directory,
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + "\n" + completed.stderr,
            )
            self.assertIn("OFFLINE_FLOW_OK", completed.stdout)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        with patch(
            "socket.socket.connect",
            side_effect=AssertionError("Network access attempted in offline test"),
        ):
            run_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
