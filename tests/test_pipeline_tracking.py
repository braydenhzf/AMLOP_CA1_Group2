"""Offline checks for MLflow logging using a temporary local store."""

import tempfile
import unittest
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from joblib import dump, load
from sklearn.linear_model import LogisticRegression

from src.pipeline_tracking import (
    create_tracking_client,
    log_model_outputs,
    tracked_model_run,
)


class PipelineTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        self.client, self.experiment_id = create_tracking_client(
            "sqlite:///:memory:",
            experiment_name=f"offline-test-{uuid4().hex}",
            artifact_location=(self.root / "artifacts").as_uri(),
        )

    def test_success_records_outputs_and_a_loadable_model(self):
        with tracked_model_run(
            self.client,
            self.experiment_id,
            "tiny_lr",
            "test-batch",
        ) as run_id:
            start = perf_counter()

            model = LogisticRegression().fit(
                [[0], [1], [2], [3]],
                [0, 0, 1, 1],
            )

            training_seconds = perf_counter() - start
            model_path = self.root / "tiny_lr.joblib"
            dump(model, model_path)

            log_model_outputs(
                self.client,
                run_id,
                parameters={"C": model.C, "threshold": 0.5},
                metrics={
                    "accuracy": model.score(
                        [[0.2], [2.8]],
                        [0, 1],
                    ),
                    "training_seconds": training_seconds,
                },
                model_path=model_path,
            )

        run = self.client.get_run(run_id)

        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.params["threshold"], "0.5")
        self.assertEqual(run.data.metrics["accuracy"], 1.0)
        self.assertGreaterEqual(
            run.data.metrics["training_seconds"],
            0,
        )
        self.assertEqual(run.data.tags["batch_id"], "test-batch")

        artifact_paths = {
            item.path
            for item in self.client.list_artifacts(run_id)
        }
        self.assertIn("validation_metrics.json", artifact_paths)

        download_dir = self.root / "downloaded"
        download_dir.mkdir()

        downloaded_path = self.client.download_artifacts(
            run_id,
            "models/tiny_lr.joblib",
            str(download_dir),
        )
        restored = load(downloaded_path)

        self.assertEqual(
            restored.predict([[0.2], [2.8]]).tolist(),
            [0, 1],
        )

    def test_failure_is_recorded_and_the_error_is_preserved(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Simulated training failure",
        ):
            with tracked_model_run(
                self.client,
                self.experiment_id,
                "broken_model",
                "test-batch",
            ) as run_id:
                raise RuntimeError("Simulated training failure")

        run = self.client.get_run(run_id)
        self.assertEqual(run.info.status, "FAILED")


if __name__ == "__main__":
    unittest.main()