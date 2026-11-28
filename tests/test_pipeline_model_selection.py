"""Explicit model selection keeps the default and reports its timing scope."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED = ("vivian_rf_v2", "isabelle_xgb_v2")


def run_selected_worker(project_root, scratch):
    sys.path.insert(0, str(project_root))
    import numpy as np
    import pandas as pd
    import run_pipeline
    from joblib import load
    from mlflow.tracking import MlflowClient
    from test_pipeline_local_flow import make_tables
    from src.models.train_random_forest import load_historical_data, train_random_forest
    from src.models.train_xgboost import train_xgboost
    from src.pipeline_splits import make_reference_splits
    from src.features import join_training_data, prepare_training_data
    from src.data_cleaning import clean_customers_data, clean_orders_data, clean_labels_data

    history_dir, incoming_dir = scratch / "history", scratch / "incoming"
    history_dir.mkdir()
    incoming_dir.mkdir()
    history, incoming = make_tables(range(80)), make_tables([0, *range(80, 88)])
    for name in history:
        history[name].to_csv(history_dir / f"{name}_base.csv", index=False)
        incoming[name].to_csv(incoming_dir / f"{name}.csv", index=False)
    reference_features, _, metadata, _ = load_historical_data(history_dir)
    manifest = scratch / "reference.csv"
    reference = make_reference_splits(metadata)
    reference.to_csv(manifest, index=False)
    arguments = ["run_pipeline.py"]
    for name in incoming:
        arguments += [f"--{name}-url", (incoming_dir / f"{name}.csv").as_uri()]
    arguments += ["--history-dir", str(history_dir), "--split-manifest", str(manifest),
                  "--train-local", "--threshold", "0.5", "--rf-jobs", "1", "--xgb-jobs", "1",
                  "--model-workers", "1", "--models", *SELECTED, "--log-mlflow-local"]
    with patch.object(sys, "argv", arguments), patch.object(run_pipeline, "__file__", str(scratch / "run_pipeline.py")):
        run_pipeline.main()
    report, = (scratch / "reports/pipeline").glob("batch-*")
    results = pd.read_csv(report / "models/model_results.csv")
    assert results.model.tolist() == list(SELECTED)
    assert set(results.training_scope) == {"selected_versions"}
    assert len(list((report / "models").glob("*.joblib"))) == 2
    selection = json.loads((report / "model_selection.json").read_text())
    assert selection["requested_models"] == list(SELECTED)
    assert set(selection["completed_models"]) == set(SELECTED)
    assert not selection["all_six_models_completed"]
    timing = json.loads((report / "performance_summary.json").read_text())
    assert timing["success"] and not timing["pbi12_runtime_check_passed"]
    assert timing["model_selection"] == selection
    assert pd.read_csv(report / "split_summary.csv").set_index("split").rows.to_dict() == {
        "train": 56, "validation": 16, "test": 16}
    client = MlflowClient(tracking_uri=f"sqlite:///{(scratch / 'reports/mlflow_local/tracking.db').as_posix()}")
    experiment = client.get_experiment_by_name("group2-pbi10")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.params["training_scope"] == "selected_versions"
        assert run.data.params["requested_models"] == ",".join(SELECTED)
        assert len(client.list_artifacts(run.info.run_id, "models")) == 1

    # Rebuild identical data and check selected models against unchanged standalone trainers.
    cleaners = {"customers": clean_customers_data, "orders": clean_orders_data, "labels": clean_labels_data}
    full = {}
    for name, cleaner in cleaners.items():
        full[name], _ = cleaner(make_tables(range(88))[name].astype("string"))
    joined, _ = join_training_data(full["customers"], full["orders"], full["labels"])
    X, y, metadata = prepare_training_data(joined)
    from src.pipeline_splits import apply_reference_splits
    splits = apply_reference_splits(metadata, reference.astype("string"))
    train, validation = splits["train"], splits["validation"]
    for name, trainer in ((SELECTED[0], train_random_forest), (SELECTED[1], train_xgboost)):
        baseline = trainer(X.loc[train], y.loc[train], version="v2", n_jobs=1)
        restored = load(report / "models" / f"{name}.joblib")
        np.testing.assert_allclose(restored.predict_proba(X.loc[validation]),
                                   baseline.predict_proba(X.loc[validation]), rtol=0, atol=1e-14)
    print("SELECTED_FLOW_OK")


class ModelSelectionTests(unittest.TestCase):
    def test_default_is_all_six_and_selection_keeps_reporting_order(self):
        from src.pipeline_training import MODEL_NAMES, resolve_model_names
        self.assertEqual(resolve_model_names(), MODEL_NAMES)
        self.assertEqual(resolve_model_names(list(reversed(SELECTED))), SELECTED)

    def test_invalid_selections_are_rejected(self):
        from src.pipeline_training import resolve_model_names
        for value in ([], [SELECTED[0], SELECTED[0]], ["unknown"], SELECTED[0], [None]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_model_names(value)

    def test_cli_rejects_selection_without_training_before_processing(self):
        import run_pipeline
        arguments = ["run_pipeline.py", "--customers-url", "unused", "--orders-url", "unused",
                     "--labels-url", "unused", "--models", *SELECTED]
        with patch.object(sys, "argv", arguments), patch.object(run_pipeline, "prepare_incoming_batch") as prepare:
            with self.assertRaises(SystemExit) as error:
                run_pipeline.main()
            self.assertEqual(error.exception.code, 2)
            prepare.assert_not_called()

    def test_under_limit_subset_is_not_reported_as_an_all_six_pass(self):
        from src.pipeline_timing import StageTimings
        with tempfile.TemporaryDirectory() as folder:
            selection = {"scope": "selected_versions", "requested_models": list(SELECTED),
                         "completed_models": list(SELECTED), "all_six_models_completed": False,
                         "selected_cloud_workflow_completed": True}
            result = StageTimings(folder).finish(success=True, elapsed=120, complete_workflow=False,
                                                model_selection=selection)
            self.assertTrue(result["within_time_limit"])
            self.assertFalse(result["pbi12_runtime_check_passed"])
            self.assertTrue(result["model_selection"]["selected_cloud_workflow_completed"])

    def test_two_selected_models_train_log_and_reload_without_prediction_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                                     str(PROJECT_ROOT), folder], cwd=folder, capture_output=True,
                                    text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            self.assertIn("SELECTED_FLOW_OK", result.stdout)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        with patch("socket.socket.connect", side_effect=AssertionError("Network access disabled in offline test")):
            run_selected_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
