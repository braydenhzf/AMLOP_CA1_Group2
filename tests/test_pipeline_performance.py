"""Offline checks for paging integrity, parallel fits and truthful timing receipts."""

import json
import re
import struct
import tempfile
import threading
import unittest
import zlib
from contextlib import contextmanager, redirect_stdout
from datetime import date, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor
from src.pipeline_rds_transport import read_page
from src.pipeline_timing import StageTimings
from src.pipeline_tracking import create_tracking_client
from src.pipeline_training import train_available_models
import src.pipeline_rds as rds


def pack(rows):
    def encode(value):
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
        raise TypeError(type(value).__name__)
    text = json.dumps(rows, default=encode).encode("utf-8")
    return struct.pack("<I", len(text)) + zlib.compress(text)


class FakeSnapshotCursor:
    def __init__(self, tables):
        self.tables = tables
        self.answer = None
        self.rows = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, parameters=()):
        name = re.search(r"FROM `([a-z_]+)`", query).group(1)
        source = self.tables[name]
        if query.startswith("SELECT COUNT(*)"):
            self.answer = (len(source),)
            return
        selected = source
        if len(parameters) == 4:
            final = (parameters[0], parameters[2])
            selected = [row for row in source if tuple(row[:2]) > final]
        self.rows = selected[:parameters[-1]]
        if query.startswith("WITH"):
            ranked = [[number, *row] for number, row in enumerate(self.rows, 1)]
            self.answer = (pack(list(reversed(ranked))), len(ranked))

    def fetchone(self):
        return self.answer

    def fetchall(self):
        return self.rows


class RdsTransferTests(unittest.TestCase):
    def test_compressed_page_restores_order_and_decimal_null_values(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (pack([
            [2, "SG", "o2", None], [1, "SG", "o1", Decimal("12.125")],
        ]), 2)
        rows, last = read_page(cursor, "orders", ["country_code", "order_id", "order_value"],
                               ["country_code", "order_id"], ("MY", "o0"), 100, "compressed")
        self.assertEqual(rows, [["SG", "o1", Decimal("12.125")], ["SG", "o2", None]])
        self.assertEqual(last, ("SG", "o2"))
        query, parameters = cursor.execute.call_args.args
        self.assertIn("ROW_NUMBER() OVER", query)
        self.assertEqual(parameters, ("MY", "MY", "o0", 100))

    def test_corrupted_or_incomplete_payload_is_rejected(self):
        cursor = MagicMock()
        payload = pack([[1, "SG", "o1"]])
        cursor.fetchone.return_value = (struct.pack("<I", 999999) + payload[4:], 1)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            read_page(cursor, "orders", ["country_code", "order_id"],
                      ["country_code", "order_id"], None, 10, "compressed")
        cursor.fetchone.return_value = (pack([[2, "SG", "o1"]]), 1)
        with self.assertRaisesRegex(ValueError, "missing or duplicate row ordinals"):
            read_page(cursor, "orders", ["country_code", "order_id"],
                      ["country_code", "order_id"], None, 10, "compressed")

    def test_full_paging_matches_normalised_row_transfer_across_countries(self):
        tables = {"customers": [], "orders": [], "labels": []}
        for i in range(5):
            country = "MY" if i < 2 else "SG"
            tables["customers"].append([
                country, f"c{i}", None if i == 0 else i % 2, 2, 0, 0,
                Decimal("12.125000"), 0, datetime(2022, 1, 1),
            ])
            tables["orders"].append([
                country, f"o{i}", "delivery", "card", Decimal("5.375000"),
                2, Decimal("0.000000"), date(2023, 1, i + 1),
            ])
            tables["labels"].append([country, f"o{i}", f"c{i}", i % 2])

        def get_tables(transfer):
            cursor = FakeSnapshotCursor(tables)
            connection = MagicMock()
            connection.cursor.return_value = cursor
            @contextmanager
            def snapshot():
                yield connection
            with patch.object(rds, "_read_snapshot", snapshot), redirect_stdout(StringIO()):
                return rds.read_rds_history(batch_size=2, transfer=transfer)

        ordinary, compressed = get_tables("rows"), get_tables("compressed")
        for name in tables:
            left, left_issues = rds.normalize_rds_table(name, ordinary[name])
            right, right_issues = rds.normalize_rds_table(name, compressed[name])
            pd.testing.assert_frame_equal(left, right)
            pd.testing.assert_frame_equal(left_issues, right_issues)
            self.assertEqual(len(right), 5)


class TimingReceiptTests(unittest.TestCase):
    def test_failure_is_recorded_and_partial_work_cannot_pass_cloud_requirement(self):
        with tempfile.TemporaryDirectory() as folder:
            timer = StageTimings(folder)
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                with timer.stage("outer"):
                    with timer.stage("inner"):
                        raise RuntimeError("original failure")
            result = timer.finish(success=False, elapsed=1, complete_workflow=True)
            self.assertFalse(result["pbi12_runtime_check_passed"])
            rows = pd.read_csv(Path(folder) / "stage_timings.csv")
            self.assertEqual(rows["status"].tolist(), ["FAILED", "FAILED"])
            self.assertEqual(rows.loc[0, "parent"], "outer")
            self.assertFalse(timer.finish(success=True, elapsed=1)["pbi12_runtime_check_passed"])
            self.assertFalse(timer.finish(success=True, elapsed=181, complete_workflow=True)["pbi12_runtime_check_passed"])
            self.assertTrue(timer.finish(success=True, elapsed=180, complete_workflow=True)["pbi12_runtime_check_passed"])


class ParallelTrainingTests(unittest.TestCase):
    def setUp(self):
        count = 24
        self.X = pd.DataFrame({
            "order_value": np.arange(count, dtype=float) + 10,
            "num_items_ordered": np.arange(count) % 3 + 1,
            "value_per_item": np.arange(count, dtype=float) + 1,
            "customer_tenure_days": np.arange(count, dtype=float) + 5,
            "country_code": ["SG", "MY"] * (count // 2),
            "collect_type": ["delivery"] * count,
            "payment_method": ["card"] * count,
            "order_weekday": ["Monday"] * count,
            "order_month": ["January"] * count,
        })
        self.y = pd.Series([0, 1] * (count // 2))

    def fit_small(self, X, y, *args, **kwargs):
        model = Pipeline([("preprocessor", build_preprocessor()),
                          ("classifier", LogisticRegression(max_iter=1000))])
        return model.fit(X, y)

    def test_two_fits_overlap_and_compressed_models_reload_without_changing_inputs(self):
        lock = threading.Lock()
        gate = threading.Barrier(2)
        calls = []
        def fit_lr(X, y, **kwargs):
            with lock:
                calls.append(kwargs["class_weight"])
            gate.wait(timeout=10)
            return self.fit_small(X, y), 0
        before = self.X.copy(deep=True)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("src.pipeline_training.train_logistic_regression", side_effect=fit_lr),
            patch("src.pipeline_training.train_random_forest", side_effect=self.fit_small),
            patch("src.pipeline_training.train_xgboost", side_effect=self.fit_small),
            redirect_stdout(StringIO()),
        ):
            models, results = train_available_models(
                self.X, self.y, self.X, self.y, Path(folder), 0.5, model_workers=2,
            )
            self.assertEqual(len(models), 6)
            self.assertEqual(len(results), 6)
            self.assertCountEqual(calls, [None, "balanced"])
            for name, model in models.items():
                restored = load(Path(folder) / f"{name}.joblib")
                np.testing.assert_allclose(restored.predict_proba(self.X), model.predict_proba(self.X))
        pd.testing.assert_frame_equal(before, self.X)

    def test_fit_failure_is_preserved_and_no_mlflow_runs_remain_running(self):
        def fit_lr(X, y, **kwargs):
            if kwargs["class_weight"] == "balanced":
                raise RuntimeError("fit failure preserved")
            return self.fit_small(X, y), 0
        with tempfile.TemporaryDirectory() as folder:
            client, experiment = create_tracking_client(
                "sqlite:///:memory:", experiment_name="parallel-failure",
                artifact_location=(Path(folder) / "artifacts").as_uri(),
            )
            with (
                patch("src.pipeline_training.train_logistic_regression", side_effect=fit_lr),
                patch("src.pipeline_training.train_random_forest", side_effect=self.fit_small),
                patch("src.pipeline_training.train_xgboost", side_effect=self.fit_small),
                redirect_stdout(StringIO()),
                self.assertRaisesRegex(RuntimeError, "fit failure preserved"),
            ):
                train_available_models(
                    self.X, self.y, self.X, self.y, Path(folder) / "models", 0.5,
                    model_workers=2, tracking_client=client,
                    experiment_id=experiment, batch_id="parallel-failure",
                )
            statuses = [run.info.status for run in client.search_runs([experiment])]
            self.assertEqual(len(statuses), 6)
            self.assertIn("FAILED", statuses)
            self.assertNotIn("RUNNING", statuses)


if __name__ == "__main__":
    unittest.main()
