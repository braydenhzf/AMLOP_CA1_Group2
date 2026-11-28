"""A cache must detect changed content, corrupt files and source changes."""

import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import pipeline_rds as rds
from src import pipeline_rds_cache as cache


class VerifiedHistoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data = {
            "customers": pd.DataFrame([["SG", "c1", 1, 2, 0, 0, 20, 0, "2022-01-01"]], columns=rds.TABLE_COLUMNS["customers"]),
            "orders": pd.DataFrame([["SG", "o1", "delivery", "card", 10, 2, 0, "2023-01-01"]], columns=rds.TABLE_COLUMNS["orders"]),
            "labels": pd.DataFrame([["SG", "o1", "c1", 1]], columns=rds.TABLE_COLUMNS["labels"]),
        }
        self.signature = {name: {"rows": 1, "sha256": "a" * 64} for name in self.data}
        self.connection = object()

    def get(self, *, identity="same-code-and-database"):
        @contextmanager
        def snapshot():
            yield self.connection
        def read(**kwargs):
            self.assertIs(kwargs["_connection"], self.connection)
            return {name: frame.copy(deep=True) for name, frame in self.data.items()}
        with (
            patch.object(rds, "_read_snapshot", snapshot),
            patch.object(rds, "read_rds_history", side_effect=read) as reader,
            patch.object(cache, "source_key", return_value=identity),
            patch.object(cache, "fingerprints", return_value=self.signature) as checker,
            redirect_stdout(StringIO()),
        ):
            tables, issues = cache.load_verified_history(self.folder, report_dir=self.folder)
        checker.assert_called_once()
        return tables, issues, reader

    def test_hit_reuses_only_live_verified_canonical_rows(self):
        first, _, reader = self.get()
        reader.assert_called_once()
        self.assertFalse(json.loads((self.folder / "rds_history_cache.json").read_text())["hit"])
        second, _, reader = self.get()
        reader.assert_not_called()
        self.assertTrue(json.loads((self.folder / "rds_history_cache.json").read_text())["hit"])
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_same_count_changed_values_force_reload(self):
        self.get()
        self.data["orders"].loc[0, "order_value"] = 99
        self.signature["orders"] = {"rows": 1, "sha256": "b" * 64}
        tables, _, reader = self.get()
        reader.assert_called_once()
        self.assertEqual(tables["orders"].loc[0, "order_value"], 99)
        self.assertFalse(json.loads((self.folder / "rds_history_cache.json").read_text())["hit"])

    def test_corrupt_payload_or_changed_source_forces_reload(self):
        self.get()
        (self.folder / "accepted_history.joblib").write_bytes(b"corrupted")
        _, _, reader = self.get()
        reader.assert_called_once()
        _, _, reader = self.get(identity="new-code-or-database")
        reader.assert_called_once()

    def test_truncated_database_fingerprint_is_rejected(self):
        from unittest.mock import MagicMock
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (100, 64, "a" * 64)
        with self.assertRaisesRegex(ValueError, "incomplete fingerprint"):
            cache.fingerprints(connection, {"orders": rds.TABLE_COLUMNS["orders"]})


if __name__ == "__main__":
    unittest.main()
