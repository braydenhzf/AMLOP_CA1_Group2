"""Offline checks for practice-batch relationships, receipts and reproducibility."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from prepare_practice_batch import COLUMNS, CLEANERS, prepare_batch
from src.features import join_training_data, prepare_training_data, chronological_split


class PracticeBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = self.root / "history"
        self.history.mkdir()
        customers, orders, labels = [], [], []
        for i in range(12):
            country = ("SG", "MY", "ID")[i % 3]
            customers.append([
                country, f"c{i}", "true", 5, 0, 0, 50.25, 0, "2022-01-01 12:00:00",
            ])
        for i in range(36):
            country = ("SG", "MY", "ID")[i % 3]
            orders.append([
                country, f"o{i}", "delivery" if i % 2 else "pickup",
                "card" if i % 3 else "cash", 20.25 + i, i % 4 + 1, 0,
                (pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            ])
            labels.append([country, f"o{i}", f"c{i % 12}", int(i % 4 == 0)])
        for name, rows in (("customers", customers), ("orders", orders), ("labels", labels)):
            pd.DataFrame(rows, columns=COLUMNS[name]).to_csv(
                self.history / f"{name}_base.csv", index=False
            )

    def generate(self, destination, **kwargs):
        with patch("socket.socket.connect", side_effect=AssertionError("Network prohibited")):
            return prepare_batch(
                self.history, destination, batch_id="test-001",
                new_orders=12, repeat_orders=4, **kwargs,
            )

    def test_real_csvs_have_new_and_repeated_keys_and_valid_relationships(self):
        output = self.root / "output"
        expected = self.generate(output)
        tables = {}
        for item in expected["datasets"]:
            name = item["dataset"]
            path = output / f"{name}_practice.csv"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected["sha256"][path.name])
            tables[name], issues = CLEANERS[name](pd.read_csv(path, dtype="string"))
            self.assertTrue(issues.empty)
            self.assertEqual(len(tables[name]), item["expected_accepted_rows"])
            original, _ = CLEANERS[name](pd.read_csv(self.history / f"{name}_base.csv", dtype="string"))
            keys = ["customer_id"] if name == "customers" else ["country_code", "order_id"]
            existing = pd.MultiIndex.from_frame(original[keys])
            incoming = pd.MultiIndex.from_frame(tables[name][keys])
            self.assertTrue(incoming.is_unique)
            actual_repeats = int(incoming.isin(existing).sum())
            self.assertEqual(actual_repeats, item["first_run_identical_records"])
            self.assertEqual(len(incoming) - actual_repeats, item["first_run_new_records"])
        joined, _ = join_training_data(tables["customers"], tables["orders"], tables["labels"])
        X, y, metadata = prepare_training_data(joined)
        self.assertEqual(len(X), 16)
        self.assertEqual(X.shape[1], 9)
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(int(y.loc[metadata["order_id"].str.startswith("pbi12-")].sum()), 3)
        self.assertEqual(expected["combined_eligible_orders_after_first_run"], 48)
        self.assertEqual(expected["required_complete_pipeline_seconds"], 180)
        self.assertEqual(len(expected["expected_model_names"]), 6)
        self.assertEqual(json.loads((output / "expected_outputs.json").read_text()), expected)

    def test_seed_reproduces_files_and_existing_output_is_preserved(self):
        first, second = self.root / "first", self.root / "second"
        self.generate(first)
        self.generate(second)
        for path in first.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
        before = (first / "orders_practice.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            self.generate(first)
        self.assertEqual((first / "orders_practice.csv").read_bytes(), before)

    def test_reference_membership_counts_are_preserved_with_new_training_orders(self):
        clean = {}
        for name, cleaner in CLEANERS.items():
            clean[name], _ = cleaner(pd.read_csv(self.history / f"{name}_base.csv", dtype="string"))
        joined, _ = join_training_data(clean["customers"], clean["orders"], clean["labels"])
        _, _, metadata = prepare_training_data(joined)
        development, test = chronological_split(metadata)
        train, validation = chronological_split(metadata.loc[development], evaluation_fraction=0.25)
        manifest = metadata[["country_code", "order_id"]].copy()
        manifest["split"] = "train"
        manifest.loc[validation, "split"] = "validation"
        manifest.loc[test, "split"] = "test"
        manifest_path = self.root / "splits.csv"
        manifest.to_csv(manifest_path, index=False)
        result = self.generate(self.root / "with-splits", split_manifest=manifest_path)
        expected = result["splits"]["after_first_run_and_repeat"]
        self.assertEqual(expected, {"train": len(train) + 12, "validation": len(validation), "test": len(test)})
        manifest.loc[0, "order_id"] = "not-a-historical-key"
        manifest.to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.generate(self.root / "invalid-reference", split_manifest=manifest_path)
        self.assertFalse((self.root / "invalid-reference").exists())


if __name__ == "__main__":
    unittest.main()
