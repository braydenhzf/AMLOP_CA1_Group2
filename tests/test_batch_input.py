import csv
import tempfile
import unittest
from pathlib import Path

from download_batch import REQUIRED_COLUMNS, download_batch, validate_csv


class BatchInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        self.sources = {}
        for name, columns in REQUIRED_COLUMNS.items():
            path = self.root / f"{name}_batch42.csv"
            self.write_csv(path, columns, [["test"] * len(columns)])
            self.sources[name] = path

    @staticmethod
    def write_csv(path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def run_batch(self):
        return download_batch(
            self.sources["customers"].as_uri(),
            self.sources["orders"].as_uri(),
            self.sources["labels"].as_uri(),
            self.root / "downloaded",
        )

    def test_valid_batch_with_different_filename_postfix(self):
        paths = self.run_batch()

        self.assertEqual(set(paths), {"customers", "orders", "labels"})
        for name, path in paths.items():
            self.assertTrue(path.is_file())
            self.assertEqual(validate_csv(path, name), 1)

    def test_batch_rejects_missing_required_column(self):
        columns = [
            column
            for column in REQUIRED_COLUMNS["labels"]
            if column != "is_fraud"
        ]
        self.write_csv(
            self.sources["labels"],
            columns,
            [["test"] * len(columns)],
        )

        with self.assertRaisesRegex(ValueError, "labels: missing.*is_fraud"):
            self.run_batch()

    def test_empty_file_is_rejected(self):
        self.sources["orders"].write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "orders:.*empty"):
            validate_csv(self.sources["orders"], "orders")

    def test_duplicate_header_is_rejected(self):
        columns = list(REQUIRED_COLUMNS["orders"]) + ["country_code"]
        self.write_csv(self.sources["orders"], columns, [])

        with self.assertRaisesRegex(ValueError, "orders: duplicate"):
            validate_csv(self.sources["orders"], "orders")

    def test_incorrect_row_length_is_rejected(self):
        self.write_csv(
            self.sources["orders"],
            REQUIRED_COLUMNS["orders"],
            [["too", "few"]],
        )

        with self.assertRaisesRegex(ValueError, "orders: line"):
            validate_csv(self.sources["orders"], "orders")

    def test_missing_source_stops_batch(self):
        self.sources["labels"] = self.root / "missing_labels.csv"

        with self.assertRaises(OSError):
            self.run_batch()


if __name__ == "__main__":
    unittest.main()