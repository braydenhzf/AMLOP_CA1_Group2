"""Test RDS reconciliation with every network connection simulated."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.pipeline_rds as rds


class RdsIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.report_dir = Path(temporary.name)

        rows = {
            "customers": [
                ["SG", "c1", 1, 2, 0, 0, 10.0, 0, "2022-01-01"],
                ["SG", "c2", 0, 2, 0, 0, 20.0, 0, "2022-01-01"],
            ],
            "orders": [
                ["SG", "o1", "delivery", "card", 5.0, 2, 0, "2023-01-01"],
                ["SG", "o2", "delivery", "card", 10.0, 2, 0, "2023-01-02"],
            ],
            "labels": [
                ["SG", "o1", "c1", 0],
                ["SG", "o2", "c2", 1],
            ],
        }

        self.history = {
            name: pd.DataFrame(
                values,
                columns=rds.TABLE_COLUMNS[name],
            )
            for name, values in rows.items()
        }

        incoming = {
            name: pd.concat(
                [frame, frame.iloc[[0]]],
                ignore_index=True,
            )
            for name, frame in self.history.items()
        }

        incoming["customers"].loc[2, "customer_id"] = "c3"
        incoming["customers"].loc[2, "total_payment_last_50days"] = 30.0
        incoming["customers"].loc[1, "total_payment_last_50days"] = 99.0

        incoming["orders"].loc[2, "order_id"] = "o3"
        incoming["orders"].loc[2, "order_value"] = 15.0
        incoming["orders"].loc[1, "order_value"] = 99.0

        incoming["labels"].loc[
            2, ["order_id", "customer_id", "is_fraud"]
        ] = ["o3", "c3", 1]
        incoming["labels"].loc[1, "is_fraud"] = 0

        self.incoming = {
            name: rds.normalize_rds_table(name, frame)[0]
            for name, frame in incoming.items()
        }

    def history_copy(self):
        return {
            name: frame.copy()
            for name, frame in self.history.items()
        }

    def reconcile(self, write_new=None, failure=None):
        with (
            patch.object(
                rds,
                "read_rds_history",
                side_effect=self.history_copy,
            ),
            patch.object(rds.pymysql, "connect") as connect,
            patch.object(rds, "get_database_settings") as settings,
            patch(
                "src.pipeline_rds_write.insert_new_records"
            ) as writer,
        ):
            settings.return_value.connection_kwargs.return_value = {}

            writer.return_value = {
                "customers": 1,
                "orders": 1,
                "labels": 1,
            }
            writer.side_effect = failure

            options = (
                {} if write_new is None
                else {"write_new": write_new}
            )

            with contextlib.redirect_stdout(io.StringIO()):
                combined, summary = rds.reconcile_rds_history(
                    self.incoming,
                    self.report_dir,
                    **options,
                )

        return combined, summary, connect, writer

    def test_default_mode_keeps_history_and_does_not_write(self):
        combined, summary, connect, writer = self.reconcile()

        connect.assert_not_called()
        writer.assert_not_called()

        self.assertEqual(
            summary["new_records"].tolist(),
            [1, 1, 1],
        )
        self.assertEqual(
            summary["identical_records"].tolist(),
            [1, 1, 1],
        )
        self.assertEqual(
            summary["conflicting_records"].tolist(),
            [1, 1, 1],
        )

        order = combined["orders"].set_index("order_id").loc["o2"]
        self.assertEqual(order["order_value"], 10.0)

    def test_write_mode_sends_only_new_records_and_saves_receipt(self):
        _, _, connect, writer = self.reconcile(write_new=True)

        writer.assert_called_once()
        selected = writer.call_args.args[1]

        self.assertEqual(
            selected["customers"]["customer_id"].tolist(),
            ["c3"],
        )
        self.assertEqual(
            selected["orders"]["order_id"].tolist(),
            ["o3"],
        )
        self.assertEqual(
            selected["labels"]["order_id"].tolist(),
            ["o3"],
        )
        self.assertFalse(connect.call_args.kwargs["autocommit"])

        receipt = pd.read_csv(
            self.report_dir / "rds_insert_summary.csv"
        )
        self.assertEqual(
            receipt["inserted_rows"].tolist(),
            [1, 1, 1],
        )

    def test_repeated_batch_does_not_open_a_write_connection(self):
        combined, _, _, _ = self.reconcile()
        self.history = combined

        _, _, connect, writer = self.reconcile(write_new=True)

        connect.assert_not_called()
        writer.assert_not_called()

        receipt = pd.read_csv(
            self.report_dir / "rds_insert_summary.csv"
        )
        self.assertEqual(receipt["inserted_rows"].sum(), 0)

    def test_insertion_failure_is_not_reported_as_success(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Simulated insertion failure",
        ):
            self.reconcile(
                write_new=True,
                failure=RuntimeError("Simulated insertion failure"),
            )

        self.assertFalse(
            (self.report_dir / "rds_insert_summary.csv").exists()
        )


if __name__ == "__main__":
    unittest.main()