"""Check Brayden's analytical SQL and rates without contacting AWS."""
from datetime import date
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analytics.brayden_analytics import (
    add_rates, analyse_customer_concentration, analyse_order_value,
    analyse_weekday_collection, publish_report, read_queries, validate_analysis,
)


class Cursor:
    def __init__(self, connection):
        self.raw = connection.cursor()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()

    def execute(self, sql, parameters):
        if not sql.lstrip().upper().startswith("SELECT"):
            raise AssertionError("Analytics attempted a non-SELECT query")
        self.raw.execute(sql.replace("%s", "?"), parameters)

    @property
    def description(self):
        return self.raw.description

    def fetchall(self):
        return self.raw.fetchall()


class Connection:
    def __init__(self):
        self.raw = sqlite3.connect(":memory:")
        self.raw.create_function("WEEKDAY", 1, lambda value: date.fromisoformat(value).weekday() if value else None)
        self.raw.executescript("""
            CREATE TABLE orders(country_code TEXT, order_id TEXT, collect_type TEXT,
                order_value REAL, order_date TEXT, PRIMARY KEY(country_code,order_id));
            CREATE TABLE labels(country_code TEXT, order_id TEXT, customer_id TEXT,
                is_fraud INTEGER, PRIMARY KEY(country_code,order_id));
        """)
        self.raw.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
            ("SG", "o1", "delivery", 10, "2023-01-02"),
            ("SG", "o2", "pickup", 50, "2023-01-02"),
            ("MY", "o1", "delivery", 200, "2023-01-03"),
            ("MY", "o4", "pickup", None, "2023-01-03"),
            ("SG", "o5", "delivery", 600, "2023-01-04"),
            ("SG", "o6", "delivery", 100, "2023-01-04"),
            ("SG", "o7", "pickup", 1000, "2023-01-05"),
            ("SG", "o8", "delivery", 100, "2023-01-05"),
            ("SG", "pbi12-practice-001-o-0001", "delivery", 90, "2023-01-06"),
        ])
        self.raw.executemany("INSERT INTO labels VALUES (?,?,?,?)", [
            ("SG", "o1", "c1", 1), ("SG", "o2", "c1", 0), ("MY", "o1", "c2", 1),
            ("SG", "o5", "c3", None), ("SG", "o6", "c3", 2), ("SG", "o7", "c3", 0),
            ("SG", "o8", "c1", 1), ("SG", "orphan-label", "c4", 1),
            ("SG", "pbi12-practice-001-o-0001", "pbi12-c1", 1),
        ])

    def cursor(self):
        return Cursor(self.raw)


class BraydenAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.connection = Connection()

    def tearDown(self):
        self.connection.raw.close()

    def payload(self, **options):
        payload = read_queries(self.connection, **options)
        payload["metadata"]["snapshot_started_utc"] = "2026-11-28T15:00:00+00:00"
        return validate_analysis(payload, minimum_labelled=1)

    def test_composite_keys_and_valid_labels_give_expected_totals(self):
        payload = self.payload()
        meta = payload["metadata"]
        self.assertEqual((meta["orders_total"], meta["labelled_orders"], meta["fraud_orders"]), (8, 5, 3))
        self.assertEqual(meta["unlabelled_or_invalid_orders"], 3)
        self.assertEqual(meta["invalid_label_orders"], 1)
        self.assertEqual(meta["fraud_order_value"], 310)
        self.assertEqual(meta["labelled_order_value"], 1360)
        self.assertEqual(meta["overall_fraud_rate"], .6)
        self.assertEqual(meta["excluded_practice_orders"], 1)

    def test_practice_records_are_only_included_on_explicit_request(self):
        meta = self.payload(include_practice=True)["metadata"]
        self.assertEqual((meta["orders_total"], meta["labelled_orders"], meta["fraud_orders"]), (9, 6, 4))
        self.assertEqual(meta["excluded_practice_orders"], 0)

    def test_date_bounds_and_customer_limit_are_applied_to_same_scope(self):
        payload = self.payload(start_date="2023-01-03", end_date="2023-01-05", top_customers=1)
        self.assertEqual(payload["metadata"]["orders_total"], 6)
        self.assertEqual(payload["metadata"]["labelled_orders"], 3)
        self.assertEqual(payload["metadata"]["fraud_orders"], 2)
        self.assertEqual(len(payload["tables"]["customer_concentration"]), 1)

    def test_invalid_dates_and_limits_are_rejected(self):
        for options in ({"start_date": "not-a-date"}, {"start_date": "2023-02-01", "end_date": "2023-01-01"},
                        {"top_customers": 0}, {"top_customers": 101}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                read_queries(self.connection, **options)

    def test_zero_labelled_denominator_is_unavailable_not_zero(self):
        payload = self.payload()
        missing = payload["tables"]["order_value"].query("value_band == 'Missing value'").iloc[0]
        self.assertTrue(pd.isna(missing.fraud_rate))
        self.assertTrue(pd.isna(missing.fraud_rate_lower_95))
        self.assertEqual(missing.labelled_orders, 0)

    def test_missing_amount_is_reported_and_does_not_remove_a_valid_label(self):
        self.connection.raw.execute("UPDATE orders SET order_value=NULL WHERE country_code='SG' AND order_id='o1'")
        metadata = self.payload()["metadata"]
        self.assertEqual(metadata["labelled_orders"], 5)
        self.assertEqual(metadata["fraud_orders"], 3)
        self.assertEqual(metadata["labelled_orders_missing_value"], 1)
        self.assertEqual(metadata["fraud_orders_missing_value"], 1)
        self.assertEqual(metadata["fraud_order_value"], 300)

    def test_rates_have_valid_intervals_and_do_not_hide_small_groups(self):
        frame = add_rates(pd.DataFrame({"labelled_orders": [0, 1, 10], "fraud_orders": [0, 1, 0]}))
        self.assertEqual(frame.low_volume.tolist(), [True, True, True])
        self.assertTrue(pd.isna(frame.fraud_rate.iloc[0]))
        self.assertAlmostEqual(frame.fraud_rate.iloc[1], 1)
        self.assertTrue((frame.fraud_rate_lower_95.dropna() >= 0).all())
        self.assertTrue((frame.fraud_rate_upper_95.dropna() <= 1).all())

    def test_top_customers_and_cumulative_share_use_global_fraud_count(self):
        customers = self.payload()["tables"]["customer_concentration"]
        self.assertEqual(customers.customer_id.tolist(), ["c1", "c2"])
        self.assertEqual(customers.fraud_orders.tolist(), [2, 1])
        np.testing.assert_allclose(customers.cumulative_fraud_share, [2/3, 1])
        self.assertAlmostEqual(customers.fraud_rate.iloc[0], 2/3)

    def test_inconsistent_group_counts_and_values_are_rejected(self):
        for column in ("fraud_orders", "fraud_order_value"):
            raw = read_queries(self.connection)
            raw["tables"]["order_value"].loc[0, column] += 1
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_analysis(raw)

    def test_empty_scope_has_no_fabricated_rates_or_customer_results(self):
        payload = self.payload(start_date="2030-01-01")
        self.assertEqual(payload["metadata"]["orders_total"], 0)
        self.assertIsNone(payload["metadata"]["overall_fraud_rate"])
        for analyse in (analyse_order_value, analyse_weekday_collection, analyse_customer_concentration):
            self.assertTrue(analyse(payload)["table"].empty)

    def test_plot_failure_does_not_replace_previous_successful_report(self):
        payload = self.payload()
        results = [analyse_order_value(payload), analyse_weekday_collection(payload), analyse_customer_concentration(payload)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = {"run_id": "previous", "report": "previous/index.html"}
            (root / "latest.json").write_text(json.dumps(previous))
            with patch("src.analytics.brayden_analytics.create_chart", side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    publish_report(payload, results, root)
            self.assertEqual(json.loads((root / "latest.json").read_text()), previous)
            self.assertEqual(list(root.glob("brayden-*")), [])


if __name__ == "__main__":
    unittest.main()
