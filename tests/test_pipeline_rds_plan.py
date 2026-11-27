"""Offline checks for choosing records to insert."""

import unittest

import pandas as pd

from src.pipeline_rds_write import select_new_records


class NewRecordSelectionTests(unittest.TestCase):
    def setUp(self):
        self.keys = ["country_code", "order_id"]

        self.history = pd.DataFrame({
            "country_code": ["SG", "SG"],
            "order_id": ["o1", "o2"],
            "value": [10, 20],
        })

        self.incoming = pd.DataFrame({
            "country_code": ["SG", "SG", "SG"],
            "order_id": ["o1", "o2", "o3"],
            "value": [10, 999, 30],
        })

    def test_only_new_keys_are_selected(self):
        original = self.history.copy()

        selected = select_new_records(
            self.history,
            self.incoming,
            self.keys,
        )

        self.assertEqual(
            selected["order_id"].tolist(),
            ["o3"],
        )
        self.assertEqual(selected["value"].tolist(), [30])

        pd.testing.assert_frame_equal(self.history, original)

    def test_repeated_batch_has_no_new_records(self):
        selected = select_new_records(
            self.history,
            self.incoming,
            self.keys,
        )

        updated_history = pd.concat(
            [self.history, selected],
            ignore_index=True,
        )

        repeated = select_new_records(
            updated_history,
            self.incoming,
            self.keys,
        )

        self.assertTrue(repeated.empty)

    def test_order_key_includes_country(self):
        incoming = self.history.iloc[[0]].copy()
        incoming.loc[0, "country_code"] = "MY"

        selected = select_new_records(
            self.history,
            incoming,
            self.keys,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected.loc[0, "country_code"],
            "MY",
        )

    def test_customer_id_is_globally_unique(self):
        history = pd.DataFrame({
            "country_code": ["SG"],
            "customer_id": ["c1"],
            "value": [10],
        })

        incoming = pd.DataFrame({
            "country_code": ["MY"],
            "customer_id": ["c1"],
            "value": [999],
        })

        selected = select_new_records(
            history,
            incoming,
            ["customer_id"],
        )

        self.assertTrue(selected.empty)


if __name__ == "__main__":
    unittest.main()