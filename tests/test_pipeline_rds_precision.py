import unittest

import pandas as pd

from src.pipeline_data import combine_accepted_data
from src.pipeline_rds import normalize_rds_precision


class RdsPrecisionTests(unittest.TestCase):
    def test_rounding_differences_match_but_real_changes_conflict(self):
        history = pd.DataFrame({
            "country_code": ["SG"] * 4,
            "order_id": [
                "rounding",
                "price_change",
                "missing",
                "refund_change",
            ],
            "order_value": pd.Series(
                [5.387819, 10.0, pd.NA, 20.0],
                dtype="Float64",
            ),
            "refund_value": pd.Series(
                [0.0, 0.0, pd.NA, 1.0],
                dtype="Float64",
            ),
        })

        incoming = history.copy()
        incoming.loc[0, "order_value"] = 5.387818896776103
        incoming.loc[1, "order_value"] = 10.01
        incoming.loc[3, "refund_value"] = 1.01
        original = incoming.copy()

        combined, conflicts, counts = combine_accepted_data(
            normalize_rds_precision("orders", history),
            normalize_rds_precision("orders", incoming),
            ["country_code", "order_id"],
        )

        self.assertEqual(counts["identical_records"], 2)
        self.assertEqual(counts["conflicting_records"], 2)
        self.assertEqual(counts["new_records"], 0)

        self.assertEqual(
            set(conflicts["order_id"]),
            {"price_change", "refund_change"},
        )

        stored = combined.set_index("order_id")

        self.assertEqual(
            stored.loc["price_change", "order_value"],
            10.0,
        )
        self.assertEqual(
            stored.loc["refund_change", "refund_value"],
            1.0,
        )
        self.assertTrue(
            pd.isna(stored.loc["missing", "order_value"])
        )

        pd.testing.assert_frame_equal(incoming, original)

    def test_customer_money_rounding_preserves_nulls(self):
        frame = pd.DataFrame({
            "total_payment_last_50days": pd.Series(
                [1.2345645, pd.NA],
                dtype="Float64",
            ),
        })

        normalized = normalize_rds_precision("customers", frame)

        self.assertEqual(
            normalized.loc[0, "total_payment_last_50days"],
            1.234565,
        )
        self.assertTrue(
            pd.isna(normalized.loc[1, "total_payment_last_50days"])
        )


if __name__ == "__main__":
    unittest.main()