import unittest

import pandas as pd

from src.pipeline_rds import normalize_rds_table


class RdsNormalizationTests(unittest.TestCase):
    def customer_frame(self):
        return pd.DataFrame({
            "country_code": ["SG", "SG", "SG"],
            "customer_id": ["c1", "c2", "c3"],
            "mobile_verified": [1.0, 0.0, None],
            "num_orders_last_50days": [2, 2, 2],
            "num_cancelled_orders_last_50days": [0, 0, 0],
            "num_refund_orders_last_50days": [0, 0, 0],
            "total_payment_last_50days": [20, 20, 20],
            "num_associated_customers": [0, 0, 0],
            "first_order_datetime": ["2023-01-01"] * 3,
        })

    def test_nullable_database_booleans_are_preserved(self):
        cleaned, _ = normalize_rds_table(
            "customers", self.customer_frame()
        )

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned["mobile_verified"].dtype, "boolean")
        self.assertTrue(cleaned.loc[0, "mobile_verified"])
        self.assertFalse(cleaned.loc[1, "mobile_verified"])
        self.assertTrue(pd.isna(cleaned.loc[2, "mobile_verified"]))

    def test_invalid_database_boolean_is_rejected(self):
        frame = self.customer_frame()
        frame.loc[0, "mobile_verified"] = 2

        with self.assertRaisesRegex(ValueError, "invalid mobile_verified"):
            normalize_rds_table("customers", frame)

    def test_duplicate_historical_records_are_not_silently_removed(self):
        frame = pd.concat(
            [self.customer_frame(), self.customer_frame().iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "excluded historical records"):
            normalize_rds_table("customers", frame)


if __name__ == "__main__":
    unittest.main()