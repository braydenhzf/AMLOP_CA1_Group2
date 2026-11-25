import unittest

import pandas as pd

from src.pipeline_splits import (
    apply_reference_splits,
    make_reference_splits,
)


class PipelineSplitTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({
            "country_code": ["SG"] * 10,
            "order_id": [f"order-{i}" for i in range(10)],
            "order_date": pd.date_range("2023-01-01", periods=10),
        })
        self.reference = make_reference_splits(self.metadata)

    def test_membership_survives_new_orders_and_row_reordering(self):
        new_order = pd.DataFrame({
            "country_code": ["SG"],
            "order_id": ["new-order"],
            "order_date": pd.to_datetime(["2023-02-01"]),
        }, index=[100])

        combined = pd.concat([self.metadata, new_order])
        combined = combined.sample(frac=1, random_state=42)

        splits = apply_reference_splits(combined, self.reference)

        for name in ("validation", "test"):
            expected = set(
                self.reference.loc[
                    self.reference["split"].eq(name), "order_id"
                ]
            )
            actual = set(combined.loc[splits[name], "order_id"])
            self.assertEqual(actual, expected)

        self.assertIn(100, splits["train"])
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertEqual(sum(map(len, splits.values())), len(combined))

    def test_missing_reference_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference orders are missing"):
            apply_reference_splits(
                self.metadata.iloc[:-1],
                self.reference,
            )

    def test_duplicate_reference_keys_are_rejected(self):
        duplicated = pd.concat(
            [self.reference, self.reference.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "duplicate order keys"):
            apply_reference_splits(self.metadata, duplicated)


if __name__ == "__main__":
    unittest.main()