import unittest

import pandas as pd

from src.pipeline_data import combine_accepted_data


class CombineAcceptedDataTests(unittest.TestCase):
    def test_new_identical_and_conflicting_records(self):
        history = pd.DataFrame({
            "id": [1, 2],
            "value": [10, 20],
        })
        incoming = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 99, 30],
        })

        combined, conflicts, summary = combine_accepted_data(
            history, incoming, keys=["id"]
        )

        # Existing value 20 is retained despite the conflicting incoming 99.
        self.assertEqual(
            combined.set_index("id")["value"].to_dict(),
            {1: 10, 2: 20, 3: 30},
        )
        self.assertEqual(conflicts["id"].tolist(), [2])
        self.assertEqual(summary["new_records"], 1)
        self.assertEqual(summary["identical_records"], 1)
        self.assertEqual(summary["conflicting_records"], 1)

        # Retrying the same input must not create more unique records.
        retried, _, retry_summary = combine_accepted_data(
            combined, incoming, keys=["id"]
        )
        self.assertEqual(len(retried), 3)
        self.assertEqual(retry_summary["new_records"], 0)


if __name__ == "__main__":
    unittest.main()