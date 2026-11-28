"""Offline insertion tests: all databases exist only in local memory."""

import sqlite3
import unittest
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from src.pipeline_rds_write import (
    TABLE_COLUMNS,
    TABLE_KEYS,
    insert_new_records,
)


class LocalCursor:
    def __init__(self, database):
        self.cursor = database.cursor()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cursor.close()

    def executemany(self, query, rows):
        converted = []

        for row in rows:
            values = []

            for value in row:
                if isinstance(value, Decimal):
                    value = str(value)
                elif isinstance(value, (date, datetime)):
                    value = value.isoformat()

                values.append(value)

            converted.append(tuple(values))

        self.cursor.executemany(
            query.replace("%s", "?"),
            converted,
        )


class LocalConnection:
    def __init__(self):
        self.database = sqlite3.connect(":memory:")
        self.begins = 0
        self.commits = 0
        self.rollbacks = 0

        for name, columns in TABLE_COLUMNS.items():
            definitions = ", ".join(
                f"`{column}` TEXT" for column in columns
            )
            keys = ", ".join(
                f"`{key}`" for key in TABLE_KEYS[name]
            )

            self.database.execute(
                f"CREATE TABLE `{name}` "
                f"({definitions}, PRIMARY KEY ({keys}))"
            )

        self.database.commit()

    def begin(self):
        self.begins += 1
        self.database.execute("BEGIN")

    def cursor(self):
        return LocalCursor(self.database)

    def commit(self):
        self.commits += 1
        self.database.commit()

    def rollback(self):
        self.rollbacks += 1
        self.database.rollback()


class RdsWriteTests(unittest.TestCase):
    def setUp(self):
        self.connection = LocalConnection()
        self.addCleanup(self.connection.database.close)

        rows = {
            "customers": [[
                "SG", "c1", None, 2, 0, 0, 12.123456789, 0,
                pd.Timestamp("2022-01-01"),
            ]],
            "orders": [[
                "SG", "o1", "delivery", "card",
                5.387818896776103, 2, 0,
                pd.Timestamp("2023-01-01"),
            ]],
            "labels": [["SG", "o1", "c1", 1]],
        }

        self.tables = {
            name: pd.DataFrame(
                values,
                columns=TABLE_COLUMNS[name],
            )
            for name, values in rows.items()
        }

    def row_counts(self):
        return {
            name: self.connection.database.execute(
                f"SELECT COUNT(*) FROM `{name}`"
            ).fetchone()[0]
            for name in TABLE_COLUMNS
        }

    def test_success_commits_all_tables_and_converts_values(self):
        counts = insert_new_records(
            self.connection,
            self.tables,
            batch_size=1,
        )

        self.assertEqual(counts, {
            "customers": 1,
            "orders": 1,
            "labels": 1,
        })
        self.assertEqual(self.row_counts(), counts)
        self.assertEqual(self.connection.commits, 1)

        order = self.connection.database.execute(
            "SELECT order_value, order_date FROM orders"
        ).fetchone()

        self.assertEqual(order, ("5.387819", "2023-01-01"))

        verified = self.connection.database.execute(
            "SELECT mobile_verified FROM customers"
        ).fetchone()[0]

        self.assertIsNone(verified)

    def test_existing_key_rolls_back_without_overwriting(self):
        insert_new_records(self.connection, self.tables)

        self.tables["customers"].loc[0, "customer_id"] = "c2"
        self.tables["orders"].loc[0, "order_value"] = 999.0
        self.tables["labels"].loc[0, "order_id"] = "o2"
        self.tables["labels"].loc[0, "customer_id"] = "c2"

        with self.assertRaises(sqlite3.IntegrityError):
            insert_new_records(
                self.connection,
                self.tables,
            )

        self.assertEqual(self.row_counts(), {
            "customers": 1,
            "orders": 1,
            "labels": 1,
        })
        self.assertEqual(self.connection.rollbacks, 1)

        stored = self.connection.database.execute(
            "SELECT order_value FROM orders WHERE order_id = 'o1'"
        ).fetchone()[0]

        self.assertEqual(stored, "5.387819")

        customer_ids = self.connection.database.execute(
            "SELECT customer_id FROM customers"
        ).fetchall()

        self.assertEqual(customer_ids, [("c1",)])

    def test_missing_column_is_rejected_before_transaction(self):
        self.tables["labels"] = self.tables["labels"].drop(
            columns=["is_fraud"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "labels: missing columns",
        ):
            insert_new_records(self.connection, self.tables)

        self.assertEqual(self.connection.begins, 0)

    def test_duplicate_input_keys_are_rejected(self):
        self.tables["orders"] = pd.concat(
            [self.tables["orders"], self.tables["orders"]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "orders: duplicate record keys",
        ):
            insert_new_records(self.connection, self.tables)

        self.assertEqual(self.connection.begins, 0)

    def test_empty_batch_does_not_start_transaction(self):
        empty = {
            name: frame.iloc[:0]
            for name, frame in self.tables.items()
        }

        self.assertEqual(
            insert_new_records(self.connection, empty),
            {"customers": 0, "orders": 0, "labels": 0},
        )
        self.assertEqual(self.connection.begins, 0)

    def test_invalid_money_rolls_back_earlier_inserts(self):
        self.tables["orders"].loc[
            0, "order_value"
        ] = 1000000000000.0

        with self.assertRaisesRegex(
            ValueError,
            "outside the RDS money range",
        ):
            insert_new_records(self.connection, self.tables)

        self.assertEqual(self.row_counts(), {
            "customers": 0,
            "orders": 0,
            "labels": 0,
        })
        self.assertEqual(self.connection.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()