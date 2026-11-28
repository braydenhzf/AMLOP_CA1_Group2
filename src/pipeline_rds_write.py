"""Insert prepared new records without overwriting database history."""

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from src.data_cleaning import (
    CUSTOMER_COLUMNS,
    ORDER_COLUMNS,
    LABEL_COLUMNS,
)


TABLE_COLUMNS = {
    "customers": CUSTOMER_COLUMNS,
    "orders": ORDER_COLUMNS,
    "labels": LABEL_COLUMNS,
}

TABLE_KEYS = {
    "customers": ["country_code", "customer_id"],
    "orders": ["country_code", "order_id"],
    "labels": ["country_code", "order_id"],
}

MONEY_COLUMNS = {
    "total_payment_last_50days",
    "order_value",
    "refund_value",
}


def _database_rows(frame, columns):
    """Convert cleaned pandas values into database-compatible values."""
    rows = []

    for values in frame.loc[:, columns].itertuples(
        index=False,
        name=None,
    ):
        row = []

        for column, value in zip(columns, values):
            if pd.isna(value):
                value = None

            elif column in MONEY_COLUMNS:
                value = Decimal(str(value)).quantize(
                    Decimal("0.000001"),
                    rounding=ROUND_HALF_UP,
                )

                if not Decimal("0") <= value <= Decimal(
                    "999999999999.999999"
                ):
                    raise ValueError(
                        f"{column}: outside the RDS money range."
                    )

            elif isinstance(value, pd.Timestamp):
                value = (
                    value.date()
                    if column == "order_date"
                    else value.to_pydatetime()
                )

            elif hasattr(value, "item"):
                value = value.item()

            row.append(value)

        rows.append(tuple(row))

    return rows


def insert_new_records(connection, new_tables, batch_size=500):
    """Insert already identified new rows using a fresh write connection.

    The caller supplies and closes the connection. This function owns the
    transaction and never updates or replaces existing records.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    if set(new_tables) != set(TABLE_COLUMNS):
        raise ValueError(
            "Supply customers, orders and labels tables."
        )

    counts = {}

    for name, columns in TABLE_COLUMNS.items():
        frame = new_tables[name]

        if not frame.columns.is_unique:
            raise ValueError(f"{name}: duplicate column names.")

        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{name}: missing columns: {missing}")

        keys = TABLE_KEYS[name]

        if frame[keys].isna().any().any():
            raise ValueError(f"{name}: missing record keys.")

        if frame.duplicated(subset=keys).any():
            raise ValueError(f"{name}: duplicate record keys.")

        counts[name] = len(frame)

    if not any(counts.values()):
        return counts

    try:
        connection.begin()

        with connection.cursor() as cursor:
            for name, columns in TABLE_COLUMNS.items():
                columns = list(columns)

                selected_columns = ", ".join(
                    f"`{column}`" for column in columns
                )
                placeholders = ", ".join(
                    ["%s"] * len(columns)
                )

                query = (
                    f"INSERT INTO `{name}` ({selected_columns}) "
                    f"VALUES ({placeholders})"
                )

                frame = new_tables[name]

                for start in range(0, len(frame), batch_size):
                    chunk = frame.iloc[start:start + batch_size]

                    cursor.executemany(
                        query,
                        _database_rows(chunk, columns),
                    )

        connection.commit()

    except BaseException:
        try:
            connection.rollback()
        except Exception:
            # Preserve the original error if the connection also failed.
            pass
        raise

    return counts

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


def select_new_records(history, incoming, keys):
    """Return incoming rows whose record keys do not exist in history."""
    keys = list(keys)

    if not keys:
        raise ValueError("At least one record key is required.")

    if set(history.columns) != set(incoming.columns):
        raise ValueError(
            "Historical and incoming columns must match."
        )

    for name, frame in (("history", history), ("incoming", incoming)):
        if not frame.columns.is_unique:
            raise ValueError(f"{name}: duplicate column names.")

        missing = sorted(set(keys) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{name}: missing key columns: {missing}"
            )

        if frame[keys].isna().any().any():
            raise ValueError(f"{name}: missing record keys.")

        if frame.duplicated(subset=keys).any():
            raise ValueError(f"{name}: duplicate record keys.")

    historical_keys = pd.MultiIndex.from_frame(history[keys])
    incoming_keys = pd.MultiIndex.from_frame(incoming[keys])

    is_new = ~incoming_keys.isin(historical_keys)

    return incoming.loc[is_new].copy().reset_index(drop=True)