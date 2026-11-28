"""RDS history reconciliation and optional insertion for PBI 10."""

from contextlib import contextmanager

import pandas as pd
import pymysql

from config import get_database_settings
from src.data_cleaning import (
    CUSTOMER_COLUMNS,
    ORDER_COLUMNS,
    LABEL_COLUMNS,
)
from src.data_cleaning import (
    clean_customers_data,
    clean_orders_data,
    clean_labels_data,
)


TABLE_COLUMNS = {
    "customers": CUSTOMER_COLUMNS,
    "orders": ORDER_COLUMNS,
    "labels": LABEL_COLUMNS,
}


@contextmanager
def _read_snapshot():
    """Keep all reads in one read-only, repeatable-read transaction."""
    settings = get_database_settings()

    with pymysql.connect(
        **settings.connection_kwargs(),
        read_timeout=30,
        write_timeout=30,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cursor.execute(
                "START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY"
            )

        try:
            yield connection
        finally:
            if connection.open:
                connection.rollback()


def get_rds_row_counts():
    """Return table counts without downloading the full datasets."""
    counts = {}

    with _read_snapshot() as connection:
        with connection.cursor() as cursor:
            for table_name in TABLE_COLUMNS:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                counts[table_name] = int(cursor.fetchone()[0])

    return counts


def read_rds_history(batch_size=10000):
    """Read all historical records in smaller queries within one snapshot."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    table_keys = {
        "customers": ("country_code", "customer_id"),
        "orders": ("country_code", "order_id"),
        "labels": ("country_code", "order_id"),
    }
    tables = {}

    with _read_snapshot() as connection:
        with connection.cursor() as cursor:
            for table_name, columns in TABLE_COLUMNS.items():
                columns = list(columns)
                first_key, second_key = table_keys[table_name]

                selected_columns = ", ".join(
                    f"`{column}`" for column in columns
                )

                cursor.execute(
                    f"SELECT COUNT(*) FROM `{table_name}`"
                )
                expected_rows = int(cursor.fetchone()[0])

                chunks = []
                received = 0
                last_key = None

                while received < expected_rows:
                    query = (
                        f"SELECT {selected_columns} "
                        f"FROM `{table_name}`"
                    )
                    parameters = []

                    if last_key is not None:
                        query += (
                            f" WHERE (`{first_key}` > %s)"
                            f" OR (`{first_key}` = %s"
                            f" AND `{second_key}` > %s)"
                        )
                        parameters.extend([
                            last_key[0],
                            last_key[0],
                            last_key[1],
                        ])

                    query += (
                        f" ORDER BY `{first_key}`, `{second_key}`"
                        " LIMIT %s"
                    )
                    parameters.append(batch_size)

                    cursor.execute(query, tuple(parameters))
                    rows = cursor.fetchall()

                    if not rows:
                        raise ValueError(
                            f"RDS {table_name}: read stopped "
                            "before all rows arrived."
                        )

                    chunks.append(
                        pd.DataFrame.from_records(
                            rows,
                            columns=columns,
                        )
                    )
                    received += len(rows)

                    last_row = rows[-1]
                    last_key = (
                        last_row[columns.index(first_key)],
                        last_row[columns.index(second_key)],
                    )

                    print(
                        f"RDS {table_name}: "
                        f"{received:,}/{expected_rows:,} rows",
                        flush=True,
                    )

                if chunks:
                    tables[table_name] = pd.concat(
                        chunks,
                        ignore_index=True,
                    )
                else:
                    tables[table_name] = pd.DataFrame(
                        columns=columns,
                    )

    return tables

def normalize_rds_table(table_name, frame):
    """Normalize database values using the team's cleaning rules."""
    cleaners = {
        "customers": clean_customers_data,
        "orders": clean_orders_data,
        "labels": clean_labels_data,
    }

    if table_name not in cleaners:
        raise ValueError(f"Unknown historical table: {table_name}")

    source = frame.copy()

    if table_name == "customers":
        values = source["mobile_verified"]

        valid = values.isna() | values.isin([0, 1])
        if not valid.all():
            raise ValueError(
                "RDS customers contains an invalid mobile_verified value."
            )

        source["mobile_verified"] = values.map({
            0: "false",
            1: "true",
        })

    cleaned, issues = cleaners[table_name](source)

    # Historical records should already have valid, unique identifiers.
    # Fail rather than silently removing records used by reference splits.
    if len(cleaned) != len(source):
        raise ValueError(
            f"RDS {table_name}: normalization excluded historical records. "
            "Review the stored identifiers and duplicates before continuing."
        )

    return cleaned, issues

def reconcile_rds_history(incoming_tables, report_dir, *, write_new=False):
    """Reconcile RDS history and optionally insert only new records."""
    from pathlib import Path
    from src.pipeline_data import combine_accepted_data
    from src.pipeline_rds_write import (
        insert_new_records,
        select_new_records,
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    historical_tables = read_rds_history()

    keys = {
        "customers": ["customer_id"],
        "orders": ["country_code", "order_id"],
        "labels": ["country_code", "order_id"],
    }

    combined_tables = {}
    new_tables = {}
    summaries = []

    for name in TABLE_COLUMNS:
        print(
            f"Reconciling RDS and incoming {name}...",
            flush=True,
        )

        history, history_issues = normalize_rds_table(
            name,
            historical_tables.pop(name),
        )

        history = normalize_rds_precision(name, history)
        incoming = normalize_rds_precision(
            name,
            incoming_tables[name],
        )

        combined, conflicts, counts = combine_accepted_data(
            history,
            incoming,
            keys[name],
        )

        new_tables[name] = select_new_records(
            history,
            incoming,
            keys[name],
        )

        combined_tables[name] = combined
        summaries.append({"dataset": name, **counts})

        history_issues.to_csv(
            report_dir / f"{name}_rds_history_issues.csv",
            index=False,
        )

        conflicts.to_csv(
            report_dir / f"{name}_rds_history_conflicts.csv",
            index=False,
        )

    reconciliation_summary = pd.DataFrame(summaries)

    reconciliation_summary.to_csv(
        report_dir / "rds_reconciliation_summary.csv",
        index=False,
    )

    if write_new:
        planned_counts = {
            name: len(frame)
            for name, frame in new_tables.items()
        }

        if any(planned_counts.values()):
            settings = get_database_settings()

            with pymysql.connect(
                **settings.connection_kwargs(),
                read_timeout=30,
                write_timeout=30,
                autocommit=False,
            ) as connection:
                inserted_counts = insert_new_records(
                    connection,
                    new_tables,
                )
        else:
            inserted_counts = planned_counts

        insertion_summary = pd.DataFrame([
            {
                "dataset": name,
                "inserted_rows": count,
            }
            for name, count in inserted_counts.items()
        ])

        insertion_summary.to_csv(
            report_dir / "rds_insert_summary.csv",
            index=False,
        )

        print(insertion_summary.to_string(index=False))

    return combined_tables, reconciliation_summary

def normalize_rds_precision(table_name, frame):
    """Match money fields to the current RDS DECIMAL(18, 6) schema."""
    from decimal import Decimal, ROUND_HALF_UP

    money_columns = {
        "customers": ("total_payment_last_50days",),
        "orders": ("order_value", "refund_value"),
        "labels": (),
    }

    if table_name not in money_columns:
        raise ValueError(f"Unknown historical table: {table_name}")

    result = frame.copy()
    quantum = Decimal("0.000001")

    def round_money(value):
        if pd.isna(value):
            return pd.NA

        return float(
            Decimal(str(value)).quantize(
                quantum,
                rounding=ROUND_HALF_UP,
            )
        )

    for column in money_columns[table_name]:
        result[column] = (
            result[column]
            .map(round_money)
            .astype("Float64")
        )

    return result