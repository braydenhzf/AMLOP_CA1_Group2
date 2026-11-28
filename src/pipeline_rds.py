"""RDS history reconciliation and optional insertion for PBI 10."""

from contextlib import contextmanager, nullcontext

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


def read_rds_history(batch_size=10000, transfer="rows", *, _connection=None):
    """Read every historical row within one consistent, read-only snapshot."""
    from src.pipeline_rds_transport import read_page

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if transfer not in {"rows", "compressed"}:
        raise ValueError("transfer must be rows or compressed.")
    table_keys = {
        "customers": ("country_code", "customer_id"),
        "orders": ("country_code", "order_id"),
        "labels": ("country_code", "order_id"),
    }
    tables = {}
    with (_read_snapshot() if _connection is None else nullcontext(_connection)) as connection:
        with connection.cursor() as cursor:
            for name, columns in TABLE_COLUMNS.items():
                columns = list(columns)
                cursor.execute(f"SELECT COUNT(*) FROM `{name}`")
                expected = int(cursor.fetchone()[0])
                chunks, received, last_key = [], 0, None
                while received < expected:
                    rows, final_key = read_page(
                        cursor, name, columns, table_keys[name],
                        last_key, batch_size, transfer,
                    )
                    if not rows or final_key is None or final_key == last_key:
                        raise ValueError(f"RDS {name}: read stopped before all rows arrived.")
                    if received + len(rows) > expected:
                        raise ValueError(f"RDS {name}: read exceeded snapshot row count.")
                    chunks.append(pd.DataFrame.from_records(rows, columns=columns))
                    received += len(rows)
                    last_key = final_key
                    print(f"RDS {name}: {received:,}/{expected:,} rows ({transfer})", flush=True)
                tables[name] = (
                    pd.concat(chunks, ignore_index=True)
                    if chunks else pd.DataFrame(columns=columns)
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

def reconcile_rds_history(incoming_tables, report_dir, *, write_new=False,
                          batch_size=None, transfer="rows", timings=None, cache_dir=None):
    """Reconcile RDS history and optionally insert only new records."""
    from pathlib import Path
    from src.pipeline_data import combine_accepted_data
    from src.pipeline_rds_write import (
        insert_new_records,
        select_new_records,
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    read_options = {}
    if batch_size is not None:
        read_options["batch_size"] = batch_size
    if transfer != "rows":
        read_options["transfer"] = transfer
    cached_issues = None
    with timings.stage("rds_read_history") if timings else nullcontext():
        if cache_dir is not None:
            from src.pipeline_rds_cache import load_verified_history
            historical_tables, cached_issues = load_verified_history(
                cache_dir, batch_size=batch_size or 100000,
                transfer=transfer, report_dir=report_dir,
            )
        else:
            historical_tables = read_rds_history(**read_options)

    keys = {
        "customers": ["customer_id"],
        "orders": ["country_code", "order_id"],
        "labels": ["country_code", "order_id"],
    }

    combined_tables = {}
    new_tables = {}
    summaries = []

    for name in TABLE_COLUMNS:
        with timings.stage(f"rds_reconcile_{name}") if timings else nullcontext():
            print(
                f"Reconciling RDS and incoming {name}...",
                flush=True,
            )

            if cached_issues is None:
                history, history_issues = normalize_rds_table(
                    name, historical_tables.pop(name),
                )
                history = normalize_rds_precision(name, history)
            else:
                history = historical_tables.pop(name)
                history_issues = cached_issues.pop(name)
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
        with timings.stage("rds_write") if timings else nullcontext():
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
