"""Manually verify live RDS inserts using session-only temporary tables.

Run from the project root. This is not an automatic unittest: it contacts RDS.
All three temporary tables must be verified before any INSERT is attempted.
The normal historical tables are shadowed only for this connection.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pymysql

from config import get_database_settings
from src.pipeline_rds_write import (
    TABLE_COLUMNS,
    TABLE_KEYS,
    insert_new_records,
    select_new_records,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_batch():
    rows = {
        "customers": [[
            "SG", "pbi10-live-c1", None, 2, 0, 0, 12.123456789, 0,
            pd.Timestamp("2022-01-01"),
        ]],
        "orders": [[
            "SG", "pbi10-live-o1", "delivery", "card",
            5.387818896776103, 2, 0, pd.Timestamp("2023-01-01"),
        ]],
        "labels": [["SG", "pbi10-live-o1", "pbi10-live-c1", 1]],
    }
    return {
        name: pd.DataFrame(rows[name], columns=columns)
        for name, columns in TABLE_COLUMNS.items()
    }


def read_tables(connection):
    tables = {}
    with connection.cursor() as cursor:
        for name, columns in TABLE_COLUMNS.items():
            selected = ", ".join(f"`{column}`" for column in columns)
            cursor.execute(f"SELECT {selected} FROM `{name}`")
            tables[name] = pd.DataFrame(cursor.fetchall(), columns=columns)
    return tables


def verify(connection):
    names = tuple(TABLE_COLUMNS)
    definitions = {}
    with connection.cursor() as cursor:
        # Capture each real schema before creating any temporary shadows.
        for name in names:
            cursor.execute(f"SHOW CREATE TABLE `{name}`")
            definition = cursor.fetchone()[1]
            prefix = f"CREATE TABLE `{name}`"
            require(definition.startswith(prefix), f"Unexpected schema: {name}")
            require("ENGINE=INNODB" in definition.upper(), f"{name}: InnoDB required")
            definitions[name] = definition.replace(
                "CREATE TABLE", "CREATE TEMPORARY TABLE", 1
            )

        for name in names:
            cursor.execute(definitions[name])

        # Fail closed: all three names must now resolve to empty temp tables.
        for name in names:
            cursor.execute(f"SHOW CREATE TABLE `{name}`")
            definition = cursor.fetchone()[1]
            require(
                definition.startswith(f"CREATE TEMPORARY TABLE `{name}`"),
                f"Temporary table verification failed: {name}",
            )
            cursor.execute(f"SELECT COUNT(*) FROM `{name}`")
            require(cursor.fetchone()[0] == 0, f"Temporary {name} is not empty")

    print("PASS: all three isolated temporary tables verified.", flush=True)
    batch = make_batch()
    inserted = insert_new_records(connection, batch, batch_size=1)
    require(inserted == dict.fromkeys(names, 1), "Incorrect insert receipt")
    stored = read_tables(connection)
    require(all(len(frame) == 1 for frame in stored.values()), "Insert readback failed")
    require(
        stored["orders"].loc[0, "order_value"] == Decimal("5.387819"),
        "Money precision check failed",
    )
    print("PASS: committed and read back one record per table.", flush=True)

    new_tables = {
        name: select_new_records(
            stored[name], batch[name],
            ["customer_id"] if name == "customers" else TABLE_KEYS[name],
        )
        for name in names
    }
    repeated = insert_new_records(connection, new_tables)
    require(repeated == dict.fromkeys(names, 0), "Repeated batch selected new records")
    after_repeat = read_tables(connection)
    for name in names:
        pd.testing.assert_frame_equal(stored[name], after_repeat[name])
    print("PASS: repeated batch inserted zero records; stored values unchanged.", flush=True)

    # Insert a new customer, then deliberately collide with the existing order.
    # The writer must roll back the earlier customer insert as well.
    conflicting = {name: frame.copy(deep=True) for name, frame in batch.items()}
    conflicting["customers"].loc[0, "customer_id"] = "pbi10-live-c2"
    conflicting["orders"].loc[0, "order_value"] = 999.0
    conflicting["labels"].loc[0, "order_id"] = "pbi10-live-o2"
    conflicting["labels"].loc[0, "customer_id"] = "pbi10-live-c2"
    try:
        insert_new_records(connection, conflicting, batch_size=1)
    except pymysql.err.IntegrityError as exc:
        require(exc.args[0] == 1062, "Expected duplicate-key error 1062")
    else:
        raise RuntimeError("Duplicate order was unexpectedly accepted")
    after_failure = read_tables(connection)
    for name in names:
        pd.testing.assert_frame_equal(stored[name], after_failure[name])
    print("PASS: duplicate-key failure rolled back all earlier inserts.", flush=True)
    return {"first_insert": inserted, "repeat_insert": repeated, "rollback_verified": True}


def main():
    evidence = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Live RDS writer and planner; session-only temporary tables",
        "data": "Synthetic verification rows; no historical table inserts",
        "status": "FAILED",
    }
    report_dir = Path(__file__).resolve().parent / "reports" / "pipeline"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"live-rds-insert-{uuid4().hex[:12]}.json"
    connection = None
    try:
        connection = pymysql.connect(
            **get_database_settings().connection_kwargs(),
            autocommit=False, read_timeout=30, write_timeout=30,
        )
        evidence.update(verify(connection))
        evidence["status"] = "PASSED"
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if connection is not None:
                connection.close()
        finally:
            evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            print(f"Verification receipt: {report_path}", flush=True)
    print("SUCCESS: live RDS temporary-table verification passed.", flush=True)


if __name__ == "__main__":
    main()
