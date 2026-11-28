"""Read bounded RDS pages, optionally compressed by MySQL before transfer.

The compressed option requires MySQL 8 with CTE/JSON/COMPRESS support.
It uses SELECT only and the caller's existing consistent snapshot. Every
row is transferred; this is not a sample or a local historical cache.
"""

import json
import re
import struct
import zlib
from decimal import Decimal


def read_page(cursor, table, columns, keys, last_key, batch_size, transfer="rows"):
    columns, keys = list(columns), list(keys)
    identifiers = [table, *columns, *keys]
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in identifiers):
        raise ValueError("Invalid SQL identifier in RDS page definition.")
    if len(keys) != 2 or not set(keys).issubset(columns):
        raise ValueError("RDS paging requires two selected key columns.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if transfer not in {"rows", "compressed"}:
        raise ValueError("transfer must be rows or compressed.")
    first, second = keys
    selected = ", ".join(f"`{column}`" for column in columns)
    query = f"SELECT {selected} FROM `{table}`"
    parameters = []
    if last_key is not None:
        query += f" WHERE (`{first}` > %s) OR (`{first}` = %s AND `{second}` > %s)"
        parameters.extend([last_key[0], last_key[0], last_key[1]])
    query += f" ORDER BY `{first}`, `{second}` LIMIT %s"
    parameters.append(batch_size)

    if transfer == "rows":
        cursor.execute(query, tuple(parameters))
        rows = list(cursor.fetchall())
        final = (
            (rows[-1][columns.index(first)], rows[-1][columns.index(second)])
            if rows else None
        )
        return rows, final

    # JSON_ARRAYAGG has no ordering guarantee. Rank the bounded page using
    # MySQL's collation, then restore that exact order after decompression.
    packed_query = (
        f"WITH `_pipeline_page` AS ({query}) "
        f"SELECT COMPRESS(JSON_ARRAYAGG(JSON_ARRAY(`_pipeline_row`, {selected}))), COUNT(*) "
        f"FROM (SELECT ROW_NUMBER() OVER (ORDER BY `{first}`, `{second}`) "
        f"AS `_pipeline_row`, {selected} FROM `_pipeline_page`) AS `_pipeline_ordered_page`"
    )
    cursor.execute(packed_query, tuple(parameters))
    payload, count = cursor.fetchone()
    count = int(count)
    if count == 0:
        return [], None
    if payload is None or len(payload) < 5:
        raise ValueError("MySQL compressed transfer returned no valid payload; use --rds-transfer rows.")
    payload = bytes(payload)
    unpacked = zlib.decompress(payload[4:])
    if len(unpacked) != struct.unpack("<I", payload[:4])[0]:
        raise ValueError("RDS compressed payload length mismatch.")
    rows = json.loads(unpacked.decode("utf-8"), parse_float=Decimal)
    if not isinstance(rows, list) or len(rows) != count or count > batch_size:
        raise ValueError("RDS compressed payload row count mismatch.")
    if any(not isinstance(row, list) or len(row) != len(columns) + 1 for row in rows):
        raise ValueError("RDS compressed payload column count mismatch.")
    if any(type(row[0]) is not int for row in rows):
        raise ValueError("RDS compressed page has invalid row ordinals.")
    rows.sort(key=lambda row: row[0])
    if any(row[0] != number for number, row in enumerate(rows, 1)):
        raise ValueError("RDS compressed page has missing or duplicate row ordinals.")
    rows = [row[1:] for row in rows]
    final = (rows[-1][columns.index(first)], rows[-1][columns.index(second)])
    return rows, final
