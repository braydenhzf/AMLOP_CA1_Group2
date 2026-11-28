"""Measure a content fingerprint of all orders without transferring their rows."""

from time import perf_counter

import pymysql

from config import get_database_settings
from src.pipeline_rds import TABLE_COLUMNS


def main():
    columns = ", ".join(f"`{name}`" for name in TABLE_COLUMNS["orders"])
    query = f"""
        SELECT row_count,
               COALESCE(OCTET_LENGTH(digests), 0),
               SHA2(COALESCE(digests, ''), 256)
        FROM (
            SELECT COUNT(*) AS row_count,
                   GROUP_CONCAT(
                       SHA2(JSON_ARRAY({columns}), 256)
                       ORDER BY country_code, order_id SEPARATOR ''
                   ) AS digests
            FROM orders
        ) AS fingerprint
    """
    print("Checking the contents of every order; only the checksum is returned...", flush=True)
    with pymysql.connect(
        **get_database_settings().connection_kwargs(), read_timeout=120,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION group_concat_max_len = 67108864")
            started = perf_counter()
            cursor.execute(query)
            count, digest_bytes, fingerprint = cursor.fetchone()
            elapsed = perf_counter() - started
            if digest_bytes != count * 64 or not fingerprint:
                raise RuntimeError("Checksum was truncated; it cannot validate a cache.")
            print(f"Order rows covered: {count:,}")
            print(f"Checksum time: {elapsed:.2f} seconds")
            print("PASS: checksum covers every order row.")
    print("No database records or pipeline code were changed.")


if __name__ == "__main__":
    main()
