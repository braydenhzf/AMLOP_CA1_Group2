"""Reuse accepted history only after full-content validation against live RDS."""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from time import perf_counter

from joblib import dump, load


def source_key():
    from config import get_database_settings
    settings = get_database_settings()
    source = {"host": settings.host, "port": settings.port, "database": settings.database}
    source["code"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_text(encoding="utf-8").encode()).hexdigest()
        for name in ("pipeline_rds.py", "pipeline_rds_cache.py", "pipeline_rds_transport.py", "data_cleaning.py")
    }
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def fingerprints(connection, columns_by_table):
    result = {}
    with connection.cursor() as cursor:
        cursor.execute("SET SESSION group_concat_max_len = 67108864")
        for name, columns in columns_by_table.items():
            selected = ", ".join(f"`{column}`" for column in columns)
            key = "customer_id" if name == "customers" else "order_id"
            query = f"""
                SELECT row_count, COALESCE(OCTET_LENGTH(digests), 0),
                       SHA2(COALESCE(digests, ''), 256)
                FROM (
                    SELECT COUNT(*) AS row_count,
                           GROUP_CONCAT(SHA2(JSON_ARRAY({selected}), 256)
                               ORDER BY country_code, `{key}` SEPARATOR '') AS digests
                    FROM `{name}`
                ) AS fingerprint
            """
            started = perf_counter()
            cursor.execute(query)
            count, length, digest = cursor.fetchone()
            if length != count * 64 or not digest:
                raise ValueError(f"RDS {name}: incomplete fingerprint; cache cannot be trusted.")
            result[name] = {"rows": int(count), "sha256": digest}
            print(f"Live {name} fingerprint verified: {count:,} rows in {perf_counter()-started:.2f}s", flush=True)
    return result







def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_verified_history(cache_dir, *, batch_size=100000, transfer="compressed", report_dir=None):
    from src import pipeline_rds as rds

    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    payload_path = folder / "accepted_history.joblib"
    identity = source_key()
    hit = False
    with rds._read_snapshot() as connection:
        current = fingerprints(connection, rds.TABLE_COLUMNS)
        payload = None
        if manifest_path.is_file() and payload_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload_bytes = payload_path.read_bytes()
                if (manifest["source_key"] == identity and manifest["fingerprints"] == current
                        and manifest["payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest()):
                    candidate = load(BytesIO(payload_bytes))
                    if set(candidate["tables"]) != set(rds.TABLE_COLUMNS):
                        raise ValueError("Cached table set mismatch")
                    for name, fields in rds.TABLE_COLUMNS.items():
                        frame = candidate["tables"][name]
                        if len(frame) != current[name]["rows"] or list(frame.columns) != list(fields):
                            raise ValueError("Cached table dimensions mismatch")
                    payload, hit = candidate, True
            except Exception:
                payload = None
        if payload is None:
            print("History cache MISS: reading and validating the full live snapshot.", flush=True)
            raw = rds.read_rds_history(batch_size=batch_size, transfer=transfer, _connection=connection)
            tables, issues = {}, {}
            for name in rds.TABLE_COLUMNS:
                frame, issues[name] = rds.normalize_rds_table(name, raw.pop(name))
                tables[name] = rds.normalize_rds_precision(name, frame)
            payload = {"tables": tables, "issues": issues}
            identifier = uuid4().hex
            temporary_payload = folder / f"{identifier}.pending.joblib"
            dump(payload, temporary_payload, compress=3)
            manifest = {"source_key": identity, "fingerprints": current,
                        "payload_sha256": file_hash(temporary_payload)}
            os.replace(temporary_payload, payload_path)
            temporary_manifest = folder / f"{identifier}.pending.json"
            temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary_manifest, manifest_path)
        else:
            print("History cache HIT: all three live content fingerprints match.", flush=True)

    receipt = {"hit": hit, "fingerprints": current,
               "policy": "Every table fingerprint is checked in one live snapshot. Changes force a full reload.",
               "cold_cache_note": "Initial builds and refreshes can exceed the three-minute target."}
    if report_dir is not None:
        destination = Path(report_dir) / "rds_history_cache.json"
        destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return payload["tables"], payload["issues"]
