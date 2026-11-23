
import sys
from pathlib import Path

# Allow imports from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from db.connection import get_engine


def test_connection():
    engine = get_engine()

    with engine.connect() as conn:
        db_name = conn.execute(
            text("SELECT DATABASE()")
        ).scalar()

    assert db_name == "spbuy"
    print("PASS: Database connection")


def test_tables():
    engine = get_engine()

    with engine.connect() as conn:
        tables = conn.execute(
            text("SHOW TABLES")
        ).fetchall()

    table_names = {row[0] for row in tables}

    expected = {"customers", "orders", "labels"}

    assert expected.issubset(table_names)
    print("PASS: Required tables exist")


def test_record_counts():
    engine = get_engine()

    with engine.connect() as conn:
        for table in ["customers", "orders", "labels"]:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()

            print(f"{table}: {count:,} records")
            assert count > 0

    print("PASS: Tables contain records")


def test_fraud_labels():
    engine = get_engine()

    with engine.connect() as conn:
        invalid_count = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM labels
                WHERE is_fraud NOT IN (0, 1)
                   OR is_fraud IS NULL
            """)
        ).scalar()

    assert invalid_count == 0
    print("PASS: Fraud labels valid")


if __name__ == "__main__":
    test_connection()
    test_tables()
    test_record_counts()
    test_fraud_labels()

    print("\nAll RDS verification tests passed!")
