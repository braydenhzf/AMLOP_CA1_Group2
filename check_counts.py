
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

with engine.connect() as conn:
    for table in ["customers", "orders", "labels"]:
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        )
        print(f"{table}: {result.scalar()} rows")
