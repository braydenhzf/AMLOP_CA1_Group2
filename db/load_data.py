
import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.mysql import insert
from db.connection import get_engine

TABLE_KEYS = {
    "customers": ["country_code", "customer_id"],
    "orders": ["country_code", "order_id"],
    "labels": ["country_code", "order_id"]
}


def load_dataframe(df, table_name, batch_size=500):
    """Insert or update records in an existing RDS table."""

    if table_name not in TABLE_KEYS:
        raise ValueError(f"Unknown table: {table_name}")

    engine = get_engine()
    metadata = MetaData()

    table = Table(
        table_name,
        metadata,
        autoload_with=engine
    )

    expected_columns = [c.name for c in table.columns]

    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for {table_name}: {missing}"
        )

    data = df[expected_columns].copy()

    keys = TABLE_KEYS[table_name]

    if data[keys].isna().any().any():
        raise ValueError("Primary key contains null values")

    if data.duplicated(subset=keys).any():
        raise ValueError("Duplicate keys in input batch")

    # Database-format conversion only.
    # Business cleaning should be done by Vivian's pipeline.
    if table_name == "customers":
        values = (
            data["mobile_verified"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False
        }

        invalid = values.notna() & ~values.isin(mapping)
        if invalid.any():
            raise ValueError("Unexpected mobile_verified value")

        data["mobile_verified"] = values.map(mapping)

        data["first_order_datetime"] = pd.to_datetime(
            data["first_order_datetime"],
            errors="raise"
        )

    elif table_name == "orders":
        data["order_date"] = pd.to_datetime(
            data["order_date"],
            errors="raise"
        )

    total = len(data)

    with engine.begin() as conn:
        for start in range(0, total, batch_size):
            chunk = data.iloc[start:start + batch_size]

            # Convert pandas missing values into SQL NULL.
            chunk = chunk.astype(object)
            chunk = chunk.where(pd.notna(chunk), None)
            records = chunk.to_dict("records")

            if not records:
                continue

            stmt = insert(table).values(records)

            updates = {
                col: stmt.inserted[col]
                for col in expected_columns
                if col not in keys
            }

            stmt = stmt.on_duplicate_key_update(**updates)
            conn.execute(stmt)

    print(f"{table_name}: processed {total} rows")
