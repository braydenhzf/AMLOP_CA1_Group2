
import time
import pandas as pd

from src.data_cleaning import (
    clean_customers_data,
    clean_orders_data,
    clean_labels_data
)

from db.load_data import load_dataframe

datasets = [
    ("customers", "customers_base.csv", clean_customers_data),
    ("orders", "orders_base.csv", clean_orders_data),
    ("labels", "labels_base.csv", clean_labels_data)
]

for table_name, filename, clean_function in datasets:
    print(f"\nProcessing {table_name}...")

    start = time.perf_counter()

    raw = pd.read_csv(
        f"data/raw/{filename}",
        nrows=10000,
        low_memory=False
    )

    cleaned, issues = clean_function(raw)

    load_dataframe(
        cleaned,
        table_name,
        batch_size=1000
    )

    elapsed = time.perf_counter() - start

    print(f"Raw rows: {len(raw):,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Issue records: {len(issues):,}")
    print(f"Elapsed seconds: {elapsed:.2f}")

print("\n10,000-record integration test completed!")
