
import pandas as pd

from src.data_cleaning import (
    clean_customers_data,
    clean_orders_data,
    clean_labels_data
)

from db.load_data import load_dataframe

datasets = [
    (
        "customers",
        "data/raw/customers_base.csv",
        clean_customers_data
    ),
    (
        "orders",
        "data/raw/orders_base.csv",
        clean_orders_data
    ),
    (
        "labels",
        "data/raw/labels_base.csv",
        clean_labels_data
    )
]

for table_name, filepath, clean_function in datasets:
    print(f"\nProcessing {table_name}...")

    raw = pd.read_csv(
        filepath,
        nrows=10,
        low_memory=False
    )

    cleaned, issues = clean_function(raw)

    print("Original rows:", len(raw))
    print("Cleaned rows:", len(cleaned))
    print("Issues found:", len(issues))

    load_dataframe(
        cleaned,
        table_name,
        batch_size=10
    )

print("\nCleaned data loading test completed!")
