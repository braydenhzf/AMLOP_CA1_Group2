
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

for table_name, filename, cleaning_function in datasets:
    print(f"\nProcessing {table_name}...")

    df = pd.read_csv(
        f"data/raw/{filename}",
        nrows=1000,
        low_memory=False
    )

    cleaned, issues = cleaning_function(df)

    print("Raw rows:", len(df))
    print("Cleaned rows:", len(cleaned))
    print("Issues:", len(issues))

    if len(issues) > 0:
        print("\nIssue report columns:")
        print(issues.columns.tolist())

        print("\nFirst 10 issue records:")
        print(issues.head(10).to_string(index=False))


    load_dataframe(
        cleaned,
        table_name,
        batch_size=500
    )

print("\n1,000-record integration test completed!")
