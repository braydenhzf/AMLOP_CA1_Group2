
import time
import pandas as pd
from pathlib import Path

from src.data_cleaning import (
    clean_customers_data,
    clean_orders_data,
    clean_labels_data
)
from db.load_data import load_dataframe

DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("customers", "customers_base.csv", clean_customers_data),
    ("orders", "orders_base.csv", clean_orders_data),
    ("labels", "labels_base.csv", clean_labels_data)
]

CHUNK_SIZE = 10000


def main():
    for table_name, filename, cleaning_function in DATASETS:
        print(f"\n=== Loading {table_name} ===")

        filepath = DATA_DIR / filename
        start = time.perf_counter()

        total_raw = 0
        total_cleaned = 0
        total_issues = 0

        issue_path = REPORT_DIR / f"{table_name}_issues.csv"

        if issue_path.exists():
            issue_path.unlink()

        for chunk_number, raw_chunk in enumerate(
            pd.read_csv(
                filepath,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                dtype={"country_code": str}
            ),
            start=1
        ):
            cleaned, issues = cleaning_function(raw_chunk)

            load_dataframe(
                cleaned,
                table_name,
                batch_size=1000
            )

            total_raw += len(raw_chunk)
            total_cleaned += len(cleaned)
            total_issues += len(issues)

            if not issues.empty:
                issues.to_csv(
                    issue_path,
                    mode="a",
                    header=not issue_path.exists(),
                    index=False
                )

            print(
                f"{table_name} chunk {chunk_number}: "
                f"{total_cleaned:,} rows loaded"
            )

        elapsed = time.perf_counter() - start

        print(f"\nFinished {table_name}")
        print(f"Raw rows: {total_raw:,}")
        print(f"Cleaned rows: {total_cleaned:,}")
        print(f"Issue records: {total_issues:,}")
        print(f"Elapsed: {elapsed:.2f} seconds")

    print("\nAll historical datasets processed!")


if __name__ == "__main__":
    main()
