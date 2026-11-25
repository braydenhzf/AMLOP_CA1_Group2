import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

from src.data_cleaning import (
    CUSTOMER_COLUMNS,
    ORDER_COLUMNS,
    LABEL_COLUMNS,
)

REQUIRED_COLUMNS = {
    "customers": CUSTOMER_COLUMNS,
    "orders": ORDER_COLUMNS,
    "labels": LABEL_COLUMNS,
}


def validate_csv(file_path, dataset_name):
    """Check CSV structure and required columns; return the data-row count."""
    required = REQUIRED_COLUMNS[dataset_name]
    row_count = 0

    try:
        with Path(file_path).open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.reader(file, strict=True)
            header = next(reader, None)

            if not header:
                raise ValueError(
                    f"{dataset_name}: the CSV is empty or has no header."
                )

            if len(header) != len(set(header)):
                raise ValueError(
                    f"{dataset_name}: duplicate column names in the header."
                )

            missing = sorted(set(required) - set(header))
            if missing:
                raise ValueError(
                    f"{dataset_name}: missing required columns: "
                    + ", ".join(missing)
                )

            for row in reader:
                # Ignore completely blank lines.
                if not row:
                    continue

                if len(row) != len(header):
                    raise ValueError(
                        f"{dataset_name}: line {reader.line_num} has "
                        f"{len(row)} fields; expected {len(header)}."
                    )

                row_count += 1

    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(
            f"{dataset_name}: unable to read a valid UTF-8 CSV: {exc}"
        ) from exc

    return row_count

def download_csv(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=15) as response:
        contents = response.read()

    destination.write_bytes(contents)
    return destination


def download_batch(customers_url, orders_url, labels_url, output_dir):
    output_dir = Path(output_dir)

    urls = {
        "customers": customers_url,
        "orders": orders_url,
        "labels": labels_url,
    }

    saved_paths = {}

        # Download all three files.
    for name, url in urls.items():
        destination = output_dir / f"{name}_practice.csv"
        try:
            saved_paths[name] = download_csv(url, destination)
        except OSError as exc:
            raise OSError(f"{name}: download failed: {exc}") from exc

    # Validate every file before returning the batch.
    for name, file_path in saved_paths.items():
        row_count = validate_csv(file_path, name)
        print(f"{name}: validated {row_count:,} data rows")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a batch of three CSV files."
    )
    parser.add_argument("--customers-url", required=True)
    parser.add_argument("--orders-url", required=True)
    parser.add_argument("--labels-url", required=True)
    args = parser.parse_args()

    try:
        saved_paths = download_batch(
            args.customers_url,
            args.orders_url,
            args.labels_url,
            Path("data/raw/incoming"),
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Batch failed: {exc}\n")

    for name, path in saved_paths.items():
        print(f"{name}: saved to {path.resolve()}")