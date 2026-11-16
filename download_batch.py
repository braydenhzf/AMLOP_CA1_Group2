import argparse
from pathlib import Path
from urllib.request import urlopen


def download_csv(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=15) as response:
        contents = response.read()

    destination.write_bytes(contents)
    return destination


def download_batch(customers_url, orders_url, labels_url, output_dir):
    urls = {
        "customers": customers_url,
        "orders": orders_url,
        "labels": labels_url,
    }

    saved_paths = {}
    for name, url in urls.items():
        destination = output_dir / f"{name}_practice.csv"
        saved_paths[name] = download_csv(url, destination)

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a batch of three CSV files.")
    parser.add_argument("--customers-url", required=True)
    parser.add_argument("--orders-url", required=True)
    parser.add_argument("--labels-url", required=True)
    args = parser.parse_args()

    saved_paths = download_batch(
        args.customers_url,
        args.orders_url,
        args.labels_url,
        Path("data/raw/incoming"),
    )

    for name, path in saved_paths.items():
        print(f"{name}: saved to {path.resolve()}")