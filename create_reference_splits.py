"""Create fixed split membership from historical base data only."""

import argparse
from pathlib import Path

from src.models.train_random_forest import load_historical_data
from src.pipeline_splits import make_reference_splits


def main():
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Create reference splits from historical base CSVs."
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=project_root / "data" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "reports" / "pipeline" / "reference_splits.csv",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error(
            "The reference file already exists. It will not be overwritten."
        )

    print("Preparing historical reference data...", flush=True)

    features, labels, metadata, audit = load_historical_data(
        args.history_dir
    )
    del features, labels

    reference = make_reference_splits(metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting an existing reference.
    with args.output.open("x", encoding="utf-8", newline="") as file:
        reference.to_csv(file, index=False)

    counts = reference["split"].value_counts().reindex(
        ["train", "validation", "test"]
    )

    print(counts.to_string())
    print(f"Reference saved to: {args.output}")
    print("No models were trained and no cloud services were contacted.")


if __name__ == "__main__":
    main()