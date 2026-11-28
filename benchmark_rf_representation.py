"""Compare sparse/dense RF speed offline; these benchmark models are not submitted."""

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features import build_preprocessor, chronological_split
from src.models.train_random_forest import load_historical_data


def main():
    root = Path(__file__).resolve().parent
    print("Preparing historical features locally; no cloud services are used...", flush=True)
    X, y, metadata, _ = load_historical_data(root / "data" / "raw")
    development, _ = chronological_split(metadata, evaluation_fraction=0.20)
    train, _ = chronological_split(metadata.loc[development], evaluation_fraction=0.25)
    sample = X.loc[train].sample(n=min(100000, len(train)), random_state=42)
    labels = y.loc[sample.index]
    preprocessor = build_preprocessor()
    sparse = preprocessor.fit_transform(sample).astype("float32")
    dense = sparse.toarray()
    print(f"Benchmark shape: {sparse.shape}; density: {sparse.nnz / (sparse.shape[0] * sparse.shape[1]):.1%}", flush=True)
    timings = {}
    for name, matrix in (("sparse", sparse), ("dense", dense)):
        model = RandomForestClassifier(
            n_estimators=20, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=8, class_weight=None,
        )
        start = perf_counter()
        model.fit(matrix, labels)
        timings[name] = perf_counter() - start
        print(f"{name}: {timings[name]:.2f} seconds", flush=True)
    result = {
        "sample_rows": len(sample), "benchmark_trees": 20,
        "scope": "Diagnostic only; final model tree counts are unchanged.",
        "timings": timings,
        "sparse_divided_by_dense": timings["sparse"] / timings["dense"],
    }
    folder = root / "reports" / "pipeline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rf_representation_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("Benchmark complete; no training code or database records were changed.")


if __name__ == "__main__":
    main()
