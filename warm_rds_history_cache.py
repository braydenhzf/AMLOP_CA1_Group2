"""One-time setup/refresh of the verified history cache; no training or RDS writes."""

from pathlib import Path
from time import perf_counter
import json

from src.pipeline_rds_cache import load_verified_history


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    reports = root / "reports/pipeline/history-cache-setup"
    reports.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    tables, _ = load_verified_history(root / "data/raw/cache/pbi12", report_dir=reports)
    elapsed = perf_counter() - started
    for name, frame in tables.items():
        print(f"Cached {name}: {len(frame):,} accepted records")
    (reports / "setup_timing.json").write_text(json.dumps({
        "setup_elapsed_seconds": elapsed,
        "included_in_subsequent_batch_timer": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"History setup/refresh took {elapsed:.2f} seconds.")
    print("Verified history setup finished. Cold setup time is separate from the subsequent batch run.")
