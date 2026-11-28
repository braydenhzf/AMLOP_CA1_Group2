"""Run Brayden's analytics flow against an explicitly configured Prefect server."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--include-practice", action="store_true")
    parser.add_argument("--top-customers", type=int, default=20)
    parser.add_argument("--minimum-labelled", type=int, default=30)
    parser.add_argument("--simulate-failure", action="store_true",
                        help="Intentionally fail before reading RDS to demonstrate failure reporting.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reports/individual/brayden-prefect")
    args = parser.parse_args()
    if not os.environ.get("PREFECT_API_URL", "").strip():
        parser.error("Set PREFECT_API_URL to your running server, for example http://127.0.0.1:4200/api.")
    from flows.brayden_analytics_flow import brayden_analytics_flow
    brayden_analytics_flow(str(args.output_dir), start_date=args.start_date, end_date=args.end_date,
                          include_practice=args.include_practice, top_customers=args.top_customers,
                          minimum_labelled=args.minimum_labelled, simulate_failure=args.simulate_failure)


if __name__ == "__main__":
    main()
