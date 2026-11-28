"""Run Brayden's PBI 13 analyses against the current shared RDS database."""
import argparse
from pathlib import Path

from src.analytics.brayden_analytics import run_analytics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", help="Optional inclusive order-date bound, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Optional inclusive order-date bound, YYYY-MM-DD.")
    parser.add_argument("--include-practice", action="store_true", help="Include synthetic pbi12- practice order IDs.")
    parser.add_argument("--top-customers", type=int, default=20)
    parser.add_argument("--minimum-labelled", type=int, default=30)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reports/individual/brayden")
    args = parser.parse_args()
    try:
        run_analytics(args.output_dir, start_date=args.start_date, end_date=args.end_date,
                      include_practice=args.include_practice, top_customers=args.top_customers,
                      minimum_labelled=args.minimum_labelled)
    except Exception as exc:
        parser.exit(1, f"Analytics failed ({type(exc).__name__}): {exc}\nPrevious published reports were not refreshed.\n")


if __name__ == "__main__":
    main()
