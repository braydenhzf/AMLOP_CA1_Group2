
import pandas as pd
from src.data_cleaning import clean_orders_data

raw = pd.read_csv(
    "data/raw/orders_base.csv",
    nrows=1000
)

cleaned, issues = clean_orders_data(raw)

problem_ids = issues.loc[
    issues["reason"] == "refund_exceeds_order",
    "order_id"
].dropna().unique()[:5]

print("=== ORIGINAL VALUES ===")
print(
    raw.loc[
        raw["order_id"].isin(problem_ids),
        ["order_id", "order_value", "refund_value"]
    ].to_string(index=False)
)

print("\n=== CLEANED VALUES ===")
print(
    cleaned.loc[
        cleaned["order_id"].isin(problem_ids),
        ["order_id", "order_value", "refund_value"]
    ].to_string(index=False)
)

print("\n=== ISSUE SUMMARY ===")
print(issues["reason"].value_counts().to_string())
