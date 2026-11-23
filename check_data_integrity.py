
import pandas as pd
from pathlib import Path

DATA = Path("data/raw")

customers = pd.read_csv(
    DATA / "customers_base.csv",
    dtype={"country_code": str, "customer_id": str},
    low_memory=False
)

orders = pd.read_csv(
    DATA / "orders_base.csv",
    dtype={"country_code": str, "order_id": str},
    low_memory=False
)

labels = pd.read_csv(
    DATA / "labels_base.csv",
    dtype={
        "country_code": str,
        "order_id": str,
        "customer_id": str
    },
    low_memory=False
)

print("=== DUPLICATE PRIMARY KEYS ===")

keys = {
    "customers": (
        customers, ["country_code", "customer_id"]
    ),
    "orders": (
        orders, ["country_code", "order_id"]
    ),
    "labels": (
        labels, ["country_code", "order_id"]
    )
}

for name, (df, key_cols) in keys.items():
    count = df.duplicated(subset=key_cols).sum()
    print(f"{name}: {count} duplicate keys")

print("\n=== UNMATCHED RECORDS ===")

order_keys = pd.MultiIndex.from_frame(
    orders[["country_code", "order_id"]]
)

label_order_keys = pd.MultiIndex.from_frame(
    labels[["country_code", "order_id"]]
)

unmatched_labels = (
    ~label_order_keys.isin(order_keys)
).sum()

unmatched_orders = (
    ~order_keys.isin(label_order_keys)
).sum()

print("Labels without matching orders:",
      unmatched_labels)
print("Orders without matching labels:",
      unmatched_orders)

customer_keys = pd.MultiIndex.from_frame(
    customers[["country_code", "customer_id"]]
)

label_customer_keys = pd.MultiIndex.from_frame(
    labels[["country_code", "customer_id"]]
)

unmatched_customers = (
    ~label_customer_keys.isin(customer_keys)
).sum()

print("Labels without matching customers:",
      unmatched_customers)

print("\n=== MOBILE VERIFIED VALUES ===")
print(
    customers["mobile_verified"]
    .value_counts(dropna=False)
    .head(20)
)

print("\n=== FRAUD LABEL VALUES ===")
print(labels["is_fraud"].value_counts(dropna=False))
