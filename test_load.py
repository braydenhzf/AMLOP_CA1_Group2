

import pandas as pd
from db.load_data import load_dataframe

files = {
    "customers": "data/raw/customers_base.csv",
    "orders": "data/raw/orders_base.csv",
    "labels": "data/raw/labels_base.csv"
}

for table_name, filepath in files.items():
    df = pd.read_csv(
        filepath,
        nrows=10,
        dtype={"country_code": str},
        low_memory=False
    )

    load_dataframe(df, table_name)

print("\nTest loading completed!")
