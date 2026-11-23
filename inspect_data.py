
import pandas as pd
from pathlib import Path

files = [
    "customers_base.csv",
    "orders_base.csv",
    "labels_base.csv"
]

for filename in files:
    path = Path("data/raw") / filename

    if not path.exists():
        print(f"\nMISSING: {path}")
        continue

    df = pd.read_csv(path)

    print(f"\n{'=' * 50}")
    print(f"FILE: {filename}")
    print(f"ROWS: {len(df)}")
    print(f"COLUMNS: {list(df.columns)}")
    print("\nDATA TYPES:")
    print(df.dtypes.to_string())
    print("\nMISSING VALUES:")
    print(df.isnull().sum().to_string())
    print("\nFIRST 3 ROWS:")
    print(df.head(3).to_string(index=False))
