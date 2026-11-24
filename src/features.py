"""Validate data relationships and prepare predictors for fraud models."""

import numpy as np
import pandas as pd

from .data_cleaning import CUSTOMER_KEY, ORDER_KEY


NUMERIC_FEATURES = [
    "order_value",
    "num_items_ordered",
    "value_per_item",
    "customer_tenure_days",
]
CATEGORICAL_FEATURES = [
    "country_code",
    "collect_type",
    "payment_method",
    "order_weekday",
    "order_month",
]


def _check_join_key(frame, keys, table):
    if frame[keys].isna().any().any():
        raise ValueError(f"{table} has missing keys. Run data cleaning before joining.")


def _eligible_for_training(joined):
    return (
        joined["label_match"].eq("both")
        & joined["is_fraud"].isin([0, 1])
        & joined["customer_match"].eq("both")
        & joined["order_date"].notna()
    )


def join_training_data(clean_customers, clean_orders, clean_labels):
    """Keep one row per order, attach labels and customer details, and audit matches.

    Look up the customer ID in the label table using the country and order ID.
    Match the customer using both country and customer ID. Orders with missing
    labels or customers stay in the joined data for inspection.
    """
    _check_join_key(clean_orders, ORDER_KEY, "Orders")
    _check_join_key(clean_labels, ORDER_KEY, "Labels")
    _check_join_key(clean_customers, CUSTOMER_KEY, "Customers")

    joined = clean_orders.merge(
        clean_labels,
        how="left",
        on=ORDER_KEY,
        validate="one_to_one",
        indicator="label_match",
    ).merge(
        clean_customers,
        how="left",
        on=CUSTOMER_KEY,
        validate="many_to_one",
        indicator="customer_match",
    )
    if len(joined) != len(clean_orders):
        raise ValueError("The joins changed the number of orders.")

    label_match = joined["label_match"].eq("both")
    customer_match = joined["customer_match"].eq("both")
    matched_labels = int(label_match.sum())
    matched_customers = int(customer_match.sum())
    first_order_day = joined["first_order_datetime"].dt.normalize()
    audit = {
        "orders": len(clean_orders),
        "joined_orders": len(joined),
        "matched_labels": matched_labels,
        "unmatched_labels": len(joined) - matched_labels,
        "orphan_labels": len(clean_labels) - matched_labels,
        "matched_customers": matched_customers,
        "unmatched_customers": len(joined) - matched_customers,
        "labelled_without_customer": int((label_match & ~customer_match).sum()),
        "orders_without_date": int(joined["order_date"].isna().sum()),
        "first_order_after_current_order": int(
            first_order_day.gt(joined["order_date"].dt.normalize()).sum()
        ),
        "eligible_training_orders": int(_eligible_for_training(joined).sum()),
    }
    return joined, audit


def make_features(frame):
    """Build predictors using order details and the customer's first-order date.

    A new order can use this function once its customer is known and those same
    input fields have been cleaned. Derive tenure from calendar days, since the
    current order has no time of day. Treat an impossible negative tenure as
    missing. Leave out amounts refunded on the order and current customer
    snapshots whose values at review time cannot be established.
    """
    order_day = frame["order_date"].dt.normalize()
    first_order_day = frame["first_order_datetime"].dt.normalize()
    order_value = frame["order_value"].astype("float64")
    item_count = frame["num_items_ordered"].astype("float64")
    tenure = (order_day - first_order_day).dt.days.astype("float64")

    features = pd.DataFrame(index=frame.index)
    features["order_value"] = order_value
    features["num_items_ordered"] = item_count
    features["value_per_item"] = order_value / item_count.where(item_count.gt(0))
    features["customer_tenure_days"] = tenure.where(tenure.ge(0))
    for column in ["country_code", "collect_type", "payment_method"]:
        features[column] = frame[column].astype("string").to_numpy(
            dtype=object, na_value=np.nan
        )
    features["order_weekday"] = order_day.dt.day_name().to_numpy(
        dtype=object, na_value=np.nan
    )
    features["order_month"] = order_day.dt.month_name().to_numpy(
        dtype=object, na_value=np.nan
    )
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def prepare_training_data(joined):
    """Return aligned predictors, binary targets and metadata for splitting.

    Require a valid fraud label, a matched customer and a valid order date.
    Identifiers and the original date stay available in metadata for evaluation.
    """
    eligible = joined.loc[_eligible_for_training(joined)]
    X = make_features(eligible)
    y = eligible["is_fraud"].astype("int8")
    metadata = eligible[["country_code", "order_id", "customer_id", "order_date"]].copy()
    return X, y, metadata


def chronological_split(metadata, evaluation_fraction=0.2):
    """Hold out the newest days, with each date assigned entirely to one split.

    Choose the date boundary closest to the requested fraction of rows.
    Returned indexes can select rows from the aligned X and y with .loc.
    """
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be strictly between 0 and 1.")
    if not metadata.index.is_unique:
        raise ValueError("Splitting requires unique row indexes.")
    order_days = metadata["order_date"].dt.normalize()
    if order_days.isna().any():
        raise ValueError("Splitting requires a valid order date on every row.")

    orders_per_day = order_days.value_counts().sort_index()
    if len(orders_per_day) < 2:
        raise ValueError("A chronological split requires orders on at least two dates.")
    possible_training_counts = orders_per_day.cumsum().iloc[:-1]
    target_training_count = len(metadata) * (1 - evaluation_fraction)
    last_training_day = (possible_training_counts - target_training_count).abs().idxmin()
    training_rows = order_days.le(last_training_day)
    return metadata.index[training_rows], metadata.index[~training_rows]


def build_preprocessor():
    """Return an unfitted transformer for scikit-learn pipelines (version 1.2+).

    Fit on the training split only. Median imputation and scaling are learned
    from training data. log1p compresses the long right tails while preserving
    legitimate extremes. Numeric missingness remains available to the model
    when observed during fitting. One-hot encoding accepts unseen categories.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median", keep_empty_features=True, add_indicator=True
                ),
            ),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    category_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value="__missing__", keep_empty_features=True
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("category", category_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )
