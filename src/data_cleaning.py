"""Clean SP-buy customer, order and fraud label tables for any input batch.

Each public function returns (cleaned, issues). The cleaned table has the
canonical input columns and a fresh row index. Each issue identifies the
one-based source data row, excluding the CSV header, and the affected key.
One source row can have multiple issues. Compare source and clean row counts
to see how many records were excluded.
"""

import pandas as pd


CUSTOMER_KEY = ["country_code", "customer_id"]
ORDER_KEY = ["country_code", "order_id"]

CUSTOMER_COUNT_COLUMNS = (
    "num_orders_last_50days",
    "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days",
    "num_associated_customers",
)
CUSTOMER_COLUMNS = (
    *CUSTOMER_KEY,
    "mobile_verified",
    *CUSTOMER_COUNT_COLUMNS[:3],
    "total_payment_last_50days",
    "num_associated_customers",
    "first_order_datetime",
)
ORDER_COLUMNS = (
    *ORDER_KEY,
    "collect_type",
    "payment_method",
    "order_value",
    "num_items_ordered",
    "refund_value",
    "order_date",
)
LABEL_COLUMNS = (*ORDER_KEY, "customer_id", "is_fraud")
ISSUE_COLUMNS = (
    "source_row", "country_code", "order_id", "customer_id", "column", "reason", "raw_value"
)

_TRUE_FALSE = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
_MONTHS = dict(zip(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(),
    (f"{month:02d}" for month in range(1, 13)),
))


def _trimmed(values):
    """Preserve missing values and treat blank strings as missing."""
    text = values.astype("string").str.strip()
    return text.mask(text.eq(""))


def _parse_dates(values, formats):
    text = _trimmed(values)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for date_format in formats:
        pending = text[parsed.isna() & text.notna()]
        if pending.empty:
            break
        parsed.loc[pending.index] = pd.to_datetime(pending, format=date_format, errors="coerce")
    return parsed


class _BatchCleaner:
    """Keep the source row positions intact while standardizing and auditing."""

    def __init__(self, frame, table, columns, identifiers):
        if not frame.columns.is_unique:
            raise ValueError(f"{table} has duplicate column names")
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")
        self.raw = frame.loc[:, list(columns)].reset_index(drop=True)
        self.clean = pd.DataFrame(index=self.raw.index, columns=list(columns))
        self.issues = []
        self.clean["country_code"] = _trimmed(self.raw["country_code"]).str.upper()
        for identifier in identifiers:
            self.clean[identifier] = _trimmed(self.raw[identifier])

        country = self.clean["country_code"]
        self.flag(country.isna(), "missing_required", "country_code")
        invalid = country.notna() & ~country.str.fullmatch(r"[A-Z]{2}", na=False)
        self.flag(invalid, "invalid_country_code", "country_code")
        self.clean["country_code"] = country.mask(invalid)
        for identifier in identifiers:
            self.flag(self.clean[identifier].isna(), "missing_required", identifier)

    def flag(self, mask, reason, column=None):
        rows = mask.index[mask.fillna(False)]
        if rows.empty:
            return
        issues = pd.DataFrame({"source_row": rows + 1}, index=rows)
        for key in ("country_code", "order_id", "customer_id"):
            issues[key] = (
                self.clean.loc[rows, key] if key in self.clean
                else pd.Series(pd.NA, index=rows, dtype="string")
            )
        issues["column"] = pd.Series(column, index=rows, dtype="string")
        issues["reason"] = pd.Series(reason, index=rows, dtype="string")
        issues["raw_value"] = (
            self.raw.loc[rows, column].astype("string") if column is not None
            else pd.Series(pd.NA, index=rows, dtype="string")
        )
        self.issues.append(issues)

    def category(self, column):
        text = _trimmed(self.raw[column]).str.lower().str.replace(r"\s+", " ", regex=True)
        self.flag(text.isna(), "missing_optional", column)
        self.clean[column] = text

    def flag_missing_and_invalid(self, column, raw, parsed, invalid_reason):
        self.flag(raw.isna(), "missing_optional", column)
        self.flag(raw.notna() & parsed.isna(), invalid_reason, column)

    def boolean(self, column):
        text = _trimmed(self.raw[column]).str.lower()
        parsed = text.map(_TRUE_FALSE).astype("boolean")
        self.flag_missing_and_invalid(column, text, parsed, "invalid_boolean")
        self.clean[column] = parsed

    def numeric(self, column, integer=False):
        raw = self.raw[column]
        if not pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
            raw = _trimmed(raw)
        parsed = pd.to_numeric(raw, errors="coerce")
        valid = parsed.ge(0) & parsed.lt(float("inf"))
        if integer:
            valid &= parsed.lt(2**63) & parsed.mod(1).eq(0)
        parsed = parsed.where(valid).astype("Int64" if integer else "Float64")
        self.flag_missing_and_invalid(column, raw, parsed, "invalid_number")
        self.clean[column] = parsed

    def dates(self, column, values, formats):
        raw = _trimmed(self.raw[column])
        parsed = _parse_dates(values, formats)
        self.flag_missing_and_invalid(column, raw, parsed, "invalid_date")
        self.clean[column] = parsed

    def finish(self, eligible, key):
        candidates = self.clean.loc[eligible]
        duplicates = candidates.duplicated(keep="first")
        self.flag(duplicates, "duplicate_record")
        unique_rows = candidates.loc[~duplicates]
        # A repeated key is safe only if all normalized field values agreed.
        conflicting = unique_rows.duplicated(subset=key, keep=False)
        self.flag(conflicting, "conflicting_key")
        results = unique_rows.loc[~conflicting].reset_index(drop=True)
        if self.issues:
            issues = pd.concat(self.issues, ignore_index=True)
            issues = issues.sort_values("source_row", kind="stable", ignore_index=True)
        else:
            issues = pd.DataFrame({
                column: pd.Series(dtype="int64" if column == "source_row" else "string")
                for column in ISSUE_COLUMNS
            })
        return results, issues


def clean_customers_data(customers):
    """Standardize customer records and quarantine conflicting customer IDs.

    Customer IDs are globally unique in the data specification. A customer
    join additionally checks the country, which should agree with the order.
    """
    batch = _BatchCleaner(customers, "customers", CUSTOMER_COLUMNS, ["customer_id"])
    batch.boolean("mobile_verified")
    for column in CUSTOMER_COUNT_COLUMNS:
        batch.numeric(column, integer=True)
    batch.numeric("total_payment_last_50days")
    batch.dates(
        "first_order_datetime", batch.raw["first_order_datetime"],
        ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"),
    )

    clean = batch.clean
    for column in ("num_cancelled_orders_last_50days", "num_refund_orders_last_50days"):
        batch.flag(clean[column].gt(clean["num_orders_last_50days"]), "history_count_exceeds_orders", column)
    return batch.finish(clean[CUSTOMER_KEY].notna().all(axis=1), ["customer_id"])


def clean_orders_data(orders):
    """Standardize orders and preserve refunds for later business analysis."""
    batch = _BatchCleaner(orders, "orders", ORDER_COLUMNS, ["order_id"])
    batch.category("collect_type")
    batch.category("payment_method")
    for column in ("order_value", "refund_value"):
        batch.numeric(column)
    batch.numeric("num_items_ordered", integer=True)

    # Translate the English month names independently of the computer locale.
    dates = _trimmed(batch.raw["order_date"]).str.replace(
        r"(?<=/)([A-Za-z]{3})(?=/)",
        lambda match: _MONTHS.get(match[1].lower(), match[1]),
        regex=True,
    )
    batch.dates("order_date", dates, ("%Y/%m/%d", "%Y-%m-%d"))
    clean = batch.clean
    batch.flag(clean["refund_value"].gt(clean["order_value"]), "refund_exceeds_order", "refund_value")
    return batch.finish(clean[ORDER_KEY].notna().all(axis=1), ORDER_KEY)


def clean_labels_data(labels):
    """Accept only 0/1 targets and quarantine records with conflicting labels."""
    batch = _BatchCleaner(labels, "labels", LABEL_COLUMNS, ["order_id", "customer_id"])
    text = _trimmed(batch.raw["is_fraud"])
    # Accept exact 0/1, including 0.0/1.0 from a CSV read as floats.
    valid_label = text.str.fullmatch(r"[01](?:\.0+)?", na=False)
    batch.flag(text.isna(), "missing_required", "is_fraud")
    batch.flag(text.notna() & ~valid_label, "invalid_fraud_label", "is_fraud")
    clean = batch.clean
    clean["is_fraud"] = text.str[0].where(valid_label).astype("Int8")

    has_order_key = clean[ORDER_KEY].notna().all(axis=1)
    valid_record = valid_label & clean["customer_id"].notna()
    eligible = has_order_key & valid_record
    # If a valid row has a malformed sibling for the same order, quarantine
    # the key until the source supplies an unambiguous target and customer.
    malformed = clean.loc[has_order_key & ~valid_record, ORDER_KEY]
    if not malformed.empty:
        malformed_keys = pd.MultiIndex.from_frame(malformed)
        blocked = pd.Series(pd.MultiIndex.from_frame(clean[ORDER_KEY]).isin(malformed_keys), index=clean.index)
        batch.flag(eligible & blocked, "blocked_label_key")
        eligible &= ~blocked
    return batch.finish(eligible, ORDER_KEY)
