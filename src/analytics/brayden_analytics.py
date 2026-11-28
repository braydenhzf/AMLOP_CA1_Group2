"""Brayden's three order-based analyses of the current RDS snapshot.

Only SELECT statements run against business tables. All joins use the full
(country_code, order_id) key. Rates exclude orders without a binary fraud
label. Value of fraud-labelled orders describes exposure, not confirmed loss.
"""
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
import html
import json
import math
import os
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

QUESTIONS = {
    "order_value": "How does fraud risk and fraud-labelled order value vary across order-value bands?",
    "weekday_collection": "Which weekday and collection-method combinations have elevated fraud rates?",
    "customer_concentration": "Which customers account for the most fraud-labelled orders, and how concentrated are those orders?",
}
VALUE_BANDS = ["Under SGD 20", "SGD 20 to under 50", "SGD 50 to under 100",
               "SGD 100 to under 200", "SGD 200 to under 500", "SGD 500 and above",
               "Missing value", "Invalid negative"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Missing date"]
COUNTS = """
    COUNT(*) AS orders_total,
    SUM(CASE WHEN l.is_fraud IN (0, 1) THEN 1 ELSE 0 END) AS labelled_orders,
    SUM(CASE WHEN l.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_orders,
    COALESCE(SUM(CASE WHEN l.is_fraud IN (0, 1) THEN o.order_value ELSE 0 END), 0) AS labelled_order_value,
    COALESCE(SUM(CASE WHEN l.is_fraud = 1 THEN o.order_value ELSE 0 END), 0) AS fraud_order_value
"""
JOIN = "FROM orders o LEFT JOIN labels l ON o.country_code = l.country_code AND o.order_id = l.order_id"
VALUE_CASE = """CASE WHEN o.order_value IS NULL THEN 'Missing value'
    WHEN o.order_value < 0 THEN 'Invalid negative'
    WHEN o.order_value < 20 THEN 'Under SGD 20'
    WHEN o.order_value < 50 THEN 'SGD 20 to under 50'
    WHEN o.order_value < 100 THEN 'SGD 50 to under 100'
    WHEN o.order_value < 200 THEN 'SGD 100 to under 200'
    WHEN o.order_value < 500 THEN 'SGD 200 to under 500'
    ELSE 'SGD 500 and above' END"""
DAY_CASE = """CASE WEEKDAY(o.order_date)
    WHEN 0 THEN 'Monday' WHEN 1 THEN 'Tuesday' WHEN 2 THEN 'Wednesday'
    WHEN 3 THEN 'Thursday' WHEN 4 THEN 'Friday' WHEN 5 THEN 'Saturday'
    WHEN 6 THEN 'Sunday' ELSE 'Missing date' END"""


def _where(start_date=None, end_date=None, include_practice=False):
    conditions, parameters = [], []
    for value in (start_date, end_date):
        if value is not None:
            date.fromisoformat(str(value))
    if start_date and end_date and str(start_date) > str(end_date):
        raise ValueError("Start date must not be after end date.")
    if not include_practice:
        conditions.append("o.order_id NOT LIKE %s")
        parameters.append("pbi12-%")
    if start_date:
        conditions.append("o.order_date >= %s")
        parameters.append(str(start_date))
    if end_date:
        conditions.append("o.order_date <= %s")
        parameters.append(str(end_date))
    return " AND ".join(conditions) or "1 = 1", parameters


def _query(connection, sql, parameters):
    with connection.cursor() as cursor:
        cursor.execute(sql, tuple(parameters))
        columns = [item[0] for item in cursor.description]
        return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)


@contextmanager
def analytics_snapshot():
    import pymysql
    from config import get_database_settings
    with pymysql.connect(**get_database_settings().connection_kwargs(), read_timeout=90, write_timeout=30) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
        try:
            yield connection
        finally:
            if connection.open:
                connection.rollback()


def read_queries(connection, *, start_date=None, end_date=None, include_practice=False, top_customers=20):
    """Run all summaries through the supplied single-snapshot connection."""
    if type(top_customers) is not int or not 1 <= top_customers <= 100:
        raise ValueError("top_customers must be between 1 and 100.")
    where, parameters = _where(start_date, end_date, include_practice)
    started = perf_counter()
    metadata = _query(connection, f"""SELECT {COUNTS},
        MIN(o.order_date) AS first_order_date, MAX(o.order_date) AS last_order_date,
        SUM(CASE WHEN l.is_fraud IS NOT NULL AND l.is_fraud NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_label_orders,
        SUM(CASE WHEN l.is_fraud IN (0, 1) AND o.order_value IS NULL THEN 1 ELSE 0 END) AS labelled_orders_missing_value,
        SUM(CASE WHEN l.is_fraud = 1 AND o.order_value IS NULL THEN 1 ELSE 0 END) AS fraud_orders_missing_value
        {JOIN} WHERE {where}""", parameters).iloc[0].to_dict()
    excluded_where, excluded_parameters = _where(start_date, end_date, True)
    excluded = _query(connection, f"""SELECT COUNT(*) AS practice_orders
        FROM orders o WHERE {excluded_where} AND o.order_id LIKE %s""",
        [*excluded_parameters, "pbi12-%"]).iloc[0, 0]
    q1 = _query(connection, f"""SELECT o.country_code, {VALUE_CASE} AS value_band, {COUNTS}
        {JOIN} WHERE {where} GROUP BY o.country_code, {VALUE_CASE}""", parameters)
    q2 = _query(connection, f"""SELECT {DAY_CASE} AS weekday,
        COALESCE(o.collect_type, 'Missing collection method') AS collect_type, {COUNTS}
        {JOIN} WHERE {where} GROUP BY {DAY_CASE}, COALESCE(o.collect_type, 'Missing collection method')""", parameters)
    q3 = _query(connection, f"""SELECT l.customer_id, COUNT(*) AS labelled_orders,
        SUM(CASE WHEN l.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_orders,
        COALESCE(SUM(CASE WHEN l.is_fraud = 1 THEN o.order_value ELSE 0 END), 0) AS fraud_order_value
        {JOIN} WHERE {where} AND l.is_fraud IN (0, 1)
            AND l.customer_id IS NOT NULL AND l.customer_id <> ''
        GROUP BY l.customer_id HAVING SUM(CASE WHEN l.is_fraud = 1 THEN 1 ELSE 0 END) > 0
        ORDER BY fraud_orders DESC, fraud_order_value DESC, l.customer_id LIMIT %s""",
        [*parameters, top_customers])
    for name in ("orders_total", "labelled_orders", "fraud_orders", "invalid_label_orders",
                 "labelled_orders_missing_value", "fraud_orders_missing_value"):
        metadata[name] = 0 if pd.isna(metadata[name]) else int(metadata[name])
    for name in ("labelled_order_value", "fraud_order_value"):
        metadata[name] = float(metadata[name] or 0)
    for name in ("first_order_date", "last_order_date"):
        metadata[name] = None if pd.isna(metadata[name]) else str(metadata[name])
    metadata.update({"include_practice": include_practice, "practice_orders_in_date_scope": int(excluded),
                     "excluded_practice_orders": 0 if include_practice else int(excluded),
                     "requested_start_date": start_date, "requested_end_date": end_date,
                     "unlabelled_or_invalid_orders": metadata["orders_total"] - metadata["labelled_orders"],
                     "query_seconds": perf_counter() - started})
    return {"metadata": metadata, "tables": {"order_value": q1, "weekday_collection": q2, "customer_concentration": q3}}


def fetch_current_analysis(**options):
    with analytics_snapshot() as connection:
        captured = datetime.now(timezone.utc).isoformat()
        result = read_queries(connection, **options)
    result["metadata"]["snapshot_started_utc"] = captured
    return result


def add_rates(frame, minimum_labelled=30):
    """Fraud-labelled distinct orders / distinct orders with a binary label."""
    frame = frame.copy()
    for name in ("orders_total", "labelled_orders", "fraud_orders"):
        if name in frame:
            values = pd.to_numeric(frame[name], errors="raise")
            if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
                raise ValueError(f"Invalid order counts: {name}")
            frame[name] = values.astype("int64")
    for name in ("labelled_order_value", "fraud_order_value"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    n, fraud = frame["labelled_orders"], frame["fraud_orders"]
    if (fraud > n).any() or ("orders_total" in frame and (n > frame["orders_total"]).any()):
        raise ValueError("Fraud counts must not exceed labelled counts or total orders.")
    denominator = n.where(n > 0)
    rate = fraud / denominator
    z = 1.959963984540054
    adjustment = 1 + z * z / denominator
    centre = (rate + z * z / (2 * denominator)) / adjustment
    half = z * np.sqrt(rate * (1 - rate) / denominator + z * z / (4 * denominator * denominator)) / adjustment
    frame["fraud_rate"] = rate
    frame["fraud_rate_lower_95"] = (centre - half).clip(lower=0)
    frame["fraud_rate_upper_95"] = (centre + half).clip(upper=1)
    frame["low_volume"] = n < minimum_labelled
    return frame


def validate_analysis(payload, minimum_labelled=30):
    if type(minimum_labelled) is not int or minimum_labelled < 1:
        raise ValueError("minimum_labelled must be positive.")
    metadata = payload["metadata"]
    tables = {name: add_rates(table, minimum_labelled) for name, table in payload["tables"].items()}
    for name in ("order_value", "weekday_collection"):
        table = tables[name]
        for column in ("orders_total", "labelled_orders", "fraud_orders"):
            if int(table[column].sum()) != metadata[column]:
                raise ValueError(f"{name}: {column} does not match snapshot totals.")
        for column in ("labelled_order_value", "fraud_order_value"):
            if not math.isclose(float(table[column].sum()), metadata[column], rel_tol=1e-10, abs_tol=0.01):
                raise ValueError(f"{name}: {column} does not match snapshot totals.")
    customers = tables["customer_concentration"]
    if customers.customer_id.duplicated().any() or customers.fraud_orders.sum() > metadata["fraud_orders"]:
        raise ValueError("Invalid customer concentration totals or duplicate customers.")
    customers["cumulative_fraud_share"] = customers.fraud_orders.cumsum() / (metadata["fraud_orders"] or np.nan)
    metadata = {**metadata, "minimum_labelled_for_comparisons": minimum_labelled,
                "overall_fraud_rate": (metadata["fraud_orders"] / metadata["labelled_orders"]
                                       if metadata["labelled_orders"] else None)}
    return {"metadata": metadata, "tables": tables}


def analyse_order_value(payload):
    table = payload["tables"]["order_value"]
    counts = table.groupby("value_band", sort=False)[["orders_total", "labelled_orders", "fraud_orders",
                                                       "labelled_order_value", "fraud_order_value"]].sum().reset_index()
    counts = add_rates(counts, payload["metadata"]["minimum_labelled_for_comparisons"])
    counts["value_band"] = pd.Categorical(counts.value_band, categories=VALUE_BANDS, ordered=True)
    counts = counts.sort_values("value_band").reset_index(drop=True)
    eligible = counts.loc[~counts.low_volume & counts.fraud_rate.notna()]
    if eligible.empty:
        insight = "No order-value band meets the labelled-order threshold for a supported rate comparison."
    else:
        row = eligible.loc[eligible.fraud_rate.idxmax()]
        insight = (f"{row.value_band} has the highest observed fraud rate among supported value bands: "
                   f"{row.fraud_rate:.1%} ({row.fraud_orders:,}/{row.labelled_orders:,} labelled orders). "
                   "Compare its order volume and value exposure before allocating review effort.")
    return {"name": "order_value", "question": QUESTIONS["order_value"], "table": table,
            "plot_table": counts, "insight": insight,
            "limitation": "Value bands are fixed operational groups. Country mix may affect global comparisons; the CSV retains country detail. Order value is not confirmed fraud loss."}


def analyse_weekday_collection(payload):
    table = payload["tables"]["weekday_collection"]
    eligible = table.loc[~table.low_volume & table.fraud_rate.notna()]
    if eligible.empty:
        insight = "No weekday/collection group meets the labelled-order threshold for a supported rate comparison."
    else:
        row = eligible.loc[eligible.fraud_rate.idxmax()]
        insight = (f"{row.weekday}, {row.collect_type}, has the highest observed rate among supported groups: "
                   f"{row.fraud_rate:.1%} ({row.fraud_orders:,}/{row.labelled_orders:,} labelled orders). "
                   "Use the pattern to plan investigation coverage alongside order volume.")
    return {"name": "weekday_collection", "question": QUESTIONS["weekday_collection"], "table": table,
            "plot_table": table, "insight": insight,
            "limitation": "Order dates support weekday analysis, not hour-of-day analysis. These associations do not establish causation or account for every country/customer difference."}


def analyse_customer_concentration(payload):
    table = payload["tables"]["customer_concentration"]
    if table.empty:
        insight = "No customers with fraud-labelled orders are available in this analysis scope."
    else:
        share = table.cumulative_fraud_share.iloc[-1]
        insight = (f"The top {len(table)} customers account for {int(table.fraud_orders.sum()):,} fraud-labelled "
                   f"orders, or {share:.1%} of all fraud-labelled orders in scope. "
                   "Prioritise investigation using repeated fraud counts together with each customer's labelled-order volume.")
    return {"name": "customer_concentration", "question": QUESTIONS["customer_concentration"], "table": table,
            "plot_table": table, "insight": insight,
            "limitation": "Ranked by fraud-labelled order count, then fraud-labelled order value. A high rate based on very few orders is uncertain. This is an investigation list, not an automated customer decision."}


def create_chart(result, metadata, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    table = result["plot_table"]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    title = {"order_value": "Fraud rate by order value", "weekday_collection": "Fraud rate by weekday and collection method",
             "customer_concentration": "Customers with the most fraud-labelled orders"}[result["name"]]
    if table.empty or table.labelled_orders.sum() == 0:
        ax.text(.5, .5, "No valid labelled orders in this scope", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    elif result["name"] == "order_value":
        rates = table.fraud_rate.to_numpy(dtype=float) * 100
        positions = np.arange(len(table))
        errors = np.vstack([rates - table.fraud_rate_lower_95.to_numpy(dtype=float) * 100,
                            table.fraud_rate_upper_95.to_numpy(dtype=float) * 100 - rates])
        ax.bar(positions, rates, color=["#8a9aa8" if low else "#236b8e" for low in table.low_volume])
        ax.errorbar(positions, rates, yerr=np.maximum(errors, 0), fmt="none", color="black", capsize=4)
        ax.set_xticks(positions, [f"{band}\nn={n:,}" for band, n in zip(table.value_band, table.labelled_orders)], rotation=20, ha="right")
        ax.set_ylabel("Fraud-labelled orders / valid labelled orders (%)")
        ax.set_xlabel("Error bars: Wilson 95% intervals; grey: fewer than the configured labelled-order minimum")
    elif result["name"] == "weekday_collection":
        rates = table.pivot(index="weekday", columns="collect_type", values="fraud_rate").reindex(WEEKDAYS).dropna(how="all") * 100
        counts = table.pivot(index="weekday", columns="collect_type", values="labelled_orders").reindex(rates.index)
        annotations = rates.copy().astype(object)
        for row in rates.index:
            for column in rates.columns:
                annotations.loc[row, column] = (f"{rates.loc[row, column]:.1f}%\nn={int(counts.loc[row, column]):,}"
                                                if pd.notna(rates.loc[row, column]) else "")
        sns.heatmap(rates, annot=annotations, fmt="", mask=rates.isna(), cmap="YlOrRd", vmin=0,
                    cbar_kws={"label": "Fraud rate (%)"}, ax=ax)
        ax.set_xlabel("Collection method")
        ax.set_ylabel("Weekday")
    else:
        shown = table.head(10).iloc[::-1]
        ax.barh(shown.customer_id.astype(str), shown.fraud_orders, color="#236b8e")
        ax.set_xlabel("Fraud-labelled order count (up to ten customers shown)")
        ax.set_ylabel("Customer ID")
    period = f"{metadata['first_order_date'] or 'unavailable'} to {metadata['last_order_date'] or 'unavailable'}"
    ax.set_title(title + "\n" + period, loc="left", pad=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def publish_report(payload, results, output_root, *, started=None, run_context=None):
    """Publish a new report only when all three analyses and charts succeed."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = "brayden-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    final = output_root / run_id
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=output_root) as folder:
        staging = Path(folder)
        fragments = []
        for result in results:
            result["table"].to_csv(staging / f"{result['name']}.csv", index=False)
            create_chart(result, payload["metadata"], staging / f"{result['name']}.png")
            table = result["table"].head(20).copy()
            for column in ("fraud_rate", "fraud_rate_lower_95", "fraud_rate_upper_95", "cumulative_fraud_share"):
                if column in table:
                    table[column] = table[column].map(lambda value: "Unavailable" if pd.isna(value) else f"{value:.1%}")
            fragments.append(f"<section><h2>{html.escape(result['question'])}</h2><p>{html.escape(result['insight'])}</p>"
                             f"<img src='{result['name']}.png' alt='{html.escape(result['question'], quote=True)}'>"
                             f"<p>{html.escape(result['limitation'])}</p><div class='table'>{table.to_html(index=False, escape=True)}</div>"
                             f"<p><a href='{result['name']}.csv'>Download complete query results</a></p></section>")
        metadata = payload["metadata"]
        summary = {"run_id": run_id, "status": "FINISHED", "metadata": metadata,
                   "orchestration": run_context,
                   "elapsed_seconds": perf_counter() - started if started is not None else None,
                   "analyses": [{"name": r["name"], "question": r["question"], "insight": r["insight"], "limitation": r["limitation"]} for r in results]}
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        overall = metadata["overall_fraud_rate"]
        overall_text = "Unavailable" if overall is None else f"{overall:.2%}"
        header = (f"<h1>Brayden fraud analytics</h1><p>Run: {html.escape(run_id)} | Snapshot start (UTC): {html.escape(metadata['snapshot_started_utc'])}</p>"
                  f"<p>Analysis period: {metadata['first_order_date']} to {metadata['last_order_date']}. "
                  f"Orders: {metadata['orders_total']:,}; valid labelled orders: {metadata['labelled_orders']:,}; "
                  f"fraud-labelled orders: {metadata['fraud_orders']:,}; fraud rate: {overall_text}.</p>"
                  f"<p>Unlabelled or invalid-label orders excluded from rate denominators: {metadata['unlabelled_or_invalid_orders']:,}. "
                  f"Known synthetic practice orders excluded: {metadata['excluded_practice_orders']:,}.</p>"
                  "<p>Only binary labels 0 and 1 enter fraud-rate denominators. Unavailable rates are not zero. "
                  "Fraud-labelled order values describe exposure, not confirmed financial loss.</p>"
                  f"<p>Value sums cover recorded amounts only. Labelled orders missing a value: {metadata['labelled_orders_missing_value']:,}; "
                  f"fraud-labelled orders missing a value: {metadata['fraud_orders_missing_value']:,}.</p>")
        document = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Brayden fraud analytics</title><style>"
        document += "body{font:16px system-ui;margin:35px auto;max-width:1200px;padding:0 20px;color:#24303d}section{margin:40px 0}img{max-width:100%}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border:1px solid #ccd3d9;text-align:left}th{background:#eaf0f4}</style><body>"
        (staging / "index.html").write_text(document + header + "".join(fragments) + "</body></html>", encoding="utf-8")
        staging.rename(final)
    pointer = output_root / f".latest-{uuid4().hex}.json"
    pointer.write_text(json.dumps({"run_id": run_id, "report": f"{run_id}/index.html"}, indent=2) + "\n", encoding="utf-8")
    os.replace(pointer, output_root / "latest.json")
    return final, summary


def run_analytics(output_root, **options):
    started = perf_counter()
    minimum_labelled = options.pop("minimum_labelled", 30)
    payload = validate_analysis(fetch_current_analysis(**options), minimum_labelled)
    results = [analyse_order_value(payload), analyse_weekday_collection(payload), analyse_customer_concentration(payload)]
    final, summary = publish_report(payload, results, output_root, started=started)
    for result in results:
        print("\n" + result["question"], flush=True)
        print(result["table"].head(20).to_string(index=False), flush=True)
        print(result["insight"], flush=True)
    print(f"\nReport: {final / 'index.html'}", flush=True)
    print(f"Execution time: {summary['elapsed_seconds']:.2f} seconds", flush=True)
    return final, summary
