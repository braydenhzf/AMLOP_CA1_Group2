"""PBI 14: refresh isabelle's individual business analytics with Prefect."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE

from src.analytics.isabelle_analytics import (
    analytics_connection,
    get_analysis_period,
    query_fraud_by_payment,
    query_fraud_by_tenure,
    query_monthly_financial_exposure,
    validate_payment_results,
    validate_financial_results,
    plot_payment_heatmap,
    plot_tenure_fraud,
    plot_financial_exposure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports" / "isabelle_prefect"
TIME_LIMIT = 180
COUNTS = (
    "total_orders", "labelled_orders", "fraud_orders",
    "non_fraud_orders", "unlabelled_orders",
)
EXPECTED_FILES = (
    "q1_payment_fraud.csv", "q1_payment_fraud_heatmap.png",
    "q2_customer_tenure_fraud.csv", "q2_customer_tenure_fraud.png",
    "q3_monthly_financial_exposure.csv", "q3_monthly_financial_exposure.png",
)


@task(name="Retrieve consistent RDS snapshot", cache_policy=NO_CACHE,
      persist_result=False, retries=0, log_prints=True)
def retrieve_analytics_snapshot():
    logger = get_run_logger()
    timings = {}
    with analytics_connection() as conn:
        snapshot_time = datetime.now(timezone.utc).isoformat()
        with conn.cursor() as cursor:
            cursor.execute("SHOW STATUS LIKE 'Ssl_cipher'")
            tls = cursor.fetchone()
            if not tls or not tls["Value"]:
                raise RuntimeError("RDS connection is not using TLS")
        logger.info("TLS verified. Consistent snapshot opened at %s", snapshot_time)
        period = get_analysis_period(conn).iloc[0]
        metadata = {
            "snapshot_opened_utc": snapshot_time,
            "analysis_start": str(period["start_date"]),
            "analysis_end": str(period["end_date"]),
            "total_orders": int(period["total_orders"]),
        }
        logger.info("Analysis period: %s to %s; orders: %s",
                    metadata["analysis_start"], metadata["analysis_end"],
                    metadata["total_orders"])
        results = {}
        for key, query in (
            ("q1", query_fraud_by_payment),
            ("q2", query_fraud_by_tenure),
            ("q3", query_monthly_financial_exposure),
        ):
            started = perf_counter()
            logger.info("Executing %s", key)
            results[key] = query(conn)
            timings[key] = round(perf_counter() - started, 3)
            logger.info("%s completed in %.2fs", key, timings[key])
    metadata["query_seconds"] = timings
    return {**results, "metadata": metadata}


@task(name="Validate and reconcile analytics", cache_policy=NO_CACHE,
      persist_result=False, retries=0, log_prints=True)
def validate_analytics_results(snapshot):
    logger = get_run_logger()
    q1, q2, q3 = (snapshot[k] for k in ("q1", "q2", "q3"))
    expected = snapshot["metadata"]["total_orders"]
    validate_payment_results(q1, expected)
    validate_payment_results(q2, expected)
    validate_financial_results(q3, expected)
    for column in COUNTS:
        if not (q1[column].sum() == q2[column].sum() == q3[column].sum()):
            raise ValueError(f"Cross-query count mismatch: {column}")
    logger.info("PASS: all three query totals match")
    for key in ("q1", "q2", "q3"):
        logger.info("%s results:\n%s", key.upper(), snapshot[key].to_string(index=False))
    return {column: int(q1[column].sum()) for column in COUNTS}


def _write_csv(df, path, metadata):
    # Preserve the original query fields and add report provenance.
    output = df.assign(
        report_run_utc=metadata["run_started_utc"],
        snapshot_opened_utc=metadata["snapshot_opened_utc"],
        analysis_start=metadata["analysis_start"],
        analysis_end=metadata["analysis_end"],
    )
    output.to_csv(path, index=False)


def _generate_report(df, metadata, directory, csv_name, plotter):
    directory = Path(directory)
    _write_csv(df, directory / csv_name, metadata)
    chart = plotter(df, metadata["analysis_start"], metadata["analysis_end"], directory)
    get_run_logger().info("Generated %s and %s", csv_name, Path(chart).name)


@task(name="Generate payment fraud report", cache_policy=NO_CACHE,
      persist_result=False, retries=0)
def generate_payment_report(snapshot, directory):
    _generate_report(snapshot["q1"], snapshot["metadata"], directory,
                     "q1_payment_fraud.csv", plot_payment_heatmap)


@task(name="Generate customer tenure report", cache_policy=NO_CACHE,
      persist_result=False, retries=0)
def generate_tenure_report(snapshot, directory):
    _generate_report(snapshot["q2"], snapshot["metadata"], directory,
                     "q2_customer_tenure_fraud.csv", plot_tenure_fraud)


@task(name="Generate financial exposure report", cache_policy=NO_CACHE,
      persist_result=False, retries=0)
def generate_financial_report(snapshot, directory):
    _generate_report(snapshot["q3"], snapshot["metadata"], directory,
                     "q3_monthly_financial_exposure.csv", plot_financial_exposure)


@task(name="Publish complete reports", cache_policy=NO_CACHE,
      persist_result=False, retries=0)
def publish_reports(staging, destination, metadata, counts, started):
    logger = get_run_logger()
    staging, destination = Path(staging), Path(destination)
    for filename in EXPECTED_FILES:
        path = staging / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty report: {filename}")
    elapsed = perf_counter() - started
    if elapsed >= TIME_LIMIT:
        raise TimeoutError(f"Report preparation exceeded {TIME_LIMIT}s")
    manifest = {
        **metadata,
        "counts": counts,
        "status": "validated",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds_before_publication": round(elapsed, 3),
        "files": list(EXPECTED_FILES),
    }
    (staging / "report_metadata.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    # Rename the complete directory on the same filesystem, then atomically
    # replace the latest pointer. Never overwrite an earlier report folder.
    pointer = destination.parent / "latest.json"
    temp_pointer = destination.parent / f".latest-{destination.name}.tmp"
    try:
        temp_pointer.write_text(json.dumps({
            "report_directory": destination.name,
            "metadata_file": f"{destination.name}/report_metadata.json",
            "run_started_utc": metadata["run_started_utc"],
            "analysis_start": metadata["analysis_start"],
            "analysis_end": metadata["analysis_end"],
        }, indent=2), encoding="utf-8")
        staging.rename(destination)
        temp_pointer.replace(pointer)
    finally:
        temp_pointer.unlink(missing_ok=True)
    logger.info("Published six reports: %s", destination)
    logger.info("Latest successful report reference: %s", pointer)
    return str(destination)


@flow(name="Isabelle - Refresh Fraud Business Analytics", retries=0,
      persist_result=False, log_prints=True)
def refresh_fraud_analytics():
    """Refresh reports from a single snapshot and publish only validated output."""
    logger = get_run_logger()
    started = perf_counter()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = REPORT_ROOT / f".staging-{run_id}"
    destination = REPORT_ROOT / run_id
    staging.mkdir()
    try:
        snapshot = retrieve_analytics_snapshot()
        snapshot["metadata"]["run_started_utc"] = now.isoformat()
        snapshot["metadata"]["run_id"] = run_id
        counts = validate_analytics_results(snapshot)
        generate_payment_report(snapshot, staging)
        generate_tenure_report(snapshot, staging)
        generate_financial_report(snapshot, staging)
        output = publish_reports(staging, destination, snapshot["metadata"], counts, started)
    except Exception:
        logger.exception("Refresh failed; unpublished staging files will be removed.")
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    elapsed = perf_counter() - started
    logger.info("Full flow body execution time: %.2f seconds", elapsed)
    if elapsed > TIME_LIMIT:
        logger.warning("Time target exceeded; inspect Prefect run duration.")
    else:
        logger.info("Flow body finished within %s seconds", TIME_LIMIT)
    logger.info("Reports: %s", output)
    return output


if __name__ == "__main__":
    refresh_fraud_analytics()
