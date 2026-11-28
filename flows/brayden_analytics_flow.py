"""PBI 14: refresh Brayden's three analyses from one current RDS snapshot."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from src.analytics.brayden_analytics import (
    fetch_current_analysis, validate_analysis, analyse_order_value,
    analyse_weekday_collection, analyse_customer_concentration, publish_report,
)

TASK_OPTIONS = dict(cache_policy=NO_CACHE, persist_result=False, retries=0, log_prints=True)


@task(name="read-current-rds-snapshot", **TASK_OPTIONS)
def read_snapshot_task(options, simulate_failure=False):
    if simulate_failure:
        raise RuntimeError("Intentional demonstration failure before RDS access; no report was refreshed.")
    payload = fetch_current_analysis(**options)
    meta = payload["metadata"]
    get_run_logger().info(
        "Snapshot UTC %s; period %s to %s; orders=%s; labelled=%s; fraud=%s; practice excluded=%s",
        meta["snapshot_started_utc"], meta["first_order_date"], meta["last_order_date"],
        meta["orders_total"], meta["labelled_orders"], meta["fraud_orders"], meta["excluded_practice_orders"],
    )
    return payload


@task(name="validate-counts-and-rates", **TASK_OPTIONS)
def validate_snapshot_task(payload, minimum_labelled):
    validated = validate_analysis(payload, minimum_labelled)
    get_run_logger().info("PASS: grouped counts and recorded values agree with snapshot totals; valid-label denominators checked.")
    return validated


def _log_analysis(result):
    logger = get_run_logger()
    logger.info("%s\n%s", result["question"], result["table"].head(20).to_string(index=False))
    logger.info("Finding: %s", result["insight"])
    logger.info("Limitation: %s", result["limitation"])
    return result


@task(name="analyse-order-value-bands", **TASK_OPTIONS)
def order_value_task(payload):
    return _log_analysis(analyse_order_value(payload))


@task(name="analyse-weekday-and-collection", **TASK_OPTIONS)
def weekday_collection_task(payload):
    return _log_analysis(analyse_weekday_collection(payload))


@task(name="analyse-customer-concentration", **TASK_OPTIONS)
def customer_concentration_task(payload):
    return _log_analysis(analyse_customer_concentration(payload))


@task(name="publish-charts-and-report", **TASK_OPTIONS)
def publish_report_task(payload, results, output_root, started, prefect_run_id):
    final, summary = publish_report(payload, results, output_root, started=started, run_context={
        "engine": "Prefect", "prefect_flow_run_id": prefect_run_id,
        "timing_scope": "Flow body through report generation; inspect Prefect run duration for engine overhead.",
    })
    get_run_logger().info("Report: %s", final / "index.html")
    get_run_logger().info("CSV tables, PNG charts and summary.json saved in %s", final)
    return {"report": str(final / "index.html"), "summary": summary}


@flow(name="brayden-fraud-analytics", log_prints=True, persist_result=False, retries=0)
def brayden_analytics_flow(output_root, start_date=None, end_date=None, include_practice=False,
                          top_customers=20, minimum_labelled=30, simulate_failure=False):
    """Read once, validate, analyse three ways, and publish only complete outputs."""
    started = perf_counter()
    prefect_run_id = flow_run.id or ("manual-" + uuid4().hex[:12])
    logger = get_run_logger()
    try:
        snapshot = read_snapshot_task.submit({
            "start_date": start_date, "end_date": end_date,
            "include_practice": include_practice, "top_customers": top_customers,
        }, simulate_failure)
        validated = validate_snapshot_task.submit(snapshot, minimum_labelled)
        results = [order_value_task.submit(validated), weekday_collection_task.submit(validated),
                   customer_concentration_task.submit(validated)]
        publication = publish_report_task.submit(validated, results, str(output_root), started, prefect_run_id)
        report = publication.result()
        logger.info("Flow main work completed in %.2f seconds. Report: %s", perf_counter() - started, report["report"])
        return report
    except BaseException as exc:
        failures = Path(output_root) / "failures"
        failures.mkdir(parents=True, exist_ok=True)
        receipt = {
            "prefect_flow_run_id": prefect_run_id, "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - started, "error_type": type(exc).__name__,
            "error": str(exc), "previous_report_not_relabelled": True,
        }
        path = failures / f"{prefect_run_id}.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        logger.error("Refresh failed; earlier reports retain their original timestamps. Failure receipt: %s", path)
        raise
