"""Check flow dependencies, freshness and publication failure without AWS."""
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_brayden_analytics import Connection
from src.analytics.brayden_analytics import read_queries
import flows.brayden_analytics_flow as workflow


class InlineFuture:
    def __init__(self, function):
        self.function = function
        self.done = False

    def result(self):
        if not self.done:
            self.value = self.function()
            self.done = True
        return self.value


def resolve(value):
    if isinstance(value, InlineFuture):
        return value.result()
    if isinstance(value, list):
        return [resolve(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    return value


def submit_for(task):
    def submit(*args, **kwargs):
        return InlineFuture(lambda: task.fn(*[resolve(arg) for arg in args],
                                           **{key: resolve(value) for key, value in kwargs.items()}))
    return submit


class BraydenPrefectTests(unittest.TestCase):
    def setUp(self):
        self.connection = Connection()

    def tearDown(self):
        self.connection.raw.close()

    def snapshot(self, **options):
        result = read_queries(self.connection, **options)
        result["metadata"]["snapshot_started_utc"] = "2026-11-28T15:00:00+00:00"
        return result

    def inline_tasks(self):
        stack = ExitStack()
        stack.enter_context(patch.object(workflow, "get_run_logger", return_value=logging.getLogger("offline-flow-test")))
        for name in ("read_snapshot_task", "validate_snapshot_task", "order_value_task", "weekday_collection_task",
                     "customer_concentration_task", "publish_report_task"):
            task = getattr(workflow, name)
            stack.enter_context(patch.object(task, "submit", side_effect=submit_for(task)))
        return stack

    def test_flow_reads_once_and_publishes_all_three_analyses(self):
        with tempfile.TemporaryDirectory() as folder, self.inline_tasks(), \
                patch.object(workflow, "fetch_current_analysis", side_effect=self.snapshot) as fetch:
            result = workflow.brayden_analytics_flow.fn(folder, minimum_labelled=1)
            fetch.assert_called_once()
            report = Path(result["report"])
            self.assertTrue(report.is_file())
            self.assertEqual(len(list(report.parent.glob("*.png"))), 3)
            self.assertEqual(len(list(report.parent.glob("*.csv"))), 3)
            self.assertEqual(result["summary"]["orchestration"]["engine"], "Prefect")
            self.assertEqual(result["summary"]["metadata"]["orders_total"], 8)

    def test_subsequent_run_reads_new_data_and_publishes_a_new_report(self):
        with tempfile.TemporaryDirectory() as folder, self.inline_tasks(), \
                patch.object(workflow, "fetch_current_analysis", side_effect=self.snapshot) as fetch:
            first = workflow.brayden_analytics_flow.fn(folder)
            self.connection.raw.execute("INSERT INTO orders VALUES('SG','new-order','delivery',100,'2023-01-07')")
            self.connection.raw.execute("INSERT INTO labels VALUES('SG','new-order','c1',1)")
            second = workflow.brayden_analytics_flow.fn(folder)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(first["summary"]["metadata"]["orders_total"], 8)
            self.assertEqual(second["summary"]["metadata"]["orders_total"], 9)
            self.assertNotEqual(first["report"], second["report"])
            pointer = json.loads((Path(folder) / "latest.json").read_text())
            self.assertEqual(pointer["run_id"], second["summary"]["run_id"])

    def test_intentional_failure_preserves_previous_report_and_records_error(self):
        with tempfile.TemporaryDirectory() as folder, self.inline_tasks(), \
                patch.object(workflow, "fetch_current_analysis") as fetch:
            previous = {"run_id": "old-success", "report": "old-success/index.html"}
            (Path(folder) / "latest.json").write_text(json.dumps(previous))
            with self.assertRaisesRegex(RuntimeError, "Intentional demonstration failure"):
                workflow.brayden_analytics_flow.fn(folder, simulate_failure=True)
            fetch.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / "latest.json").read_text()), previous)
            receipt, = (Path(folder) / "failures").glob("*.json")
            self.assertEqual(json.loads(receipt.read_text())["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
