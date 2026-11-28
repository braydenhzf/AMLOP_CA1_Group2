"""Persist workflow stage timings, including failed stages, for PBI 12."""

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter


class StageTimings:
    def __init__(self, report_dir, *, started=None):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.started = perf_counter() if started is None else started
        self.rows = []
        self.stack = []

    def _save(self):
        with (self.report_dir / "stage_timings.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "stage", "parent", "start_seconds", "end_seconds",
                "duration_seconds", "status", "error_type",
            ])
            writer.writeheader()
            writer.writerows(self.rows)

    @contextmanager
    def stage(self, name):
        beginning = perf_counter()
        parent = self.stack[-1] if self.stack else ""
        self.stack.append(name)
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            ending = perf_counter()
            self.stack.pop()
            self.rows.append({
                "stage": name, "parent": parent,
                "start_seconds": beginning - self.started,
                "end_seconds": ending - self.started,
                "duration_seconds": ending - beginning,
                "status": "FAILED" if error is not None else "FINISHED",
                "error_type": type(error).__name__ if error is not None else "",
            })
            try:
                self._save()
            except Exception:
                if error is None:
                    raise

    def call(self, name, function, *args, **kwargs):
        with self.stage(name):
            return function(*args, **kwargs)

    def finish(self, *, success, elapsed=None, complete_workflow=False):
        elapsed = perf_counter() - self.started if elapsed is None else elapsed
        result = {
            "success": bool(success), "elapsed_seconds": elapsed,
            "time_limit_seconds": 180,
            "within_time_limit": bool(success and elapsed <= 180),
            "complete_cloud_workflow": bool(complete_workflow),
            "pbi12_runtime_check_passed": bool(success and complete_workflow and elapsed <= 180),
            "timing_scope": "Main workflow; Python startup and external verification are excluded.",
            "note": "Nested stages overlap their parents; do not sum every CSV row.",
        }
        self._save()
        (self.report_dir / "performance_summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
