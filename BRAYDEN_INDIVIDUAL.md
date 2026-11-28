# Brayden individual analytics

PBIs 13 and 14 use the same three analytical functions. The analytics read
current orders and labels from the configured shared RDS MySQL database.
They do not use MLflow and do not write to business tables.

## PBI 13 business questions

1. How does fraud risk and fraud-labelled order value vary across order-value bands?
2. Which weekday and collection-method combinations have elevated fraud rates?
3. Which customers account for the most fraud-labelled orders, and how concentrated are those orders?

The SQL, validation, analysis and chart functions are in
`src/analytics/brayden_analytics.py`. Run them with:

```powershell
python run_brayden_analytics.py
```

Default output: `reports/individual/brayden/<run-id>/index.html`, three CSV
tables, three PNG charts and `summary.json`. The output root's `latest.json`
points to the most recently published successful report. Generated reports
are ignored by Git; include the chosen report and screenshots in the
individual submission separately.

The first live run, `brayden-20261128T154157Z-aa6b89f4`, took 41.48 seconds.
It analysed 949,889 real orders, including 949,874 valid labelled orders and
104,422 fraud-labelled orders, for an overall fraud rate of about 10.99%.
It excluded 3,000 known synthetic practice orders and excluded 15 orders
without valid labels from rate denominators. The data period was
2022-11-12 to 2023-05-11.

Observed findings from that run:

- The SGD 100 to under 200 value band had 103 fraud-labelled orders out of
  180 valid labelled orders (57.2%). This is a small share of all orders;
  review risk together with counts, intervals and country composition.
- Thursday pickup had 509 fraud-labelled orders out of 1,666 valid labelled
  orders (30.6%). Weekday and collection associations do not establish a cause.
- The top 20 customers by fraud-labelled order count accounted for 759
  fraud-labelled orders, about 0.73% of the total. Targeted investigations
  alone would therefore cover only a small part of overall fraud volume.

Rates are fraud-labelled distinct orders divided by distinct orders with a
valid binary label. Joins use `(country_code, order_id)` so repeated IDs in
different countries do not multiply rows. The database tables have primary
keys on those fields. Unavailable rates are not reported as zero. Tables
include Wilson 95% intervals and flag groups below 30 labelled orders.
Amounts are recorded order values associated with fraud labels, not confirmed
fraud loss. Missing order values are counted separately.

Known `pbi12-` synthetic order IDs are excluded by default. Tutor-supplied
records with other IDs are included automatically. Use `--include-practice`
only when intentionally including the synthetic records for a rehearsal.

## PBI 14 local Prefect workflow

Install the pinned orchestration dependency:

```powershell
python -m pip install -r requirements-brayden-prefect.txt
```

In a dedicated project terminal, start the local service and leave it running:

```powershell
.\.venv\Scripts\prefect.exe server start --host 127.0.0.1 --port 4200
```

In the runner terminal:

```powershell
$env:PREFECT_API_URL = 'http://127.0.0.1:4200/api'
python run_brayden_prefect.py
```

Open `http://127.0.0.1:4200` and inspect the `brayden-fraud-analytics` flow run.
The entry point is `run_brayden_prefect.py`; the flow and tasks are in
`flows/brayden_analytics_flow.py`.

Task order:

1. `read-current-rds-snapshot`: execute all analytical SQL in one read-only,
   repeatable-read transaction and record the snapshot time and analysis period.
2. `validate-counts-and-rates`: check grouped counts, recorded value totals
   and valid-label denominators.
3. `analyse-order-value-bands`, `analyse-weekday-and-collection`, and
   `analyse-customer-concentration`: calculate and log the three results and
   their interpretations from that same snapshot. These three tasks may overlap.
4. `publish-charts-and-report`: create all outputs, publish the complete report
   and update the success pointer only after chart/report generation succeeds.

Tasks explicitly disable result caching and persistence, so every invocation
reads RDS again. Data added after the snapshot begins is included in a later run.

Default flow reports are under `reports/individual/brayden-prefect/<run-id>/`.
Query tables and findings also appear in Prefect task logs. `summary.json`
records the Prefect flow-run ID and local computation timing. Use the Prefect
UI's run duration as evidence of orchestration duration, including engine overhead.
The live Prefect run `wisteria-weasel` completed all six tasks successfully.
Flow ID: `92b4faf8-0883-4d5d-8959-fd2a2ffe7ed2`. Its logged main-work
duration was 44.13 seconds; its report summary records 44.12 seconds through
report generation. The report is `brayden-20261128T155124Z-0dd6c065`.
The deliberate failure run `7ec3d076-7f6c-4372-8b2c-a604aa2d7929` failed
before RDS access, returned exit code 1 and left the success pointer unchanged.
The read task records the intentional RuntimeError; downstream tasks become
NotReady and the flow records UnfinishedRun when the blocked publication
result is requested. Capture screenshots and complete the individual template
with your interpretation of the saved results before submission.

For a controlled failure before any RDS access:

```powershell
python run_brayden_prefect.py --simulate-failure
```

Expect a failed flow/task and a nonzero process exit. A failure receipt is
saved under the output root's `failures/` folder; the previous successful
report and its timestamp remain unchanged. Capture the failure state/logs.

## Evidence for the individual template

For A1, include each question, its function/file, a real chart/table, your
interpretation, the business action it supports and its limitations.

For A2, include the flow/entry files, task order, how to open the report,
screenshots of the local Prefect service, a completed flow graph and task
logs with query results. Record the actual run duration and capture a failure
example. A successful local test alone is not live Prefect evidence.

PBI 15 is the optional AWS challenge. It is not implemented or verified by
these files. It requires a hosted Prefect server, an RDS PostgreSQL backend
for Prefect metadata, and a separate cloud runner that reads the existing
business RDS MySQL database.

## Offline checks

```powershell
python -m unittest discover -s tests -p "test_brayden_*.py" -v
```

Eleven analytical checks and three flow-function checks passed using a small
local fixture. These cover denominator rules, composite keys, synthetic-data
scope, missing values, date bounds, concentration calculations, freshness,
and preserving previous reports after a failed refresh. The flow-function
checks do not start a real Prefect server; record a live server run separately.
