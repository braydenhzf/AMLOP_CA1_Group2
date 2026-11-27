# AMLOP CA1 Group 2 — SP-buy Fraud Detection

This is our ST1519 Agile MLOps CA1 project. We use historical customer,
order, and fraud-label data to develop fraud detection models. The final
workflow will process new data, load it into our shared database, and
retrain the team's models.

## First-time setup on Windows

### 1. Open the project

Clone the `braydenhzf/AMLOP_CA1_Group2` repository using GitHub Desktop.
If you already cloned it, select the existing repository instead.

In GitHub Desktop, choose **Repository → Open in Visual Studio Code**.
In VS Code, install the **Python** extension. Install the **Jupyter**
extension if you will be working with notebooks.

### 2. Set up Python

Open **Terminal → New Terminal** in VS Code. Use PowerShell and make
sure the prompt ends in `AMLOP_CA1_Group2>`.

Run these commands one at a time:

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

The `.venv` folder is your local Python environment for this project.
Each teammate creates their own. You do not need to activate it to run
the commands in this README.

In VS Code, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**,
and select the project's `.venv`. If you open a notebook, use **Select
Kernel** in the top right and select that same environment.

### 3. Add the base CSV files

Create a folder named `data` in the project folder, then create a folder
named `raw` inside `data`. Copy the three CA1 base CSVs into it:

- `data/raw/customers_base.csv`
- `data/raw/orders_base.csv`
- `data/raw/labels_base.csv`

The CSVs are kept on each teammate's laptop. They are not downloaded
when you clone or pull the repository.

### 4. Check that setup works

From the same VS Code terminal, run:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/raw/orders_base.csv', nrows=2); print(f'Read {len(df)} rows from orders_base.csv')"
```

Expected output: `Read 2 rows from orders_base.csv`

If pandas is missing, rerun the package installation command in Step 2.
If the CSV cannot be found, check that you are in the project folder
and that the file is named exactly `orders_base.csv`.

## Data preparation for PBI 1–3

- `notebooks/01_data_exploration.ipynb` profiles the historical CSVs and records cleaning recommendations.
- `notebooks/02_preparation_validation.ipynb` runs the cleaning and feature preparation on the historical files and a practice batch.
- `src/data_cleaning.py` provides the three cleaners and issue reports for database loading.
- `src/features.py` provides join checks, features, and preprocessing for model training.

Open the notebooks in order from this repository's VS Code workspace. Both look
for the historical files in `data/raw/` by default. To reuse the code in a
script run from the repository root, import from `src.data_cleaning` and
`src.features`.

## Service configuration (PBI 5)

From the project folder, install dependencies and create your local settings:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
# Run only if you do not already have a .env file:
Copy-Item .env.example .env
```

Edit `.env` and supply `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
`DB_PASSWORD` and `MLFLOW_TRACKING_URI`. Use the shared RDS hostname and
database name from your team, and the HTTP(S) URL of the MLflow service.
No real credentials belong in `.env.example`. Quote passwords containing
spaces or `#` in `.env`. Dollar signs are read literally.

The application reads `.env` relative to `config.py`, so it also works when
launched from another folder. Environment variables override `.env` values.
An explicitly empty environment variable causes a missing-setting error;
remove it if you want to use the value from `.env`. No source edits are
needed when a connection setting changes. Restart running scripts or rerun
the setup function in a notebook after changing settings.

`DB_SSL_CA` is optional and defaults to `certs/global-bundle.pem`. Relative
certificate paths are resolved from the project folder. Keep verified TLS
enabled for RDS connections.

Validate configuration without connecting to either service:

```powershell
.\.venv\Scripts\python.exe config.py --service database
.\.venv\Scripts\python.exe config.py --service mlflow
.\.venv\Scripts\python.exe config.py
```

Missing required settings and invalid ports/URLs produce clear errors without
printing passwords. Database-only tasks do not require MLflow settings.
If PBI 6 is not ready, validate the database now and supply the actual MLflow
URL once the shared service has been deployed.

Use this shared configuration in database code:

```python
import pymysql
from config import get_database_settings

connection = pymysql.connect(**get_database_settings().connection_kwargs())
```

In training scripts, call this before starting an MLflow experiment:

```python
from config import configure_mlflow

configure_mlflow()
```

When running a notebook, ensure the project folder is on Python's import path.

Place your RDS CA certificate at `certs/global-bundle.pem`, or set `DB_SSL_CA`
to the certificate path on your machine. The certificate and existing local
`test_team_access.py` helper are not part of this PBI change.

Run offline tests and verify which files Git will ignore:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_config.py" -v
git check-ignore .env .env.local
git check-ignore .env.example
git ls-files .env
```

The first Git check should list both private files; the last two commands
should print nothing (`check-ignore` exits 1 for the trackable example).
Do not force-add private configuration. Ignore rules do not remove a file
that was already tracked. The offline tests use dummy values and verify
configuration changes, environment precedence, validation and MLflow setup.
They do not establish that deployed services are reachable.

## Working on a task

1. Open your assigned task on the [Group 2 GitHub Project board](https://github.com/users/braydenhzf/projects/2).
2. Pull the latest `main` in GitHub Desktop.
3. Create a branch for your task. For example, `feat/batch-input`.
4. Make and test your changes in VS Code.
5. Commit and push your branch, then open a pull request and ask a teammate to review it.

Do not commit `.venv`, the raw CSVs, or `.env` files containing credentials.

## Random Forest for PBI 7

Model training code lives in `src/models/`. The Random Forest script reads
the three base CSVs from `data/raw/`, runs the
shared cleaning and feature code, and trains on eligible orders with a valid
fraud label. Both versions use order details and customer tenure available at
review time. The fitted pipeline includes preprocessing and the forest.

### Run and compare the two versions

From the repository root, run V1 on the validation dates. The latest dates
inside the training period are reserved for validation. The shared evaluation
period starts on 4 April 2023 for the historical dataset.

```powershell
.\.venv\Scripts\python.exe -m src.models.train_random_forest --version v1 --split validation --threshold 0.5
.\.venv\Scripts\python.exe -m src.models.train_random_forest --version v2 --split validation --threshold 0.5
```

V1 uses unweighted training. V2 keeps the same settings and uses
`class_weight="balanced_subsample"` to give the underrepresented fraud class
more weight during fitting. Compare fraud recall and missed cases alongside
fraud precision and false alarms to assess the trade-off. Average precision
compares the ranking of orders across thresholds.

Use the team’s agreed threshold for both final evaluation runs. Each fit uses
all the eligible training dates, and both versions are scored on the same later
orders. A flagged order goes to Trust & Safety for review.

```powershell
$threshold = Read-Host "Team-agreed fraud threshold"
.\.venv\Scripts\python.exe -m src.models.train_random_forest --version v1 --split evaluation --threshold $threshold
.\.venv\Scripts\python.exe -m src.models.train_random_forest --version v2 --split evaluation --threshold $threshold
```

Reports under `reports/pbi7-random-forest/` contain the split dates, model
parameters, training time, metrics, confusion-matrix counts and join audit.
Evaluation runs also save the fitted pipeline as a `.joblib` file. Use
`--n-jobs 4` if you want to limit parallel training on a laptop.

### Log the evaluation to shared MLflow

Set `MLFLOW_TRACKING_URI` in your local `.env` to the team's service URL.
Validate the setting, then use the team’s experiment name in both runs.

```powershell
.\.venv\Scripts\python.exe config.py --service mlflow
$experiment = Read-Host "Shared MLflow experiment name"
.\.venv\Scripts\python.exe -m src.models.train_random_forest --version v1 --split evaluation --threshold $threshold --log-mlflow --experiment $experiment --run-name vivian-random-forest-v1
.\.venv\Scripts\python.exe -m src.models.train_random_forest --version v2 --split evaluation --threshold $threshold --log-mlflow --experiment $experiment --run-name vivian-random-forest-v2
```

Open both runs in MLflow and check the parameters, metrics, `evaluation.json`
and logged model. For scoring, the model returns the probabilities for classes
0 and 1 in that order. Apply the logged threshold to the class 1 probability.

### Local model comparison

The local trials used scikit-learn 1.7.2 and a provisional threshold of 0.50.
V1 caught 11.4% of fraud on the validation dates and missed 17,622 fraud
orders. V2 tested whether giving fraud cases more weight could reduce those
misses. Validation recall rose to 80.0% and misses fell to 3,975, while false
alarms rose from 811 to 26,680.

Both final local fits used 758,025 orders through 3 April 2023. They were
scored on the same 191,847 orders from 4 April through 11 May 2023.

| Version | Average precision | Fraud precision | Fraud recall | Fraud F1 | Missed fraud | False alarms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 unweighted | 0.4295 | 58.7% | 18.8% | 0.2845 | 19,539 | 3,176 |
| V2 weighted | 0.4421 | 32.0% | 75.3% | 0.4496 | 5,934 | 38,442 |

At 0.50, V2 caught 13,605 additional fraud orders and sent 35,266 additional
legitimate orders for review. Use the team-agreed threshold and review
capacity when assessing this trade-off in the shared experiment.

## Incoming data batches (PBI 9)

From the project root, supply URLs for the customer, order and label CSVs:

```powershell
python download_batch.py --customers-url "CUSTOMERS_CSV_URL" --orders-url "ORDERS_CSV_URL" --labels-url "LABELS_CSV_URL"
```

Replace the three placeholder values with actual accessible CSV URLs.
Source filenames may have different postfixes.

Files are saved under `data/raw/incoming/` as:
- `customers_practice.csv`
- `orders_practice.csv`
- `labels_practice.csv`

Existing files with those names are overwritten.

Before reporting success, the downloader checks all three files for:
- Readable UTF-8 CSV content.
- A header containing the required columns.
- Duplicate column names.
- Consistent field counts in data rows.

Extra columns are allowed. Business-value cleaning is handled separately
by the team's cleaning functions.

Download or validation failures produce an error and a non-zero exit code.
Files downloaded before failure may remain on disk. Downstream processing
must use the returned paths only after `download_batch()` succeeds.

Run the automated checks:

```powershell
python -m unittest discover -s tests -p "test_batch_input.py" -v
```

Verification performed:
- Six automated tests passed.
- All three historical base files downloaded and validated through a local
  HTTP server.
- A missing HTTP resource produced a clear error and exit code 1.

External public practice URLs have not yet been verified.

## Automated pipeline (PBI 10)

Run commands from the project root. The pipeline downloads, validates and
cleans the three input CSVs, reconciles history, prepares features and can
train the available models with automatic MLflow logging.

Currently integrated:
- Brayden's Logistic Regression V1 and V2.
- Vivian's Random Forest V1 and V2.
- Isabelle's XGBoost V1 and V2.

All six versions use the pipeline's shared training and validation rows.

### Local rehearsal without AWS

Create the historical split manifest once. Skip this command if the
manifest already exists; the script refuses to overwrite it.

```powershell
.\.venv\Scripts\python.exe create_reference_splits.py --history-dir data/raw
```

The manifest is saved to `reports/pipeline/reference_splits.csv`.
Historical train, validation and test membership stays fixed. Previously
unseen order keys are assigned to training. Reported model metrics use
the fixed validation set; this command does not score the test set.

Start the local CSV server in one terminal and leave it running:

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory data/raw
```

In another terminal, run:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --customers-url http://127.0.0.1:8765/customers_base.csv --orders-url http://127.0.0.1:8765/orders_base.csv --labels-url http://127.0.0.1:8765/labels_base.csv --history-dir data/raw --split-manifest reports/pipeline/reference_splits.csv --train-local --threshold 0.5 --rf-jobs 4 --xgb-jobs 4 --log-mlflow-local
```

The threshold of 0.5 is a rehearsal setting. Use the team's agreed
threshold for final comparisons.

This rehearsal resubmits the historical base files, so reconciliation
should find identical records and no new records.

### History, database writes and logging

| Option | Behaviour |
| --- | --- |
| `--history-dir data/raw` | Reconcile against local historical base CSVs. |
| `--use-rds` | Read historical records from RDS using the shared configuration. |
| `--write-rds` | Insert only new records into RDS; requires `--use-rds`. |
| `--train-local` | Train the available models on this computer. |
| `--log-mlflow-local` | Save experiments locally under `reports/mlflow_local/`. |
| `--log-mlflow` | Upload experiments to the configured shared MLflow service. |
| `--rf-jobs N` | Set Random Forest CPU workers; -1 uses all available CPUs. |
| `--xgb-jobs N` | Set XGBoost CPU workers; -1 uses all available CPUs. |

Choose either local history or RDS history. Choose either local MLflow
logging or shared MLflow logging. Both logging options require training.

RDS writes are disabled by default. To use shared history, replace
`--history-dir data/raw` with `--use-rds`. Add `--write-rds` when database
insertion is intended.

Database reconciliation:
- Keeps existing identical records without inserting duplicates.
- Reports conflicting incoming records and retains historical values.
- Compares money fields at the RDS schema's six-decimal precision.
- Passes only previously unseen record keys to the insertion function.
- Uses a transaction for inserts and attempts rollback if insertion fails.
- Rejects duplicate-key inserts instead of overwriting existing records.

RDS access requires the database settings and TLS certificate described
in the service-configuration section. Shared logging additionally
requires an accessible `MLFLOW_TRACKING_URI`.

### Outputs and failure reports

Each execution gets a batch ID. Its reports are saved under
`reports/pipeline/<batch-id>/`, including cleaning, reconciliation,
join and split summaries where applicable.

Training saves model files and `models/model_results.csv`. When logging
is enabled, the CSV also records the corresponding MLflow run IDs.

When the insertion step is requested, `rds_insert_summary.csv` records
inserted counts, including zero counts for a repeated batch.

Processing failures save the batch ID, stage and error in `failure.txt`
and return a non-zero exit code. Invalid command options are rejected
before processing starts.

### Offline checks and verification status

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_pipeline_*.py" -v
```

All 26 offline pipeline tests passed. The end-to-end test combines
80 historical records with eight new records and one repeated record.
All six model versions train and log locally, and their saved models
can be reloaded for prediction. Validation and test membership stays fixed.

A full-size local rehearsal processed 949,872 eligible labelled orders
and trained/logged all six versions in 225.11 seconds. It used threshold
0.5, eight Random Forest workers and eight XGBoost workers.

Live RDS insertion, the full RDS check after the precision fix, shared
MLflow logging, and verification with externally supplied incoming
batches remain pending. PBI 10 is still in progress.