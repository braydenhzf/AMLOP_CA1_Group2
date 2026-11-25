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