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

## Working on a task

1. Open your assigned task on the [Group 2 GitHub Project board](https://github.com/users/braydenhzf/projects/2).
2. Pull the latest `main` in GitHub Desktop.
3. Create a branch for your task. For example, `feat/batch-input`.
4. Make and test your changes in VS Code.
5. Commit and push your branch, then open a pull request and ask a teammate to review it.

Do not commit `.venv`, the raw CSVs, or `.env` files containing credentials.