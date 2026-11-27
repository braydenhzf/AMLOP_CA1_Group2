
"""isabelle's individual PBI 13 fraud business analytics."""

from contextlib import contextmanager
from time import perf_counter

import pandas as pd
import pymysql

from config import get_database_settings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

@contextmanager
def analytics_connection():
    """Open a secure, consistent, read-only database snapshot."""

    settings = get_database_settings()

    conn = pymysql.connect(
        **settings.connection_kwargs(),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cursor.execute(
                "START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY"
            )

        yield conn

    finally:
        conn.rollback()
        conn.close()


def execute_query(conn, sql):
    """Execute a SQL query and return a pandas DataFrame."""

    with conn.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def get_analysis_period(conn):
    """Retrieve the available order analysis period."""

    sql = """
        SELECT
            MIN(order_date) AS start_date,
            MAX(order_date) AS end_date,
            COUNT(*) AS total_orders
        FROM orders
        WHERE order_date IS NOT NULL
    """

    return execute_query(conn, sql)


def query_fraud_by_payment(conn):
    """Q1: Analyse fraud risk by country and payment method."""

    sql = """
        SELECT
            o.country_code,
            o.payment_method,

            COUNT(*) AS total_orders,

            SUM(
                CASE
                    WHEN l.is_fraud IN (0, 1) THEN 1
                    ELSE 0
                END
            ) AS labelled_orders,

            SUM(
                CASE
                    WHEN l.is_fraud = 1 THEN 1
                    ELSE 0
                END
            ) AS fraud_orders,

            SUM(
                CASE
                    WHEN l.is_fraud = 0 THEN 1
                    ELSE 0
                END
            ) AS non_fraud_orders,

            SUM(
                CASE
                    WHEN l.is_fraud IS NULL
                         OR l.is_fraud NOT IN (0, 1)
                    THEN 1
                    ELSE 0
                END
            ) AS unlabelled_orders,

            ROUND(
                100.0 *
                SUM(
                    CASE
                        WHEN l.is_fraud = 1 THEN 1
                        ELSE 0
                    END
                )
                /
                NULLIF(
                    SUM(
                        CASE
                            WHEN l.is_fraud IN (0, 1) THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ),
                4
            ) AS fraud_rate_pct

        FROM orders AS o

        LEFT JOIN labels AS l
            ON o.country_code = l.country_code
            AND o.order_id = l.order_id

        WHERE o.order_date IS NOT NULL

        GROUP BY
            o.country_code,
            o.payment_method

        ORDER BY fraud_rate_pct DESC
    """

    df = execute_query(conn, sql)

    numeric_columns = [
        "total_orders",
        "labelled_orders",
        "fraud_orders",
        "non_fraud_orders",
        "unlabelled_orders",
        "fraud_rate_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    return df


def validate_payment_results(df, expected_orders):
    """Validate the consistency of Query 1 calculations."""

    assert not df.empty, "Query returned no records."

    assert (
        df["fraud_orders"] + df["non_fraud_orders"]
        == df["labelled_orders"]
    ).all(), "Labelled order counts do not match."

    assert (
        df["labelled_orders"] + df["unlabelled_orders"]
        == df["total_orders"]
    ).all(), "Total order counts do not match."

    assert (
        df["total_orders"].sum() == expected_orders
    ), "Grouped counts do not match database totals."

    valid = df["labelled_orders"] > 0

    expected_rate = (
        df.loc[valid, "fraud_orders"]
        / df.loc[valid, "labelled_orders"]
        * 100
    )

    difference = (
        df.loc[valid, "fraud_rate_pct"] - expected_rate
    ).abs()

    assert (
        difference <= 0.0001
    ).all(), "Fraud rate calculation is incorrect."

    assert df.loc[
        ~valid, "fraud_rate_pct"
    ].isna().all(), "Unavailable fraud rates must remain null."

    print("PASS: Fraud rate calculations validated.")
    print("PASS: Order counts validated.")
    print("PASS: Unlabelled orders handled correctly.")


def plot_payment_heatmap(
    df,
    start_date,
    end_date,
    output_dir,
    min_labelled_orders=1000,
):
    """Generate a fraud-risk heatmap with minimum sample filtering."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()

    plot_df["payment_method"] = (
        plot_df["payment_method"]
        .fillna("Unknown")
    )

    # Pivot fraud rates and labelled transaction counts.
    rates = plot_df.pivot(
        index="country_code",
        columns="payment_method",
        values="fraud_rate_pct",
    )

    counts = plot_df.pivot(
        index="country_code",
        columns="payment_method",
        values="labelled_orders",
    )

    # Mask groups with insufficient labelled transactions.
    filtered_rates = rates.where(
        counts >= min_labelled_orders
    )

    fig, ax = plt.subplots(
        figsize=(
            max(14, len(rates.columns) * 1.3),
            max(5, len(rates.index) * 1.2),
        )
    )

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e5e7eb")

    image = ax.imshow(
        np.ma.masked_invalid(
            filtered_rates.to_numpy(dtype=float)
        ),
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(range(len(rates.columns)))

    ax.set_xticklabels(
        rates.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(range(len(rates.index)))
    ax.set_yticklabels(rates.index)

    # Annotate fraud rates and sample sizes.
    for row in range(len(rates.index)):
        for col in range(len(rates.columns)):

            count = counts.iloc[row, col]
            rate = rates.iloc[row, col]

            if pd.isna(count):
                annotation = "-"
                colour = "#64748b"

            elif pd.isna(rate):
                annotation = "N/A"
                colour = "#64748b"

            elif count < min_labelled_orders:
                annotation = f"Low n\n({int(count):,})"
                colour = "#64748b"

            else:
                annotation = (
                    f"{rate:.1f}%\n"
                    f"n={int(count):,}"
                )
                colour = (
                    "white" if rate >= 65 else "black"
                )

            ax.text(
                col,
                row,
                annotation,
                ha="center",
                va="center",
                fontsize=8,
                color=colour,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Fraud rate (%)")

    ax.set_title(
        "Fraud Risk by Country and Payment Method\n"
        f"{start_date} to {end_date}",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlabel("Payment method")
    ax.set_ylabel("Country")

    fig.text(
        0.5,
        0.01,
        f"Low n: fewer than {min_labelled_orders:,} "
        "labelled orders. "
        "N/A: unavailable fraud rate. "
        "-: no matching orders.",
        ha="center",
        fontsize=9,
        color="#64748b",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = (
        output_dir / "q1_payment_fraud_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

def query_fraud_by_tenure(conn):
    """Q2: Analyse fraud risk by customer purchasing history."""

    sql = """
        WITH order_tenure AS (
            SELECT
                o.country_code,
                o.order_id,
                l.is_fraud,

                DATEDIFF(
                    o.order_date,
                    c.first_order_datetime
                ) AS tenure_days

            FROM orders AS o

            LEFT JOIN labels AS l
                ON o.country_code = l.country_code
                AND o.order_id = l.order_id

            LEFT JOIN customers AS c
                ON l.country_code = c.country_code
                AND l.customer_id = c.customer_id

            WHERE o.order_date IS NOT NULL
        ),

        tenure_groups AS (
            SELECT
                *,
                CASE
                    WHEN tenure_days IS NULL
                         OR tenure_days < 0
                        THEN 'Unknown'

                    WHEN tenure_days = 0
                        THEN '0 days'

                    WHEN tenure_days <= 7
                        THEN '1-7 days'

                    WHEN tenure_days <= 30
                        THEN '8-30 days'

                    WHEN tenure_days <= 90
                        THEN '31-90 days'

                    WHEN tenure_days <= 180
                        THEN '91-180 days'

                    ELSE '181+ days'
                END AS tenure_group

            FROM order_tenure
        )

        SELECT
            tenure_group,

            COUNT(*) AS total_orders,

            SUM(
                CASE WHEN is_fraud IN (0, 1)
                    THEN 1 ELSE 0 END
            ) AS labelled_orders,

            SUM(
                CASE WHEN is_fraud = 1
                    THEN 1 ELSE 0 END
            ) AS fraud_orders,

            SUM(
                CASE WHEN is_fraud = 0
                    THEN 1 ELSE 0 END
            ) AS non_fraud_orders,

            SUM(
                CASE WHEN is_fraud IS NULL
                       OR is_fraud NOT IN (0, 1)
                    THEN 1 ELSE 0 END
            ) AS unlabelled_orders,

            ROUND(
                100.0 *
                SUM(
                    CASE WHEN is_fraud = 1
                        THEN 1 ELSE 0 END
                )
                /
                NULLIF(
                    SUM(
                        CASE WHEN is_fraud IN (0, 1)
                            THEN 1 ELSE 0 END
                    ),
                    0
                ),
                4
            ) AS fraud_rate_pct

        FROM tenure_groups

        GROUP BY tenure_group

        ORDER BY FIELD(
            tenure_group,
            '0 days',
            '1-7 days',
            '8-30 days',
            '31-90 days',
            '91-180 days',
            '181+ days',
            'Unknown'
        )
    """

    df = execute_query(conn, sql)

    numeric_columns = [
        "total_orders",
        "labelled_orders",
        "fraud_orders",
        "non_fraud_orders",
        "unlabelled_orders",
        "fraud_rate_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    return df

def plot_tenure_fraud(
    df,
    start_date,
    end_date,
    output_dir,
):
    """Visualise fraud rates and volumes by customer tenure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()

    labels = plot_df["tenure_group"].tolist()

    positions = np.arange(len(labels))

    rates = plot_df["fraud_rate_pct"].to_numpy(dtype=float)
    fraud_counts = plot_df["fraud_orders"].to_numpy(dtype=float)

    overall_rate = (
        plot_df["fraud_orders"].sum()
        / plot_df["labelled_orders"].sum()
        * 100
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
    )

    # Chart 1: Fraud rate.
    ax = axes[0]

    colours = [
        "#94a3b8" if label == "Unknown"
        else "#dc2626" if rate > overall_rate
        else "#2563eb"
        for label, rate in zip(labels, rates)
    ]

    bars = ax.bar(
        positions,
        np.nan_to_num(rates, nan=0.0),
        color=colours,
        alpha=0.85,
    )

    ax.axhline(
        overall_rate,
        color="black",
        linestyle="--",
        label=f"Overall fraud rate: {overall_rate:.2f}%",
    )

    for bar, rate, count in zip(
        bars,
        rates,
        plot_df["labelled_orders"],
    ):
        annotation = (
            f"{rate:.2f}%\nn={int(count):,}"
            if not np.isnan(rate)
            else "N/A"
        )

        ax.annotate(
            annotation,
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title(
        "Fraud Rate by Customer Tenure",
        fontweight="bold",
    )

    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.margins(y=0.25)

    # Chart 2: Fraudulent order count.
    ax = axes[1]

    bars = ax.bar(
        positions,
        fraud_counts,
        color="#2563eb",
        alpha=0.85,
    )

    for bar, count in zip(bars, fraud_counts):
        ax.annotate(
            f"{int(count):,}",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_title(
        "Fraudulent Order Volume by Customer Tenure",
        fontweight="bold",
    )

    ax.set_ylabel("Number of fraudulent orders")
    ax.set_xlabel("Time since first recorded order")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.2)
    ax.margins(y=0.2)

    fig.suptitle(
        "Customer Tenure and Fraud Risk\n"
        f"{start_date} to {end_date}",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.01,
        "Unknown includes missing customer records, "
        "missing first-order dates and negative tenure.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    output_path = (
        output_dir / "q2_customer_tenure_fraud.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def query_monthly_financial_exposure(conn):
    """Q3: Analyse monthly fraud-related financial exposure by country."""

    sql = """
        WITH monthly_summary AS (
            SELECT
                o.country_code,

                DATE_FORMAT(
                    o.order_date, '%Y-%m'
                ) AS order_month,

                COUNT(*) AS total_orders,

                SUM(
                    CASE WHEN l.is_fraud IN (0, 1)
                        THEN 1 ELSE 0 END
                ) AS labelled_orders,

                SUM(
                    CASE WHEN l.is_fraud = 1
                        THEN 1 ELSE 0 END
                ) AS fraud_orders,

                SUM(
                    CASE WHEN l.is_fraud = 0
                        THEN 1 ELSE 0 END
                ) AS non_fraud_orders,

                SUM(
                    CASE WHEN l.is_fraud IS NULL
                           OR l.is_fraud NOT IN (0, 1)
                        THEN 1 ELSE 0 END
                ) AS unlabelled_orders,

                SUM(
                    CASE
                        WHEN l.is_fraud = 1
                         AND o.order_value >= 0
                        THEN o.order_value
                        ELSE NULL
                    END
                ) AS fraud_order_value,

                SUM(
                    CASE
                        WHEN l.is_fraud = 1
                         AND o.refund_value >= 0
                        THEN o.refund_value
                        ELSE NULL
                    END
                ) AS fraud_refund_value,

                SUM(
                    CASE
                        WHEN l.is_fraud = 1
                         AND (
                            o.order_value IS NULL
                            OR o.order_value < 0
                         )
                        THEN 1 ELSE 0
                    END
                ) AS invalid_order_value_count,

                SUM(
                    CASE
                        WHEN l.is_fraud = 1
                         AND (
                            o.refund_value IS NULL
                            OR o.refund_value < 0
                         )
                        THEN 1 ELSE 0
                    END
                ) AS invalid_refund_value_count,

                SUM(
                    CASE
                        WHEN l.is_fraud = 1
                         AND o.order_value >= 0
                         AND o.refund_value >= 0
                         AND o.refund_value > o.order_value
                        THEN 1 ELSE 0
                    END
                ) AS refund_exceeds_order_count

            FROM orders AS o

            LEFT JOIN labels AS l
                ON o.country_code = l.country_code
                AND o.order_id = l.order_id

            WHERE o.order_date IS NOT NULL

            GROUP BY
                o.country_code,
                DATE_FORMAT(o.order_date, '%Y-%m')
        )

        SELECT
            *,

            ROUND(
                100.0 * fraud_orders
                / NULLIF(labelled_orders, 0),
                4
            ) AS fraud_rate_pct,

            ROUND(
                fraud_order_value
                / NULLIF(
                    fraud_orders - invalid_order_value_count,
                    0
                ),
                4
            ) AS avg_fraud_order_value

        FROM monthly_summary

        ORDER BY country_code, order_month
    """

    df = execute_query(conn, sql)

    numeric_columns = [
        "total_orders",
        "labelled_orders",
        "fraud_orders",
        "non_fraud_orders",
        "unlabelled_orders",
        "fraud_order_value",
        "fraud_refund_value",
        "invalid_order_value_count",
        "invalid_refund_value_count",
        "refund_exceeds_order_count",
        "fraud_rate_pct",
        "avg_fraud_order_value",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    return df

def validate_financial_results(df, expected_orders):
    """Validate fraud counts and financial exposure calculations."""

    # Reuse the existing fraud-rate and count checks.
    validate_payment_results(df, expected_orders)

    for column in [
        "fraud_order_value",
        "fraud_refund_value",
        "avg_fraud_order_value",
    ]:
        assert (
            df[column].dropna() >= 0
        ).all(), f"Negative financial value: {column}"

    for column in [
        "invalid_order_value_count",
        "invalid_refund_value_count",
        "refund_exceeds_order_count",
    ]:
        assert (
            df[column] >= 0
        ).all(), f"Negative anomaly count: {column}"

        assert (
            df[column] <= df["fraud_orders"]
        ).all(), f"Invalid anomaly count: {column}"

    valid_value_count = (
        df["fraud_orders"]
        - df["invalid_order_value_count"]
    )

    valid = valid_value_count > 0

    expected_average = (
        df.loc[valid, "fraud_order_value"]
        / valid_value_count.loc[valid]
    )

    assert np.allclose(
        df.loc[valid, "avg_fraud_order_value"],
        expected_average,
        atol=0.0001,
        rtol=0,
    ), "Average fraudulent order value is incorrect."

    assert df.loc[
        ~valid, "avg_fraud_order_value"
    ].isna().all(), "Unavailable averages must remain null."

    print("PASS: Financial values validated.")
    print("PASS: Average fraudulent order values validated.")
    print("PASS: Financial anomaly counts validated.")

def plot_financial_exposure(
    df,
    start_date,
    end_date,
    output_dir,
):
    """Visualise monthly fraud-related values separately by country."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = sorted(df["country_code"].unique())

    country_names = {
        "BD": "Bangladesh",
        "MY": "Malaysia",
        "PH": "Philippines",
        "PK": "Pakistan",
        "TH": "Thailand",
    }

    fig, axes = plt.subplots(
        len(countries),
        1,
        figsize=(13, 3.5 * len(countries)),
        squeeze=False,
    )

    for index, country in enumerate(countries):

        ax = axes[index, 0]

        country_df = (
            df[df["country_code"] == country]
            .sort_values("order_month")
        )

        months = country_df["order_month"]

        ax.plot(
            months,
            country_df["fraud_order_value"],
            marker="o",
            linewidth=2.2,
            color="#2563eb",
            label="Fraudulent order value",
        )

        ax.plot(
            months,
            country_df["fraud_refund_value"],
            marker="s",
            linewidth=2.2,
            color="#d97706",
            label="Fraud-associated refund value",
        )

        ax.set_title(
            country_names.get(country, country),
            fontweight="bold",
            fontsize=13,
        )

        ax.set_ylabel("Recorded monetary value")
        ax.set_xlabel("Order month")

        ax.ticklabel_format(
            style="plain",
            axis="y",
        )

        from matplotlib.ticker import StrMethodFormatter

        ax.yaxis.set_major_formatter(
            StrMethodFormatter("{x:,.0f}")
        )

        ax.grid(alpha=0.25)

        ax.set_ylim(bottom=0)

        ax.legend(loc="best")

    fig.suptitle(
        "Monthly Financial Exposure Associated with Fraud\n"
        f"{start_date} to {end_date}",
        fontsize=17,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.015,
        "Values are shown separately by country. "
        "Refund amounts do not represent confirmed fraud losses.\n"
        "Missing and negative amounts are excluded from their "
        "respective totals. First and last months may be incomplete.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    output_path = (
        output_dir / "q3_monthly_financial_exposure.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

def main():

    started = perf_counter()

    output_dir = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "isabelle_analytics"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting securely to AWS RDS...", flush=True)

    with analytics_connection() as conn:

        with conn.cursor() as cursor:
            cursor.execute("SHOW STATUS LIKE 'Ssl_cipher'")
            tls = cursor.fetchone()

            if not tls or not tls["Value"]:
                raise RuntimeError(
                    "RDS connection is not using TLS."
                )

            print("PASS: TLS connection verified.")

        period = get_analysis_period(conn)

        start_date = period.iloc[0]["start_date"]
        end_date = period.iloc[0]["end_date"]
        expected_orders = int(
            period.iloc[0]["total_orders"]
        )

        print("\n========== ANALYSIS PERIOD ==========")
        print(period.to_string(index=False))

        # Query 1
        print("\nRunning Query 1...", flush=True)

        q1_started = perf_counter()

        q1 = query_fraud_by_payment(conn)

        validate_payment_results(q1, expected_orders)

        q1_seconds = perf_counter() - q1_started

        # Query 2
        print("\nRunning Query 2...", flush=True)

        q2_started = perf_counter()

        q2 = query_fraud_by_tenure(conn)

        # Reuse the same count and fraud rate validation.
        validate_payment_results(q2, expected_orders)

        q2_seconds = perf_counter() - q2_started

        # Cross-check both queries.
        for column in [
            "total_orders",
            "labelled_orders",
            "fraud_orders",
            "non_fraud_orders",
            "unlabelled_orders",
        ]:
            assert (
                q1[column].sum() == q2[column].sum()
            ), f"Cross-query count mismatch: {column}"

        print("PASS: Query 1 and Query 2 totals match.")

        # Query 3
        print("\nRunning Query 3...", flush=True)

        q3_started = perf_counter()

        q3 = query_monthly_financial_exposure(conn)

        validate_financial_results(q3, expected_orders)

        q3_seconds = perf_counter() - q3_started

        # Cross-check all three analytical queries.
        for column in [
            "total_orders",
            "labelled_orders",
            "fraud_orders",
            "non_fraud_orders",
            "unlabelled_orders",
        ]:
            assert (
                q1[column].sum()
                == q2[column].sum()
                == q3[column].sum()
            ), f"Cross-query count mismatch: {column}"

        print("PASS: All three query totals match.")


    # Database snapshot closed. Generate outputs locally.

    print("\n========== QUERY 1 RESULTS ==========")
    print(q1.head(20).to_string(index=False))

    q1.to_csv(
        output_dir / "q1_payment_fraud.csv",
        index=False,
    )

    q1_chart = plot_payment_heatmap(
        q1,
        start_date,
        end_date,
        output_dir,
    )

    print(f"\nQuery 1 execution time: {q1_seconds:.2f}s")
    print(f"Heatmap saved to: {q1_chart}")

    print("\n========== QUERY 2 RESULTS ==========")
    print(q2.to_string(index=False))

    q2.to_csv(
        output_dir / "q2_customer_tenure_fraud.csv",
        index=False,
    )

    q2_chart = plot_tenure_fraud(
        q2,
        start_date,
        end_date,
        output_dir,
    )

    print(f"\nQuery 2 execution time: {q2_seconds:.2f}s")
    print(f"Tenure chart saved to: {q2_chart}")

    print("\n========== QUERY 3 RESULTS ==========")

    print(q3.to_string(index=False))

    q3.to_csv(
        output_dir / "q3_monthly_financial_exposure.csv",
        index=False,
    )

    q3_chart = plot_financial_exposure(
        q3,
        start_date,
        end_date,
        output_dir,
    )

    print(
        f"\nQuery 3 execution time: {q3_seconds:.2f}s"
    )

    print(f"Financial exposure chart saved to: {q3_chart}")

    print("\n========== FINANCIAL DATA QUALITY ==========")

    print(
        "Fraud orders with invalid order values:",
        int(q3["invalid_order_value_count"].sum()),
    )

    print(
        "Fraud orders with invalid refund values:",
        int(q3["invalid_refund_value_count"].sum()),
    )

    print(
        "Fraud orders with refund exceeding order value:",
        int(q3["refund_exceeds_order_count"].sum()),
    )


    print("\n========== OVERALL SUMMARY ==========")

    print(f"Total orders: {expected_orders:,}")
    print(f"Labelled orders: {int(q1['labelled_orders'].sum()):,}")
    print(f"Fraud orders: {int(q1['fraud_orders'].sum()):,}")
    print(f"Unlabelled orders: {int(q1['unlabelled_orders'].sum()):,}")

    print(
        f"\nTotal execution time: "
        f"{perf_counter() - started:.2f} seconds"
    )


if __name__ == "__main__":
    main()
