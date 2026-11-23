
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

queries = {
    "Record counts": """
        SELECT 'customers' AS table_name, COUNT(*) AS total
        FROM customers
        UNION ALL
        SELECT 'orders', COUNT(*) FROM orders
        UNION ALL
        SELECT 'labels', COUNT(*) FROM labels
    """,

    "Duplicate customer IDs": """
        SELECT COUNT(*) AS duplicates
        FROM (
            SELECT country_code, customer_id
            FROM customers
            GROUP BY country_code, customer_id
            HAVING COUNT(*) > 1
        ) t
    """,

    "Missing payment methods": """
        SELECT COUNT(*) AS missing_payments
        FROM orders
        WHERE payment_method IS NULL
    """,

    "Fraud distribution": """
        SELECT is_fraud, COUNT(*) AS total
        FROM labels
        GROUP BY is_fraud
    """,

    "Refund anomalies": """
        SELECT COUNT(*) AS anomalies
        FROM orders
        WHERE refund_value > order_value
    """
}

with engine.connect() as conn:
    for name, query in queries.items():
        print(f"\n=== {name} ===")

        result = conn.execute(text(query))

        for row in result:
            print(tuple(row))

print("\nRDS verification completed!")
