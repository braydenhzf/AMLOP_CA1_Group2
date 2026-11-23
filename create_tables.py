
from sqlalchemy import text, inspect
from db.connection import get_engine

engine = get_engine()

tables = [
    """
    CREATE TABLE IF NOT EXISTS customers (
        country_code VARCHAR(10) NOT NULL,
        customer_id VARCHAR(100) NOT NULL,
        mobile_verified BOOLEAN,
        num_orders_last_50days INT,
        num_cancelled_orders_last_50days INT,
        num_refund_orders_last_50days INT,
        total_payment_last_50days DECIMAL(15,2),
        num_associated_customers INT,
        first_order_datetime DATETIME,
        PRIMARY KEY (country_code, customer_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        country_code VARCHAR(10) NOT NULL,
        order_id VARCHAR(100) NOT NULL,
        collect_type VARCHAR(100),
        payment_method VARCHAR(100),
        order_value DECIMAL(15,2),
        num_items_ordered INT,
        refund_value DECIMAL(15,2),
        order_date DATE,
        PRIMARY KEY (country_code, order_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS labels (
        country_code VARCHAR(10) NOT NULL,
        order_id VARCHAR(100) NOT NULL,
        customer_id VARCHAR(100) NOT NULL,
        is_fraud TINYINT,
        PRIMARY KEY (country_code, order_id)
    )
    """
]

try:
    with engine.begin() as conn:
        for sql in tables:
            conn.execute(text(sql))

    print("Tables created successfully!")

    inspector = inspect(engine)

    for table_name in ["customers", "orders", "labels"]:
        columns = inspector.get_columns(table_name)
        print(f"\n{table_name}:")
        for column in columns:
            print(f"  {column['name']} - {column['type']}")

except Exception as e:
    print("Error creating tables:", e)
