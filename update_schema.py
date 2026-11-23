
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

statements = [
    """
    ALTER TABLE customers
    MODIFY total_payment_last_50days DECIMAL(18,6)
    """,
    """
    ALTER TABLE orders
    MODIFY order_value DECIMAL(18,6)
    """,
    """
    ALTER TABLE orders
    MODIFY refund_value DECIMAL(18,6)
    """
]

with engine.begin() as conn:
    for statement in statements:
        conn.execute(text(statement))

print("Monetary columns updated successfully!")
