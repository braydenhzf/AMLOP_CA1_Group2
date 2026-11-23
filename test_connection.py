
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT DATABASE(), VERSION()")
        )
        print("Connected successfully!")
        print("Database:", result.fetchone())

except Exception as e:
    print("Connection failed:", e)
