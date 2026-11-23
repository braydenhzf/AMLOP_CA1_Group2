
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

try:
    connection = pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=10
    )

    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS spbuy"
        )
        print("Database 'spbuy' created successfully!")

    connection.close()

except Exception as e:
    print("Error:", e)
