
import os
import ssl
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()

def get_engine():
    url = URL.create(
        drivername="mysql+pymysql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", 3306)),
        database=os.environ["DB_NAME"]
    )

    ca_path = Path(
        os.getenv("DB_SSL_CA", "certs/global-bundle.pem")
    ).resolve()

    ssl_context = ssl.create_default_context(
        cafile=str(ca_path)
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={
            "ssl": ssl_context,
            "connect_timeout": 10
        }
    )
