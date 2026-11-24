"""Read service settings from the project .env or process environment."""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """A required service setting is missing or invalid."""


def _settings(env_file=None):
    # Environment variables take priority, including explicitly empty values.
    path = PROJECT_ROOT / ".env" if env_file is None else Path(env_file)
    return {**dotenv_values(path, interpolate=False), **os.environ}


def _required(settings, names):
    missing = [name for name in names if not settings.get(name, "")
               or not settings[name].strip()]
    if missing:
        raise ConfigurationError(
            "Missing required settings: " + ", ".join(missing)
            + ". Set them in the project .env or environment; see .env.example."
        )


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str
    ssl_ca: Path

    def connection_kwargs(self):
        """Arguments for pymysql.connect, including verified RDS TLS."""
        import ssl

        return dict(
            host=self.host, port=self.port, user=self.user,
            password=self.password, database=self.database,
            ssl=ssl.create_default_context(cafile=str(self.ssl_ca)),
            connect_timeout=10,
        )


def get_database_settings(env_file=None):
    settings = _settings(env_file)
    _required(settings, ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"))
    try:
        port = int(settings["DB_PORT"])
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigurationError("DB_PORT must be an integer from 1 to 65535.") from None
    ca_path = Path(settings.get("DB_SSL_CA") or "certs/global-bundle.pem")
    if not ca_path.is_absolute():
        ca_path = PROJECT_ROOT / ca_path
    if not ca_path.is_file():
        raise ConfigurationError(
            "DB_SSL_CA must point to an existing CA certificate file. "
            "Relative paths are resolved from the project folder."
        )
    return DatabaseSettings(
        host=settings["DB_HOST"].strip(), port=port,
        user=settings["DB_USER"].strip(), password=settings["DB_PASSWORD"],
        database=settings["DB_NAME"].strip(), ssl_ca=ca_path,
    )


def get_mlflow_tracking_uri(env_file=None):
    settings = _settings(env_file)
    _required(settings, ("MLFLOW_TRACKING_URI",))
    uri = settings["MLFLOW_TRACKING_URI"].strip()
    try:
        parts = urlsplit(uri)
        port = parts.port
        valid = (
            parts.scheme in ("http", "https") and parts.hostname
            and parts.username is None and parts.password is None
            and not parts.query and not parts.fragment
            and not any(c.isspace() for c in uri)
            and (port is None or 1 <= port <= 65535)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(
            "MLFLOW_TRACKING_URI must be an http:// or https:// server URL "
            "without embedded credentials, query parameters or fragments."
        )
    return uri


def configure_mlflow(env_file=None):
    """Call before starting an experiment in a training script or notebook."""
    uri = get_mlflow_tracking_uri(env_file)
    import mlflow

    mlflow.set_tracking_uri(uri)
    return uri


def main():
    parser = argparse.ArgumentParser(description="Validate settings without connecting to services.")
    parser.add_argument("--service", choices=("database", "mlflow", "all"), default="all")
    args = parser.parse_args()
    try:
        if args.service in ("database", "all"):
            get_database_settings()
        if args.service in ("mlflow", "all"):
            get_mlflow_tracking_uri()
    except ConfigurationError as exc:
        parser.exit(1, f"Configuration error: {exc}\n")
    print(f"Configuration valid ({args.service}); no service connection attempted.")


if __name__ == "__main__":
    main()
