"""Offline configuration checks using only temporary files and dummy settings."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import config


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ca = self.root / "ca.pem"
        self.ca.touch()
        self.env = self.root / ".env"
        self.values = dict(
            DB_HOST="first.example.test", DB_PORT="3306", DB_NAME="fraud_test",
            DB_USER="dummy_user", DB_PASSWORD="dummy_password",
            DB_SSL_CA=str(self.ca), MLFLOW_TRACKING_URI="http://first.example.test:5000",
        )
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.write_env()

    def write_env(self):
        self.env.write_text("\n".join(f"{k}='{v}'" for k, v in self.values.items()), encoding="utf-8")

    def test_reads_database_settings_and_hides_password_in_repr(self):
        settings = config.get_database_settings(self.env)
        self.assertEqual((settings.host, settings.port, settings.database),
                         ("first.example.test", 3306, "fraud_test"))
        self.assertNotIn("dummy_password", repr(settings))
        with patch("ssl.create_default_context") as context:
            kwargs = settings.connection_kwargs()
        context.assert_called_once_with(cafile=str(self.ca))
        self.assertEqual(kwargs["password"], "dummy_password")
        self.assertEqual(kwargs["database"], "fraud_test")
        self.assertIs(kwargs["ssl"], context.return_value)

    def test_configuration_changes_are_read_without_source_edits(self):
        config.get_database_settings(self.env)
        self.values.update(DB_HOST="second.example.test", DB_PORT="3307",
                           DB_USER="second_user", DB_PASSWORD="second_password",
                           MLFLOW_TRACKING_URI="https://second.example.test")
        self.write_env()
        settings = config.get_database_settings(self.env)
        self.assertEqual((settings.host, settings.port, settings.user, settings.password),
                         ("second.example.test", 3307, "second_user", "second_password"))
        self.assertEqual(config.get_mlflow_tracking_uri(self.env), "https://second.example.test")

    def test_environment_takes_precedence(self):
        os.environ.update(DB_HOST="override.example.test", DB_PORT="3308")
        settings = config.get_database_settings(self.env)
        self.assertEqual((settings.host, settings.port), ("override.example.test", 3308))
        self.assertNotIn("DB_PASSWORD", os.environ)

    def test_each_required_database_setting_produces_clear_error(self):
        for name in ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"):
            with self.subTest(name=name), patch.dict(os.environ, {name: ""}):
                with self.assertRaisesRegex(config.ConfigurationError, name) as caught:
                    config.get_database_settings(self.env)
                self.assertNotIn("dummy_password", str(caught.exception))

    def test_invalid_ports_are_rejected(self):
        for value in ("abc", "0", "65536", "3.5"):
            with self.subTest(value=value), patch.dict(os.environ, {"DB_PORT": value}):
                with self.assertRaisesRegex(config.ConfigurationError, "DB_PORT"):
                    config.get_database_settings(self.env)

    def test_service_settings_are_validated_independently(self):
        self.values.pop("MLFLOW_TRACKING_URI")
        self.write_env()
        config.get_database_settings(self.env)
        with self.assertRaisesRegex(config.ConfigurationError, "MLFLOW_TRACKING_URI"):
            config.get_mlflow_tracking_uri(self.env)
        self.env.write_text("MLFLOW_TRACKING_URI=https://tracking.example.test\n", encoding="utf-8")
        self.assertEqual(config.get_mlflow_tracking_uri(self.env), "https://tracking.example.test")

    def test_invalid_mlflow_urls_do_not_leak_values(self):
        for uri in ("file:///tmp/mlruns", "example.test", "https://host:bad",
                    "http://host:0", "https://user:secret@host", "https://host?token=secret",
                    "https://ho st", "https://host#secret"):
            with self.subTest(uri=uri), patch.dict(os.environ, {"MLFLOW_TRACKING_URI": uri}):
                with self.assertRaisesRegex(config.ConfigurationError, "MLFLOW_TRACKING_URI") as caught:
                    config.get_mlflow_tracking_uri(self.env)
                self.assertNotIn("secret", str(caught.exception))

    def test_password_characters_are_preserved(self):
        self.values["DB_PASSWORD"] = "space # ${not_a_variable} $literal"
        self.write_env()
        self.assertEqual(config.get_database_settings(self.env).password, self.values["DB_PASSWORD"])

    def test_default_env_and_certificate_paths_are_project_relative(self):
        self.values["DB_SSL_CA"] = "ca.pem"
        self.write_env()
        with patch.object(config, "PROJECT_ROOT", self.root):
            self.assertEqual(config.get_database_settings().ssl_ca, self.ca)
        with patch.dict(os.environ, {"DB_SSL_CA": str(self.root / "missing.pem")}):
            with self.assertRaisesRegex(config.ConfigurationError, "DB_SSL_CA"):
                config.get_database_settings(self.env)

    def test_mlflow_receives_configured_url_without_network(self):
        fake_mlflow = types.ModuleType("mlflow")
        with patch.object(fake_mlflow, "set_tracking_uri", create=True) as set_uri:
            with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
                config.configure_mlflow(self.env)
        set_uri.assert_called_once_with("http://first.example.test:5000")

    def test_cli_missing_setting_is_actionable(self):
        env = dict(os.environ, DB_HOST="", DB_PASSWORD="dummy_password")
        result = subprocess.run(
            [sys.executable, str(Path(config.__file__).resolve()), "--service", "database"],
            env=env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("DB_HOST", result.stderr)
        self.assertNotIn("dummy_password", result.stderr)


if __name__ == "__main__":
    unittest.main()
