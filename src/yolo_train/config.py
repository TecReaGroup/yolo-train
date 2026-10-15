"""Application configuration loaded from TOML and environment variables."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENVIRONMENT = "development"
DEFAULT_LOG_LEVEL = "INFO"
PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIRECTORY / "config" / "config.toml"
LOG_LEVELS = ("DEBUG", "PERF", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings."""

    environment: str = DEFAULT_ENVIRONMENT
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(cls) -> "Settings":
        """Load TOML settings with environment variable overrides."""
        configured_level = DEFAULT_LOG_LEVEL
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("rb") as config_file:
                configured_level = tomllib.load(config_file).get("logging", {}).get(
                    "level", DEFAULT_LOG_LEVEL
                )
        log_level = os.getenv("LOG_LEVEL", configured_level)
        if not isinstance(log_level, str) or log_level.upper() not in LOG_LEVELS:
            raise ValueError(f"logging.level must be one of {', '.join(LOG_LEVELS)}")
        return cls(
            environment=os.getenv("APP_ENV", DEFAULT_ENVIRONMENT),
            log_level=log_level.upper(),
        )
