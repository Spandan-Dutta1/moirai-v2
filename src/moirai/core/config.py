"""
Moirai configuration.

Precedence, lowest to highest:
    1. Defaults declared below
    2. Values from a .env file
    3. Environment variables (MOIRAI_* prefix)

Two rules make this safe for a reproducible research platform:

* Every setting is typed and validated at load time. A bad value fails
  at startup, not three hours into a simulation.
* The resolved settings are serialisable, so the exact configuration that
  produced a result is recorded in that run's ledger entry.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any, ClassVar, Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from moirai.core.exceptions import ConfigError
from moirai.core.paths import get_paths

ENV_PREFIX = "MOIRAI_"


class Environment(StrEnum):
    """Which mode Moirai is running in."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Validated runtime configuration for Moirai."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8-sig",  # tolerate a BOM
        case_sensitive=False,
        extra="forbid",  # an unknown MOIRAI_* var is a typo, not a feature
        frozen=True,  # settings cannot drift mid-run
        validate_default=True,
    )

    # ---- identity ----
    environment: Environment = Environment.DEVELOPMENT
    project_name: str = "moirai"

    # ---- observability ----
    log_level: LogLevel = LogLevel.INFO
    log_json: bool = Field(
        default=False,
        description="Emit structured JSON logs instead of human-readable text.",
    )

    # ---- reproducibility ----
    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
        description="Master seed. Every stochastic component derives from this.",
    )
    strict_reproducibility: bool = Field(
        default=True,
        description="Refuse to run if the environment cannot guarantee Class A reproducibility.",
    )

    # ---- data fabric ----
    warehouse_filename: str = "moirai.duckdb"
    default_vintage: Literal["latest", "as_of"] = "latest"

    # ---- simulation ----
    n_households: int = Field(default=10_000, gt=0, le=100_000_000)
    n_firms: int = Field(default=500, gt=0, le=10_000_000)
    use_gpu: bool = False

    # ---- AI layer ----
    llm_enabled: bool = False
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = Field(default=4096, gt=0, le=200_000)
    anthropic_api_key: str | None = Field(default=None, repr=False)

    # ---- data sources ----
    fred_api_key: str | None = Field(default=None, repr=False)
    http_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    http_max_retries: int = Field(default=3, ge=0, le=10)

    # ---- validators ----

    @field_validator("project_name")
    @classmethod
    def _project_name_is_filesystem_safe(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("project_name must not be empty")
        if any(ch in cleaned for ch in '/\\:*?"<>|'):
            raise ValueError(f"project_name contains unsafe characters: {value!r}")
        return cleaned

    @field_validator("warehouse_filename")
    @classmethod
    def _warehouse_is_a_bare_filename(cls, value: str) -> str:
        if "/" in value or "\\" in value:
            raise ValueError("warehouse_filename must be a filename, not a path")
        if not value.endswith(".duckdb"):
            raise ValueError("warehouse_filename must end with .duckdb")
        return value

    @model_validator(mode="after")
    def _llm_requires_a_key(self) -> Settings:
        if self.llm_enabled and not self.anthropic_api_key:
            raise ValueError("llm_enabled is true but anthropic_api_key is not set")
        return self

    @model_validator(mode="after")
    def _production_forbids_debug_logging(self) -> Settings:
        if self.environment is Environment.PRODUCTION and self.log_level is LogLevel.DEBUG:
            raise ValueError("DEBUG logging is not permitted in production")
        return self

    # ---- derived values ----

    @property
    def warehouse_path(self):
        """Absolute path to the DuckDB file. Derived, never configured directly."""
        return get_paths().warehouse / self.warehouse_filename

    @property
    def is_testing(self) -> bool:
        return self.environment is Environment.TESTING

    #: Fields never written to a run ledger or any serialised output.
    SECRET_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"anthropic_api_key", "fred_api_key"}
    )

    def to_ledger_dict(self) -> dict[str, Any]:
        """Serialisable snapshot for the Run Ledger, with secrets removed."""
        payload = self.model_dump(mode="json")
        for secret in self.SECRET_FIELDS:
            payload.pop(secret, None)
        return payload


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once per process.

    Raises
    ------
    ConfigError
        If any setting is missing, malformed, or mutually inconsistent.
    """
    try:
        return Settings()
    except ValidationError as err:
        raise ConfigError(f"Invalid Moirai configuration:\n{err}") from err


def reset_settings_cache() -> None:
    """Clear the memoised settings. Used by tests and explicit reloads."""
    get_settings.cache_clear()