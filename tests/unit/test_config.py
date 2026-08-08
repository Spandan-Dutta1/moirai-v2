"""Tests for Moirai configuration.

Isolation strategy:

* The autouse fixture strips every MOIRAI_* variable and clears the
  settings cache, so no test can see configuration set by another.
* Each test constructs `Settings(_env_file=None)` directly, which bypasses
  any real `.env` on disk. Tests must not depend on the developer's machine.
"""

import json
import os

import pytest

from moirai.core.config import (
    ENV_PREFIX,
    Environment,
    LogLevel,
    Settings,
    get_settings,
    reset_settings_cache,
)
from moirai.core.exceptions import ConfigError


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Start and end every test with no MOIRAI_* vars and a cold cache."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


# --- defaults --------------------------------------------------------------

def test_defaults_are_sane():
    settings = Settings(_env_file=None)
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.log_level is LogLevel.INFO
    assert settings.random_seed == 42
    assert settings.n_households == 10_000
    assert settings.llm_enabled is False


def test_settings_are_frozen():
    settings = Settings(_env_file=None)
    with pytest.raises(Exception):
        settings.random_seed = 99  # type: ignore[misc]


# --- environment variable overrides ----------------------------------------

def test_env_var_overrides_a_default(monkeypatch):
    monkeypatch.setenv("MOIRAI_RANDOM_SEED", "1234")
    assert Settings(_env_file=None).random_seed == 1234


def test_env_var_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("moirai_n_households", "555")
    assert Settings(_env_file=None).n_households == 555


def test_strings_are_coerced_to_the_declared_type(monkeypatch):
    monkeypatch.setenv("MOIRAI_USE_GPU", "true")
    monkeypatch.setenv("MOIRAI_N_FIRMS", "42")
    settings = Settings(_env_file=None)
    assert settings.use_gpu is True
    assert settings.n_firms == 42


def test_unknown_setting_is_rejected():
    """extra='forbid' turns a typo into an immediate failure."""
    with pytest.raises(Exception):
        Settings(_env_file=None, totally_made_up_setting=1)


# --- range validation ------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_seed", "-1"),
        ("n_households", "0"),
        ("n_firms", "-10"),
        ("llm_max_tokens", "0"),
    ],
)
def test_out_of_range_values_are_rejected(monkeypatch, field, value):
    monkeypatch.setenv(f"{ENV_PREFIX}{field.upper()}", value)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_invalid_enum_value_is_rejected(monkeypatch):
    monkeypatch.setenv("MOIRAI_LOG_LEVEL", "VERBOSE")
    with pytest.raises(Exception):
        Settings(_env_file=None)


# --- custom field validators -----------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "my/project", "a:b", "x?y"])
def test_unsafe_project_names_are_rejected(monkeypatch, bad):
    monkeypatch.setenv("MOIRAI_PROJECT_NAME", bad)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_project_name_is_stripped(monkeypatch):
    monkeypatch.setenv("MOIRAI_PROJECT_NAME", "  moirai  ")
    assert Settings(_env_file=None).project_name == "moirai"


@pytest.mark.parametrize(
    "bad", ["data/moirai.duckdb", "moirai.sqlite", "sub\\moirai.duckdb"]
)
def test_warehouse_filename_must_be_a_bare_duckdb_file(monkeypatch, bad):
    monkeypatch.setenv("MOIRAI_WAREHOUSE_FILENAME", bad)
    with pytest.raises(Exception):
        Settings(_env_file=None)


# --- cross-field validators ------------------------------------------------

def test_llm_enabled_without_a_key_is_rejected(monkeypatch):
    monkeypatch.setenv("MOIRAI_LLM_ENABLED", "true")
    with pytest.raises(Exception, match="anthropic_api_key"):
        Settings(_env_file=None)


def test_llm_enabled_with_a_key_is_accepted(monkeypatch):
    monkeypatch.setenv("MOIRAI_LLM_ENABLED", "true")
    monkeypatch.setenv("MOIRAI_ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    assert Settings(_env_file=None).llm_enabled is True


def test_production_rejects_debug_logging(monkeypatch):
    monkeypatch.setenv("MOIRAI_ENVIRONMENT", "production")
    monkeypatch.setenv("MOIRAI_LOG_LEVEL", "DEBUG")
    with pytest.raises(Exception, match="DEBUG"):
        Settings(_env_file=None)


def test_development_allows_debug_logging(monkeypatch):
    monkeypatch.setenv("MOIRAI_ENVIRONMENT", "development")
    monkeypatch.setenv("MOIRAI_LOG_LEVEL", "DEBUG")
    assert Settings(_env_file=None).log_level is LogLevel.DEBUG


# --- derived values --------------------------------------------------------

def test_warehouse_path_is_derived_from_the_layout():
    settings = Settings(_env_file=None)
    assert settings.warehouse_path.name == "moirai.duckdb"
    assert settings.warehouse_path.parent.name == "warehouse"


def test_is_testing_reflects_the_environment(monkeypatch):
    monkeypatch.setenv("MOIRAI_ENVIRONMENT", "testing")
    assert Settings(_env_file=None).is_testing is True


# --- ledger serialisation --------------------------------------------------

def test_ledger_dict_is_json_serialisable():
    payload = Settings(_env_file=None).to_ledger_dict()
    json.dumps(payload)  # must not raise


def test_ledger_dict_omits_the_api_key(monkeypatch):
    monkeypatch.setenv("MOIRAI_LLM_ENABLED", "true")
    monkeypatch.setenv("MOIRAI_ANTHROPIC_API_KEY", "sk-secret-value")
    payload = Settings(_env_file=None).to_ledger_dict()
    assert "anthropic_api_key" not in payload
    assert "sk-secret-value" not in str(payload)


def test_repr_does_not_leak_the_api_key(monkeypatch):
    monkeypatch.setenv("MOIRAI_LLM_ENABLED", "true")
    monkeypatch.setenv("MOIRAI_ANTHROPIC_API_KEY", "sk-secret-value")
    assert "sk-secret-value" not in repr(Settings(_env_file=None))


# --- get_settings wrapper --------------------------------------------------

def test_get_settings_wraps_validation_errors_in_config_error(monkeypatch):
    monkeypatch.setenv("MOIRAI_RANDOM_SEED", "-1")
    reset_settings_cache()
    with pytest.raises(ConfigError):
        get_settings()


def test_get_settings_is_cached(monkeypatch):
    reset_settings_cache()
    first = get_settings()
    monkeypatch.setenv("MOIRAI_RANDOM_SEED", "999")
    assert get_settings() is first


def test_resetting_the_cache_reloads_settings(monkeypatch):
    reset_settings_cache()
    assert get_settings().random_seed == 42
    monkeypatch.setenv("MOIRAI_RANDOM_SEED", "777")
    reset_settings_cache()
    assert get_settings().random_seed == 777