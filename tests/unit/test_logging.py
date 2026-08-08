"""Tests for structured logging.

Every test writes into an in-memory StringIO rather than stderr, so
assertions run against exact bytes and nothing pollutes the test output.
"""

import io
import json
import logging

import pytest

from moirai.core.config import LogLevel, reset_settings_cache
from moirai.core.logging import (
    LOGGER_ROOT,
    ConsoleFormatter,
    JsonFormatter,
    configure_logging,
    current_context,
    get_logger,
    log_context,
)


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture(autouse=True)
def _clean_logging():
    """Tear down handlers so no test inherits another's configuration."""
    reset_settings_cache()
    yield
    root = logging.getLogger(LOGGER_ROOT)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    reset_settings_cache()


def _lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line.strip()]


def _json_lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in _lines(stream)]


# --- logger naming ---------------------------------------------------------

@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, "moirai"),
        ("moirai", "moirai"),
        ("engine.causal", "moirai.engine.causal"),
        ("moirai.engine.causal", "moirai.engine.causal"),
    ],
)
def test_logger_names_live_under_the_moirai_namespace(given, expected):
    assert get_logger(given).name == expected


def test_moirai_logger_does_not_propagate_to_the_global_root():
    """Otherwise every library's root handler would duplicate our output."""
    configure_logging("INFO", json_output=True, stream=io.StringIO())
    assert logging.getLogger(LOGGER_ROOT).propagate is False


# --- JSON output -----------------------------------------------------------

def test_json_record_has_the_expected_shape(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    get_logger("demo").info("var_estimated", n_lags=4, seed=42)

    record = _json_lines(stream)[0]
    assert record["event"] == "var_estimated"
    assert record["level"] == "INFO"
    assert record["logger"] == "moirai.demo"
    assert record["n_lags"] == 4
    assert record["seed"] == 42
    assert "ts" in record


def test_json_timestamp_is_utc_iso8601(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    get_logger("demo").info("event")
    ts = _json_lines(stream)[0]["ts"]
    assert ts.endswith("+00:00")


def test_each_json_record_is_one_line(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    log = get_logger("demo")
    log.info("first")
    log.info("second")
    assert len(_lines(stream)) == 2


def test_unserialisable_values_do_not_crash_the_call(stream):
    """default=str means a Path or a custom object degrades, never explodes."""
    from pathlib import Path

    configure_logging("INFO", json_output=True, stream=stream)
    get_logger("demo").info("wrote_file", path=Path("data/raw/x.csv"))
    assert "x.csv" in _json_lines(stream)[0]["path"]


# --- console output --------------------------------------------------------

def test_console_line_contains_level_event_and_fields(stream):
    configure_logging("INFO", json_output=False, stream=stream)
    get_logger("demo").info("var_estimated", n_lags=4)
    line = _lines(stream)[0]
    assert "INFO" in line
    assert "var_estimated" in line
    assert "n_lags=4" in line


def test_console_output_is_not_json(stream):
    configure_logging("INFO", json_output=False, stream=stream)
    get_logger("demo").info("event")
    with pytest.raises(json.JSONDecodeError):
        json.loads(_lines(stream)[0])


# --- level filtering -------------------------------------------------------

def test_messages_below_the_level_are_dropped(stream):
    configure_logging("WARNING", json_output=True, stream=stream)
    log = get_logger("demo")
    log.debug("dropped")
    log.info("dropped")
    log.warning("kept")
    log.error("kept")

    events = [r["event"] for r in _json_lines(stream)]
    assert events == ["kept", "kept"]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
    ],
)
def test_each_method_logs_at_its_level(stream, method, expected):
    configure_logging("DEBUG", json_output=True, stream=stream)
    getattr(get_logger("demo"), method)("event")
    assert _json_lines(stream)[0]["level"] == expected


def test_level_accepts_a_log_level_enum(stream):
    configure_logging(LogLevel.ERROR, json_output=True, stream=stream)
    log = get_logger("demo")
    log.warning("dropped")
    log.error("kept")
    assert [r["event"] for r in _json_lines(stream)] == ["kept"]


# --- ambient context -------------------------------------------------------

def test_context_fields_are_attached_to_records(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    log = get_logger("demo")
    with log_context(run_id="run-001", layer="causal"):
        log.info("var_estimated", n_lags=4)

    record = _json_lines(stream)[0]
    assert record["run_id"] == "run-001"
    assert record["layer"] == "causal"
    assert record["n_lags"] == 4


def test_context_is_removed_after_the_block(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    log = get_logger("demo")
    with log_context(run_id="run-001"):
        log.info("inside")
    log.info("outside")

    inside, outside = _json_lines(stream)
    assert inside["run_id"] == "run-001"
    assert "run_id" not in outside


def test_nested_contexts_merge_with_inner_winning(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    log = get_logger("demo")
    with log_context(run_id="run-001", layer="fabric"):
        with log_context(layer="causal"):
            log.info("event")

    record = _json_lines(stream)[0]
    assert record["run_id"] == "run-001"
    assert record["layer"] == "causal"


def test_context_is_restored_even_when_the_body_raises(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    with pytest.raises(ValueError):  # noqa: SIM117
        with log_context(run_id="run-001"):
            raise ValueError("boom")
    assert current_context() == {}


def test_explicit_field_overrides_the_context(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    with log_context(layer="fabric"):
        get_logger("demo").info("event", layer="causal")
    assert _json_lines(stream)[0]["layer"] == "causal"


def test_current_context_returns_a_copy():
    with log_context(run_id="run-001"):
        snapshot = current_context()
        snapshot["run_id"] = "mutated"
        assert current_context()["run_id"] == "run-001"


# --- exceptions ------------------------------------------------------------

def test_exception_logging_includes_the_traceback(stream):
    configure_logging("INFO", json_output=True, stream=stream)
    log = get_logger("demo")
    try:
        raise ValueError("underlying failure")
    except ValueError:
        log.exception("ingestion_failed", source="rbi")

    record = _json_lines(stream)[0]
    assert record["level"] == "ERROR"
    assert "ValueError" in record["exception"]
    assert "underlying failure" in record["exception"]
    assert record["source"] == "rbi"


def test_exception_traceback_survives_json_encoding(stream):
    """Multi-line tracebacks must not break the one-record-per-line contract."""
    configure_logging("INFO", json_output=True, stream=stream)
    try:
        raise RuntimeError("multi\nline")
    except RuntimeError:
        get_logger("demo").exception("failed")
    assert len(_lines(stream)) == 1


# --- configure_logging -----------------------------------------------------

def test_configure_is_idempotent(stream):
    """Repeated configuration must not duplicate handlers or output."""
    configure_logging("INFO", json_output=True, stream=stream)
    configure_logging("INFO", json_output=True, stream=stream)
    get_logger("demo").info("event")

    assert len(logging.getLogger(LOGGER_ROOT).handlers) == 1
    assert len(_lines(stream)) == 1


def test_arguments_override_settings(stream, monkeypatch):
    monkeypatch.setenv("MOIRAI_LOG_LEVEL", "ERROR")
    reset_settings_cache()
    configure_logging("DEBUG", json_output=True, stream=stream)
    get_logger("demo").debug("kept")
    assert len(_lines(stream)) == 1


def test_settings_supply_the_defaults(stream, monkeypatch):
    monkeypatch.setenv("MOIRAI_LOG_LEVEL", "ERROR")
    reset_settings_cache()
    configure_logging(stream=stream, json_output=True)
    log = get_logger("demo")
    log.warning("dropped")
    log.error("kept")
    assert [r["event"] for r in _json_lines(stream)] == ["kept"]


# --- formatters in isolation ----------------------------------------------

def test_json_formatter_handles_a_bare_record():
    record = logging.LogRecord("moirai.x", logging.INFO, "f.py", 1, "event", None, None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "event"
    assert payload["logger"] == "moirai.x"


def test_console_formatter_handles_a_bare_record():
    record = logging.LogRecord("moirai.x", logging.INFO, "f.py", 1, "event", None, None)
    assert "event" in ConsoleFormatter().format(record)