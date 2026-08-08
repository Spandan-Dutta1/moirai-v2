"""
Structured logging for Moirai.

Logs here are not prose, they are records. Every call attaches typed
key-value context, which the console renders for humans and the JSON
formatter renders for the Run Ledger.

    log.info("var_estimated", n_lags=4, seed=42, duration_ms=1203)

Console:  14:22:01 INFO     var_estimated  n_lags=4 seed=42 duration_ms=1203
JSON:     {"ts": "...", "level": "INFO", "event": "var_estimated", ...}

Why this matters for Moirai: a log line saying "estimated the model" cannot
support a provenance claim. A record carrying the seed, the lag order and
the data vintage can.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from moirai.core.config import LogLevel, get_settings

LOGGER_ROOT = "moirai"

# Attributes LogRecord always carries. Anything else was added by us.
_RESERVED: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}

# Ambient context merged into every record: run_id, experiment_hash, layer...
_context: ContextVar[dict[str, Any]] = ContextVar("moirai_log_context", default={})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach fields to every log record emitted inside this block.

    Nested blocks merge; the inner values win. Restoration is guaranteed
    even if the body raises.

        with log_context(run_id=run_id, layer="causal"):
            log.info("var_estimated", n_lags=4)
    """
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> dict[str, Any]:
    """Read the ambient logging context. Returns a copy."""
    return dict(_context.get())


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Structured fields attached to a record, excluding stdlib internals."""
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, suitable for the Run Ledger."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_extras(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str so Paths, datetimes and enums never crash a log call.
        return json.dumps(payload, default=str, sort_keys=True)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable line with trailing key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        head = f"{stamp} {record.levelname:<8} {record.getMessage()}"
        extras = _extras(record)
        if extras:
            head += "  " + " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        if record.exc_info:
            head += "\n" + self.formatException(record.exc_info)
        return head


class ContextFilter(logging.Filter):
    """Merge the ambient context into each record before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _context.get().items():
            if key not in record.__dict__:
                record.__dict__[key] = value
        return True


class MoiraiLogger:
    """Thin wrapper turning keyword arguments into structured fields.

    Exists so call sites read `log.info("event", key=value)` instead of the
    stdlib's `logger.info("event", extra={"key": value})`, which is verbose
    enough that people stop adding context.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(self, level: int, event: str, exc_info: bool = False, **fields: Any) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, event, extra=fields, exc_info=exc_info, stacklevel=3)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def critical(self, event: str, **fields: Any) -> None:
        self._log(logging.CRITICAL, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        """Log at ERROR with the active exception's traceback attached."""
        self._log(logging.ERROR, event, exc_info=True, **fields)


def configure_logging(
    level: LogLevel | str | None = None,
    *,
    json_output: bool | None = None,
    stream: Any = None,
) -> logging.Logger:
    """Configure the `moirai` logger. Idempotent: safe to call repeatedly.

    Arguments override settings; settings supply the defaults.
    """
    settings = get_settings()
    resolved_level = LogLevel(level) if level is not None else settings.log_level
    use_json = settings.log_json if json_output is None else json_output

    root = logging.getLogger(LOGGER_ROOT)
    root.setLevel(resolved_level.value)
    root.propagate = False  # do not leak into the global root logger

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if use_json else ConsoleFormatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)

    return root


def get_logger(name: str | None = None) -> MoiraiLogger:
    """Return a structured logger under the `moirai` namespace.

        log = get_logger(__name__)
    """
    if name is None or name == LOGGER_ROOT:
        full_name = LOGGER_ROOT
    elif name.startswith(f"{LOGGER_ROOT}."):
        full_name = name
    else:
        full_name = f"{LOGGER_ROOT}.{name}"
    return MoiraiLogger(logging.getLogger(full_name))