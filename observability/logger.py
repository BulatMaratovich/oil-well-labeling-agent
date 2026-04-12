"""
observability/logger.py — Structured pipeline logger.

Wraps Python's stdlib logging with structured key-value context so every
pipeline event carries run_id, task_id, asset_id, stage.

Usage
-----
    from observability.logger import get_logger
    log = get_logger(__name__)
    log.info("stage_complete", stage="input_normalizer", rows=1024)

When structlog is installed it is used automatically; otherwise falls back
to stdlib logging with a JSON formatter.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Stdlib JSON formatter (fallback)
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # Attach any extra kwargs passed via log.info("msg", extra={...})
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ) and not key.startswith("_"):
                data[key] = val
        return json.dumps(data, ensure_ascii=False, default=str)


def _configure_stdlib() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Structlog integration (optional)
# ---------------------------------------------------------------------------

def _try_configure_structlog() -> bool:
    try:
        import structlog  # type: ignore[import]

        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return True
    except ImportError:
        return False


_structlog_configured = _try_configure_structlog()
if not _structlog_configured:
    _configure_stdlib()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class _StdlibAdapter:
    """Thin wrapper: log.info("event", key=val) → stdlib logger."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    # LogRecord reserves these attribute names — prefix conflicts to avoid KeyError
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def _log(self, level: int, event: str, **kw: Any) -> None:
        safe = {
            (f"kw_{k}" if k in self._RESERVED else k): v
            for k, v in kw.items()
        }
        self._logger.log(level, event, extra=safe)

    def debug(self, event: str, **kw: Any) -> None:
        self._log(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._log(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._log(logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._log(logging.ERROR, event, **kw)

    def exception(self, event: str, **kw: Any) -> None:
        self._logger.exception(event, extra=kw)


def get_logger(name: str):
    """Return a structured logger for *name*.

    Returns a structlog BoundLogger when structlog is available,
    otherwise a stdlib adapter with the same interface.
    """
    if _structlog_configured:
        import structlog  # type: ignore[import]
        return structlog.get_logger(name)
    return _StdlibAdapter(name)
