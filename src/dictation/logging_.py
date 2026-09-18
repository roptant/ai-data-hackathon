"""Structured logging that cannot carry dictated text.

Plan sections 8 and 9 require that API errors, logs and operational metadata
never contain transcript content.  A convention would not hold, so the log
enforces it: values must be numbers, booleans, enums or short token-shaped
strings.  Anything with whitespace or of prose length is refused at the call
site, which turns a privacy leak into a test failure.

Use :func:`content_digest` when a log line genuinely needs to refer to some
text, for example to correlate a duplicate.  It is keyed with a per-install
salt, so digests are not comparable across machines and are not reversible by
guessing short strings.
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import time
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

LOGGER_NAME = "dictation"

#: Token-shaped strings only: identifiers, enum values, reason codes, versions.
_SAFE_STRING = re.compile(r"^[A-Za-z0-9_.:@/+-]{0,96}$")

_FORBIDDEN_KEYS = frozenset(
    {
        "text",
        "transcript",
        "prompt",
        "words",
        "audio",
        "clip_text",
        "spans",
        "token",
        "secret",
        "password",
    }
)


class ForbiddenLogField(ValueError):
    """Raised when a log call would put content into a log record."""


_salt: bytes | None = None


def _install_salt() -> bytes:
    global _salt
    if _salt is None:
        _salt = secrets.token_bytes(32)
    return _salt


def set_install_salt(salt: bytes) -> None:
    """Install the persistent digest salt (called by the store at startup)."""
    global _salt
    if len(salt) < 16:
        raise ValueError("digest salt must be at least 16 bytes")
    _salt = salt


def content_digest(text: str, *, length: int = 16) -> str:
    """Salted, truncated digest of text.  Safe to log; not reversible."""
    mac = hmac.new(_install_salt(), text.encode("utf-8"), "sha256").hexdigest()
    return mac[:length]


def _check_key(key: str) -> None:
    if key.lower() in _FORBIDDEN_KEYS:
        raise ForbiddenLogField(f"field {key!r} may not be logged")


def _check_value(key: str, value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (list, tuple)):
        return [_check_value(key, item) for item in value]
    if isinstance(value, str):
        if not _SAFE_STRING.match(value):
            raise ForbiddenLogField(
                f"field {key!r} is not token-shaped; log a reason code or "
                f"content_digest() instead of free text"
            )
        return value
    raise ForbiddenLogField(f"field {key!r} has unloggable type {type(value).__name__}")


def sanitize(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a field mapping, raising on anything content-shaped."""
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        _check_key(key)
        clean[key] = _check_value(key, value)
    return clean


class EventLog:
    """Emitter for operational events.

    Events are the audit trail the plan asks for: decisions, reason codes,
    state transitions and counts, with no payload content.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(LOGGER_NAME)

    def emit(self, event: str, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
        record = {"event": event, "ts": round(time.time(), 3), **sanitize(fields)}
        self._logger.log(level, json.dumps(record, sort_keys=True), extra={"dictation": record})
        return record

    def warn(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit(event, logging.WARNING, **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit(event, logging.ERROR, **fields)


events = EventLog()


def configure(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Attach a formatter that prints only the structured record."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
