"""Safe, engine-neutral serialization for exception diagnostics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

ERROR_DIAGNOSTIC_SCHEMA = "voicerender.error-diagnostic.v1"
_MAX_MESSAGE_LENGTH = 4096
_MAX_DETAIL_STRING_LENGTH = 2048
_MAX_DETAIL_ITEMS = 100
_MAX_DETAIL_DEPTH = 8
_OMIT = object()

_SIGNED_URL_QUERY = re.compile(r"(?P<url>https?://[^\s?#]+)\?[^\s#]*", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(hf[_-]?token|access[_-]?token|api[_-]?key|authorization|"
    r"x-amz-(?:signature|credential|security-token)|signature|token)(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)


def sanitize_error_message(value: object, *, limit: int = _MAX_MESSAGE_LENGTH) -> str:
    """Bound diagnostic text and redact signed URL queries and inline credentials."""
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - diagnostics must not mask the original error.
        text = f"<{type(value).__name__} message unavailable>"
    text = _SIGNED_URL_QUERY.sub(r"\g<url>?[REDACTED]", text)
    text = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _limited_text(value: object, limit: int) -> str:
    return sanitize_error_message(value, limit=limit)


def _sensitive_key(key: str) -> bool:
    normalized = "".join(character for character in key.lower() if character.isalnum())
    sensitive_markers = (
        "audio",
        "environment",
        "phoneme",
        "speech",
        "token",
        "transcript",
        "utterance",
    )
    return (
        normalized in {"env", "envvars", "text"}
        or any(marker in normalized for marker in sensitive_markers)
        or (normalized.endswith("text") and normalized != "context")
    )


def sanitize_json_value(
    value: object,
    *,
    max_depth: int = _MAX_DETAIL_DEPTH,
    _seen: set[int] | None = None,
) -> object:
    """Return JSON-native data, limiting size and omitting sensitive fields."""
    seen = set() if _seen is None else _seen
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            return _limited_text(value, _MAX_DETAIL_STRING_LENGTH)
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return _limited_text(value.as_posix(), _MAX_DETAIL_STRING_LENGTH)

    value_type = type(value)
    if value_type.__module__.startswith("numpy"):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except Exception:  # noqa: BLE001
                scalar = _OMIT
            if scalar is not _OMIT and scalar is not value:
                return sanitize_json_value(scalar, max_depth=max_depth, _seen=seen)

    if max_depth <= 0:
        return f"[omitted nested {value_type.__name__}]"

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return "[cycle omitted]"
        seen.add(identity)
        sanitized: dict[str, object] = {}
        try:
            for index, (key, child) in enumerate(value.items()):
                if index >= _MAX_DETAIL_ITEMS:
                    sanitized["_truncated"] = True
                    break
                if not isinstance(key, str) or _sensitive_key(key):
                    continue
                child_value = sanitize_json_value(
                    child, max_depth=max_depth - 1, _seen=seen
                )
                if child_value is not _OMIT:
                    sanitized[_limited_text(key, 128)] = child_value
        except Exception:  # noqa: BLE001 - hostile mappings are omitted safely.
            return f"[omitted unreadable {value_type.__name__}]"
        finally:
            seen.discard(identity)
        return sanitized

    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return "[cycle omitted]"
        seen.add(identity)
        result: list[object] = []
        try:
            for index, child in enumerate(value):
                if index >= _MAX_DETAIL_ITEMS:
                    result.append("[truncated]")
                    break
                child_value = sanitize_json_value(
                    child, max_depth=max_depth - 1, _seen=seen
                )
                if child_value is not _OMIT:
                    result.append(child_value)
        except Exception:  # noqa: BLE001
            return f"[omitted unreadable {value_type.__name__}]"
        finally:
            seen.discard(identity)
        return result

    return f"[omitted unsupported {value_type.__module__}.{value_type.__name__}]"


def _safe_attribute(error: BaseException, name: str) -> str | None:
    try:
        value = getattr(error, name, None)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, (str, int, float, bool)):
        return _limited_text(value, _MAX_DETAIL_STRING_LENGTH)
    return None


def _exception_node(error: BaseException) -> dict[str, object]:
    error_type = type(error)
    node: dict[str, object] = {
        "type": error_type.__name__,
        "module": error_type.__module__,
        "message": _limited_text(error, _MAX_MESSAGE_LENGTH),
    }
    for attribute in ("code", "stage", "native_error_type", "native_error_message"):
        value = _safe_attribute(error, attribute)
        if value is not None:
            node[attribute] = value

    try:
        raw_details = getattr(error, "details", None)
    except Exception:  # noqa: BLE001
        raw_details = None
    if isinstance(raw_details, Mapping):
        details = sanitize_json_value(raw_details)
        if isinstance(details, dict):
            engine_diagnostic = details.pop("native_diagnostic", None)
            if details:
                node["details"] = details
            if isinstance(engine_diagnostic, (dict, list)):
                node["engine_diagnostic"] = engine_diagnostic
    return node


def exception_diagnostic(
    error: BaseException,
    *,
    max_depth: int = 8,
) -> dict[str, object]:
    """Serialize a bounded Python cause/context chain without traceback locals."""
    max_depth = max(max_depth, 1)
    diagnostic: dict[str, object] = {
        "schema": ERROR_DIAGNOSTIC_SCHEMA,
    }
    seen: set[int] = set()
    current: BaseException | None = error
    previous: dict[str, object] | None = None

    for index in range(max_depth):
        if current is None:
            break
        identity = id(current)
        if identity in seen:
            if previous is not None:
                previous["cause"] = {
                    "type": type(current).__name__,
                    "module": type(current).__module__,
                    "message": "[exception cycle omitted]",
                }
            break
        seen.add(identity)
        node = _exception_node(current)
        if previous is None:
            diagnostic.update(node)
            previous = diagnostic
        else:
            previous["cause"] = node
            previous = node

        explicit_cause = current.__cause__
        if explicit_cause is not None:
            current = explicit_cause
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
        if current is not None and index == max_depth - 1:
            previous["cause_truncated"] = True

    return diagnostic


def root_cause(diagnostic: Mapping[str, object]) -> Mapping[str, object]:
    """Return the deepest mapping in a serialized diagnostic cause chain."""
    current: Mapping[str, object] = diagnostic
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        cause = current.get("cause")
        if not isinstance(cause, Mapping):
            break
        current = cause
    return current


def diagnostic_stage(diagnostic: Mapping[str, object]) -> str | None:
    """Find the most specific available engine/error stage in a diagnostic tree."""
    pending: list[Mapping[str, object]] = [diagnostic]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        stage = current.get("stage")
        if isinstance(stage, str) and stage:
            return stage
        details = current.get("details")
        if isinstance(details, Mapping):
            stage = details.get("native_stage") or details.get("stage")
            if isinstance(stage, str) and stage:
                return stage
        engine_diagnostic = current.get("engine_diagnostic")
        cause = current.get("cause")
        if isinstance(engine_diagnostic, Mapping):
            pending.append(engine_diagnostic)
        if isinstance(cause, Mapping):
            pending.append(cause)
    return None


def diagnostic_summary_fields(
    diagnostic: Mapping[str, object] | None,
) -> dict[str, str | None]:
    """Return compact stage and deepest-cause fields for report/API results."""
    if diagnostic is None:
        return {
            "error_stage": None,
            "error_root_cause_type": None,
            "error_root_cause_message": None,
        }
    cause = root_cause(diagnostic)
    error_type = cause.get("type")
    error_message = cause.get("message")
    return {
        "error_stage": diagnostic_stage(diagnostic),
        "error_root_cause_type": error_type if isinstance(error_type, str) else None,
        "error_root_cause_message": (
            error_message if isinstance(error_message, str) else None
        ),
    }


def root_cause_summary(diagnostic: Mapping[str, object]) -> str | None:
    """Return a compact root-cause line, unless it repeats the outer message."""
    cause = root_cause(diagnostic)
    message = cause.get("message")
    if not isinstance(message, str) or not message:
        return None
    outer_message = diagnostic.get("message")
    if message == outer_message:
        return None
    error_type = cause.get("type")
    if isinstance(error_type, str) and error_type:
        return f"{error_type}: {message}"
    return message
