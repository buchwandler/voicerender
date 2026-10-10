"""Canonical engine-catalog metadata normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GENDERS = frozenset({"female", "male", "neutral", "unknown"})


def normalize_language_key(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("language must be a non-empty string")
    return normalized


def normalize_locale_tag(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("code", "")
    if not isinstance(value, str) or not value.strip():
        return ""
    return normalize_language_key(value)


def language_base(value: Any) -> str:
    return normalize_locale_tag(value).partition("-")[0]


def language_tags_match(requested: str, available: str) -> bool:
    requested_tag = normalize_locale_tag(requested)
    available_tag = normalize_locale_tag(available)
    if not requested_tag or not available_tag:
        return False
    requested_base = language_base(requested_tag)
    available_base = language_base(available_tag)
    return requested_base == available_base and (
        requested_tag in (available_tag, requested_base)
        or available_tag == available_base
    )


def normalize_gender(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in GENDERS:
        return value.strip().lower()
    return "unknown"


__all__ = [
    "GENDERS",
    "language_base",
    "language_tags_match",
    "normalize_gender",
    "normalize_language_key",
    "normalize_locale_tag",
]
