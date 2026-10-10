"""Public neutral metadata type for discovered pronunciation lexicons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LexiconCatalogEntry:
    """A discovered lexicon selector and its optional asset metadata."""

    selector: str
    engine: str
    language: str
    locale: str
    asset_id: str | None
    data_backend: str | None
    default: bool
    installed: bool | None
    models: tuple[str, ...] = ()
    model_support: str = "unknown"
    display_name: str | None = None
    phoneme_encoding: str | None = None
    data_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "engine": self.engine,
            "language": self.language,
            "locale": self.locale,
            "asset_id": self.asset_id,
            "data_backend": self.data_backend,
            "default": self.default,
            "installed": self.installed,
            "models": list(self.models),
            "model_support": self.model_support,
            "display_name": self.display_name,
            "phoneme_encoding": self.phoneme_encoding,
            "data_version": self.data_version,
        }
