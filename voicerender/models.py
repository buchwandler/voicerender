"""Engine-neutral request, profile and result contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from .errors import PlanError, ProfileError

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

SCHEMA_PLAN = "utterplan.voiceplan.v1"
SCHEMA_PROFILE = "voicerender.voice-profile.v1"
SCHEMA_MANIFEST = "voicerender.render-manifest.v1"


def _mapping(value: Any, what: str, error: type[ValueError]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise error(f"{what} must be a table/object")
    return value


def _nonempty(value: Any, what: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{what} must be a non-empty string")
    return value


def _options(value: Any, what: str) -> dict[str, Any]:
    opts = dict(_mapping(value, what, ProfileError))
    try:
        json.dumps(opts, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProfileError(
            f"{what} must contain finite, JSON-compatible values"
        ) from exc
    return opts


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    id: str
    text: str
    language: str = "en-US"
    role: str = "narrator"
    scope_id: str = "default"

    def __post_init__(self) -> None:
        for key in ("id", "role", "scope_id", "language"):
            _nonempty(getattr(self, key), f"segment.{key}", PlanError)
        _nonempty(self.text, "segment.text", PlanError)

    def as_dict(self) -> dict[str, str]:
        return {
            k: getattr(self, k) for k in ("id", "scope_id", "role", "language", "text")
        }


@dataclass(frozen=True, slots=True)
class VoiceBinding:
    engine: str
    target_id: str
    voice: str | None = None
    speaker: str | int | None = None
    reference_audio: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    model_revision: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, base_dir: Path) -> VoiceBinding:
        engine = _nonempty(data.get("engine"), "role.engine", ProfileError)
        target = _nonempty(data.get("target_id"), "role.target_id", ProfileError)
        voice = data.get("voice")
        reference = data.get("reference_audio")
        if voice is not None:
            _nonempty(voice, "role.voice", ProfileError)
        if reference is not None:
            _nonempty(reference, "role.reference_audio", ProfileError)
            reference = (base_dir / Path(reference).expanduser()).resolve()
        if reference is not None and voice is not None:
            raise ProfileError("Choose either voice or reference_audio for a role")
        if reference is not None and engine != "pocket":
            raise ProfileError(
                f"Engine {engine!r} does not support reference_audio in this MVP"
            )
        speaker = data.get("speaker")
        if speaker is not None and (
            isinstance(speaker, bool) or not isinstance(speaker, (str, int))
        ):
            raise ProfileError("speaker must be a name or integer")
        revision = data.get("model_revision")
        if revision is not None:
            _nonempty(revision, "role.model_revision", ProfileError)
        known = {
            "engine",
            "target_id",
            "voice",
            "speaker",
            "reference_audio",
            "options",
            "model_revision",
        }
        extra = set(data).difference(known)
        if extra:
            raise ProfileError(f"Unknown role fields: {', '.join(sorted(extra))}")
        return cls(
            engine=engine,
            target_id=target,
            voice=voice,
            speaker=speaker,
            reference_audio=reference,
            options=_options(data.get("options", {}), "role.options"),
            model_revision=revision,
        )


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    name: str
    roles: Mapping[str, VoiceBinding]
    schema: str = SCHEMA_PROFILE

    @classmethod
    def load(cls, path: str | Path) -> VoiceProfile:
        location = Path(path).expanduser().resolve()
        try:
            with location.open("rb") as stream:
                data = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ProfileError(f"Cannot read voice profile {location}: {exc}") from exc
        if data.get("schema") != SCHEMA_PROFILE:
            raise ProfileError(f"Expected profile schema {SCHEMA_PROFILE!r}")
        name = _nonempty(data.get("name"), "profile.name", ProfileError)
        roles = _mapping(data.get("roles"), "profile.roles", ProfileError)
        if not roles:
            raise ProfileError("At least one role binding is required")
        unknown = set(data).difference({"schema", "name", "roles"})
        if unknown:
            raise ProfileError(
                f"Unknown top-level profile fields: {', '.join(sorted(unknown))}"
            )
        return cls(
            name=name,
            roles={
                _nonempty(role, "role name", ProfileError): VoiceBinding.from_mapping(
                    _mapping(raw, f"roles.{role}", ProfileError),
                    base_dir=location.parent,
                )
                for role, raw in roles.items()
            },
        )


@dataclass(frozen=True, slots=True)
class VoicePlan:
    segments: tuple[SpeechSegment, ...]
    source_plan_id: str | None = None
    source_plan: str | None = None
    schema: str = SCHEMA_PLAN

    @classmethod
    def load(cls, path: str | Path) -> VoicePlan:
        location = Path(path)
        try:
            with location.open("rb") as stream:
                data = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PlanError(f"Cannot read voiceplan {location}: {exc}") from exc
        if data.get("schema") != SCHEMA_PLAN:
            raise PlanError(f"Expected voiceplan schema {SCHEMA_PLAN!r}")
        unknown_root = set(data).difference(
            {"schema", "source_plan_id", "source_plan", "segments"}
        )
        if unknown_root:
            raise PlanError(f"Unknown voiceplan fields: {sorted(unknown_root)}")
        for key in ("source_plan_id", "source_plan"):
            if data.get(key) is not None:
                _nonempty(data[key], key, PlanError)
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise PlanError("voiceplan must include one or more [[segments]]")
        segments: list[SpeechSegment] = []
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(raw_segments):
            item = _mapping(item, f"segments[{index}]", PlanError)
            unknown = set(item).difference(
                {"id", "scope_id", "text", "language", "role"}
            )
            if unknown:
                raise PlanError(
                    f"Unsupported speech metadata in segment {index}: {sorted(unknown)}; do not drop semantics"
                )
            segment = SpeechSegment(
                id=item.get("id"),
                scope_id=item.get("scope_id", "default"),
                text=item.get("text"),
                role=item.get("role", "narrator"),
                language=item.get("language", "en-US"),
            )
            identity = (segment.scope_id, segment.id)
            if identity in seen:
                raise PlanError(f"Duplicate logical speech segment {identity}")
            seen.add(identity)
            segments.append(segment)
        return cls(
            segments=tuple(segments),
            source_plan_id=data.get("source_plan_id"),
            source_plan=data.get("source_plan"),
        )


@dataclass(frozen=True, slots=True)
class RenderResult:
    manifest_path: Path
    rendered_count: int
    reused_count: int
    failed_count: int
    selected_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "rendered": self.rendered_count,
            "reused": self.reused_count,
            "failed": self.failed_count,
            "selected": self.selected_count,
        }
