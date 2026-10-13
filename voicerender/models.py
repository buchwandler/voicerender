"""Engine-neutral request, profile and result contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import PlanError, ProfileError

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

SCHEMA_PLAN = "utterplan.voiceplan.v1"
SCHEMA_PROFILE = "voicerender.voice-profile.v1"
SCHEMA_MANIFEST = "voicerender.render-manifest.v1"
SCHEMA_VOICE_CONFIG = "voicerender.voice-config.v1"


class ModelDiscoveryError(ProfileError):
    """A model catalog could not be inspected or contained invalid choices."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "engine.catalog_unavailable",
        installed_version: str | None = None,
        distribution_version: str | None = None,
        module_version: str | None = None,
        module_path: str | None = None,
        missing_dependency: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.installed_version = installed_version
        self.distribution_version = distribution_version
        self.module_version = module_version
        self.module_path = module_path
        self.missing_dependency = missing_dependency


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
class SourceRef:
    plan_id: str
    flow_index: int
    segment_index: int
    flow_hash: str
    locator: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "source.plan_id", PlanError)
        _nonempty(self.flow_hash, "source.flow_hash", PlanError)
        if type(self.flow_index) is not int or self.flow_index < 0:
            raise PlanError("source.flow_index must be a non-negative integer")
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise PlanError("source.segment_index must be a non-negative integer")
        object.__setattr__(
            self, "locator", f"f{self.flow_index:06d}.s{self.segment_index:03d}"
        )


@dataclass(frozen=True, slots=True)
class RenderSegment:
    source: SourceRef
    text: str
    language: str
    logical_voice: str | None
    directives: Mapping[str, Any] = field(default_factory=dict)
    tokens: tuple[Mapping[str, Any], ...] = ()
    markers: tuple[str, ...] = ()
    heading: int | None = None
    pause_before: Any = None
    pause_after: Any = None
    semantic_segment_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef):
            raise PlanError("render segment source must be a SourceRef")
        if not isinstance(self.text, str):
            raise PlanError("segment.text must be a string")
        _nonempty(self.language, "segment.language", PlanError)
        if self.logical_voice is not None:
            _nonempty(self.logical_voice, "segment.logical_voice", PlanError)
        _mapping(self.directives, "segment.directives", PlanError)
        if not isinstance(self.tokens, tuple) or any(
            not isinstance(token, Mapping) for token in self.tokens
        ):
            raise PlanError("segment.tokens must be a tuple of mappings")
        if not isinstance(self.markers, tuple) or any(
            not isinstance(marker, str) or not marker for marker in self.markers
        ):
            raise PlanError("segment.markers must be a tuple of non-empty strings")
        if self.heading is not None and (
            type(self.heading) is not int or self.heading < 1
        ):
            raise PlanError("segment.heading must be a positive integer")
        if self.semantic_segment_hash and not self.semantic_segment_hash.startswith(
            "sha256:"
        ):
            raise PlanError("segment semantic hash must use the sha256: prefix")


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
        if (
            "target_id" in data
            and "model" in data
            and data["target_id"] != data["model"]
        ):
            raise ProfileError(
                "role.model and role.target_id must agree when both are set"
            )
        target = _nonempty(
            data.get("target_id", data.get("model")), "role.model", ProfileError
        )
        voice = data.get("voice")
        reference = data.get("reference_audio")
        if voice is not None:
            _nonempty(voice, "role.voice", ProfileError)
        if reference is not None:
            _nonempty(reference, "role.reference_audio", ProfileError)
            reference = (base_dir / Path(reference).expanduser()).resolve()
        if reference is not None and voice is not None:
            raise ProfileError("Choose either voice or reference_audio for a role")
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
            "model",
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
    defaults: Mapping[str, str] = field(default_factory=dict)
    backend_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def default_role(self) -> str | None:
        role = self.defaults.get("role")
        return role if isinstance(role, str) else None

    @property
    def default_language(self) -> str | None:
        language = self.defaults.get("language")
        return language if isinstance(language, str) else None

    @classmethod
    def load(cls, path: str | Path) -> VoiceProfile:
        location = Path(path).expanduser().resolve()
        try:
            with location.open("rb") as stream:
                data = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ProfileError(f"Cannot read voice profile {location}: {exc}") from exc

        legacy = data.get("schema") == SCHEMA_PROFILE
        if legacy:
            allowed_top = {"schema", "name", "roles"}
            defaults: dict[str, str] = {}
            backend_data: dict[str, dict[str, Any]] = {}
            profile_schema = SCHEMA_PROFILE
        else:
            if (
                type(data.get("schema_version")) is not int
                or data["schema_version"] != 1
            ):
                raise ProfileError(
                    "Expected legacy voice-profile schema or voice.toml schema_version = 1"
                )
            allowed_top = {"schema_version", "name", "defaults", "roles", "backend"}
            raw_defaults = _mapping(data.get("defaults", {}), "defaults", ProfileError)
            extra_defaults = set(raw_defaults).difference({"role", "language"})
            if extra_defaults:
                raise ProfileError(
                    f"Unknown voice defaults: {', '.join(sorted(extra_defaults))}"
                )
            defaults = {
                key: _nonempty(value, f"defaults.{key}", ProfileError)
                for key, value in raw_defaults.items()
            }
            raw_backend = _mapping(data.get("backend", {}), "backend", ProfileError)
            backend_data = {
                _nonempty(engine, "backend engine", ProfileError): _options(
                    options, f"backend.{engine}"
                )
                for engine, options in raw_backend.items()
            }
            profile_schema = SCHEMA_VOICE_CONFIG

        name = _nonempty(data.get("name"), "profile.name", ProfileError)
        raw_roles = _mapping(data.get("roles"), "profile.roles", ProfileError)
        if not raw_roles:
            raise ProfileError("At least one role binding is required")
        unknown = set(data).difference(allowed_top)
        if unknown:
            raise ProfileError(
                f"Unknown top-level profile fields: {', '.join(sorted(unknown))}"
            )

        roles: dict[str, VoiceBinding] = {}
        for role, raw in raw_roles.items():
            role_name = _nonempty(role, "role name", ProfileError)
            role_data = dict(_mapping(raw, f"roles.{role_name}", ProfileError))
            engine = _nonempty(
                role_data.get("engine"), f"roles.{role_name}.engine", ProfileError
            )
            role_options = _options(
                role_data.get("options", {}), f"roles.{role_name}.options"
            )
            merged_options = {**backend_data.get(engine, {}), **role_options}
            role_data["options"] = merged_options
            roles[role_name] = VoiceBinding.from_mapping(
                role_data, base_dir=location.parent
            )

        default_role = defaults.get("role")
        if default_role is not None and default_role not in roles:
            raise ProfileError(f"Default role {default_role!r} is not configured")
        return cls(
            name=name,
            roles=roles,
            schema=profile_schema,
            defaults=defaults,
            backend_defaults=backend_data,
            source_path=location,
        )


# The v5-facing name describes the new role-routing configuration; the alias
# keeps one implementation and preserves existing VoiceProfile consumers.
VoiceConfig = VoiceProfile


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
            segment_id = _nonempty(item.get("id"), f"segments[{index}].id", PlanError)
            text = _nonempty(item.get("text"), f"segments[{index}].text", PlanError)
            scope_id = _nonempty(
                item.get("scope_id", "default"),
                f"segments[{index}].scope_id",
                PlanError,
            )
            role = _nonempty(
                item.get("role", "narrator"), f"segments[{index}].role", PlanError
            )
            language = _nonempty(
                item.get("language", "en-US"), f"segments[{index}].language", PlanError
            )
            segment = SpeechSegment(
                id=segment_id,
                scope_id=scope_id,
                text=text,
                role=role,
                language=language,
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


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Engine-neutral model projection retained for adapter selection helpers."""

    id: str
    source: str
    languages: tuple[str, ...]
    voices: tuple[str, ...]
    default_voice: str
    qualities: tuple[str, ...]
    g2p_backend: str | None
    lexicons: tuple[str, ...] | None
    frontend: str
    status: str
    experimental: bool
    runtime_available: bool
    redistribution_allowed: bool
    distribution_id: str | None
    provider: str | None
    distribution_provider: str | None
    engine: str
    sample_rate: int | None
    max_tokens: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_target(cls, target: Any) -> ModelInfo:
        metadata = dict(target.metadata)
        voices = tuple(target.voices)
        lexicons = metadata.get("lexicons")
        return cls(
            id=target.id,
            source=str(metadata.get("source") or target.engine),
            languages=tuple(target.languages),
            voices=voices,
            default_voice=str(target.default_voice or (voices[0] if voices else "")),
            qualities=tuple(target.qualities),
            g2p_backend=metadata.get("g2p_backend"),
            lexicons=tuple(lexicons) if lexicons is not None else None,
            frontend=str(metadata.get("frontend") or ""),
            status=target.status,
            experimental=bool(metadata.get("experimental", False)),
            runtime_available=bool(target.runtime_available),
            redistribution_allowed=bool(metadata.get("redistribution_allowed", False)),
            distribution_id=metadata.get("distribution_id"),
            provider=metadata.get("provider"),
            distribution_provider=metadata.get("distribution_provider"),
            engine=target.engine,
            sample_rate=target.sample_rate or metadata.get("sample_rate"),
            max_tokens=metadata.get("max_tokens"),
            metadata=metadata,
        )


def language_matches(requested: str, declared: tuple[str, ...]) -> bool:
    """Match normalized language tags with base-language fallback."""
    normalized = requested.strip().lower().replace("_", "-")
    base = normalized.partition("-")[0]
    return any(
        normalized == item.strip().lower().replace("_", "-")
        or base == item.strip().lower().replace("_", "-").partition("-")[0]
        for item in declared
    )


def get_model_info(
    model_id: str,
    *,
    offline: bool = False,
    refresh: bool = False,
    preference: str = "auto",
    engine: str | None = None,
) -> tuple[ModelInfo, Any]:
    """Resolve one target using the VoiceRender engine registry."""
    from .engines.registry import discover_targets, normalize_engine_id

    engine_id = normalize_engine_id(engine or "kokoro")
    targets = discover_targets(
        engine_id,
        offline=offline,
        refresh=refresh,
        preference=preference,
    )
    for target in targets:
        if target.id == model_id or model_id in target.aliases:
            return ModelInfo.from_target(target), targets
    raise ModelDiscoveryError(
        f"Unknown {engine_id} model {model_id!r}.", code=f"{engine_id}.model_not_found"
    )
