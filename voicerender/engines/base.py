"""Engine-neutral synthesis contracts owned by VoiceRender."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from renderplan import CapacityConstraint

from .api_probe import EngineApiProbe

if TYPE_CHECKING:
    from .catalog import CatalogRequest, SynthesisTarget
    from .selection import EngineRequest


def distribution_version(distribution: str) -> str | None:
    """Return the installed distribution version, or None when unavailable."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def voice_level_metadata(
    value: Any,
    mode: str,
    *,
    identity_keys: tuple[str, ...],
    revision_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Map an engine voice-level calibration record to VoiceRender metadata."""
    metadata = dict(value) if isinstance(value, Mapping) else {}

    def first_present(keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in metadata:
                return metadata[key]
        return None

    return {
        "mode": metadata.get("mode", mode),
        "applied": bool(metadata.get("applied", False)),
        "gain_db": metadata.get("gain_db"),
        "source": metadata.get("source", "none"),
        "reason": metadata.get("reason"),
        "calibration_identity": first_present(identity_keys),
        "calibration_revision": first_present(revision_keys),
    }


@dataclass(frozen=True, slots=True)
class SpeechToken:
    """One linguistic token with offsets local to a synthesis request."""

    start: int
    end: int
    text: str
    pos: str | None = None
    tag: str | None = None
    lemma: str | None = None
    morph: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class PronunciationSpan:
    """A request-local pronunciation override."""

    start: int
    end: int
    phonemes: str | None = None
    language: str | None = None
    alphabet: str | None = None


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    """Prepared text and optional linguistic context for one engine synthesis."""

    id: str
    text: str
    language: str
    voice: str | None = None
    speaker: str | int | None = None
    pronunciation_overrides: tuple[PronunciationSpan, ...] = ()
    tokens: tuple[SpeechToken, ...] = ()
    whole_request_phonemes: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpeechWordTiming:
    """A word timing expressed in request text and rendered-audio coordinates."""

    text: str
    char_start: int
    char_end: int
    start_sample: int
    end_sample: int


@dataclass(slots=True)
class RenderedSpeech:
    """Mono float32 audio and metadata for one independent speech request."""

    id: str
    audio: Any
    sample_rate: int
    warnings: tuple[str, ...] = ()
    word_timings: tuple[SpeechWordTiming, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RequestMeasure:
    """Capacity measurement for one exact VoiceRender speech request."""

    fits: bool | None
    amount: int | None
    maximum: int | None
    unit: Literal["model_tokens", "phoneme_ids", "unknown"]
    source: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Synthesis features an engine adapter can represent natively."""

    id: str = ""
    voice_binding_namespace: str | None = None
    voice_binding_scope: Literal["request", "target"] = "request"
    option_names: frozenset[str] = frozenset()
    supports_named_voices: bool = False
    supports_reference_voice: bool = False
    supports_speakers: bool = False
    supports_pronunciation_overrides: bool = False
    pronunciation_alphabets: frozenset[str] = frozenset()
    supports_linguistic_tokens: bool = False
    supports_whole_request_phonemes: bool = False
    supports_lexicons: bool = False
    supports_model_sources: bool = False
    supports_qualities: bool = False
    supports_live: bool = False
    supports_timestamps: bool = False

    supports_voice_level_calibration: bool = False
    # True means a side-effect-free measurement API exists; result actionability is in RequestMeasure.
    supports_request_measurement: bool = False
    # Compatibility aliases retained for the original VoiceRender renderer.
    supports_builtin_voices: bool = True
    supports_reference_audio: bool = False
    supports_speaker_selection: bool = False
    supports_word_timings: bool = False
    options: frozenset[str] = frozenset()
    native_semantics: frozenset[str] = frozenset()
    composition_semantics: frozenset[str] = frozenset()
    supports_synthesis_request: bool = False


@dataclass(frozen=True, slots=True)
class EngineDiagnostic:
    """Stable, machine-readable adapter selection or capability diagnostic."""

    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    field: str | None = None
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.field is not None:
            value["field"] = self.field
        if self.details:
            value["details"] = dict(self.details)
        return value


@dataclass(frozen=True, slots=True)
class EngineSelection:
    """Concrete engine selection produced during plan resolution."""

    engine: str
    target_id: str
    language: str
    voice: str | None = None
    speaker: str | int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    offline: bool = False
    refresh: bool = False


class EngineSession(Protocol):
    """Open synthesis session that renders one VoiceRender speech request at a time."""

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        """Measure request capacity without performing acoustic inference."""
        ...

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        """Synthesize one independent request without composing other requests."""
        ...


@dataclass(frozen=True, slots=True)
class RenderProfileFragment:
    """Engine-derived facts needed to seed a shared RenderProfile role."""

    binding: Mapping[str, Any]
    token_fields: tuple[str, ...]
    frontend: str
    pronunciation: str
    renderer_capacity: CapacityConstraint | None = None
    planner_capacity: CapacityConstraint | None = None


class EngineAdapter(Protocol):
    """Discovery, target resolution, and session opening for one synthesis engine."""

    id: str
    package_name: str

    def probe_api(self) -> EngineApiProbe:
        """Return the structured public API compatibility diagnosis."""
        ...

    def compatible_api(self) -> bool:
        """Return whether the public API is compatible (legacy wrapper)."""
        ...

    def version(self) -> str | None:
        """Return the installed engine package version, or None."""
        ...

    def capabilities(self) -> EngineCapabilities:
        """Return the synthesis features supported by this adapter."""
        ...

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        """Describe engine-derived binding, semantics, and capacity facts."""
        ...

    def discover(
        self,
        request: CatalogRequest,
    ) -> tuple[SynthesisTarget, ...]:
        """Discover available targets for this engine."""
        ...

    def resolve(
        self,
        request: EngineRequest,
    ) -> tuple[EngineSelection, tuple[Any, ...]]:
        """Resolve and validate a concrete engine selection."""
        ...

    def validate_selection(
        self, selection: EngineSelection
    ) -> tuple[EngineDiagnostic, ...]:
        """Validate resolved engine/model/voice choices without synthesizing."""
        ...

    def canonical_synthesis_identity(
        self,
        selection: EngineSelection,
    ) -> Mapping[str, Any]:
        """Return identity inputs for canonical speech synthesis."""
        ...

    def open(
        self,
        selection: EngineSelection,
    ) -> AbstractContextManager[EngineSession]:
        """Open a rendering session for the given concrete target."""
        ...


def validate_word_timings(
    *,
    text: str,
    frame_count: int,
    timings: Sequence[SpeechWordTiming],
    context: Mapping[str, object] | None = None,
) -> tuple[SpeechWordTiming, ...]:
    """Validate request-text and rendered-audio coordinates for native or derived words."""
    from numbers import Integral

    label = ""
    if context:
        label = (
            " ("
            + ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
            + ")"
        )

    def invalid(reason: str) -> ValueError:
        return ValueError(f"{reason}{label}")

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, Integral)
        or frame_count < 0
    ):
        raise invalid("frame count must be a non-negative integer")
    previous_char_start = previous_char_end = -1
    previous_sample_start = previous_sample_end = -1
    for timing in timings:
        if not isinstance(timing.text, str):
            raise invalid("word timing text must be a string")
        coordinates = (
            timing.char_start,
            timing.char_end,
            timing.start_sample,
            timing.end_sample,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in coordinates
        ):
            raise invalid("word timing coordinates must be integers")
        char_start, char_end, sample_start, sample_end = (
            int(value) for value in coordinates
        )
        if not (0 <= char_start <= char_end <= len(text)):
            raise invalid("word timing character range is outside the request text")
        if not (0 <= sample_start <= sample_end <= frame_count):
            raise invalid("word timing sample range is outside the rendered audio")
        if char_start < previous_char_start or char_end < previous_char_end:
            raise invalid("word timings are not monotonic in request text")
        if sample_start < previous_sample_start or sample_end < previous_sample_end:
            raise invalid("word timings are not monotonic in rendered audio")
        previous_char_start, previous_char_end = char_start, char_end
        previous_sample_start, previous_sample_end = sample_start, sample_end
    return tuple(timings)


def validate_rendered_speech(
    request: SpeechRequest,
    result: RenderedSpeech,
    *,
    engine: str | None = None,
    engine_version: str | None = None,
    target_id: str | None = None,
) -> RenderedSpeech:
    """Validate and normalize an engine result before VoiceRender consumes it."""
    from numbers import Integral

    import numpy as np

    from ..errors import EngineBackendError

    def invalid(reason: str, cause: Exception | None = None) -> EngineBackendError:
        error = EngineBackendError(
            f"Engine returned invalid rendered speech: {reason}",
            engine=engine,
            engine_version=engine_version,
            target_id=target_id,
            language=request.language,
            voice=request.voice,
            speaker=request.speaker,
            request_id=request.id,
        )
        if cause is not None:
            error.__cause__ = cause
        return error

    if result.id != request.id:
        raise invalid(
            f"result id {result.id!r} does not match request id {request.id!r}"
        )
    if isinstance(result.sample_rate, bool) or not isinstance(
        result.sample_rate, Integral
    ):
        raise invalid("sample rate must be a positive integer")
    sample_rate = int(result.sample_rate)
    if sample_rate <= 0:
        raise invalid("sample rate must be a positive integer")

    try:
        audio = np.asarray(result.audio, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise invalid("audio must convert to float32", exc) from exc
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim != 1:
        raise invalid("audio must be mono")
    if audio.size == 0:
        raise invalid("audio must be non-empty")
    if not np.isfinite(audio).all():
        raise invalid("audio must contain only finite samples")

    peak = float(np.max(np.abs(audio)))
    if peak > float(np.float32(1.00001)):
        raise invalid(f"audio exceeds PCM full-scale range [-1, 1] (peak={peak:.6f})")

    try:
        result.word_timings = validate_word_timings(
            text=request.text,
            frame_count=int(audio.size),
            timings=result.word_timings,
        )
    except ValueError as exc:
        raise invalid(str(exc)) from exc

    result.audio = audio
    result.sample_rate = sample_rate
    return result


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Compatibility request for the original VoiceRender v5 renderer."""

    request_id: str
    text: str
    language: str
    directives: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SynthesisOutput:
    """Compatibility result for adapters registered by legacy integrations."""

    audio: Any
    sample_rate: int
    word_timings: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "EngineAdapter",
    "EngineCapabilities",
    "EngineDiagnostic",
    "EngineSelection",
    "EngineSession",
    "PronunciationSpan",
    "RenderProfileFragment",
    "RenderedSpeech",
    "RequestMeasure",
    "SpeechRequest",
    "SpeechToken",
    "SpeechWordTiming",
    "SynthesisOutput",
    "SynthesisRequest",
    "validate_rendered_speech",
]
