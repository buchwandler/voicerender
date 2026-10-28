"""VoiceRender-owned request fitting, subdivision, and result merging."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .engines.base import (
    PronunciationSpan,
    RenderedSpeech,
    RequestMeasure,
    SayAsSpan,
    SpeechRequest,
    SpeechToken,
    SpeechWordTiming,
    validate_rendered_speech,
)
from .errors import EngineBackendError, SpeechRequestTooLongError
from .jsonutil import json_value


@dataclass(frozen=True, slots=True)
class SemanticBoundary:
    """A planner-authorized request split position and its provenance."""

    position: int
    kind: str
    semantic_boundary_id: str | None = None
    origin: str | None = None


LOWERING_SCHEMA = "voicerender.atomic-lowering.v1"
CAPACITY_SCHEMA = "voicerender.capacity-fitting.v1"


def _json_value(value: Any) -> Any:
    return json_value(value, label="capacity metadata")


@dataclass(frozen=True, slots=True)
class CapacityContext:
    """VoiceRender-owned planning hints for request-capacity subdivision."""

    semantic_boundaries: tuple[SemanticBoundary, ...] = ()
    linguistic_token_ranges: tuple[tuple[int, int], ...] = ()
    protected_ranges: tuple[tuple[int, int], ...] = ()
    planned_semantics: bool = False


@dataclass(frozen=True, slots=True)
class CapacityBoundary:
    """One legal split candidate with priority and provenance."""

    position: int
    priority: int
    kind: str
    source: str
    semantic_boundary_id: str | None = None
    origin: str | None = None


@dataclass(frozen=True, slots=True)
class AtomicRequest:
    """One exact child request rendered for a semantic parent request."""

    request: SpeechRequest
    parent_id: str
    index: int
    char_start: int
    char_end: int
    measure: RequestMeasure | None = None
    render: Mapping[str, Any] = field(default_factory=dict)
    split: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        measure = self.measure
        request = self.request
        return {
            "id": request.id,
            "parent_id": self.parent_id,
            "index": self.index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": request.text,
            "request": _json_value(
                {
                    "id": request.id,
                    "text": request.text,
                    "language": request.language,
                    "voice": request.voice,
                    "speaker": request.speaker,
                    "pronunciation_overrides": request.pronunciation_overrides,
                    "say_as": request.say_as,
                    "tokens": request.tokens,
                    "whole_request_phonemes": request.whole_request_phonemes,
                    "options": request.options,
                }
            ),
            "measure": (
                {
                    "fits": measure.fits,
                    "amount": measure.amount,
                    "maximum": measure.maximum,
                    "unit": measure.unit,
                    "source": measure.source,
                    "details": _json_value(measure.details),
                }
                if measure is not None
                else None
            ),
            "render": _json_value(self.render),
            "split": _json_value(self.split),
        }


@dataclass(frozen=True, slots=True)
class AtomicRender:
    """Merged result plus the exact child requests used to produce it."""

    result: RenderedSpeech
    requests: tuple[AtomicRequest, ...]

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": LOWERING_SCHEMA,
            "capacity_schema": CAPACITY_SCHEMA,
            "requests": [item.to_dict() for item in self.requests],
        }


@dataclass(frozen=True, slots=True)
class _RenderedChild:
    request: SpeechRequest
    start: int
    end: int
    measure: RequestMeasure
    result: RenderedSpeech
    split: Mapping[str, Any] = field(default_factory=dict)


def render_atomic_request(
    session: Any,
    request: SpeechRequest,
    *,
    capacity: CapacityContext | None = None,
    protected_ranges: Iterable[tuple[int, int]] = (),
) -> AtomicRender:
    """Measure, fit, render, and merge one exact VoiceRender speech request."""
    capacity_context = capacity if capacity is not None else CapacityContext()
    protected = _collect_protected_ranges(
        request,
        (
            *capacity_context.linguistic_token_ranges,
            *capacity_context.protected_ranges,
            *protected_ranges,
        ),
    )
    measured: dict[tuple[int, int], RequestMeasure] = {}

    def child_request(start: int, end: int) -> SpeechRequest:
        return _slice_request(request, start, end)

    def measure_range(start: int, end: int) -> RequestMeasure:
        key = (start, end)
        if key in measured:
            return measured[key]
        candidate = child_request(start, end)
        measure_method = getattr(session, "measure", None)
        if not callable(measure_method):
            result = RequestMeasure(
                fits=None,
                amount=None,
                maximum=None,
                unit="unknown",
                source="measurement.unavailable",
            )
        else:
            try:
                result = measure_method(candidate)
            except SpeechRequestTooLongError as exc:
                result = _measure_from_error(exc, candidate, "measurement.too_long")
            if not isinstance(result, RequestMeasure):
                result = RequestMeasure(
                    fits=None,
                    amount=None,
                    maximum=None,
                    unit="unknown",
                    source="measurement.indeterminate",
                )
        measured[key] = result
        return result

    def render_range(
        start: int,
        end: int,
        split_context: Mapping[str, Any] | None = None,
    ) -> list[_RenderedChild]:
        candidate = child_request(start, end)
        capacity_measure = measure_range(start, end)
        if capacity_measure.fits is not False:
            try:
                result = session.synthesize(candidate)
            except SpeechRequestTooLongError as exc:
                capacity_measure = _measure_from_error(
                    exc, candidate, "synthesis.too_long"
                )
                measured[(start, end)] = capacity_measure
            else:
                validate_rendered_speech(candidate, result)
                return [
                    _RenderedChild(
                        candidate,
                        start,
                        end,
                        capacity_measure,
                        result,
                        dict(split_context or {}),
                    )
                ]
        if request.whole_request_phonemes is not None:
            raise _capacity_error(
                candidate,
                capacity_measure,
                "whole-request phonemes cannot be subdivided safely",
            )
        boundaries = _candidate_boundaries(
            request,
            start,
            end,
            protected,
            capacity_context,
        )
        if not boundaries:
            raise _capacity_error(
                candidate, capacity_measure, "no legal source-text split boundary"
            )
        split_boundary = _choose_boundary(
            request,
            start,
            end,
            boundaries,
            capacity_measure,
            measure_range,
        )
        split_position = split_boundary.position
        if split_position <= start or split_position >= end:
            raise _capacity_error(
                candidate, capacity_measure, "no legal source-text split boundary"
            )
        split_provenance = _split_provenance(split_boundary)
        return render_range(start, split_position, split_provenance) + render_range(
            split_position,
            end,
            split_provenance,
        )

    children = render_range(0, len(request.text))
    if "".join(child.request.text for child in children) != request.text:
        raise AssertionError("capacity fitting did not preserve exact request text")

    atomic_requests = tuple(
        AtomicRequest(
            request=child.request,
            parent_id=request.id,
            index=index,
            char_start=child.start,
            char_end=child.end,
            measure=child.measure,
            render={
                "sample_rate": child.result.sample_rate,
                "frames": int(np.asarray(child.result.audio).size),
                "warnings": list(child.result.warnings),
                "metadata": _json_value(child.result.metadata),
            },
            split=child.split,
        )
        for index, child in enumerate(children)
    )
    result = _merge_results(request, children, atomic_requests)
    validate_rendered_speech(request, result)
    return AtomicRender(result=result, requests=atomic_requests)


def _collect_protected_ranges(
    request: SpeechRequest,
    extra: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ranges = [(token.start, token.end) for token in request.tokens]
    ranges.extend((span.start, span.end) for span in request.pronunciation_overrides)
    ranges.extend((span.start, span.end) for span in request.say_as)
    ranges.extend(extra)
    for start, end in ranges:
        if start < 0 or start >= end or end > len(request.text):
            raise ValueError(
                "protected request range is outside the exact request text"
            )
    return tuple(sorted(set(ranges)))


def _slice_request(request: SpeechRequest, start: int, end: int) -> SpeechRequest:
    text = request.text[start:end]
    if start == 0 and end == len(request.text):
        child_id = request.id
    else:
        child_id = f"{request.id}.part-{start:06d}-{end:06d}"
    tokens = tuple(
        SpeechToken(
            start=token.start - start,
            end=token.end - start,
            text=token.text,
            pos=token.pos,
            tag=token.tag,
            lemma=token.lemma,
            morph=token.morph,
            language=token.language,
        )
        for token in request.tokens
        if token.start >= start and token.end <= end
    )
    overrides = tuple(
        PronunciationSpan(
            start=span.start - start,
            end=span.end - start,
            phonemes=span.phonemes,
            language=span.language,
            alphabet=span.alphabet,
        )
        for span in request.pronunciation_overrides
        if span.start >= start and span.end <= end
    )
    say_as: list[SayAsSpan] = []
    for span in request.say_as:
        if span.start >= start and span.end <= end:
            say_as.append(
                SayAsSpan(
                    start=span.start - start,
                    end=span.end - start,
                    interpret_as=span.interpret_as,
                    format=span.format,
                    detail=span.detail,
                )
            )
        elif span.start < end and span.end > start:
            raise ValueError("capacity child partially overlaps a say_as span")
    return SpeechRequest(
        id=child_id,
        text=text,
        language=request.language,
        voice=request.voice,
        speaker=request.speaker,
        pronunciation_overrides=overrides,
        say_as=tuple(say_as),
        tokens=tokens,
        whole_request_phonemes=request.whole_request_phonemes,
        options=dict(request.options),
    )


def _candidate_boundaries(
    request: SpeechRequest,
    start: int,
    end: int,
    protected: tuple[tuple[int, int], ...],
    capacity: CapacityContext,
) -> tuple[CapacityBoundary, ...]:
    text = request.text
    candidates: dict[int, CapacityBoundary] = {}
    planned = capacity.planned_semantics

    def add(
        position: int,
        priority: int,
        kind: str,
        source: str = "voicerender.heuristic",
        *,
        semantic_boundary_id: str | None = None,
        origin: str | None = None,
    ) -> None:
        if not start < position < end:
            return
        if any(left < position < right for left, right in protected):
            return
        if not text[start:position].strip() or not text[position:end].strip():
            return
        candidate = CapacityBoundary(
            position=position,
            priority=priority,
            kind=kind,
            source=source,
            semantic_boundary_id=semantic_boundary_id,
            origin=origin,
        )
        existing = candidates.get(position)
        if existing is None or _prefer_candidate(candidate, existing):
            candidates[position] = candidate

    if planned:
        semantic_priorities = {"clause": 0, "parenthetical": 1}
        for boundary in capacity.semantic_boundaries:
            priority = semantic_priorities.get(boundary.kind)
            if priority is None:
                continue
            add(
                boundary.position,
                priority,
                f"semantic_{boundary.kind}",
                "planner",
                semantic_boundary_id=boundary.semantic_boundary_id,
                origin=boundary.origin,
            )

    index = start
    while index < end:
        if text[index] in "\r\n":
            position = index + 1
            while position < end and text[position] in "\r\n":
                position += 1
            add(position, 2 if planned else 0, "newline")
            index = position
            continue
        index += 1

    if not planned:
        sentence_marks = ".!?。！？"
        closing = "\"'’”»)]}"
        index = start
        while index < end:
            if text[index] in sentence_marks:
                position = index + 1
                while position < end and text[position] in sentence_marks + closing:
                    position += 1
                while position < end and text[position] in " \t":
                    position += 1
                add(position, 1, "sentence_punctuation")
            index += 1

    clause_marks = ";:,，；：、"
    index = start
    while index < end:
        if text[index] in clause_marks:
            position = index + 1
            while position < end and text[position] in " \t":
                position += 1
            add(
                position,
                3 if planned else 2,
                "punctuation_clause",
            )
        index += 1

    token_priority = 4 if planned else 3
    for token_start, token_end in capacity.linguistic_token_ranges:
        add(token_start, token_priority, "token_edge", "planner")
        add(token_end, token_priority, "token_edge", "planner")
    for token in request.tokens:
        add(token.start, token_priority, "token_edge", "request.tokens")
        add(token.end, token_priority, "token_edge", "request.tokens")

    index = start
    while index < end:
        if text[index].isspace():
            position = index + 1
            while position < end and text[position].isspace():
                position += 1
            add(position, 5 if planned else 4, "whitespace")
            index = position
            continue
        index += 1

    if not planned:
        for position in range(start + 1, end):
            add(position, 5, "character")
    return tuple(
        sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.priority,
                candidate.position,
                candidate.kind,
            ),
        )
    )


def _prefer_candidate(candidate: CapacityBoundary, current: CapacityBoundary) -> bool:
    if candidate.priority != current.priority:
        return candidate.priority < current.priority
    if candidate.source == "planner" and current.source != "planner":
        return True
    if current.source == "planner" and candidate.source != "planner":
        return False
    return (candidate.kind, candidate.semantic_boundary_id or "") < (
        current.kind,
        current.semantic_boundary_id or "",
    )


def _choose_boundary(
    request: SpeechRequest,
    start: int,
    end: int,
    boundaries: tuple[CapacityBoundary, ...],
    parent_measure: RequestMeasure,
    measure_range: Any,
) -> CapacityBoundary:
    groups: dict[int, list[CapacityBoundary]] = {}
    for boundary in boundaries:
        groups.setdefault(boundary.priority, []).append(boundary)
    ordered_groups = [
        sorted(groups[priority], key=lambda boundary: boundary.position)
        for priority in sorted(groups)
    ]
    fallback = min(
        ordered_groups[0],
        key=lambda boundary: (
            abs(boundary.position - (start + end) / 2),
            boundary.position,
        ),
    )
    if parent_measure.fits is None:
        return fallback

    for boundaries_at_priority in ordered_groups:
        low = 0
        high = len(boundaries_at_priority) - 1
        while low <= high:
            middle = (low + high) // 2
            boundary = boundaries_at_priority[middle]
            left = measure_range(start, boundary.position)
            right = measure_range(boundary.position, end)
            if left.fits is True and right.fits is True:
                return boundary
            if left.fits is None or right.fits is None:
                break
            if left.fits is False and right.fits is True:
                high = middle - 1
            elif left.fits is True and right.fits is False:
                low = middle + 1
            else:
                break
    return fallback


def _split_provenance(boundary: CapacityBoundary) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "reason": "model_capacity",
        "boundary_kind": boundary.kind,
        "source": boundary.source,
    }
    if boundary.semantic_boundary_id is not None:
        provenance["semantic_boundary_id"] = boundary.semantic_boundary_id
    if boundary.origin is not None:
        provenance["origin"] = boundary.origin
    return provenance


def _measure_from_error(
    error: SpeechRequestTooLongError,
    request: SpeechRequest,
    source: str,
) -> RequestMeasure:
    if error.unit == "model_tokens":
        unit: Literal["model_tokens", "phoneme_ids", "unknown"] = "model_tokens"
    elif error.unit == "phoneme_ids":
        unit = "phoneme_ids"
    else:
        unit = "unknown"
    return RequestMeasure(
        fits=False,
        amount=error.amount,
        maximum=error.maximum,
        unit=unit,
        source=source,
        details={
            "native_error_type": getattr(error, "native_error_type", None),
            "request_id": request.id,
            "text_length": len(request.text),
        },
    )


def _capacity_error(
    request: SpeechRequest,
    measure: RequestMeasure,
    reason: str,
) -> SpeechRequestTooLongError:
    return SpeechRequestTooLongError(
        f"VoiceRender cannot fit request {request.id!r}: {reason}.",
        language=request.language,
        voice=request.voice,
        speaker=request.speaker,
        request_id=request.id,
        amount=measure.amount,
        maximum=measure.maximum,
        unit=measure.unit,
        text_length=len(request.text),
        source="voicerender.capacity",
        details={
            "capacity_source": measure.source,
            "capacity_reason": reason,
        },
    )


def _merge_results(
    request: SpeechRequest,
    children: list[_RenderedChild],
    atomic_requests: tuple[AtomicRequest, ...],
) -> RenderedSpeech:
    sample_rate = children[0].result.sample_rate
    arrays: list[np.ndarray] = []
    timings: list[SpeechWordTiming] = []
    warnings: list[str] = []
    sample_offset = 0
    for child in children:
        result = child.result
        if result.sample_rate != sample_rate:
            raise EngineBackendError(
                "Atomic request children returned different sample rates.",
                language=request.language,
                voice=request.voice,
                speaker=request.speaker,
                request_id=request.id,
                details={
                    "expected_sample_rate": sample_rate,
                    "actual_sample_rate": result.sample_rate,
                    "child_request_id": child.request.id,
                },
            )
        audio = np.asarray(result.audio, dtype=np.float32)
        arrays.append(audio)
        timings.extend(
            SpeechWordTiming(
                text=timing.text,
                char_start=child.start + timing.char_start,
                char_end=child.start + timing.char_end,
                start_sample=sample_offset + timing.start_sample,
                end_sample=sample_offset + timing.end_sample,
            )
            for timing in result.word_timings
        )
        for warning in result.warnings:
            if warning not in warnings:
                warnings.append(warning)
        sample_offset += int(audio.size)

    metadata = dict(children[0].result.metadata)
    metadata["atomic_lowering"] = {
        "schema": LOWERING_SCHEMA,
        "capacity_schema": CAPACITY_SCHEMA,
        "requests": [item.to_dict() for item in atomic_requests],
    }
    return RenderedSpeech(
        id=request.id,
        audio=np.concatenate(arrays).astype(np.float32, copy=False),
        sample_rate=sample_rate,
        warnings=tuple(warnings),
        word_timings=tuple(timings),
        metadata=metadata,
    )


__all__ = [
    "CAPACITY_SCHEMA",
    "LOWERING_SCHEMA",
    "AtomicRender",
    "AtomicRequest",
    "CapacityBoundary",
    "CapacityContext",
    "SemanticBoundary",
    "render_atomic_request",
]
