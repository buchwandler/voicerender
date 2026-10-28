from __future__ import annotations

import numpy as np
import pytest

from voicerender import synthesize
from voicerender.capacity import (
    CapacityContext,
    SemanticBoundary,
    _slice_request,
    render_atomic_request,
)
from voicerender.engines.base import (
    PronunciationSpan,
    RenderedSpeech,
    RequestMeasure,
    SayAsSpan,
    SpeechRequest,
    SpeechToken,
    SpeechWordTiming,
)
from voicerender.errors import EngineBackendError, SpeechRequestTooLongError


class _MeasuredSession:
    def __init__(self, maximum: int, *, sample_rate: int = 24_000) -> None:
        self.maximum = maximum
        self.sample_rate = sample_rate
        self.calls: list[SpeechRequest] = []
        self.measure_calls: list[SpeechRequest] = []

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        self.measure_calls.append(request)
        amount = len(request.text)
        return RequestMeasure(
            fits=amount <= self.maximum,
            amount=amount,
            maximum=self.maximum,
            unit="model_tokens",
            source="test.length",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        self.calls.append(request)
        assert len(request.text) <= self.maximum
        return _rendered(request, self.sample_rate, len(self.calls))


class _UnknownSession(_MeasuredSession):
    def measure(self, request: SpeechRequest) -> RequestMeasure:
        return RequestMeasure(
            fits=None,
            amount=None,
            maximum=None,
            unit="unknown",
            source="test.unknown",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        self.calls.append(request)
        if len(request.text) > self.maximum:
            raise SpeechRequestTooLongError(
                "request exceeds fake limit",
                request_id=request.id,
                amount=len(request.text),
                maximum=self.maximum,
                unit="model_tokens",
                text_length=len(request.text),
            )
        return _rendered(request, self.sample_rate, len(self.calls))


def _rendered(request: SpeechRequest, sample_rate: int, value: int) -> RenderedSpeech:
    audio = np.full(len(request.text), value / (value + 1), dtype=np.float32)
    start = len(request.text) - len(request.text.lstrip())
    end = len(request.text.rstrip())
    timings = ()
    if start < end:
        timings = (
            SpeechWordTiming(
                text=request.text[start:end],
                char_start=start,
                char_end=end,
                start_sample=0,
                end_sample=len(audio),
            ),
        )
    return RenderedSpeech(
        id=request.id,
        audio=audio,
        sample_rate=sample_rate,
        warnings=("shared warning", f"render {value}"),
        word_timings=timings,
        metadata={"child_value": value},
    )


def _request(text: str, **kwargs) -> SpeechRequest:
    return SpeechRequest(id="segment-1", text=text, language="en", **kwargs)


def test_capacity_fitting_prefers_sentence_boundaries_and_preserves_exact_text():
    request = _request("Hello. World is a test.")
    session = _MeasuredSession(10)

    rendered = render_atomic_request(session, request)

    assert [child.text for child in session.calls] == [
        "Hello. ",
        "World is ",
        "a test.",
    ]
    assert "".join(child.text for child in session.calls) == request.text
    assert rendered.result.id == request.id
    assert [item.char_start for item in rendered.requests] == [0, 7, 16]
    assert [item.char_end for item in rendered.requests] == [7, 16, len(request.text)]
    assert (
        rendered.result.metadata["atomic_lowering"]["requests"][0]["text"] == "Hello. "
    )


def test_fitting_request_is_rendered_as_one_unchanged_child():
    request = _request("  already fits.  ")
    session = _MeasuredSession(100)

    rendered = render_atomic_request(session, request)

    assert len(session.calls) == 1
    assert session.calls[0].text == request.text
    assert rendered.requests[0].request.text == request.text
    assert rendered.requests[0].request.id == request.id


def test_capacity_fitting_preserves_newline_and_tab_whitespace():
    request = _request("first line \t\n\nsecond line")
    session = _MeasuredSession(15)

    rendered = render_atomic_request(session, request)

    assert [child.text for child in session.calls] == [
        "first line \t\n\n",
        "second line",
    ]
    assert "".join(child.text for child in session.calls) == request.text
    assert [(item.char_start, item.char_end) for item in rendered.requests] == [
        (0, 14),
        (14, len(request.text)),
    ]


def test_capacity_fitting_uses_clause_fallback_before_words():
    request = _request("First clause, second clause, final words.")
    session = _MeasuredSession(15)

    rendered = render_atomic_request(session, request)

    assert session.calls[0].text == "First clause, "
    assert all("," in child.text for child in session.calls[:-1])
    assert "".join(child.text for child in session.calls) == request.text
    assert all(item.measure.fits is True for item in rendered.requests)


def test_planned_capacity_prefers_semantic_clause_and_records_provenance() -> None:
    text = "Dr. Smith arrived early, while the audience waited quietly."
    position = text.index(", ") + 2
    boundary = SemanticBoundary(
        position=position,
        kind="clause",
        semantic_boundary_id="semantic-boundary-000004",
        origin="phrasplit",
    )
    capacity = CapacityContext(
        semantic_boundaries=(boundary,),
        planned_semantics=True,
    )
    session = _MeasuredSession(34)

    rendered = render_atomic_request(session, _request(text), capacity=capacity)

    assert [child.text for child in session.calls] == [text[:position], text[position:]]
    assert [child.text for child in session.calls] != ["Dr. ", text[4:]]
    split = rendered.manifest["requests"][0]["split"]
    assert rendered.manifest["schema"] == "voicerender.atomic-lowering.v1"
    assert rendered.manifest["capacity_schema"] == "voicerender.capacity-fitting.v1"
    assert split == {
        "reason": "model_capacity",
        "boundary_kind": "semantic_clause",
        "source": "planner",
        "semantic_boundary_id": "semantic-boundary-000004",
        "origin": "phrasplit",
    }
    assert "".join(child.text for child in session.calls) == text


def test_planned_capacity_uses_clause_punctuation_when_semantics_are_unavailable() -> (
    None
):
    text = "Dr. Smith arrived early, while the audience waited quietly."
    position = text.index(", ") + 2
    session = _MeasuredSession(34)

    rendered = render_atomic_request(
        session,
        _request(text),
        capacity=CapacityContext(planned_semantics=True),
    )

    assert session.calls[0].text == text[:position]
    assert rendered.manifest["requests"][0]["split"] == {
        "reason": "model_capacity",
        "boundary_kind": "punctuation_clause",
        "source": "voicerender.heuristic",
    }


def test_planned_capacity_uses_independent_linguistic_token_ranges() -> None:
    request = _request("alpha beta gamma")
    capacity = CapacityContext(
        linguistic_token_ranges=((0, 5), (6, 10), (11, 16)),
        planned_semantics=True,
    )
    session = _MeasuredSession(10)

    rendered = render_atomic_request(session, request, capacity=capacity)

    assert request.tokens == ()
    assert [child.text for child in session.calls] == ["alpha ", "beta gamma"]
    assert rendered.manifest["requests"][0]["split"] == {
        "reason": "model_capacity",
        "boundary_kind": "token_edge",
        "source": "planner",
    }
    assert "".join(child.text for child in session.calls) == request.text


def test_planned_capacity_disables_arbitrary_character_fallback() -> None:
    request = _request("longword")
    session = _MeasuredSession(4)

    with pytest.raises(
        SpeechRequestTooLongError, match="no legal source-text split boundary"
    ):
        render_atomic_request(
            session,
            request,
            capacity=CapacityContext(planned_semantics=True),
        )

    assert session.calls == []


def test_planned_semantic_boundary_inside_pronunciation_span_is_rejected() -> None:
    text = "red blue green"
    boundary = SemanticBoundary(
        position=6,
        kind="clause",
        semantic_boundary_id="semantic-boundary-000005",
        origin="phrasplit",
    )
    request = _request(
        text,
        pronunciation_overrides=(
            PronunciationSpan(start=4, end=8, phonemes="bluː", alphabet="ipa"),
        ),
    )
    capacity = CapacityContext(
        semantic_boundaries=(boundary,),
        planned_semantics=True,
    )
    session = _MeasuredSession(8)

    rendered = render_atomic_request(session, request, capacity=capacity)

    assert [child.text for child in session.calls] == ["red ", "blue ", "green"]
    override_children = [
        child for child in session.calls if child.pronunciation_overrides
    ]
    assert len(override_children) == 1
    override = override_children[0].pronunciation_overrides[0]
    assert override_children[0].text[override.start : override.end] == "blue"
    assert all(
        item["split"].get("semantic_boundary_id") != boundary.semantic_boundary_id
        for item in rendered.manifest["requests"]
    )


def test_capacity_say_as_span_is_atomic_rebased_and_serialized() -> None:
    text = "one two A R T three four"
    span_start = text.index("A R T")
    span = SayAsSpan(span_start, span_start + len("A R T"), "characters")
    request = _request(text, say_as=(span,))
    session = _MeasuredSession(10)

    rendered = render_atomic_request(session, request)

    assert "".join(child.text for child in session.calls) == text
    assert len(rendered.requests) > 1
    span_children = [
        child.request for child in rendered.requests if child.request.say_as
    ]
    assert len(span_children) == 1
    child = span_children[0]
    child_span = child.say_as[0]
    assert child.text[child_span.start : child_span.end] == "A R T"
    assert child_span.start == span_start - next(
        item.char_start for item in rendered.requests if item.request is child
    )

    child_by_id = {item.request.id: item.request for item in rendered.requests}
    for request_id, child_request in child_by_id.items():
        measured = [item for item in session.measure_calls if item.id == request_id]
        assert measured
        assert all(
            (item.text, item.say_as) == (child_request.text, child_request.say_as)
            for item in measured
        )

    diagnostic = next(
        item.to_dict() for item in rendered.requests if item.request.say_as
    )
    assert diagnostic["request"]["say_as"] == [
        {
            "start": child_span.start,
            "end": child_span.end,
            "interpret_as": "characters",
            "format": None,
            "detail": None,
        }
    ]

    with pytest.raises(ValueError, match="partially overlaps a say_as span"):
        _slice_request(request, span_start + 1, len(text))


def test_fitting_request_with_say_as_stays_one_child() -> None:
    text = "A R T fits."
    request = _request(
        text,
        say_as=(SayAsSpan(0, len("A R T"), "characters"),),
    )
    session = _MeasuredSession(100)

    rendered = render_atomic_request(session, request)

    assert len(session.calls) == 1
    assert session.calls[0].text == text
    assert session.calls[0].say_as == request.say_as
    assert rendered.requests[0].request.say_as == request.say_as
    assert "".join(child.text for child in session.calls) == text


def test_live_acoustic_request_identity_includes_say_as() -> None:
    request = _request(
        "A R T",
        say_as=(SayAsSpan(0, 5, "characters", format="spell-out"),),
    )
    session = _MeasuredSession(100)
    session.synthesis_identity = {"engine": "test"}

    result = synthesize(session, request)

    identity = result.metadata["acoustic_identity"]["request"]
    assert identity["say_as"] == [
        {
            "start": 0,
            "end": 5,
            "interpret_as": "characters",
            "format": "spell-out",
            "detail": None,
        }
    ]


def test_capacity_fitting_preserves_leading_trailing_and_repeated_whitespace():
    request = _request("  alpha \t beta  ")
    session = _MeasuredSession(10)

    rendered = render_atomic_request(session, request)

    assert [child.text for child in session.calls] == ["  alpha \t ", "beta  "]
    assert "".join(child.text for child in session.calls) == request.text
    assert rendered.requests[-1].char_end == len(request.text)


def test_capacity_fitting_rebases_complete_tokens_and_pronunciation_spans():
    request = _request(
        "red blue green",
        tokens=(SpeechToken(start=4, end=8, text="blue", lemma="blue"),),
        pronunciation_overrides=(
            PronunciationSpan(start=4, end=8, phonemes="bluː", alphabet="ipa"),
        ),
    )
    session = _MeasuredSession(8)

    rendered = render_atomic_request(session, request)

    assert "".join(child.text for child in session.calls) == request.text
    assert all(
        token.start == 0 or token.end == len(child.text)
        for child in session.calls
        for token in child.tokens
    )
    token_child = next(child for child in session.calls if child.tokens)
    assert token_child.tokens[0].text == "blue"
    assert token_child.tokens[0].start == token_child.pronunciation_overrides[0].start
    assert token_child.tokens[0].end == token_child.pronunciation_overrides[0].end
    assert rendered.requests[0].char_start == 0


def test_unknown_measurement_uses_typed_too_long_fallback_and_merges_timings():
    request = _request("alpha beta gamma")
    session = _UnknownSession(6)

    rendered = render_atomic_request(session, request)

    assert [len(call.text) for call in session.calls] == [16, 6, 10, 5, 5]
    assert [item.request.text for item in rendered.requests] == [
        "alpha ",
        "beta ",
        "gamma",
    ]
    assert "".join(item.request.text for item in rendered.requests) == request.text
    assert rendered.result.audio.size == len(request.text)
    assert rendered.result.word_timings[-1].char_start == request.text.index("gamma")
    assert rendered.result.word_timings[-1].start_sample == 11
    assert rendered.result.warnings[0] == "shared warning"
    assert rendered.result.warnings.count("shared warning") == 1
    assert len(rendered.requests) == 3


def test_oversized_whole_request_phonemes_are_not_split():
    request = _request("alpha beta", whole_request_phonemes="ɑl.fə beɪ.tə")
    session = _MeasuredSession(5)

    with pytest.raises(SpeechRequestTooLongError, match="cannot be subdivided safely"):
        render_atomic_request(session, request)

    assert session.calls == []


def test_protected_range_without_a_legal_boundary_fails_before_inference():
    request = _request("longword")
    session = _MeasuredSession(4)

    with pytest.raises(
        SpeechRequestTooLongError, match="no legal source-text split boundary"
    ):
        render_atomic_request(
            session, request, protected_ranges=((0, len(request.text)),)
        )

    assert session.calls == []


def test_sample_rate_mismatch_between_children_is_a_typed_backend_error():
    request = _request("alpha beta")
    session = _MeasuredSession(6)
    calls: list[SpeechRequest] = []

    def synthesize(request: SpeechRequest) -> RenderedSpeech:
        calls.append(request)
        rate = 22_050 if request.text.startswith("alpha") else 24_000
        return _rendered(request, rate, len(calls))

    session.synthesize = synthesize

    with pytest.raises(EngineBackendError, match="different sample rates"):
        render_atomic_request(session, request)

    assert [call.text for call in calls] == ["alpha ", "beta"]


def test_public_synthesize_uses_the_shared_capacity_spine():
    request = _request("alpha beta gamma")
    session = _MeasuredSession(6)

    result = synthesize(
        session,
        request,
        capacity=CapacityContext(planned_semantics=True),
    )

    assert [child.text for child in session.calls] == ["alpha ", "beta ", "gamma"]
    assert "".join(child.text for child in session.calls) == request.text
    assert (
        result.metadata["atomic_lowering"]["capacity_schema"]
        == "voicerender.capacity-fitting.v1"
    )
