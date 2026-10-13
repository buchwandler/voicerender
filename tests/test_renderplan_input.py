from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from renderplan import (
    CapacityConstraint,
    SourceSpan,
    SpeechControl,
    SpeechItem,
    TokenBundle,
)

from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineDiagnostic,
    EngineSelection,
    PronunciationSpan,
    RenderProfileFragment,
    RequestMeasure,
)
from voicerender.engines.registry import _ADAPTERS
from voicerender.engines.selection import EngineRequest
from voicerender.errors import (
    EngineSelectionError,
    InvalidSpeechRequestError,
    ProfileError,
    SpeechRequestTooLongError,
    UnsupportedSynthesisFeatureError,
)
from voicerender.renderplan_input import (
    PreflightDisposition,
    classify_measurement,
    engine_selection_from_binding,
    speech_request_from_renderplan,
)


class _Adapter:
    id = "renderplan-input-test"
    package_name = "renderplan-input-test"

    def __init__(self) -> None:
        self.required_fields: tuple[str, ...] = ("span", "pos", "tag", "lemma", "morph")
        self.native_semantics: frozenset[str] = frozenset()
        self.pronunciation = True

    def probe_api(self) -> EngineApiProbe:
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            supports_pronunciation_overrides=self.pronunciation,
            pronunciation_alphabets=frozenset({"ipa"}),
            native_semantics=self.native_semantics,
        )

    def version(self) -> str:
        return "1.0"

    def resolve(
        self, request: EngineRequest
    ) -> tuple[EngineSelection, tuple[EngineDiagnostic, ...]]:
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or "target-v1",
                language=request.language or "en-US",
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(
        self, selection: EngineSelection
    ) -> tuple[EngineDiagnostic, ...]:
        if selection.target_id != "target-v1":
            return (
                EngineDiagnostic(
                    code="test.target_unknown",
                    severity="error",
                    message="unknown target",
                ),
            )
        return ()

    def target_metadata(self, selection: EngineSelection) -> dict[str, Any]:
        return {"source_revision": "runtime-r1"}

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return RenderProfileFragment(
            binding={
                "renderer": self.id,
                "target_id": selection.target_id,
                "voice": selection.voice,
                "options": dict(selection.options),
            },
            token_fields=self.required_fields,
            frontend="test-frontend",
            pronunciation="test-pronunciation",
            renderer_capacity=CapacityConstraint(
                metric="model_tokens", maximum=128, guarantee="hard"
            ),
        )

    def lower_renderplan_control(
        self, control: SpeechControl, selection: EngineSelection
    ) -> dict[str, Any]:
        return {
            "start": control.start,
            "end": control.end,
            "options": {"value": control.params.get("value")},
        }

    @contextmanager
    def open(self, selection: EngineSelection) -> Iterator[object]:
        yield object()


def _install(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
    adapter = _Adapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    return adapter


def _item(
    *,
    text: str = "  Hello, world!  ",
    tokens: TokenBundle | None = None,
    controls: tuple[SpeechControl, ...] = (),
    binding_extra: dict[str, Any] | None = None,
) -> SpeechItem:
    binding: dict[str, Any] = {
        "renderer": "renderplan-input-test",
        "target_id": "target-v1",
        "voice": "voice-a",
        "speaker": 2,
        "options": {"speed": 1.1},
        "renderer_metadata": {"frontend": "test-frontend"},
    }
    binding.update(binding_extra or {})
    return SpeechItem(
        number=1,
        text=text,
        language="en-US",
        role="narrator",
        binding=binding,
        source=(SourceSpan(locator="f000000.s000", start=0, end=len(text)),),
        tokens=tokens,
        controls=controls,
    )


def _complete_tokens() -> TokenBundle:
    return TokenBundle(
        span=((2, 7), (7, 8), (9, 14)),
        pos=("NOUN", "PUNCT", "NOUN"),
        tag=("NN", ",", "NN"),
        lemma=((0, "hello"), (1, ","), (2, "world")),
        morph=((0, "Number=Sing"), (1, "PunctType=Comm"), (2, "Number=Sing")),
    )


def test_request_conversion_preserves_exact_text_binding_and_token_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    item = _item(tokens=_complete_tokens())

    request = speech_request_from_renderplan(item)

    assert request.id == item.computed_id()
    assert request.text == "  Hello, world!  "
    assert request.language == "en-US"
    assert request.voice == "voice-a"
    assert request.speaker == 2
    assert request.options == {"speed": 1.1}
    assert [token.text for token in request.tokens] == ["Hello", ",", "world"]
    assert [(token.start, token.end) for token in request.tokens] == [
        (2, 7),
        (7, 8),
        (9, 14),
    ]
    assert [token.pos for token in request.tokens] == ["NOUN", "PUNCT", "NOUN"]
    assert [token.tag for token in request.tokens] == ["NN", ",", "NN"]
    assert [token.lemma for token in request.tokens] == ["hello", ",", "world"]
    assert [token.morph for token in request.tokens] == [
        "Number=Sing",
        "PunctType=Comm",
        "Number=Sing",
    ]
    assert all(token.language == "en-US" for token in request.tokens)


def test_binding_resolver_rejects_unknown_target_and_reference_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    with pytest.raises(EngineSelectionError, match="unknown target"):
        engine_selection_from_binding(
            {"renderer": "renderplan-input-test", "target_id": "bad"},
            language="en-US",
        )

    with pytest.raises(
        ProfileError, match="Reference-audio bindings are not supported"
    ):
        engine_selection_from_binding(
            {
                "renderer": "renderplan-input-test",
                "target_id": "target-v1",
                "options": {
                    "voice_source": {"kind": "reference", "value": "old-profile.wav"}
                },
            },
            language="en-US",
        )


def test_missing_profile_required_token_fields_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.required_fields = ("span", "pos")
    item = _item(tokens=TokenBundle(span=((2, 7),)))

    with pytest.raises(ProfileError, match="incomplete required token fields: pos"):
        speech_request_from_renderplan(item)


def test_pronunciation_control_preserves_exact_request_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    item = _item(
        tokens=_complete_tokens(),
        controls=(
            SpeechControl(
                start=9,
                end=14,
                name="pronunciation",
                params={"phonemes": "wɝːld", "alphabet": "ipa"},
            ),
        ),
    )

    request = speech_request_from_renderplan(item)

    assert request.pronunciation_overrides == (
        PronunciationSpan(
            start=9,
            end=14,
            phonemes="wɝːld",
            language="en-US",
            alphabet="ipa",
        ),
    )


def test_prosody_control_requires_adapter_declaration_and_exact_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    control = SpeechControl(start=2, end=7, name="prosody", params={"value": "slow"})
    item = _item(tokens=_complete_tokens(), controls=(control,))

    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="does not declare support"
    ):
        speech_request_from_renderplan(item)

    adapter.native_semantics = frozenset({"prosody"})
    request = speech_request_from_renderplan(item)
    assert request.options["renderplan_controls"] == (
        {"name": "prosody", "start": 2, "end": 7, "options": {"value": "slow"}},
    )


def test_emphasis_and_unknown_controls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    emphasis = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="emphasis", params={}),),
    )
    with pytest.raises(UnsupportedSynthesisFeatureError):
        speech_request_from_renderplan(emphasis)

    adapter.native_semantics = frozenset({"extension:example"})
    unknown = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="mystery", params={}),),
    )
    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="Unsupported RenderPlan speech control"
    ):
        speech_request_from_renderplan(unknown)


def test_measurement_outcomes_and_typed_too_long_exception_are_centralized() -> None:
    assert (
        classify_measurement(
            RequestMeasure(True, 12, 20, "model_tokens", "fake.prepare")
        ).disposition
        is PreflightDisposition.OK
    )
    assert (
        classify_measurement(
            RequestMeasure(False, 21, 20, "model_tokens", "fake.prepare")
        ).disposition
        is PreflightDisposition.TOO_LONG
    )
    assert (
        classify_measurement(
            RequestMeasure(None, None, None, "unknown", "fake.no_measure")
        ).disposition
        is PreflightDisposition.REJECT
    )

    too_long = SpeechRequestTooLongError(
        "request too long",
        amount=21,
        maximum=20,
        unit="model_tokens",
        source="fake.prepare",
    )
    outcome = classify_measurement(too_long)
    assert outcome.disposition is PreflightDisposition.TOO_LONG
    assert outcome.measurement is not None
    assert outcome.measurement.amount == 21
    assert outcome.measurement.maximum == 20
    assert outcome.measurement.source == "fake.prepare"
    bare_outcome = classify_measurement(SpeechRequestTooLongError("request too long"))
    assert bare_outcome.disposition is PreflightDisposition.TOO_LONG
    assert bare_outcome.measurement is not None
    assert bare_outcome.measurement.unit == "unknown"

    invalid = InvalidSpeechRequestError("bad request")
    assert classify_measurement(invalid).disposition is PreflightDisposition.REJECT
