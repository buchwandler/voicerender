from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from renderplan import (
    CapacityConstraint,
    RenderPlan,
    SourceSpan,
    SpeechControl,
    SpeechItem,
    TokenBundle,
)
from renderplan.errors import ValidationError

from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineDiagnostic,
    EngineSelection,
    PronunciationSpan,
    RenderProfileFragment,
    RequestMeasure,
    SayAsSpan,
    SpeechRequest,
    SpeechToken,
)
from voicerender.engines.inflectsynth import (
    InflectSynthEngineAdapter,
    InflectSynthEngineSession,
)
from voicerender.engines.kittensynth import KittenSynthEngineAdapter
from voicerender.engines.kokoro import PyKokoroEngineAdapter
from voicerender.engines.pipersynth import PiperSynthEngineAdapter
from voicerender.engines.pocketsynth import (
    PocketSynthEngineAdapter,
    PocketSynthEngineSession,
)
from voicerender.engines.registry import _ADAPTERS
from voicerender.engines.selection import EngineRequest
from voicerender.engines.supertonicsynth import (
    SupertonicSynthEngineAdapter,
    SupertonicSynthEngineSession,
)
from voicerender.errors import (
    EngineBackendError,
    EngineSelectionError,
    InvalidSpeechRequestError,
    ProfileError,
    SpeechRequestTooLongError,
    UnsupportedSynthesisFeatureError,
)
from voicerender.preflight import preflight_render_plan
from voicerender.renderplan_input import (
    PreflightDisposition,
    classify_measurement,
    engine_selection_from_binding,
    speech_request_conversion_from_renderplan,
    speech_request_from_renderplan,
)
from voicerender.voice_sources import local_reference_source


class _Adapter:
    id = "renderplan-input-test"
    package_name = "renderplan-input-test"

    def __init__(self) -> None:
        self.required_fields: tuple[str, ...] = ("span", "pos", "tag", "lemma", "morph")
        self.supports_tokens = True
        self.native_semantics: frozenset[str] = frozenset()
        self.pronunciation = True
        self.supports_say_as = False
        self.say_as_interpretations: frozenset[str] = frozenset()
        self.lowered_controls: list[SpeechControl] = []

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
            supports_say_as=self.supports_say_as,
            say_as_interpretations=self.say_as_interpretations,
            supports_linguistic_tokens=self.supports_tokens,
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
        self.lowered_controls.append(control)
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
    speaker: str | int | None = 2,
    binding_extra: dict[str, Any] | None = None,
) -> SpeechItem:
    binding: dict[str, Any] = {
        "renderer": "renderplan-input-test",
        "target_id": "target-v1",
        "voice": "voice-a",
        "options": {"speed": 1.1},
        "renderer_metadata": {"frontend": "test-frontend"},
    }
    if speaker is not None:
        binding["speaker"] = speaker
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


def test_optional_tokens_are_dropped_for_adapter_without_token_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.supports_tokens = False
    adapter.required_fields = ()
    item = _item(tokens=_complete_tokens())
    original_binding = dict(item.binding)
    original_tokens = item.tokens
    original_id = item.computed_id()

    request = speech_request_from_renderplan(item)

    assert request.tokens == ()
    assert request.text == item.text
    assert request.options == {"speed": 1.1}
    assert item.binding == original_binding
    assert item.tokens is original_tokens
    assert item.computed_id() == original_id


def test_optional_tokens_are_preserved_for_supporting_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.supports_tokens = True
    adapter.required_fields = ()
    item = _item(tokens=_complete_tokens())

    request = speech_request_from_renderplan(item)

    assert [token.text for token in request.tokens] == ["Hello", ",", "world"]
    assert [token.pos for token in request.tokens] == ["NOUN", "PUNCT", "NOUN"]


def test_sparse_optional_kitten_tokens_convert_without_morphology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.supports_tokens = True
    adapter.required_fields = ()
    item = _item(
        text="read",
        tokens=TokenBundle(
            span=((0, 4),),
            lemma=((0, "read"),),
            morph=(),
        ),
    )

    request = speech_request_from_renderplan(item)

    assert len(request.tokens) == 1
    assert request.tokens[0].text == "read"
    assert request.tokens[0].lemma == "read"
    assert request.tokens[0].pos is None
    assert request.tokens[0].tag is None
    assert request.tokens[0].morph is None


def test_required_tokens_rejected_when_adapter_declares_no_token_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.supports_tokens = False
    adapter.required_fields = ("span",)

    with pytest.raises(ProfileError, match="requires linguistic token fields"):
        speech_request_from_renderplan(_item(tokens=_complete_tokens()))


def _builtin_renderplan_request(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    *,
    speaker: str | int | None = 2,
    tokens: TokenBundle | None = None,
    text: str = "  Hello, world!  ",
) -> SpeechRequest:
    from voicerender import renderplan_input

    probe = EngineApiProbe(
        engine=adapter.id,
        package=adapter.package_name,
        compatible=True,
        status="ready",
        expected_api_version=1,
        api_version=1,
        contract_source="explicit",
    )
    monkeypatch.setattr(renderplan_input, "get_adapter", lambda renderer: adapter)
    monkeypatch.setattr(
        renderplan_input,
        "resolve_voice",
        lambda request: (
            EngineSelection(
                engine=request.engine,
                target_id=request.target_id or "target-v1",
                language=request.language or "en-US",
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        ),
    )
    monkeypatch.setattr(adapter, "probe_api", lambda: probe)
    monkeypatch.setattr(adapter, "target_metadata", lambda selection: {})
    item = _item(
        tokens=tokens if tokens is not None else _complete_tokens(),
        text=text,
        binding_extra={"renderer": adapter.id},
        speaker=speaker,
    )
    return speech_request_from_renderplan(item)


def test_kittensynth_profile_contract_converts_sparse_optional_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = KittenSynthEngineAdapter()
    assert adapter.capabilities().supports_linguistic_tokens is True
    request = _builtin_renderplan_request(
        monkeypatch,
        adapter,
        speaker=None,
        text="read",
        tokens=TokenBundle(
            span=((0, 4),),
            lemma=((0, "read"),),
            morph=(),
        ),
    )

    assert len(request.tokens) == 1
    assert request.tokens[0].text == "read"
    assert request.tokens[0].lemma == "read"
    assert request.tokens[0].morph is None


@pytest.mark.parametrize(
    ("adapter_factory", "supports_tokens"),
    [
        pytest.param(InflectSynthEngineAdapter, False, id="inflect"),
        pytest.param(PocketSynthEngineAdapter, False, id="pocket"),
        pytest.param(SupertonicSynthEngineAdapter, False, id="supertonic"),
        pytest.param(PyKokoroEngineAdapter, True, id="kokoro"),
        pytest.param(PiperSynthEngineAdapter, True, id="piper"),
    ],
)
def test_builtin_adapter_capability_controls_renderplan_token_lowering(
    monkeypatch: pytest.MonkeyPatch,
    adapter_factory: Any,
    supports_tokens: bool,
) -> None:
    adapter = adapter_factory()
    assert adapter.capabilities().supports_linguistic_tokens is supports_tokens

    request = _builtin_renderplan_request(monkeypatch, adapter)

    if supports_tokens:
        assert [token.text for token in request.tokens] == ["Hello", ",", "world"]
    else:
        assert request.tokens == ()
    assert request.text == "  Hello, world!  "
    assert request.options == {"speed": 1.1}


def test_pocket_renderplan_request_reaches_capacity_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = PocketSynthEngineAdapter()
    request = _builtin_renderplan_request(monkeypatch, adapter, speaker=None)
    module = ModuleType("pocketsynth")
    module.SynthesisRequest = lambda **values: SimpleNamespace(**values)
    monkeypatch.setitem(sys.modules, "pocketsynth", module)
    measured_requests: list[Any] = []

    def measure_request(native: Any) -> Any:
        measured_requests.append(native)
        return SimpleNamespace(fits=True, amount=3, maximum=20)

    session = PocketSynthEngineSession(
        SimpleNamespace(measure_request=measure_request),
        EngineSelection(engine="pocket", target_id="target-v1", language="en-US"),
        generation=None,
        voice_level=None,
    )

    measured = session.measure(request)

    assert measured.fits is True
    assert measured.amount == 3
    assert measured_requests[0].text == request.text


def test_inflect_renderplan_optional_tokens_reach_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = InflectSynthEngineAdapter()
    request = _builtin_renderplan_request(monkeypatch, adapter, speaker=None)
    monkeypatch.setitem(sys.modules, "inflectsynth", ModuleType("inflectsynth"))
    calls: list[str] = []

    class Voice:
        def synthesize_prepared(self, text: str, **kwargs: Any) -> Any:
            calls.append(text)
            raise RuntimeError("synthesis reached")

    session = InflectSynthEngineSession(
        Voice(),
        EngineSelection(
            engine="inflect", target_id="target-v1", language="en-US", voice="voice-a"
        ),
        config=None,
    )

    with pytest.raises(EngineBackendError, match="synthesis reached"):
        session.synthesize(request)

    assert calls == [request.text]


def test_supertonic_renderplan_optional_tokens_reach_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SupertonicSynthEngineAdapter()
    request = _builtin_renderplan_request(monkeypatch, adapter, speaker=None)
    module = ModuleType("supertonicsynth")
    module.SynthesisRequest = lambda **values: SimpleNamespace(**values)
    monkeypatch.setitem(sys.modules, "supertonicsynth", module)
    calls: list[Any] = []

    def synthesize(native: Any, **kwargs: Any) -> Any:
        calls.append(native)
        raise RuntimeError("synthesis reached")

    session = SupertonicSynthEngineSession(
        SimpleNamespace(synthesize=synthesize),
        EngineSelection(
            engine="supertonic",
            target_id="target-v1",
            language="en-US",
            voice="voice-a",
        ),
        generation=None,
        voice_level=SimpleNamespace(mode="off"),
    )

    with pytest.raises(EngineBackendError, match="synthesis reached"):
        session.synthesize(request)

    assert calls[0].text == request.text


def test_unsupported_engine_sessions_retain_direct_token_rejection() -> None:
    request = SpeechRequest(
        id="direct-token-request",
        text="hello",
        language="en-US",
        tokens=(SpeechToken(start=0, end=5, text="hello"),),
    )
    inflect = InflectSynthEngineSession(
        object(),
        EngineSelection(engine="inflect", target_id="target-v1", language="en-US"),
        config=None,
    )
    pocket = PocketSynthEngineSession(
        object(),
        EngineSelection(engine="pocket", target_id="target-v1", language="en-US"),
        generation=None,
        voice_level=None,
    )
    supertonic = SupertonicSynthEngineSession(
        object(),
        EngineSelection(engine="supertonic", target_id="target-v1", language="en-US"),
        generation=None,
        voice_level=None,
    )

    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="does not support linguistic tokens"
    ):
        inflect.synthesize(request)
    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="does not support linguistic tokens"
    ):
        pocket._native_request(request, object())
    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="does not support linguistic tokens"
    ):
        supertonic.synthesize(request)


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


def test_binding_resolver_accepts_pinned_local_reference_and_rejects_unknown_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install(monkeypatch)
    with pytest.raises(EngineSelectionError, match="unknown target"):
        engine_selection_from_binding(
            {"renderer": "renderplan-input-test", "target_id": "bad"},
            language="en-US",
        )

    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"reference audio")
    source = local_reference_source(reference)
    selection = engine_selection_from_binding(
        {
            "renderer": "renderplan-input-test",
            "target_id": "target-v1",
            "options": {"voice_source": source},
        },
        language="en-US",
    )
    assert selection.options["voice_source"] == source


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


def test_emphasis_without_downstream_policy_and_unknown_controls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    emphasis = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="emphasis", params={}),),
    )
    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="explicitly declared downstream handler"
    ):
        speech_request_from_renderplan(emphasis)

    unknown = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="mystery", params={}),),
    )
    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="Unsupported RenderPlan speech control"
    ):
        speech_request_from_renderplan(unknown)

    extension = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="extension:example", params={}),),
    )
    with pytest.raises(
        UnsupportedSynthesisFeatureError,
        match="does not declare support for 'extension:example'",
    ):
        speech_request_from_renderplan(extension, emphasis_policy="defer-to-compose")


def test_unsupported_say_as_characters_is_rejected_during_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    adapter.required_fields = ()
    item = _item(
        text="A R T",
        controls=(SpeechControl(0, 5, "say_as", {"interpret_as": "characters"}),),
    )
    plan = RenderPlan(
        source_plan_id="sha256:" + "1" * 64,
        render_profile_id="sha256:" + "2" * 64,
        flow=(item,),
    ).with_identity()

    report = preflight_render_plan(plan, tmp_path / "say-as-preflight.toml")

    assert report.status == "reject"
    assert len(report.checks) == 1
    check = report.checks[0]
    assert check.status == "error"
    assert check.detail is not None
    assert "does not support say_as='characters'" in check.detail
    assert not adapter.lowered_controls


def test_say_as_control_lowers_to_typed_span_without_changing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.required_fields = ()
    adapter.supports_say_as = True
    adapter.say_as_interpretations = frozenset({"characters"})
    text = "Because someone called A R T finally."
    start = text.index("A R T")
    item = _item(
        text=text,
        controls=(
            SpeechControl(
                start=start,
                end=start + len("A R T"),
                name="say_as",
                params={"interpret_as": "characters"},
            ),
        ),
    )

    request = speech_request_from_renderplan(item)

    assert request.text == text
    assert request.say_as == (
        SayAsSpan(
            start=start,
            end=start + len("A R T"),
            interpret_as="characters",
        ),
    )
    assert "renderplan_controls" not in request.options


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"interpret_as": "  "},
        {"interpret_as": "characters", "format": 1},
        {"interpret_as": "characters", "detail": None},
        {"interpret_as": "characters", "extra": "value"},
    ],
)
def test_say_as_invalid_parameters_are_rejected(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    adapter = _install(monkeypatch)
    adapter.required_fields = ()
    adapter.supports_say_as = True
    adapter.say_as_interpretations = frozenset({"characters"})
    control = SpeechControl(0, 5, "say_as", {"interpret_as": "characters"})
    item = _item(text="A R T", controls=(control,))
    object.__setattr__(control, "params", params)

    with pytest.raises(UnsupportedSynthesisFeatureError, match="say_as"):
        speech_request_from_renderplan(item)


def test_say_as_rejects_unsupported_interpretation_and_pronunciation_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    adapter.required_fields = ()
    adapter.supports_say_as = True
    adapter.say_as_interpretations = frozenset({"date"})
    item = _item(
        text="A R T",
        controls=(SpeechControl(0, 5, "say_as", {"interpret_as": "characters"}),),
    )
    with pytest.raises(
        UnsupportedSynthesisFeatureError,
        match="does not support say_as='characters'",
    ):
        speech_request_from_renderplan(item)

    adapter.say_as_interpretations = frozenset({"characters"})
    overlapping = _item(
        text="A R T",
        controls=(
            SpeechControl(0, 5, "say_as", {"interpret_as": "characters"}),
            SpeechControl(2, 5, "pronunciation", {"phonemes": "ɑːɹ tiː"}),
        ),
    )
    with pytest.raises(
        UnsupportedSynthesisFeatureError,
        match="cannot overlap pronunciation",
    ):
        speech_request_from_renderplan(overlapping)


def test_speech_request_rejects_say_as_with_whole_request_phonemes() -> None:
    with pytest.raises(ValueError, match="whole-request phonemes"):
        SpeechRequest(
            id="request-1",
            text="A R T",
            language="en-US",
            whole_request_phonemes="ɑːɹ tiː",
            say_as=(SayAsSpan(0, 5, "characters"),),
        )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"level": "reduced"},
        {"level": "moderate"},
        {"level": "strong"},
    ],
)
def test_emphasis_is_deferred_without_native_lowering(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    adapter = _install(monkeypatch)
    item = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="emphasis", params=params),),
    )
    original_controls = item.controls
    original_id = item.computed_id()

    conversion = speech_request_conversion_from_renderplan(
        item, emphasis_policy="defer-to-compose"
    )

    assert conversion.deferred_capability == "audiocompose.emphasis.gain.v1"
    assert conversion.deferred_control_count == 1
    assert conversion.request.text == item.text
    assert conversion.request.id == original_id
    assert "renderplan_controls" not in conversion.request.options
    assert adapter.lowered_controls == []
    assert item.controls == original_controls
    assert item.computed_id() == original_id


@pytest.mark.parametrize(
    "params",
    [
        {"volume": 2},
        {"rate": 1.1},
        {"level": "extreme"},
        {"level": 1},
        {"level": True},
        {"level": "strong", "volume": 2},
    ],
)
def test_invalid_emphasis_parameters_fail_closed(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    _install(monkeypatch)
    item = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="emphasis", params=params),),
    )
    with pytest.raises(UnsupportedSynthesisFeatureError):
        speech_request_from_renderplan(item, emphasis_policy="defer-to-compose")


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 2), (2, 2), (7, 2), (True, 4)],
)
def test_invalid_control_ranges_are_rejected_by_renderplan_model(
    start: int, end: int
) -> None:
    with pytest.raises(ValidationError, match="control"):
        SpeechControl(start=start, end=end, name="emphasis", params={})


def test_out_of_text_control_range_fails_closed_during_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    item = _item(
        tokens=_complete_tokens(),
        controls=(SpeechControl(start=2, end=7, name="emphasis", params={}),),
    )
    object.__setattr__(item.controls[0], "end", len(item.text) + 1)

    with pytest.raises(UnsupportedSynthesisFeatureError, match="invalid range"):
        speech_request_from_renderplan(item, emphasis_policy="defer-to-compose")


def test_speech_item_rejects_control_ranges_past_text() -> None:
    control = SpeechControl(
        start=2, end=len(_item().text) + 1, name="emphasis", params={}
    )
    with pytest.raises(ValidationError, match="exceeds speech text length"):
        _item(controls=(control,))


def test_overlapping_emphasis_controls_are_only_deferred_once_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    controls = (
        SpeechControl(start=2, end=7, name="emphasis", params={}),
        SpeechControl(start=4, end=8, name="emphasis", params={"level": "strong"}),
    )
    item = _item(tokens=_complete_tokens(), controls=controls)

    conversion = speech_request_conversion_from_renderplan(
        item, emphasis_policy="defer-to-compose"
    )

    assert conversion.deferred_control_count == 2
    assert conversion.deferred_capability == "audiocompose.emphasis.gain.v1"
    assert conversion.request.options == {"speed": 1.1}
    assert adapter.lowered_controls == []
    assert item.controls == controls


def test_native_pronunciation_is_preserved_alongside_deferred_emphasis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    item = _item(
        tokens=_complete_tokens(),
        controls=(
            SpeechControl(
                start=2, end=7, name="emphasis", params={"level": "moderate"}
            ),
            SpeechControl(
                start=9,
                end=14,
                name="pronunciation",
                params={"phonemes": "wɝːld", "alphabet": "ipa"},
            ),
        ),
    )

    conversion = speech_request_conversion_from_renderplan(
        item, emphasis_policy="defer-to-compose"
    )

    assert conversion.deferred_control_count == 1
    assert conversion.request.pronunciation_overrides == (
        PronunciationSpan(
            start=9,
            end=14,
            phonemes="wɝːld",
            language="en-US",
            alphabet="ipa",
        ),
    )
    assert "renderplan_controls" not in conversion.request.options
    assert adapter.lowered_controls == []


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
    invalid_too_long = classify_measurement(
        SpeechRequestTooLongError(
            "malformed overflow",
            amount=21,
            maximum=0,
            unit="model_tokens",
            source="fake.prepare",
        )
    )
    assert invalid_too_long.disposition is PreflightDisposition.REJECT
    assert invalid_too_long.error == "Engine returned an invalid capacity measurement"
    assert invalid_too_long.measurement is not None
    assert invalid_too_long.measurement.source == "fake.prepare.invalid_measurement"

    invalid = InvalidSpeechRequestError("bad request")
    assert classify_measurement(invalid).disposition is PreflightDisposition.REJECT


def test_measured_amount_without_authoritative_fit_is_rejected_with_details() -> None:
    measurement = RequestMeasure(
        fits=None,
        amount=80,
        maximum=None,
        unit="model_tokens",
        source="inflectsynth.measure_prepared",
    )

    outcome = classify_measurement(measurement)

    assert outcome.disposition is PreflightDisposition.REJECT
    assert outcome.measurement == measurement
    assert outcome.error == (
        "Engine measured the exact request size but did not provide an "
        "authoritative maximum/fit result"
    )


def test_invalid_adapter_measurement_has_stable_rejection() -> None:
    outcome = classify_measurement(
        RequestMeasure(
            fits=None,
            amount=80,
            maximum=None,
            unit="model_tokens",
            source="inflectsynth.invalid_measurement",
            details={"invalid_fields": ["maximum"]},
        )
    )

    assert outcome.disposition is PreflightDisposition.REJECT
    assert outcome.error == "Engine returned an invalid capacity measurement"


def test_fit_without_exact_amount_is_not_accepted() -> None:
    outcome = classify_measurement(
        RequestMeasure(
            fits=True,
            amount=None,
            maximum=100,
            unit="model_tokens",
            source="fake.measure",
        )
    )

    assert outcome.disposition is PreflightDisposition.REJECT
    assert outcome.error == "Engine did not provide an exact request measurement"


def test_zero_maximum_is_invalid() -> None:
    outcome = classify_measurement(
        RequestMeasure(
            fits=None,
            amount=1,
            maximum=0,
            unit="model_tokens",
            source="fake.measure",
        )
    )

    assert outcome.disposition is PreflightDisposition.REJECT
    assert outcome.error == "Engine returned an invalid capacity measurement"


def test_discovery_runtime_capacity_mismatch_is_rejected_with_both_values() -> None:
    outcome = classify_measurement(
        RequestMeasure(
            fits=None,
            amount=12,
            maximum=90,
            unit="model_tokens",
            source="inflectsynth.capacity_metadata_mismatch",
            details={
                "discovered_max_input_tokens": 100,
                "runtime_maximum": 90,
            },
        )
    )

    assert outcome.disposition is PreflightDisposition.REJECT
    assert outcome.measurement is not None
    assert "discovered max_input_tokens=100" in (outcome.error or "")
    assert "runtime maximum=90" in (outcome.error or "")
