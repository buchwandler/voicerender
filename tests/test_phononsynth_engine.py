"""Adapter tests for the public PhononSynth API."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from voicerender.engines.base import (
    EngineSelection,
    PronunciationSpan,
    RequestMeasure,
    SayAsSpan,
    SpeechRequest,
    SpeechToken,
)
from voicerender.engines.catalog import CatalogRequest
from voicerender.engines.phononsynth import (
    PHONON_OPTION_NAMES,
    PhononSelectionError,
    PhononSynthEngineAdapter,
    PhononSynthEngineSession,
    _target_from_model,
)
from voicerender.engines.selection import EngineRequest
from voicerender.errors import (
    EmptySpeechTextError,
    EngineBackendError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
    UnsupportedSynthesisFeatureError,
)


def _model() -> Any:
    return SimpleNamespace(
        id="phonon-demo",
        display_name="Phonon demo",
        languages=("en-US", "de-DE"),
        quality_choices=("q4", "fp32"),
        sample_rate=24000,
        voices=(
            SimpleNamespace(
                id="voice-a",
                display_name="Voice A",
                gender="female",
                language="en-US",
                locale="en-US",
                languages=("en-US",),
            ),
            SimpleNamespace(
                id="voice-b",
                display_name="Voice B",
                gender="male",
                language="de-DE",
                locale="de-DE",
                languages=("de-DE",),
            ),
        ),
        default_voice="voice-a",
        source_revision="rev-1",
        license_spdx="CC-BY-4.0",
        commercial_use=True,
        release_version="1.2.3",
        release_tags={"q4": "release-q4"},
        source_revisions={"q4": "rev-q4"},
        metadata={"generation": {"temperature": 0.3}, "source_revision": "wrong"},
    )


def _fake_package(
    *,
    api_version: Any = 2,
    contract: Any = None,
    missing_method: str | None = None,
) -> Any:
    module: Any = ModuleType("phononsynth")
    module.__version__ = "0.1.2"
    module.REQUEST_API_VERSION = api_version
    contract_value = contract or {
        "version": 2,
        "engine": "phonon",
        "entrypoint": "PhononVoice.synthesize_prepared",
        "stream_entrypoint": "PhononVoice.stream_prepared",
        "measure_entrypoint": "PhononVoice.measure_prepared",
        "input": "prepared_text",
        "measurement_unit": "model_tokens",
        "max_input_tokens": None,
        "measurement_fit_is_authoritative": True,
        "features": {
            "named_voices": True,
            "streaming": True,
            "voice_cloning": False,
            "word_timings": False,
            "automatic_chunking": False,
        },
    }
    module.request_api_contract = lambda: contract_value
    module.SynthesisConfig = type(
        "SynthesisConfig",
        (),
        {
            "__init__": lambda self, **kwargs: vars(self).update(kwargs),
            "validated": lambda self: self,
        },
    )
    module.SynthesisResult = type("SynthesisResult", (), {})
    module.SynthesisMeasure = type("SynthesisMeasure", (), {})
    module.DiscoveredModel = type("DiscoveredModel", (), {})
    module.DescribedVoice = type("DescribedVoice", (), {})
    module.runtime_identity = lambda: {"runtime": "phonon.runtime.v1"}
    module.discover_models = lambda **_kwargs: (_model(),)

    class FakeVoice:
        opens: ClassVar[list[tuple[tuple[Any, ...], dict[str, Any]]]] = []
        measure_calls: ClassVar[list[str]] = []
        synthesize_calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
        close_calls = 0
        open_error: Exception | None = None
        measure_error: Exception | None = None
        synthesize_error: Exception | None = None

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeVoice:
            cls.opens.append((args, kwargs))
            if cls.open_error is not None:
                raise cls.open_error
            return cls()

        def measure_prepared(self, text: str) -> Any:
            type(self).measure_calls.append(text)
            if self.measure_error is not None:
                raise self.measure_error
            return SimpleNamespace(
                token_count=17, max_tokens=None, fits=True, model_id="phonon-demo"
            )

        def synthesize_prepared(self, text: str, **kwargs: Any) -> Any:
            type(self).synthesize_calls.append((text, kwargs))
            if self.synthesize_error is not None:
                raise self.synthesize_error
            return SimpleNamespace(
                audio=np.asarray([1.2, -1.1, 0.25], dtype=np.float32),
                sample_rate=24000,
                voice=kwargs["voice"],
                model_ref="phonon://phonon-demo",
                metadata={"token_count": 17, "generation": {"temperature": 0.3}},
            )

        def close(self) -> None:
            type(self).close_calls += 1

    if missing_method == "measure_prepared":
        delattr(FakeVoice, "measure_prepared")
    module.PhononVoice = FakeVoice
    module.EmptyTextError = type("EmptyTextError", (Exception,), {})
    module.InvalidSynthesisConfigError = type(
        "InvalidSynthesisConfigError", (Exception,), {}
    )
    module.UnsupportedModelError = type("UnsupportedModelError", (Exception,), {})
    module.CatalogUnavailableError = type("CatalogUnavailableError", (Exception,), {})
    module.CatalogDiscoveryError = type("CatalogDiscoveryError", (Exception,), {})
    module.TokenizerError = type("TokenizerError", (Exception,), {})
    module.ManagedModelDependencyError = type(
        "ManagedModelDependencyError", (Exception,), {}
    )
    module.OnnxVoiceContractError = type("OnnxVoiceContractError", (Exception,), {})
    module.ModelInferenceError = type("ModelInferenceError", (Exception,), {})
    return module


def _selection(**kwargs: Any) -> EngineSelection:
    values: dict[str, Any] = {
        "engine": "phonon",
        "target_id": "phonon-demo",
        "language": "en-US",
        "voice": "voice-a",
        "options": {"quality": "q4"},
        "metadata": {
            "source_revision": "rev-1",
            "release_version": "1.2.3",
            "release_tags": {"q4": "release-q4"},
            "source_revisions": {"q4": "rev-q4"},
        },
    }
    values.update(kwargs)
    return EngineSelection(**values)


def test_api_probe_accepts_explicit_phonon_v2_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "phononsynth", _fake_package())

    probe = PhononSynthEngineAdapter().probe_api()

    assert probe.compatible
    assert probe.status == "ready"
    assert probe.api_version == 2
    assert probe.details["request_api_contract"]["engine"] == "phonon"


def test_api_probe_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "phononsynth":
            raise ModuleNotFoundError("missing", name="phononsynth")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "package_missing"


@pytest.mark.parametrize(
    ("api_version", "status"),
    [(1, "api_version_incompatible"), ("1", "api_probe_failed")],
)
def test_api_probe_rejects_bad_version(
    monkeypatch: pytest.MonkeyPatch, api_version: Any, status: str
) -> None:
    monkeypatch.setitem(
        sys.modules, "phononsynth", _fake_package(api_version=api_version)
    )

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == status



def test_api_probe_rejects_v1_with_actionable_error_without_opening_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package(api_version=1)
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "api_version_incompatible"
    assert probe.expected_api_version == 2
    assert "authoritative request-measurement contract" in probe.error_message
    assert "phononsynth>=0.1.2,<0.2" in probe.error_message
    assert package.PhononVoice.opens == []


@pytest.mark.parametrize("field_value", [None, False])
def test_api_probe_rejects_missing_or_false_authoritative_fit_contract(
    monkeypatch: pytest.MonkeyPatch, field_value: Any
) -> None:
    package = _fake_package()
    contract = package.request_api_contract()
    if field_value is None:
        del contract["measurement_fit_is_authoritative"]
    else:
        contract["measurement_fit_is_authoritative"] = field_value
    package = _fake_package(contract=contract)
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert probe.details["contract_mismatches"]["measurement_fit_is_authoritative"] == {
        "expected": True,
        "actual": field_value,
    }
    assert "phononsynth>=0.1.2,<0.2" in probe.error_message
    assert package.PhononVoice.opens == []

def test_api_probe_rejects_missing_required_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "phononsynth", _fake_package(missing_method="measure_prepared")
    )

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert "PhononVoice.measure_prepared" in probe.missing_methods


def test_api_probe_rejects_missing_required_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    del package.DiscoveredModel
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert "DiscoveredModel" in probe.missing_symbols


def test_api_probe_fails_closed_on_contradictory_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _fake_package().request_api_contract()
    contract["features"]["voice_cloning"] = True
    monkeypatch.setitem(sys.modules, "phononsynth", _fake_package(contract=contract))

    probe = PhononSynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.failed_stage == "request_api_contract"
    assert "voice_cloning" in probe.details["contract_mismatches"]


def test_target_mapping_preserves_normalized_voices_and_provenance() -> None:
    target = _target_from_model(_model())

    assert target.engine == "phonon"
    assert target.id == "phonon-demo"
    assert target.languages == ("en", "de")
    assert target.qualities == ("q4", "fp32")
    assert target.voices == ("voice-a", "voice-b")
    assert target.voice_details[0].gender == "female"
    assert target.voice_details[0].locale == "en-us"
    assert target.default_voice == "voice-a"
    assert target.sample_rate == 24000
    assert target.metadata["source_revision"] == "rev-1"
    assert target.metadata["release_version"] == "1.2.3"
    assert target.metadata["release_tags"] == {"q4": "release-q4"}
    assert target.metadata["source_revisions"] == {"q4": "rev-q4"}
    assert target.metadata["license_spdx"] == "CC-BY-4.0"
    assert target.metadata["commercial_use"] is True
    assert target.metadata["voice_display_names"]["voice-a"] == "Voice A"


def test_discovery_passes_offline_and_refresh_without_opening_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    calls: list[dict[str, Any]] = []
    package.discover_models = lambda **kwargs: calls.append(kwargs) or (_model(),)
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    targets = PhononSynthEngineAdapter().discover(
        CatalogRequest(engine="phonon", language="en-US", offline=True, refresh=True)
    )

    assert [target.id for target in targets] == ["phonon-demo"]
    assert calls == [
        {
            "language": "en",
            "offline": True,
            "refresh": True,
            "cache_dir": None,
            "catalog_url": None,
        }
    ]
    assert package.PhononVoice.opens == []


def test_resolution_checks_target_language_quality_and_default_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    calls: list[dict[str, Any]] = []
    package.discover_models = lambda **kwargs: calls.append(kwargs) or (_model(),)
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    selection, diagnostics = PhononSynthEngineAdapter().resolve(
        EngineRequest(
            engine="phonon",
            target_id="phonon-demo",
            language="en-US",
            engine_options={"quality": "q4", "temperature": 0.7},
            offline=True,
        )
    )

    assert diagnostics == ()
    assert selection.voice == "voice-a"
    assert selection.options["quality"] == "q4"
    assert selection.options["temperature"] == 0.7
    assert selection.metadata["source_revision"] == "rev-1"
    assert calls[0]["offline"] is True


@pytest.mark.parametrize(
    ("engine_request", "code"),
    [
        (
            EngineRequest(engine="phonon", target_id="missing", language="en"),
            "phonon.model_not_found",
        ),
        (
            EngineRequest(
                engine="phonon",
                target_id="phonon-demo",
                language="en",
                engine_options={"quality": "q8"},
            ),
            "phonon.quality_unavailable",
        ),
        (
            EngineRequest(
                engine="phonon", target_id="phonon-demo", language="en", voice="bad"
            ),
            "phonon.voice_not_found",
        ),
        (
            EngineRequest(engine="phonon", target_id="phonon-demo", language="fr-FR"),
            "engine_language_incompatible",
        ),
        (
            EngineRequest(
                engine="phonon", target_id="phonon-demo", language="en", speaker=1
            ),
            "phonon.speaker_unsupported",
        ),
        (
            EngineRequest(
                engine="phonon",
                target_id="phonon-demo",
                language="en",
                options={"voice_source": {"kind": "reference"}},
            ),
            "phonon.reference_voice_unsupported",
        ),
    ],
)
def test_resolution_rejects_invalid_and_unsupported_selection(
    monkeypatch: pytest.MonkeyPatch, engine_request: EngineRequest, code: str
) -> None:
    monkeypatch.setitem(sys.modules, "phononsynth", _fake_package())

    with pytest.raises(PhononSelectionError) as raised:
        PhononSynthEngineAdapter().resolve(engine_request)

    assert raised.value.diagnostic_code == code


def test_validate_selection_rediscovers_catalog_without_opening_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    monkeypatch.setitem(sys.modules, "phononsynth", package)
    adapter = PhononSynthEngineAdapter()

    assert adapter.validate_selection(_selection()) == ()
    invalid = adapter.validate_selection(
        _selection(language="fr-FR", options={"quality": "q8"}, voice="missing")
    )

    assert {item.code for item in invalid} == {
        "engine_language_incompatible",
        "phonon.quality_unavailable",
        "phonon.voice_not_found",
    }
    assert package.PhononVoice.opens == []


def test_profile_has_only_named_voice_quality_and_no_capacity() -> None:
    fragment = PhononSynthEngineAdapter().renderprofile_fragment(_selection())

    assert fragment.frontend == "phononsynth"
    assert fragment.pronunciation == "unsupported"
    assert fragment.token_fields == ()
    assert fragment.renderer_capacity is None
    assert fragment.planner_capacity is None
    assert fragment.binding["renderer"] == "phonon"
    assert fragment.binding["voice"] == "voice-a"
    assert fragment.binding["options"]["quality"] == "q4"
    assert "voice_source" not in fragment.binding["options"]


def test_capabilities_are_truthful_and_options_are_canonical() -> None:
    capabilities = PhononSynthEngineAdapter().capabilities()

    assert capabilities.option_names == PHONON_OPTION_NAMES
    assert capabilities.supports_named_voices
    assert capabilities.supports_qualities
    assert capabilities.supports_request_measurement
    assert not capabilities.supports_reference_voice
    assert not capabilities.supports_speakers
    assert not capabilities.supports_pronunciation_overrides
    assert not capabilities.supports_say_as
    assert not capabilities.supports_linguistic_tokens
    assert not capabilities.supports_whole_request_phonemes
    assert not capabilities.supports_live
    assert not capabilities.supports_timestamps
    assert not capabilities.supports_model_sources


def test_measurement_preserves_authoritative_fit_and_exact_prepared_text() -> None:
    class Voice:
        def __init__(self) -> None:
            self.text: str | None = None

        def measure_prepared(self, text: str) -> Any:
            self.text = text
            return SimpleNamespace(
                token_count=23, max_tokens=None, fits=True, model_id="phonon-demo"
            )

    voice = Voice()
    text = "  leading spaces and exact punctuation!\n"
    request = SpeechRequest(id="request-1", text=text, language="en-US")

    measured = PhononSynthEngineSession(voice, _selection(), None).measure(request)

    assert measured == RequestMeasure(
        fits=True,
        amount=23,
        maximum=None,
        unit="model_tokens",
        source="phononsynth.measure_prepared",
        details={"model_id": "phonon-demo"},
    )
    assert voice.text == text


@pytest.mark.parametrize(
    "measurement",
    [
        SimpleNamespace(token_count=23, max_tokens=None, fits=None),
        SimpleNamespace(token_count=True, max_tokens=None, fits=True),
        SimpleNamespace(token_count=-1, max_tokens=None, fits=True),
        SimpleNamespace(token_count=23, max_tokens=0, fits=True),
        SimpleNamespace(token_count=23, max_tokens=20, fits=True),
        SimpleNamespace(token_count=23, max_tokens=30, fits=False),
    ],
    ids=("unknown-fit", "boolean-count", "negative-count", "zero-maximum", "too-long-fit", "too-short-fit"),
)
def test_measurement_rejects_invalid_contract_values(measurement: Any) -> None:
    class Voice:
        def measure_prepared(self, _text: str) -> Any:
            return measurement

    request = SpeechRequest(id="invalid-measurement", text="hello", language="en-US")

    with pytest.raises(EngineBackendError) as raised:
        PhononSynthEngineSession(Voice(), _selection(), None).measure(request)

    assert raised.value.code == "phonon.measurement_invalid"
    assert raised.value.details["token_count"] == measurement.token_count
    assert raised.value.details["max_tokens"] == measurement.max_tokens
    assert raised.value.details["fits"] == measurement.fits

def test_measurement_translates_empty_text_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    monkeypatch.setitem(sys.modules, "phononsynth", package)
    voice = package.PhononVoice()
    voice.measure_error = package.EmptyTextError("empty")

    with pytest.raises(EmptySpeechTextError) as raised:
        PhononSynthEngineSession(voice, _selection(), None).measure(
            SpeechRequest(id="request-id", text="", language="en-US")
        )

    assert raised.value.code == "phonon.empty_text"
    assert raised.value.details["request_id"] == "request-id"


def test_synthesis_passes_exact_text_config_voice_and_clamps_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    monkeypatch.setitem(sys.modules, "phononsynth", package)
    selection = _selection()
    adapter = PhononSynthEngineAdapter()
    text = "\tExact text stays unchanged.  "
    request = SpeechRequest(id="request-2", text=text, language="en-US")

    with adapter.open(selection) as session:
        rendered = session.synthesize(request)

    voice_type = package.PhononVoice
    assert voice_type.synthesize_calls[0][0] == text
    assert voice_type.synthesize_calls[0][1]["voice"] == "voice-a"
    assert voice_type.synthesize_calls[0][1]["config"].temperature is None
    assert rendered.id == request.id
    assert rendered.sample_rate == 24000
    assert np.array_equal(
        rendered.audio, np.asarray([1.0, -1.0, 0.25], dtype=np.float32)
    )
    assert rendered.metadata["pcm_range"]["clipped_samples"] == 2
    assert rendered.metadata["quality"] == "q4"
    assert rendered.metadata["model_ref"] == "phonon://phonon-demo"
    assert rendered.warnings
    assert voice_type.close_calls == 1


def test_open_passes_runtime_controls_and_generation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    monkeypatch.setitem(sys.modules, "phononsynth", package)
    selection = _selection(
        options={
            "quality": "q4",
            "duration_delta": 0.2,
            "temperature": 0.6,
            "seed": 9,
            "eos_threshold": 2.5,
            "frames_after_eos": 3,
            "min_frames_before_eos": 4,
            "max_frames": 99,
            "cache_dir": "/tmp/phonon-cache",
            "catalog_url": "https://example.invalid/catalog.json",
            "providers": ["CPUExecutionProvider"],
            "provider_options": {
                "CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}
            },
            "session_options": {"intra_op_num_threads": 1},
            "force_download": True,
        },
        offline=True,
        refresh=True,
    )

    with PhononSynthEngineAdapter().open(selection):
        pass

    args, kwargs = package.PhononVoice.opens[0]
    assert args == ("phonon-demo",)
    assert kwargs == {
        "quality": "q4",
        "cache_dir": "/tmp/phonon-cache",
        "offline": True,
        "refresh_catalog": True,
        "force_download": True,
        "catalog_url": "https://example.invalid/catalog.json",
        "providers": ["CPUExecutionProvider"],
        "provider_options": {
            "CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}
        },
        "session_options": {"intra_op_num_threads": 1},
    }
    assert package.PhononVoice.close_calls == 1


def test_open_translates_config_error_before_opening_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()

    class Config:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def validated(self) -> Any:
            raise package.InvalidSynthesisConfigError("bad generation control")

    package.SynthesisConfig = Config
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    with pytest.raises(InvalidEngineOptionError):
        PhononSynthEngineAdapter().open(_selection())

    assert package.PhononVoice.opens == []


def test_synthesis_translates_backend_and_invalid_voice_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    monkeypatch.setitem(sys.modules, "phononsynth", package)
    selection = _selection()
    request = SpeechRequest(id="request-3", text="hello", language="en-US")
    voice = package.PhononVoice()
    voice.synthesize_error = package.ModelInferenceError("inference failed")
    with pytest.raises(EngineBackendError) as backend_error:
        PhononSynthEngineSession(voice, selection, None).synthesize(request)
    assert backend_error.value.code == "phonon.inference_failed"
    assert backend_error.value.details["target_id"] == "phonon-demo"
    assert backend_error.value.details["request_id"] == "request-3"

    voice.synthesize_error = ValueError("Unknown Phonon voice 'bad'")
    with pytest.raises(InvalidEngineVoiceError) as voice_error:
        PhononSynthEngineSession(voice, selection, None).synthesize(request)
    assert voice_error.value.code == "phonon.voice_not_found"


def test_runtime_closes_when_synthesis_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    package = _fake_package()
    package.PhononVoice.synthesize_error = package.ModelInferenceError("failed")
    monkeypatch.setitem(sys.modules, "phononsynth", package)

    with (
        pytest.raises(EngineBackendError),
        PhononSynthEngineAdapter().open(_selection()) as session,
    ):
        session.synthesize(SpeechRequest(id="failure", text="hello", language="en-US"))

    assert package.PhononVoice.close_calls == 1


@pytest.mark.parametrize(
    "speech_request",
    [
        SpeechRequest(
            id="tokens",
            text="hello",
            language="en-US",
            tokens=(SpeechToken(start=0, end=5, text="hello"),),
        ),
        SpeechRequest(
            id="pronunciation",
            text="hello",
            language="en-US",
            pronunciation_overrides=(
                PronunciationSpan(start=0, end=5, phonemes="həˈloʊ"),
            ),
        ),
        SpeechRequest(
            id="say-as",
            text="hello",
            language="en-US",
            say_as=(SayAsSpan(start=0, end=5, interpret_as="characters"),),
        ),
        SpeechRequest(
            id="phonemes",
            text="hello",
            language="en-US",
            whole_request_phonemes="həˈloʊ",
        ),
        SpeechRequest(id="speaker", text="hello", language="en-US", speaker="narrator"),
    ],
)
def test_synthesis_rejects_unsupported_request_semantics(
    speech_request: SpeechRequest,
) -> None:
    class Voice:
        def synthesize_prepared(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("unsupported semantics must fail before inference")

    with pytest.raises(UnsupportedSynthesisFeatureError):
        PhononSynthEngineSession(Voice(), _selection(), None).synthesize(speech_request)


def test_identity_tracks_output_inputs_but_not_cache_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _fake_package()
    monkeypatch.setitem(sys.modules, "phononsynth", package)
    adapter = PhononSynthEngineAdapter()
    baseline = _selection()

    def identity(selection: EngineSelection) -> Any:
        return adapter.canonical_synthesis_identity(selection)

    baseline_identity = identity(baseline)
    assert baseline_identity == identity(baseline)
    assert baseline_identity["engine_identity"] == {"runtime": "phonon.runtime.v1"}
    assert baseline_identity["source_revision"] == "rev-q4"
    assert identity(baseline) != identity(_selection(options={"quality": "fp32"}))
    assert identity(baseline) != identity(_selection(voice="voice-b"))
    for name, value in {
        "duration_delta": 0.2,
        "temperature": 0.6,
        "seed": 9,
        "eos_threshold": 2.5,
        "frames_after_eos": 3,
        "min_frames_before_eos": 4,
        "max_frames": 99,
    }.items():
        assert identity(baseline) != identity(
            _selection(options={"quality": "q4", name: value})
        )
    assert identity(baseline) != identity(
        _selection(options={"quality": "q4", "providers": ["CUDAExecutionProvider"]})
    )
    assert identity(baseline) != identity(
        _selection(
            metadata={**dict(baseline.metadata), "source_revisions": {"q4": "rev-new"}}
        )
    )
    assert identity(baseline) != identity(
        _selection(metadata={**dict(baseline.metadata), "release_version": "1.2.4"})
    )
    assert identity(baseline) == identity(
        _selection(options={"quality": "q4", "cache_dir": "/different/cache"})
    )
    assert baseline_identity == identity(_selection(offline=True, refresh=True))
    package.runtime_identity = lambda: {"runtime": "phonon.runtime.v2"}
    assert baseline_identity != identity(baseline)


def test_profile_and_identity_do_not_claim_reference_or_capacity_support() -> None:
    adapter = PhononSynthEngineAdapter()
    fragment = adapter.renderprofile_fragment(_selection())

    assert fragment.binding["renderer_metadata"]["frontend"] == "phononsynth"
    assert fragment.renderer_capacity is None
    assert fragment.planner_capacity is None
