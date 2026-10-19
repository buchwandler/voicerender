from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voicerender.engines.base import (
    PCM_RANGE_POLICY,
    EngineSelection,
    RenderedSpeech,
    SpeechRequest,
    validate_rendered_speech,
)
from voicerender.engines.kokoro import (
    PyKokoroEngineAdapter,
    _normalize_short_sentence,
    _normalize_voice_level,
    _target_from_capabilities,
    _version_supported,
)
from voicerender.engines.selection import EngineRequest
from voicerender.errors import EngineBackendError, EngineSelectionError
from voicerender.voice_sources import local_reference_source


def test_pykokoro_minimum_version_is_0105() -> None:
    assert not _version_supported("0.10.4")
    assert _version_supported("0.10.5")
    assert _version_supported("0.10.10")
    assert not _version_supported("0.11.0")


def _model(
    *,
    voice_mode: str = "static",
    reference_enrollment: bool = False,
    enrollers: tuple[dict[str, Any], ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        id="kokoro-test",
        source="github",
        languages=("en", "fr"),
        voices=("af_heart",) if voice_mode == "static" else (),
        default_voice="af_heart" if voice_mode == "static" else "",
        qualities=("fp32",),
        sample_rate=24000,
        max_tokens=512,
        distribution_id="kokoro-test-r1",
        metadata={
            "voice_mode": voice_mode,
            "supports_reference_enrollment": reference_enrollment,
            "supports_voice_enrollment": reference_enrollment or bool(enrollers),
            "voice_enrollers": enrollers,
        },
    )


def _managed_source() -> dict[str, str]:
    return {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:demo",
        "sha256": "b" * 64,
        "source_revision": "voice-r4",
        "source_repository": "voice-donations",
        "source_path": "voices/demo.wav",
    }


def _install(
    monkeypatch: pytest.MonkeyPatch,
    model: SimpleNamespace,
) -> tuple[PyKokoroEngineAdapter, dict[str, Any]]:
    from voicerender import models

    monkeypatch.setattr(models, "get_model_info", lambda *args, **kwargs: (model, ()))
    state: dict[str, Any] = {
        "config": None,
        "audio": np.ones(24, dtype=np.float32),
        "diagnostics": (),
        "measurements": [],
        "enrollments": [],
        "syntheses": [],
        "closed": 0,
    }
    module = ModuleType("pykokoro")
    module.__version__ = "0.10.5"
    module.__file__ = "/fake/pykokoro/__init__.py"

    class FakeSynthesisRequest:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class FakeConfig:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class FakeInnoOptions:
        def __init__(self, *, fmax: float | None = None) -> None:
            self.fmax = fmax

    class FakeSynthesizer:
        def __init__(self, config: Any) -> None:
            state["config"] = config

        def prepare(self, request: Any) -> None:
            raise AssertionError("measurement must not prepare or enroll a voice")

        def measure_request(self, request: Any, *, maximum: int | None = None) -> Any:
            state["measurements"].append((request, maximum))
            return SimpleNamespace(amount=9, maximum=maximum)

        def enroll_voice(
            self,
            audio: str | Path,
            reference_text: str | None = None,
            *,
            engine: str,
            sample_rate: int | None = None,
            language: str,
            name: str | None = None,
            options: Any = None,
        ) -> Any:
            voice = SimpleNamespace(enrolled=len(state["enrollments"]) + 1)
            state["enrollments"].append(
                {
                    "audio": audio,
                    "reference_text": reference_text,
                    "engine": engine,
                    "sample_rate": sample_rate,
                    "language": language,
                    "name": name,
                    "options": options,
                    "voice": voice,
                }
            )
            return voice

        def synthesize(self, request: Any) -> Any:
            state["syntheses"].append(request)
            return SimpleNamespace(
                id=request.id,
                audio=state["audio"],
                sample_rate=24000,
                diagnostics=state["diagnostics"],
                word_timings=(),
                language=request.language,
                voice=request.voice,
                phonemes="t e s t",
                token_ids=(1, 2),
                trace=None,
                voice_level_applications=(),
            )

        def close(self) -> None:
            state["closed"] += 1

    module.SynthesisRequest = FakeSynthesisRequest
    module.GenerationConfig = FakeConfig
    module.TokenizerConfig = FakeConfig
    module.ShortSentenceConfig = FakeConfig
    module.SynthesisConfig = FakeConfig
    module.VoiceLevelConfig = FakeConfig
    module.InnoEnrollmentOptions = FakeInnoOptions
    module.KokoroSynthesizer = FakeSynthesizer
    monkeypatch.setitem(sys.modules, "pykokoro", module)
    return PyKokoroEngineAdapter(), state


def test_discovered_capabilities_preserve_reference_and_enroller_metadata() -> None:
    static = _target_from_capabilities(
        SimpleNamespace(
            model_id="static-v1",
            languages=("en",),
            voices=("af_heart",),
            voice_details=(),
            default_voice="af_heart",
            qualities=("fp32",),
            aliases=(),
            status="ready",
            runtime_available=True,
            sample_rate=24000,
            source="github",
            g2p_backend="espeak",
            frontend="kokoro",
            lexicons=(),
            experimental=False,
            redistribution_allowed=True,
            voice_mode="static",
            supports_reference_enrollment=False,
            supports_voice_enrollment=True,
            voice_enrollers=(
                SimpleNamespace(
                    id="inno-v0.2",
                    kind="zero-shot-tuner",
                    transcript_required=False,
                    min_seconds=2.0,
                    max_seconds=15.0,
                    recommended_seconds=8.0,
                    output_format="voice-pack",
                ),
            ),
        )
    )
    reference = _target_from_capabilities(
        SimpleNamespace(
            model_id="en-akinvox-cloning-v1",
            languages=("en",),
            voices=(),
            voice_details=(),
            default_voice=None,
            qualities=("fp32",),
            aliases=(),
            status="ready",
            runtime_available=True,
            sample_rate=24000,
            source="github",
            g2p_backend="espeak",
            frontend="kokoro",
            lexicons=(),
            experimental=False,
            redistribution_allowed=True,
            voice_mode="reference",
            supports_reference_enrollment=True,
            supports_voice_enrollment=True,
            voice_enrollers=(),
        )
    )

    assert "named_voices" in static.capabilities
    assert static.metadata["voice_enrollers"][0]["kind"] == "zero-shot-tuner"
    assert "reference_voice" in reference.capabilities
    assert "named_voices" not in reference.capabilities
    assert reference.metadata["voice_mode"] == "reference"


def test_inno_reference_is_lazily_enrolled_reused_and_measured_without_enrollment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _model(
        enrollers=(
            {
                "id": "inno-v0.2",
                "kind": "zero-shot-tuner",
                "transcript_required": False,
            },
        )
    )
    adapter, state = _install(monkeypatch, model)
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"pinned local reference audio")
    source = local_reference_source(audio)
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="kokoro",
            target_id="kokoro-test",
            language="en-US",
            options={
                "voice_source": source,
                "voice_enrollment": {"engine": "auto", "fmax": 19000},
            },
        )
    )

    assert diagnostics == ()
    assert selection.options["voice_enrollment"] == {
        "engine": "inno",
        "fmax": 19000.0,
    }
    assert adapter.validate_selection(selection) == ()
    identity_json = json.dumps(
        adapter.canonical_synthesis_identity(selection), sort_keys=True
    )
    assert str(audio) not in identity_json
    assert source["sha256"] in identity_json

    with adapter.open(selection) as session:
        measured = session.measure(
            SpeechRequest(id="measure", text="measure this", language="en-US")
        )
        assert measured.amount == 9
        assert measured.source == "pykokoro.measure_request"
        assert state["enrollments"] == []
        first = session.synthesize(
            SpeechRequest(id="one", text="First request", language="en-US")
        )
        second = session.synthesize(
            SpeechRequest(id="two", text="Second request", language="en-US")
        )

    assert len(state["enrollments"]) == 1
    enrollment = state["enrollments"][0]
    assert enrollment["audio"] == audio.resolve()
    assert enrollment["engine"] == "inno"
    assert enrollment["language"] == "en-US"
    assert enrollment["reference_text"] is None
    assert enrollment["options"].fmax == 19000.0
    assert state["syntheses"][0].voice is enrollment["voice"]
    assert state["syntheses"][1].voice is enrollment["voice"]
    assert first.metadata["reference_voice_identity"]["sha256"] == source["sha256"]
    assert second.id == "two"
    assert state["closed"] == 1


def test_akinvox_managed_source_is_not_fetched_in_preflight_and_identity_hashes_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicerender.engines import kokoro

    model = _model(voice_mode="reference", reference_enrollment=True)
    adapter, state = _install(monkeypatch, model)
    source = _managed_source()
    validation_calls: list[Any] = []
    resolution_calls: list[Any] = []
    resolved_path = Path("/cache/voicefolio/demo.wav")
    monkeypatch.setattr(
        kokoro,
        "validate_voice_source",
        lambda *args, **kwargs: validation_calls.append((args, kwargs)) or (),
    )

    def resolve_reference(*args: Any, **kwargs: Any) -> Any:
        resolution_calls.append((args, kwargs))
        return SimpleNamespace(path=resolved_path)

    monkeypatch.setattr(kokoro, "resolve_reference_audio", resolve_reference)
    transcript = "Exact reference transcript, including punctuation!"
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="kokoro",
            target_id="kokoro-test",
            language="en-US",
            options={
                "voice_source": source,
                "voice_enrollment": {
                    "engine": "auto",
                    "reference_text": transcript,
                },
            },
        )
    )

    assert diagnostics == ()
    assert selection.options["voice_enrollment"]["engine"] == "akinvox"
    assert adapter.validate_selection(selection) == ()
    assert len(validation_calls) == 1
    assert resolution_calls == []
    identity = adapter.canonical_synthesis_identity(selection)
    identity_json = json.dumps(identity, sort_keys=True)
    assert transcript not in identity_json
    assert hashlib.sha256(transcript.encode("utf-8")).hexdigest() in identity_json
    assert "voices/demo.wav" not in identity_json

    with adapter.open(selection) as session:
        measured = session.measure(
            SpeechRequest(id="measure", text="ready", language="en-US")
        )
        assert measured.amount == 9
        assert resolution_calls == []
        output = session.synthesize(
            SpeechRequest(id="speech", text="Speak this", language="en-US")
        )

    assert len(resolution_calls) == 1
    enrollment = state["enrollments"][0]
    assert enrollment["audio"] == resolved_path
    assert enrollment["engine"] == "akinvox"
    assert enrollment["language"] == "en-US"
    assert enrollment["reference_text"] == transcript
    assert enrollment["options"] is None
    assert state["syntheses"][0].voice is enrollment["voice"]
    assert output.metadata["reference_enrollment_identity"]


def test_akinvox_requires_english_and_nonempty_reference_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(voice_mode="reference", reference_enrollment=True)
    adapter, _ = _install(monkeypatch, model)
    source = _managed_source()
    from voicerender.engines import kokoro

    monkeypatch.setattr(kokoro, "validate_voice_source", lambda *args, **kwargs: ())

    selection, _ = adapter.resolve(
        EngineRequest(
            engine="kokoro",
            target_id="kokoro-test",
            language="fr",
            options={
                "voice_source": source,
                "voice_enrollment": {"engine": "akinvox", "reference_text": "texte"},
            },
        )
    )
    diagnostics = adapter.validate_selection(selection)
    assert [item.code for item in diagnostics] == [
        "pykokoro.reference_text_invalid_language"
    ]

    selection, _ = adapter.resolve(
        EngineRequest(
            engine="kokoro",
            target_id="kokoro-test",
            language="en",
            options={"voice_source": source, "voice_enrollment": {"engine": "akinvox"}},
        )
    )
    diagnostics = adapter.validate_selection(selection)
    assert [item.code for item in diagnostics] == ["pykokoro.reference_text_required"]


def _synthesize_pcm_audio(
    monkeypatch: pytest.MonkeyPatch,
    audio: Any,
    *,
    diagnostics: tuple[str, ...] = (),
) -> Any:
    adapter, state = _install(monkeypatch, _model())
    state["audio"] = audio
    state["diagnostics"] = diagnostics
    selection = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        voice="af_heart",
    )
    with adapter.open(selection) as session:
        return session.synthesize(
            SpeechRequest(id="pcm", text="PCM regression", language="en-US")
        )


def test_kokoro_pcm_range_preserves_in_range_audio_and_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _synthesize_pcm_audio(monkeypatch, np.array([0.25, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(exact.audio, np.array([0.25, 1.0], dtype=np.float32))
    assert exact.warnings == ()
    assert exact.metadata["pcm_range"] == {
        "policy": PCM_RANGE_POLICY,
        "peak_before_clip": 1.0,
        "clipped_samples": 0,
    }

    tolerant = _synthesize_pcm_audio(
        monkeypatch, np.array([0.25, 1.000005], dtype=np.float32)
    )
    assert float(np.max(np.abs(tolerant.audio))) == 1.0
    assert tolerant.metadata["pcm_range"]["peak_before_clip"] == pytest.approx(1.000005)
    assert "clamped 1 sample(s)" in tolerant.warnings[0]


def test_kokoro_observed_pcm_overshoot_is_clamped_with_diagnostics_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = np.asarray([0.0, -0.4, 1.008497], dtype=np.float32)
    rendered = _synthesize_pcm_audio(
        monkeypatch, audio, diagnostics=("PyKokoro diagnostic",)
    )

    assert float(np.max(np.abs(rendered.audio))) == 1.0
    assert rendered.audio[-1] == 1.0
    assert rendered.metadata["pcm_range"] == {
        "policy": PCM_RANGE_POLICY,
        "peak_before_clip": pytest.approx(1.008497),
        "clipped_samples": 1,
    }
    assert rendered.warnings[0] == "PyKokoro diagnostic"
    assert "PyKokoro returned audio above full scale" in rendered.warnings[1]
    assert "clamped 1 sample(s)" in rendered.warnings[1]
    assert "1.008497" in rendered.warnings[1]


def test_kokoro_pcm_clamps_negative_and_multiple_overshoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _synthesize_pcm_audio(
        monkeypatch, np.array([-1.2, 0.0, 1.008497, -1.1], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        rendered.audio, np.array([-1.0, 0.0, 1.0, -1.0], dtype=np.float32)
    )
    assert rendered.metadata["pcm_range"]["clipped_samples"] == 3
    assert "clamped 3 sample(s)" in rendered.warnings[0]


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf])
def test_kokoro_pcm_does_not_sanitize_nonfinite_audio(
    monkeypatch: pytest.MonkeyPatch, bad_sample: float
) -> None:
    with pytest.raises(EngineBackendError, match="finite"):
        _synthesize_pcm_audio(monkeypatch, np.array([0.0, bad_sample]))


def test_kokoro_pcm_does_not_hide_empty_or_multidimensional_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(EngineBackendError, match="non-empty"):
        _synthesize_pcm_audio(monkeypatch, np.array([], dtype=np.float32))
    with pytest.raises(EngineBackendError, match="mono"):
        _synthesize_pcm_audio(monkeypatch, np.ones((2, 2), dtype=np.float32))


def test_kokoro_pcm_does_not_hide_nonconvertible_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(EngineBackendError, match="audio must convert to float32"):
        _synthesize_pcm_audio(monkeypatch, object())


def test_common_validator_keeps_full_scale_contract_strict() -> None:
    request = SpeechRequest(id="strict", text="strict", language="en-US")
    result = RenderedSpeech(
        id="strict",
        audio=np.array([1.008497], dtype=np.float32),
        sample_rate=24000,
    )
    with pytest.raises(EngineBackendError, match="exceeds PCM full-scale range"):
        validate_rendered_speech(request, result, engine="kokoro")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "off",
            {
                "enabled": False,
                "resolve_mode": "randomized-phrase",
                "min_phoneme_length": 30,
                "phoneme_pretext": "—",
                "phrase_fallback_tries": 1,
            },
        ),
        (
            "phrase",
            {
                "enabled": True,
                "resolve_mode": "phrase",
                "min_phoneme_length": 30,
                "phoneme_pretext": "—",
                "phrase_fallback_tries": 1,
            },
        ),
        (
            {
                "enabled": True,
                "resolve_mode": "wrap",
                "min_phoneme_length": 45,
                "phoneme_pretext": "—",
                "phrase_fallback_tries": 2,
            },
            {
                "enabled": True,
                "resolve_mode": "wrap",
                "min_phoneme_length": 45,
                "phoneme_pretext": "—",
                "phrase_fallback_tries": 2,
            },
        ),
    ],
)
def test_short_sentence_options_normalize_shorthand_and_tables(
    value: Any, expected: dict[str, Any]
) -> None:
    assert _normalize_short_sentence(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        "invalid-mode",
        {"resolve_mode": "off"},
        {"resolve_mode": "typo"},
        {"min_phoneme_length": -1},
        {"min_phoneme_length": True},
        {"phrase_fallback_tries": -1},
        {"phrase_fallback_tries": False},
        {"enabled": 1},
        {"phoneme_pretext": ""},
        {"unknown": 1},
    ],
)
def test_short_sentence_options_reject_invalid_values(value: Any) -> None:
    from voicerender.errors import InvalidEngineOptionError

    with pytest.raises(InvalidEngineOptionError, match="short_sentence"):
        _normalize_short_sentence(value)


def test_voice_level_options_normalize_gain_and_preserve_shorthand() -> None:
    assert _normalize_voice_level("off") == {"mode": "off", "gain_db": None}
    assert _normalize_voice_level("calibrated") == {
        "mode": "calibrated",
        "gain_db": None,
    }
    assert _normalize_voice_level({"mode": "off", "gain_db": 1.5}) == {
        "mode": "off",
        "gain_db": 1.5,
    }


@pytest.mark.parametrize(
    "value",
    [
        "unknown",
        {"mode": "unknown"},
        {"gain_db": True},
        {"gain_db": float("nan")},
        {"gain_db": float("inf")},
        {"unknown": 1},
    ],
)
def test_voice_level_options_reject_invalid_values(value: Any) -> None:
    from voicerender.errors import InvalidEngineOptionError

    with pytest.raises(InvalidEngineOptionError, match="voice_level"):
        _normalize_voice_level(value)


def test_kokoro_shorthand_and_equivalent_tables_share_synthesis_identity() -> None:
    adapter = PyKokoroEngineAdapter()
    shorthand = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={"short_sentence": "phrase", "voice_level": "calibrated"},
    )
    tables = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={
            "short_sentence": {"enabled": True, "resolve_mode": "phrase"},
            "voice_level": {"mode": "calibrated"},
        },
    )
    assert (
        adapter.canonical_synthesis_identity(shorthand)["options"]
        == adapter.canonical_synthesis_identity(tables)["options"]
    )


def test_kokoro_opens_with_structured_short_sentence_and_voice_level_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, state = _install(monkeypatch, _model())
    selection = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={
            "short_sentence": {
                "resolve_mode": "randomized-phrase",
                "min_phoneme_length": 45,
                "phoneme_pretext": "—",
                "phrase_fallback_tries": 2,
            },
            "voice_level": {"mode": "off", "gain_db": -1.5},
        },
    )
    with adapter.open(selection) as session:
        config = state["config"]
        rendered = session.synthesize(
            SpeechRequest(id="structured", text="Configured speech", language="en-US")
        )
    assert config.short_sentence_config.enabled is True
    assert config.short_sentence_config.resolve_mode == "randomized-phrase"
    assert config.short_sentence_config.min_phoneme_length == 45
    assert config.short_sentence_config.phoneme_pretext == "—"
    assert config.short_sentence_config.phrase_fallback_tries == 2
    assert config.voice_level.mode == "off"
    assert config.voice_level.gain_db == -1.5
    assert rendered.metadata["voice_level"]["mode"] == "off"


def test_kokoro_invalid_option_is_reported_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _install(monkeypatch, _model())
    selection, _ = adapter.resolve(
        EngineRequest(
            engine="kokoro",
            target_id="kokoro-test",
            language="en-US",
            options={"short_sentence": {"resolve_mode": "typo"}},
        )
    )
    diagnostics = adapter.validate_selection(selection)
    assert any(item.code == "pykokoro.invalid_options" for item in diagnostics)
    with pytest.raises(Exception, match="short_sentence.resolve_mode"):
        adapter.open(selection)


def test_voice_level_gain_changes_synthesis_identity() -> None:
    adapter = PyKokoroEngineAdapter()
    first = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={"voice_level": {"mode": "off", "gain_db": 0}},
    )
    second = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={"voice_level": {"mode": "off", "gain_db": 2}},
    )
    assert (
        adapter.canonical_synthesis_identity(first)["options"]
        != adapter.canonical_synthesis_identity(second)["options"]
    )


def test_akinvox_rejects_non_unit_speed_during_selection_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(voice_mode="reference", reference_enrollment=True)
    adapter, _ = _install(monkeypatch, model)
    selection, _ = adapter.resolve(
        EngineRequest(
            engine="kokoro",
            target_id="kokoro-test",
            language="en-US",
            options={
                "voice_source": _managed_source(),
                "voice_enrollment": {
                    "engine": "akinvox",
                    "reference_text": "Exact transcript.",
                },
                "speed": 1.2,
            },
        )
    )

    diagnostics = adapter.validate_selection(selection)

    speed_diagnostic = next(
        item
        for item in diagnostics
        if item.code == "pykokoro.reference_speed_unsupported"
    )
    assert speed_diagnostic.field == "render.options.speed"


def test_akinvox_binding_rejects_conflicting_explicit_transcript() -> None:
    model = _model(voice_mode="reference", reference_enrollment=True)
    adapter = PyKokoroEngineAdapter()

    with pytest.raises(EngineSelectionError, match="Conflicting transcript") as caught:
        adapter.bind_reference_voice(
            target=model,
            language="en-US",
            voice_source={},
            transcript="Clone transcript.",
            options={
                "voice_enrollment": {
                    "engine": "akinvox",
                    "reference_text": "Explicit nested transcript.",
                }
            },
        )
    assert caught.value.code == "pykokoro.reference_text_conflict"


def test_inno_binding_rejects_clone_transcript() -> None:
    model = _model(enrollers=({"id": "inno-v0.2", "transcript_required": False},))
    adapter = PyKokoroEngineAdapter()

    with pytest.raises(
        EngineSelectionError, match="does not accept reference_text"
    ) as caught:
        adapter.bind_reference_voice(
            target=model,
            language="en-US",
            voice_source={},
            transcript="Transcript is not accepted by Inno.",
            options={"voice_enrollment": {"engine": "auto"}},
        )
    assert caught.value.code == "pykokoro.reference_text_unsupported"


def test_real_pykokoro_public_api_contract_smoke_without_runtime_load() -> None:
    pykokoro = pytest.importorskip("pykokoro")
    adapter = PyKokoroEngineAdapter()

    probe = adapter.probe_api()
    contract = pykokoro.request_api_contract()

    assert pykokoro.REQUEST_API_VERSION == 1
    assert probe.compatible, probe
    assert probe.api_version == 1
    assert contract.supports_request_measurement is True
    assert contract.supports_reference_voice is True
    assert contract.supports_voice_enrollment is True
