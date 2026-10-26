from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from voicerender.engines.base import (
    PCM_RANGE_POLICY,
    EngineSelection,
    RenderedSpeech,
    SpeechRequest,
    validate_rendered_speech,
)
from voicerender.engines.kokoro import (
    PYKOKORO_REQUIRED,
    PyKokoroEngineAdapter,
    _normalize_short_sentence,
    _normalize_voice_level,
    _pykokoro_discovery,
    _target_from_capabilities,
    _translate_pykokoro_error,
    _version_supported,
)
from voicerender.engines.selection import EngineRequest
from voicerender.errors import EngineBackendError, EngineSelectionError
from voicerender.models import ModelDiscoveryError
from voicerender.voice_sources import local_reference_source


def test_pykokoro_minimum_version_is_0106() -> None:
    assert not _version_supported("0.10.5")
    assert _version_supported("0.10.6")
    assert _version_supported("0.10.10")
    assert not _version_supported("0.11.0")


def test_pykokoro_incompatibility_diagnostic_matches_package_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
    declared_requirement = metadata["project"]["optional-dependencies"]["kokoro"][0]
    assert declared_requirement == f"pykokoro{PYKOKORO_REQUIRED}"

    module = ModuleType("pykokoro")
    module.__version__ = "0.10.5"
    module.__file__ = "/fake/pykokoro/__init__.py"
    monkeypatch.setitem(sys.modules, "pykokoro", module)

    with pytest.raises(ModelDiscoveryError) as caught:
        _pykokoro_discovery()

    assert caught.value.code == "pykokoro.version_unsupported"
    assert f"required: {PYKOKORO_REQUIRED}" in str(caught.value)


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {"mode": "off"}),
        ("off", {"mode": "off"}),
        ("calibrated", {"mode": "calibrated"}),
        ({"mode": "off"}, {"mode": "off"}),
        ({"mode": "off", "gain_db": None}, {"mode": "off"}),
        ({"mode": "off", "gain_db": 0}, {"mode": "off", "gain_db": 0.0}),
        (
            {"mode": "calibrated", "gain_db": 1.5},
            {"mode": "calibrated", "gain_db": 1.5},
        ),
        ({"mode": "off", "gain_db": -1.5}, {"mode": "off", "gain_db": -1.5}),
    ],
)
def test_voice_level_options_normalize_gain_and_preserve_shorthand(
    value: Any, expected: dict[str, Any]
) -> None:
    assert _normalize_voice_level(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "unknown",
        {"mode": "unknown"},
        {"gain_db": True},
        {"gain_db": float("nan")},
        {"gain_db": float("inf")},
        {"gain_db": float("-inf")},
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


def test_kokoro_forwards_persisted_g2lex_tables_as_public_file_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, state = _install(monkeypatch, _model())
    module = sys.modules["pykokoro"]

    class FakeG2LexFileRef:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    module.G2LexFileRef = FakeG2LexFileRef
    selection = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={
            "lexicons": ["lexhint"],
            "g2lex_files": [
                {
                    "path": "/tmp/custom.g2lex",
                    "sha256": "sha256:" + "a" * 64,
                    "logical_sha256": "sha256:" + "b" * 64,
                    "language": "en-us",
                    "phoneme_encoding": "ipa",
                }
            ],
        },
    )

    with adapter.open(selection):
        pass

    refs = state["config"].tokenizer_config.g2lex_files
    assert len(refs) == 1
    assert refs[0].path == "/tmp/custom.g2lex"
    assert refs[0].sha256 == "sha256:" + "a" * 64
    assert refs[0].logical_sha256 == "sha256:" + "b" * 64


def test_validate_selection_rejects_unusable_local_g2lex_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _ = _install(monkeypatch, _model())
    valid_path = tmp_path / "valid.g2lex"
    valid_path.write_bytes(b"valid source")
    malformed_path = tmp_path / "malformed.g2lex"
    malformed_path.write_bytes(b"broken source")
    missing_path = tmp_path / "missing.g2lex"
    actual_sha = "sha256:" + hashlib.sha256(valid_path.read_bytes()).hexdigest()

    def inspect(path: str) -> Any:
        if Path(path) == malformed_path.resolve():
            raise ValueError("invalid G2Lex manifest")
        return SimpleNamespace(
            path=path,
            sha256=actual_sha,
            logical_sha256="sha256:" + "b" * 64,
            metadata={"language": "en-us", "phoneme_encoding": "ipa"},
        )

    monkeypatch.setattr(adapter, "inspect_g2lex_file", inspect)

    def diagnostics_for(sources: list[dict[str, Any]]) -> set[str]:
        selection = EngineSelection(
            engine="kokoro",
            target_id="kokoro-test",
            language="en-US",
            options={"g2lex_files": sources},
        )
        return {item.code for item in adapter.validate_selection(selection)}

    assert diagnostics_for([{"path": str(missing_path)}]) == {
        "pykokoro.g2lex_file_missing"
    }
    assert diagnostics_for([{"path": str(malformed_path)}]) == {
        "pykokoro.g2lex_file_invalid"
    }
    assert diagnostics_for(
        [{"path": str(valid_path), "sha256": "sha256:" + "a" * 64}]
    ) == {"pykokoro.g2lex_sha256_mismatch"}
    assert "pykokoro.g2lex_file_duplicate" in diagnostics_for(
        [{"path": str(valid_path)}, {"path": str(valid_path)}]
    )

    def incompatible_inspection(path: str) -> Any:
        return SimpleNamespace(
            path=path,
            sha256=actual_sha,
            logical_sha256=None,
            metadata={"language": "fr", "phoneme_encoding": "x-unknown"},
        )

    monkeypatch.setattr(adapter, "inspect_g2lex_file", incompatible_inspection)
    incompatible = diagnostics_for([{"path": str(valid_path)}])
    assert "pykokoro.g2lex_language_mismatch" in incompatible
    assert "pykokoro.g2lex_encoding_unsupported" in incompatible


def test_validate_selection_checks_named_sources_and_installed_only_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _install(monkeypatch, _model())
    entry = SimpleNamespace(
        selector="gold",
        default=True,
        installed=False,
        model_support="unsupported",
        phoneme_encoding="x-unknown",
    )
    monkeypatch.setattr(
        adapter, "discover_lexicons", lambda **_kwargs: ((entry,), None)
    )

    def codes(options: dict[str, Any]) -> set[str]:
        selection = EngineSelection(
            engine="kokoro",
            target_id="kokoro-test",
            language="en-US",
            options=options,
        )
        return {item.code for item in adapter.validate_selection(selection)}

    unavailable = codes({"lexicons": ["missing"]})
    assert "pykokoro.lexicon_unavailable" in unavailable
    unsupported = codes({"lexicons": ["gold"]})
    assert "pykokoro.lexicon_model_unsupported" in unsupported
    assert "pykokoro.lexicon_encoding_unsupported" in unsupported
    assert "pykokoro.lexicon_not_installed" in codes(
        {"lexicons": ["gold"], "lexicon_data_policy": "installed-only"}
    )
    assert "pykokoro.lexicon_duplicate" in codes({"lexicons": ["gold", "gold"]})


def test_local_g2lex_content_hash_changes_synthesis_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _ = _install(monkeypatch, _model())
    path = tmp_path / "mutable.g2lex"
    path.write_bytes(b"first")

    def inspect(source_path: str) -> Any:
        resolved = Path(source_path).resolve()
        return SimpleNamespace(
            path=str(resolved),
            sha256="sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest(),
            logical_sha256=None,
            metadata={"language": "en-us", "phoneme_encoding": "ipa"},
        )

    monkeypatch.setattr(adapter, "inspect_g2lex_file", inspect)
    selection = EngineSelection(
        engine="kokoro",
        target_id="kokoro-test",
        language="en-US",
        options={"g2lex_files": [{"path": str(path)}]},
    )
    first = adapter.canonical_synthesis_identity(selection)
    path.write_bytes(b"second")
    second = adapter.canonical_synthesis_identity(selection)

    assert first["local_pronunciation_sources"] != second["local_pronunciation_sources"]


def test_pronunciation_stack_reports_effective_sources_and_local_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = PyKokoroEngineAdapter()
    path = tmp_path / "editorial.g2lex"
    path.write_bytes(b"editorial source")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    inspection = SimpleNamespace(
        path=str(path.resolve()),
        sha256=digest,
        logical_sha256="sha256:" + "b" * 64,
        metadata={"language": "en-us", "phoneme_encoding": "ipa"},
    )
    entry = SimpleNamespace(
        selector="gold",
        default=True,
        installed=True,
        model_support="supported",
        phoneme_encoding="ipa",
    )
    monkeypatch.setattr(adapter, "inspect_g2lex_file", lambda _path: inspection)
    monkeypatch.setattr(
        adapter, "discover_lexicons", lambda **_kwargs: ((entry,), None)
    )

    stack = adapter.pronunciation_stack(
        language="en-US",
        target_id="kokoro-test",
        options={
            "lexicons": ["gold"],
            "g2lex_files": [{"path": str(path), "sha256": digest}],
        },
    )

    assert stack["lexicon_catalog_available"] is True
    assert stack["local_g2lex"][0]["status"] == "valid"
    assert stack["named_lexicons"][0]["available"] is True
    assert stack["precedence"] == [
        "local G2Lex file 1",
        "contextual heteronym rules",
        "gold",
        "espeak",
    ]


@pytest.mark.parametrize(
    ("seed_options", "expected_seed"),
    [({}, None), ({"random_seed": 17}, 17)],
)
def test_akinvox_reference_generation_preserves_optional_seed(
    monkeypatch: pytest.MonkeyPatch,
    seed_options: dict[str, int],
    expected_seed: int | None,
) -> None:
    from voicerender.engines import kokoro

    adapter, state = _install(
        monkeypatch, _model(voice_mode="reference", reference_enrollment=True)
    )
    reference_audio = Path("/cache/reference.wav")
    monkeypatch.setattr(
        kokoro,
        "resolve_reference_audio",
        lambda *args, **kwargs: SimpleNamespace(path=reference_audio),
    )
    options = {
        "voice_source": _managed_source(),
        "voice_enrollment": {
            "engine": "akinvox",
            "reference_text": "Reference transcript",
        },
        **seed_options,
    }
    selection = EngineSelection(
        engine="kokoro",
        target_id="en-akinvox-cloning-v1",
        language="en-US",
        voice="akinvox",
        options=options,
    )

    with adapter.open(selection) as session:
        session.synthesize(
            SpeechRequest(
                id="reference",
                text="Reference-conditioned speech",
                language="en-US",
            )
        )

    config = state["config"]
    assert config.model_variant == "en-akinvox-cloning-v1"
    assert config.voice is None
    assert config.generation.random_seed == expected_seed
    assert len(state["enrollments"]) == 1
    enrollment = state["enrollments"][0]
    assert enrollment["engine"] == "akinvox"
    assert enrollment["reference_text"] == "Reference transcript"
    assert state["syntheses"][0].voice is enrollment["voice"]


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


def test_akinvox_binding_rejects_missing_transcript_before_profile_persistence() -> (
    None
):
    adapter = PyKokoroEngineAdapter()
    with pytest.raises(
        EngineSelectionError, match="requires non-empty reference_text"
    ) as caught:
        adapter.bind_reference_voice(
            target=_model(voice_mode="reference", reference_enrollment=True),
            language="en-US",
            voice_source=_managed_source(),
            transcript=None,
            options={"voice_enrollment": {"engine": "auto"}},
        )
    assert caught.value.code == "pykokoro.reference_text_required"


def test_kokoro_reference_binding_uses_transcript_and_preserves_advanced_options() -> (
    None
):
    adapter = PyKokoroEngineAdapter()
    akinvox_target = _model(voice_mode="reference", reference_enrollment=True)
    bound = adapter.bind_reference_voice(
        target=akinvox_target,
        language="en-US",
        voice_source=_managed_source(),
        transcript="Catalog or fallback transcript.",
        options={"voice_enrollment": {"engine": "auto"}, "speed": 1.0},
    )
    assert bound["voice_enrollment"] == {
        "engine": "akinvox",
        "reference_text": "Catalog or fallback transcript.",
    }
    assert bound["speed"] == 1.0

    inno_target = _model(enrollers=({"id": "inno-v0.2", "transcript_required": False},))
    inno_bound = adapter.bind_reference_voice(
        target=inno_target,
        language="en-US",
        voice_source=_managed_source(),
        transcript=None,
        options={"voice_enrollment": {"engine": "inno", "fmax": 19000}},
    )
    assert inno_bound["voice_enrollment"] == {"engine": "inno", "fmax": 19000.0}


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


class BackendError(Exception):
    stage = "inference"
    native_error_message = "bad shape"


def _translation_selection(*, options: dict[str, Any] | None = None) -> EngineSelection:
    return EngineSelection(
        engine="kokoro",
        target_id="en-akinvox-cloning-v1",
        language="en-US",
        voice="akinvox",
        speaker=1,
        options=options or {},
    )


@pytest.mark.parametrize(
    ("seed_options", "expected_configured", "expected_seed"),
    [({}, False, None), ({"random_seed": 17}, True, 17)],
)
def test_translate_pykokoro_error_preserves_native_diagnostics_and_cause(
    monkeypatch: pytest.MonkeyPatch,
    seed_options: dict[str, int],
    expected_configured: bool,
    expected_seed: int | None,
) -> None:
    from voicerender.engines import kokoro

    versions = {
        "pykokoro": "0.10.6",
        "onnxvoice": "0.2.12",
        "onnxruntime": "1.20.1",
    }
    monkeypatch.setattr(kokoro, "distribution_version", lambda name: versions.get(name))
    module = ModuleType("pykokoro")
    module.BackendError = BackendError
    native_diagnostic = {
        "schema": "pykokoro.error-diagnostic.v1",
        "type": "BackendError",
        "stage": "inference",
        "cause": {"type": "RuntimeContractError", "message": "bad shape"},
    }
    module.error_diagnostic = lambda _error: native_diagnostic
    monkeypatch.setitem(sys.modules, "pykokoro", module)

    native = BackendError(
        "Kokoro model inference failed: RuntimeContractError: bad shape"
    )
    native.__cause__ = RuntimeError("provider execution failed")
    selection = _translation_selection(
        options={
            "voice_source": {
                "kind": "managed_reference",
                "resolver": "voicefolio",
                "value": "catalog:private-source",
                "access_token": "private-token",
                "path": "/private/reference.wav",
            },
            "voice_enrollment": {
                "engine": "akinvox",
                "reference_text": "private transcript",
            },
            **seed_options,
        }
    )
    translated = _translate_pykokoro_error(native, selection)

    assert isinstance(translated, EngineBackendError)
    assert translated.code == "synthesis.backend_error"
    assert str(translated) == str(native)
    assert translated.native_error_type == "BackendError"
    assert translated.details["native_stage"] == "inference"
    assert translated.details["native_error_message"] == "bad shape"
    assert translated.details["native_diagnostic"] == native_diagnostic
    assert translated.details["pykokoro_version"] == "0.10.6"
    assert translated.details["onnxvoice_version"] == "0.2.12"
    assert translated.details["onnxruntime_version"] == "1.20.1"
    assert translated.details["python_version"]
    assert translated.details["platform_system"]
    assert translated.details["platform_machine"]
    assert translated.details["target_id"] == "en-akinvox-cloning-v1"
    assert translated.details["reference_conditioning"] is True
    assert translated.details["random_seed_configured"] is expected_configured
    assert translated.details["random_seed"] == expected_seed
    assert translated.details["backend_boundary"] == "pykokoro->onnxvoice"
    serialized = repr(translated.details)
    for secret in (
        "catalog:private-source",
        "private-token",
        "/private/reference.wav",
        "private transcript",
    ):
        assert secret not in serialized
    with pytest.raises(EngineBackendError) as caught:
        raise translated from native
    assert caught.value.__cause__ is native
    assert caught.value.__cause__.__cause__.__str__() == "provider execution failed"


def test_translate_pykokoro_reference_enrollment_error_keeps_specific_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pykokoro")
    module.BackendError = BackendError
    module.error_diagnostic = lambda _error: {"stage": "reference_enrollment"}
    monkeypatch.setitem(sys.modules, "pykokoro", module)

    native = BackendError("Kokoro reference enrollment failed")
    native.stage = "reference_enrollment"
    translated = _translate_pykokoro_error(
        native,
        _translation_selection(),
        failure_code="pykokoro.reference_enrollment_failed",
    )

    assert translated.code == "pykokoro.reference_enrollment_failed"
    assert translated.details["native_stage"] == "reference_enrollment"
    assert translated.details["native_diagnostic"]["stage"] == "reference_enrollment"


def test_translate_pykokoro_error_falls_back_without_public_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pykokoro")
    module.BackendError = BackendError
    monkeypatch.setitem(sys.modules, "pykokoro", module)

    native = BackendError("older PyKokoro backend message")
    translated = _translate_pykokoro_error(native, _translation_selection())

    assert str(translated) == "older PyKokoro backend message"
    assert translated.code == "synthesis.backend_error"
    assert translated.details["native_stage"] == "inference"
    assert "native_diagnostic" not in translated.details
