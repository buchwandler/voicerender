from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voicerender.engines.base import SpeechRequest
from voicerender.engines.kokoro import (
    PyKokoroEngineAdapter,
    _target_from_capabilities,
    _version_supported,
)
from voicerender.engines.selection import EngineRequest
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
                audio=np.ones(24, dtype=np.float32),
                sample_rate=24000,
                diagnostics=(),
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
