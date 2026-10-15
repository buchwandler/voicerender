"""Mocked released-package public contracts: no models or network required."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from voicerender.engines.base import (
    EngineSelection,
    PronunciationSpan,
    SpeechRequest,
    SpeechToken,
    SynthesisRequest,
)
from voicerender.engines.piper import PiperAdapter
from voicerender.engines.piper import discover_voices as piper_voices
from voicerender.engines.pipersynth import PiperSynthEngineSession
from voicerender.engines.pocket import PocketAdapter
from voicerender.engines.pocket import discover_voices as pocket_voices
from voicerender.models import VoiceBinding


def test_piper_public_api(monkeypatch):
    events = []

    class FakePiperVoice:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            events.append(("load", args, kwargs))
            return cls()

        def synthesize(self, request, *, config):
            events.append(
                ("synthesize", request.text, request.speaker, config.length_scale)
            )
            return SimpleNamespace(
                audio=np.ones(80, dtype=np.float32) / 4, sample_rate=16000, warnings=()
            )

        def close(self):
            events.append(("close",))

    class SynthesisConfig:
        def __init__(self, **kwargs):
            self.length_scale = kwargs["length_scale"]

    class VoiceAssetManager:
        def __init__(self, offline):
            pass

        def list_voices(self, language):
            return [SimpleNamespace(id="model", name="Model", language_code="en-US")]

    fake = SimpleNamespace(
        PiperVoice=FakePiperVoice,
        SynthesisConfig=SynthesisConfig,
        SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        VoiceLevelConfig=lambda mode: SimpleNamespace(mode=mode),
        VoiceAssetManager=VoiceAssetManager,
    )
    monkeypatch.setitem(sys.modules, "pipersynth", fake)
    adapter = PiperAdapter()
    binding = VoiceBinding(
        engine="piper", target_id="model", speaker=1, options={"native_speed": 2.0}
    )
    with adapter.open(binding) as session:
        result = session.synthesize(request_id="1", text="hello", language="en-US")
        assert result.sample_rate == 16000
        typed_result = session.synthesize_request(
            SynthesisRequest("2", "typed", "en-US")
        )
        assert typed_result.sample_rate == 16000
    assert events[1] == ("synthesize", "hello", 1, 0.5)
    assert events[-1] == ("close",)
    assert piper_voices(language="en-US")[0]["target_id"] == "model"


def test_pocket_reference_audio_public_api(monkeypatch, tmp_path):
    events = []

    class Runtime:
        @classmethod
        def from_resolved(cls, bundle, offline):
            events.append(("runtime", bundle))
            return cls()

        def prepare_voice(self, voice_source):
            events.append(("prepare", str(voice_source)))
            return "prepared"

        def synthesize(self, request, *, voice, config, voice_level):
            events.append(("synthesize", request.text, voice))
            return SimpleNamespace(
                audio=np.ones(40, dtype=np.float32) / 4, sample_rate=24000, warnings=()
            )

        def close(self):
            events.append(("close",))

    class BundleManager:
        def __init__(self, offline):
            events.append(("bundle_manager", offline))

        def resolve_bundle(self, name, precision):
            return name

        def list_bundles(self, language):
            return [SimpleNamespace(id="english", predefined_voices=("alba", "alex"))]

    fake = SimpleNamespace(
        PocketRuntime=Runtime,
        BundleAssetManager=BundleManager,
        SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        GenerationConfig=lambda **kwargs: kwargs,
        VoiceLevelConfig=lambda mode: SimpleNamespace(mode=mode),
    )
    monkeypatch.setitem(sys.modules, "pocketsynth", fake)
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"fixture")
    binding = VoiceBinding(engine="pocket", target_id="english", reference_audio=ref)
    with PocketAdapter().open(binding) as session:
        result = session.synthesize(request_id="1", text="cloned", language="en-US")
        assert result.sample_rate == 24000
        typed_result = session.synthesize_request(
            SynthesisRequest("2", "typed", "en-US")
        )
        assert typed_result.sample_rate == 24000
    assert ("prepare", str(ref)) in events
    assert ("synthesize", "cloned", "prepared") in events
    assert events[-1] == ("close",)
    assert pocket_voices()[0]["voices"] == ["alba", "alex"]


def test_pipersynth_measure_and_synthesize_share_native_request(monkeypatch):
    events = []

    class FakeVoice:
        config = SimpleNamespace(
            espeak_voice="en-US", phoneme_type=SimpleNamespace(value="espeak")
        )

        def measure(self, native):
            events.append(("measure", native))
            return SimpleNamespace(
                fits=None,
                phoneme_count=42,
                max_phonemes=None,
                model_id="voice-model",
                metadata={
                    "frontend": "piper-g2p",
                    "capacity_kind": "variable",
                    "capacity_source": "onnx.input.shape",
                },
            )

        def synthesize(self, native, *, config):
            events.append(("synthesize", native, config))
            return SimpleNamespace(
                id=native.id,
                audio=np.ones(80, dtype=np.float32) / 4,
                sample_rate=16000,
                warnings=(),
                metadata={},
            )

    fake = SimpleNamespace(
        SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        LinguisticToken=lambda **kwargs: SimpleNamespace(**kwargs),
        PronunciationOverride=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "pipersynth", fake)

    request = SpeechRequest(
        id="request-7",
        text="hello world",
        language="en",
        speaker=3,
        tokens=(SpeechToken(0, 5, "hello", pos="NOUN", lemma="hello"),),
        pronunciation_overrides=(
            PronunciationSpan(
                0, 5, phonemes="həˈloʊ", language="en-US", alphabet="ipa"
            ),
        ),
    )
    voice = FakeVoice()
    session = PiperSynthEngineSession(
        voice,
        config=SimpleNamespace(length_scale=1.0),
        selection=EngineSelection(
            engine="piper", target_id="voice-model", language="en-US"
        ),
    )

    measured = session.measure(request)
    session.synthesize(request)

    measured_native = events[0][1]
    synthesized_native = events[1][1]
    assert vars(measured_native) == vars(synthesized_native)
    assert measured_native.id == request.id
    assert measured_native.text == request.text
    assert measured_native.language == "en-US"
    assert measured_native.speaker == 3
    assert measured_native.tokens == (
        SimpleNamespace(
            start=0,
            end=5,
            text="hello",
            pos="NOUN",
            tag=None,
            lemma="hello",
            language=None,
            morph=None,
        ),
    )
    assert measured_native.pronunciation_overrides == (
        SimpleNamespace(start=0, end=5, phonemes="həˈloʊ", language="en-US"),
    )
    assert measured.amount == 42
    assert measured.maximum is None
    assert measured.unit == "phoneme_ids"
    assert measured.source == "pipersynth.measure"
    assert measured.details == {
        "model_id": "voice-model",
        "native_metadata": {
            "frontend": "piper-g2p",
            "capacity_kind": "variable",
            "capacity_source": "onnx.input.shape",
        },
    }
