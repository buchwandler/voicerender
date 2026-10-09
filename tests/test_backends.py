"""Mocked released-package public contracts: no models or network required."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from voicerender.engines.base import SynthesisRequest
from voicerender.engines.piper import PiperAdapter
from voicerender.engines.piper import discover_voices as piper_voices
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


def test_cli_render_with_fake_piper_runtime(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from voicerender.cli import main
    from voicerender.storage import wav_info

    class Voice:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def synthesize(self, request, *, config):
            return SimpleNamespace(
                audio=np.full(96, 0.1, dtype=np.float32), sample_rate=16000, warnings=()
            )

        def close(self):
            return None

    fake = SimpleNamespace(
        PiperVoice=Voice,
        SynthesisConfig=lambda **kwargs: kwargs,
        SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        VoiceLevelConfig=lambda mode: mode,
    )
    monkeypatch.setitem(sys.modules, "pipersynth", fake)
    plan = tmp_path / "plan.toml"
    plan.write_text(
        'schema = "utterplan.voiceplan.v1"\n[[segments]]\nid = "s1"\ntext = "Hello"\n'
    )
    profile = tmp_path / "profile.toml"
    profile.write_text(
        'schema = "voicerender.voice-profile.v1"\nname = "simple"\n[roles.narrator]\nengine = "piper"\ntarget_id = "en_US-amy-medium"\n'
    )
    out = tmp_path / "renders"
    args = [
        "render",
        str(plan),
        "--profile",
        str(profile),
        "--out-dir",
        str(out),
        "--take",
        "1",
        "--json",
    ]
    assert main(args) == 0
    manifest = __import__("json").loads((out / "manifest.json").read_text())
    assert manifest["complete"]
    assert wav_info(out / manifest["segments"][0]["path"])["frames"] == 96
    assert main(args) == 0
    assert main(["inspect", str(out / "manifest.json")]) == 0
