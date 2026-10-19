from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voicerender.engines.base import EngineCapabilities
from voicerender.engines.catalog import SynthesisTarget
from voicerender.errors import EngineSelectionError, ProfileError
from voicerender.voice_clone import VoiceCloneRequest, clone_voice_profile


def _target(*, reference: bool = True) -> SynthesisTarget:
    return SynthesisTarget(
        engine="fake",
        id="clone-model",
        display_name="Clone model",
        languages=("en-US",),
        capabilities=frozenset({"reference_voice"}) if reference else frozenset(),
        metadata={},
    )


def test_clone_rejects_target_without_reference_voice_support(
    monkeypatch, tmp_path: Path
) -> None:
    class Adapter:
        id = "fake"

        def capabilities(self):
            return EngineCapabilities(id="fake", supports_reference_voice=False)

    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"wav reference")
    monkeypatch.setattr("voicerender.voice_clone.get_adapter", lambda engine: Adapter())
    monkeypatch.setattr(
        "voicerender.voice_clone.discover_targets",
        lambda *a, **k: (_target(reference=False),),
    )

    with pytest.raises(EngineSelectionError) as error:
        clone_voice_profile(
            VoiceCloneRequest("fake", "clone-model", "en-US", reference_wav=reference)
        )

    assert error.value.code == "voice_clone.reference_unsupported"


def test_clone_delegates_transcript_validation_and_never_synthesizes(
    monkeypatch, tmp_path: Path
) -> None:
    class Adapter:
        id = "fake"

        def capabilities(self):
            return EngineCapabilities(
                id="fake",
                supports_reference_voice=True,
                option_names=frozenset({"reference_text", "temperature"}),
            )

    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"wav reference")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("voicerender.voice_clone.get_adapter", lambda engine: Adapter())
    monkeypatch.setattr(
        "voicerender.voice_clone.discover_targets", lambda *a, **k: (_target(),)
    )

    def create_profile(**kwargs):
        calls.append(kwargs)
        if "reference_text" not in kwargs["options"]:
            raise EngineSelectionError(
                "Exact reference transcript required",
                engine="fake",
                code="fake.reference_text_required",
            )
        return kwargs

    monkeypatch.setattr("voicerender.voice_clone.create_render_profile", create_profile)
    monkeypatch.setattr(
        "voicerender.engines.registry.open_session",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("clone must not synthesize")
        ),
    )

    with pytest.raises(EngineSelectionError) as error:
        clone_voice_profile(
            VoiceCloneRequest("fake", "clone-model", "en-US", reference_wav=reference)
        )
    assert error.value.code == "fake.reference_text_required"

    result = clone_voice_profile(
        VoiceCloneRequest(
            "fake",
            "clone-model",
            "en-US",
            reference_wav=reference,
            transcript="Exact transcript.",
            options={"temperature": 0.7},
        ),
        role="narrator",
    )

    assert result["options"]["reference_text"] == "Exact transcript."
    assert result["options"]["temperature"] == 0.7
    assert result["role"] == "narrator"
    assert result["voice_source"]["kind"] == "reference"
    assert len(calls) == 2


def test_clone_loads_pinned_source_sidecar_and_uses_metadata_transcript(
    monkeypatch, tmp_path: Path
) -> None:
    from voicerender.voice_source_codec import save_voice_source
    from voicerender.voice_sources import local_reference_source

    class Adapter:
        id = "fake"

        def capabilities(self):
            return EngineCapabilities(
                id="fake",
                supports_reference_voice=True,
                option_names=frozenset({"transcript"}),
            )

    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"pinned")
    source: dict[str, Any] = dict(local_reference_source(reference))
    source["metadata"] = {"transcript": "From the source."}
    sidecar = tmp_path / "speaker.toml"
    save_voice_source(source, sidecar)
    monkeypatch.setattr("voicerender.voice_clone.get_adapter", lambda engine: Adapter())
    monkeypatch.setattr(
        "voicerender.voice_clone.discover_targets", lambda *a, **k: (_target(),)
    )
    monkeypatch.setattr(
        "voicerender.voice_clone.create_render_profile", lambda **kwargs: kwargs
    )

    result = clone_voice_profile(
        VoiceCloneRequest("fake", "clone-model", "en-US", voice_source=sidecar)
    )

    assert result["options"]["transcript"] == "From the source."
    assert result["voice_source"]["sha256"] == source["sha256"]


def test_clone_uses_kokoro_adapter_to_bind_akinvox_transcript(
    monkeypatch, tmp_path: Path
) -> None:
    from voicerender.engines.kokoro import PyKokoroEngineAdapter

    target = SynthesisTarget(
        engine="kokoro",
        id="clone-model",
        display_name="AkinVox model",
        languages=("en-US",),
        capabilities=frozenset({"reference_voice", "voice_enrollment"}),
        metadata={
            "voice_mode": "reference",
            "supports_reference_enrollment": True,
            "supports_voice_enrollment": True,
        },
    )
    adapter = PyKokoroEngineAdapter()
    monkeypatch.setattr("voicerender.voice_clone.get_adapter", lambda engine: adapter)
    monkeypatch.setattr(
        "voicerender.voice_clone.discover_targets", lambda *args, **kwargs: (target,)
    )
    monkeypatch.setattr(
        "voicerender.voice_clone.validate_voice_source", lambda *a, **k: ()
    )
    monkeypatch.setattr(
        "voicerender.voice_clone.create_render_profile", lambda **kwargs: kwargs
    )
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"reference audio")

    profile = clone_voice_profile(
        VoiceCloneRequest(
            "kokoro",
            "clone-model",
            "en-US",
            reference_wav=reference,
            transcript="Exact AkinVox transcript.",
            options={"voice_enrollment": {"engine": "akinvox"}},
        )
    )

    assert profile["options"]["voice_enrollment"] == {
        "engine": "akinvox",
        "reference_text": "Exact AkinVox transcript.",
    }


def test_clone_rejects_conflicting_source_and_cli_transcripts(
    tmp_path: Path,
) -> None:
    from voicerender.voice_sources import local_reference_source

    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"reference audio")
    source: dict[str, Any] = dict(local_reference_source(reference))
    source["metadata"] = {"transcript": "Source transcript."}

    with pytest.raises(ProfileError, match="Conflicting transcript"):
        clone_voice_profile(
            VoiceCloneRequest(
                "fake",
                "clone-model",
                "en-US",
                voice_source=source,
                transcript="Different CLI transcript.",
            )
        )
