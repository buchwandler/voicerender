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
    monkeypatch.setattr(
        "voicerender.renderprofile.get_adapter", lambda engine: Adapter()
    )
    monkeypatch.setattr(
        "voicerender.renderprofile.discover_targets",
        lambda *a, **k: (_target(reference=False),),
    )

    with pytest.raises(EngineSelectionError) as error:
        clone_voice_profile(
            VoiceCloneRequest("fake", "clone-model", "en-US", reference_wav=reference)
        )

    assert error.value.code == "voice_clone.reference_unsupported"


def test_clone_delegates_binding_to_shared_profile_flow(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"wav reference")
    calls: list[dict[str, Any]] = []

    def create_profile(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr("voicerender.voice_clone.create_render_profile", create_profile)
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

    assert result["transcript"] == "Exact transcript."
    assert result["options"] == {"temperature": 0.7}
    assert result["role"] == "narrator"
    assert result["voice_source"]["kind"] == "reference"
    assert len(calls) == 1


def test_clone_preserves_sidecar_metadata_for_shared_binding(
    monkeypatch, tmp_path: Path
) -> None:
    from voicerender.voice_source_codec import save_voice_source
    from voicerender.voice_sources import local_reference_source

    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"pinned")
    source: dict[str, Any] = dict(local_reference_source(reference))
    source["metadata"] = {"transcript": "From the source."}
    sidecar = tmp_path / "speaker.toml"
    save_voice_source(source, sidecar)
    monkeypatch.setattr(
        "voicerender.voice_clone.create_render_profile", lambda **kwargs: kwargs
    )

    result = clone_voice_profile(
        VoiceCloneRequest("fake", "clone-model", "en-US", voice_source=sidecar)
    )

    assert result["transcript"] is None
    assert result["voice_source"]["metadata"]["transcript"] == "From the source."
    assert result["voice_source"]["sha256"] == source["sha256"]


def test_kokoro_adapter_binds_generic_transcript_to_akinvox() -> None:
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
    options = PyKokoroEngineAdapter().bind_reference_voice(
        target=target,
        language="en-US",
        voice_source={"kind": "reference"},
        transcript="Exact AkinVox transcript.",
        options={"voice_enrollment": {"engine": "akinvox"}},
    )

    assert options["voice_enrollment"] == {
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
