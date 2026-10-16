from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.qwen3synth import Qwen3SynthEngineAdapter
from voicerender.errors import (
    ArtifactError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
)
from voicerender.voice_design import Qwen3VoiceDesignRecipe, design_qwen3_reference
from voicerender.voice_source_codec import load_voice_source

_RUNTIME_IDENTITY = {
    "engine": "qwen3synth",
    "engine_version": "0.1.0",
    "request_api_version": 1,
    "onnxvoice_version": "0.1.0",
    "layouts": ("qwen3-wavekat-v1",),
    "sample_rate": 24_000,
}


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []

    class FakeRuntime:
        model_id = "wavekat/voice-design-1.7b"
        model_revision = "a" * 40

        @classmethod
        def from_installed(cls, target_id: str, **kwargs: Any) -> FakeRuntime:
            events.append(("open", (target_id, kwargs)))
            return cls()

        def design_voice(self, request: Any, *, config: Any) -> Any:
            events.append(("design", (request, config)))
            return SimpleNamespace(
                samples=np.asarray([0.0, 0.2, -0.2, 0.0], dtype=np.float32),
                sample_rate=24_000,
            )

        def close(self) -> None:
            events.append(("close", None))

    module = ModuleType("qwen3synth")
    module.__version__ = "0.1.0"
    module.Qwen3Runtime = FakeRuntime
    module.VoiceDesignRequest = lambda **values: SimpleNamespace(**values)
    module.GenerationConfig = lambda **values: SimpleNamespace(**values)
    module.runtime_identity = lambda: dict(_RUNTIME_IDENTITY)
    monkeypatch.setitem(sys.modules, "qwen3synth", module)
    monkeypatch.setattr(
        Qwen3SynthEngineAdapter,
        "probe_api",
        lambda self: EngineApiProbe(
            engine="qwen3",
            package="qwen3synth",
            compatible=True,
            status="ready",
            module_version="0.1.0",
            details={"supports_voice_design": True},
        ),
    )
    return events


def test_design_materializes_pinned_reference_and_recipe_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _install_fake(monkeypatch)
    output = tmp_path / "designed.wav"
    sidecar = tmp_path / "designed.voice-source.toml"
    recipe = Qwen3VoiceDesignRecipe(
        prompt="Warm, measured baritone.",
        reference_text="This sample demonstrates the voice.",
        language="en",
        seed=42,
        temperature=0.7,
        top_k=24,
        repetition_penalty=1.2,
        max_new_tokens=128,
    )

    designed = design_qwen3_reference(
        recipe,
        output,
        cache_dir=tmp_path / "cache",
        catalog=tmp_path / "catalog.toml",
        source_output=sidecar,
    )

    assert [event[0] for event in events] == ["open", "design", "close"]
    assert events[0][1] == (
        "voice-design-1.7b",
        {
            "quality": "int4",
            "provider": "cpu",
            "catalog": tmp_path / "catalog.toml",
            "cache_dir": tmp_path / "cache",
        },
    )
    request, generation = events[1][1]
    assert request.id == "voicerender-voice-design"
    assert request.text == recipe.reference_text
    assert request.description == recipe.prompt
    assert request.language == recipe.language
    assert generation.temperature == recipe.temperature
    assert generation.top_k == recipe.top_k
    assert generation.repetition_penalty == recipe.repetition_penalty
    assert generation.max_new_tokens == recipe.max_new_tokens
    assert generation.seed == recipe.seed

    expected_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    assert designed.path == output.resolve()
    assert designed.sha256 == expected_hash
    assert designed.source["kind"] == "reference"
    assert designed.source["value"] == str(output.resolve())
    assert designed.source["sha256"] == expected_hash
    assert designed.source["origin"] == {
        "schema": "voicerender.voice-origin.qwen3-design.v1",
        "generator": "qwen3",
        "target_id": "voice-design-1.7b",
        "model_revision": "a" * 40,
        "model_id": "wavekat/voice-design-1.7b",
        "qwen3synth_version": "0.1.0",
        "onnxvoice_version": "0.1.0",
        "prompt": recipe.prompt,
        "reference_text": recipe.reference_text,
        "language": recipe.language,
        "quality": recipe.quality,
        "provider": recipe.provider,
        "temperature": recipe.temperature,
        "top_k": recipe.top_k,
        "repetition_penalty": recipe.repetition_penalty,
        "max_new_tokens": recipe.max_new_tokens,
        "seed": recipe.seed,
        "runtime_identity": _RUNTIME_IDENTITY,
    }
    assert designed.model_id == "wavekat/voice-design-1.7b"
    assert designed.runtime_identity == _RUNTIME_IDENTITY
    assert load_voice_source(sidecar) == designed.source


def test_design_rejects_collisions_without_force_and_hash_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _install_fake(monkeypatch)
    output = tmp_path / "designed.wav"
    sidecar = tmp_path / "designed.toml"
    recipe = Qwen3VoiceDesignRecipe(
        prompt="Warm voice.",
        reference_text="This sample demonstrates the voice.",
    )
    designed = design_qwen3_reference(recipe, output, source_output=sidecar)
    original_event_count = len(events)

    with pytest.raises(ArtifactError) as collision:
        design_qwen3_reference(recipe, output, source_output=sidecar)
    assert collision.value.code == "qwen3.voice_design_output_exists"
    assert len(events) == original_event_count

    output.write_bytes(b"changed pinned audio")
    with pytest.raises(InvalidEngineVoiceError) as stale_sidecar:
        load_voice_source(sidecar)
    assert stale_sidecar.value.code == "voice_source.reference_changed"
    assert designed.sha256 != hashlib.sha256(output.read_bytes()).hexdigest()


def test_recipe_rejects_invalid_generation_settings() -> None:
    with pytest.raises(InvalidEngineOptionError) as invalid_temperature:
        Qwen3VoiceDesignRecipe(
            prompt="Warm voice.",
            reference_text="This sample demonstrates the voice.",
            temperature=0,
        )
    assert invalid_temperature.value.code == "qwen3.voice_design_recipe_invalid"
