from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voicerender.engines.base import SpeechRequest
from voicerender.engines.catalog import CatalogRequest
from voicerender.engines.registry import normalize_engine_id
from voicerender.engines.selection import EngineRequest
from voicerender.errors import EngineSelectionError, InvalidEngineVoiceError
from voicerender.voice_sources import (
    ResolvedReferenceAudio,
    local_prepared_voice_source,
    local_reference_source,
)


def _managed_source() -> dict[str, str]:
    return {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:voice-a",
        "sha256": "a" * 64,
        "source_revision": "voice-revision-1",
        "source_repository": "voice-donations",
        "source_path": "voices/a.wav",
    }


def _fake_soprosynth() -> tuple[ModuleType, dict[str, Any]]:
    module = ModuleType("soprosynth")
    module.__version__ = "0.1.0"
    module.__file__ = "/fake/soprosynth/__init__.py"
    state: dict[str, Any] = {
        "discoveries": [],
        "loads": [],
        "prepared": [],
        "measurements": [],
        "syntheses": [],
        "closed": 0,
        "from_pretrained": 0,
    }

    class SoproSynthError(Exception):
        pass

    class FakeSynthesisRequest:
        def __init__(self, *, id: str, text: str, language: str | None) -> None:
            self.id = id
            self.text = text
            self.language = language

    class FakeGenerationConfig:
        def __init__(self, **values: Any) -> None:
            fields = {
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "steps": None,
                "max_seconds": None,
                "min_seconds": None,
                "seed": None,
                "text_chunk_size": None,
            }
            fields.update(values)
            self.__dict__.update(fields)

    class FakeVoiceLevelConfig:
        def __init__(self, *, mode: str = "off", gain_db: float | None = None) -> None:
            self.mode = mode
            self.gain_db = gain_db

    class FakeOutputConfig:
        def __init__(self, *, mode: str = "native") -> None:
            self.mode = mode

    class FakeRuntime:
        model_id = "model-v1"
        source_revision = "bundle-revision-1"

        def measure_request(self, request: Any, *, maximum: int | None = None) -> Any:
            state["measurements"].append((request, maximum))
            return SimpleNamespace(amount=7, maximum=maximum, fits=None)

        def prepare_voice(
            self,
            audio: str | Path,
            *,
            reference_seconds: float | None = None,
            resampler: str | None = None,
        ) -> Any:
            state["prepared"].append((audio, reference_seconds, resampler))
            return SimpleNamespace(
                identity={
                    "kind": "reference",
                    "sha256": "a" * 64,
                    "model_id": self.model_id,
                }
            )

        def synthesize(
            self,
            request: Any,
            *,
            voice: Any,
            config: Any,
            voice_level: Any,
            output: Any,
        ) -> Any:
            state["syntheses"].append((request, voice, config, voice_level, output))
            return SimpleNamespace(
                id=request.id,
                audio=np.ones(32, dtype=np.float32),
                sample_rate=24000,
                warnings=(),
                metadata={
                    "model_id": self.model_id,
                    "model_revision": self.source_revision,
                    "effective_generation": {"seed": config.seed},
                    "sopro_output": {"mode": output.mode},
                    "voice_level_application": {"mode": voice_level.mode},
                    "runtime_identity": {"engine_version": "0.1.0"},
                },
            )

        def close(self) -> None:
            state["closed"] += 1

    class FakePublicRuntime:
        @classmethod
        def load(cls, ref: str, **kwargs: Any) -> FakeRuntime:
            state["loads"].append((ref, kwargs))
            return FakeRuntime()

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            state["from_pretrained"] += 1
            raise AssertionError("VoiceRender must not acquire a Sopro model")

        def prepare_voice(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

        def measure_request(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

        def synthesize(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

        def close(self) -> None:
            raise NotImplementedError

    bundle = SimpleNamespace(
        id="model-v1",
        ref="sopro:model-v1",
        display_name="Sopro model v1",
        aliases=("model-alias",),
        languages=("en", "pt", "fr", "de"),
        sample_rate=24000,
        profiles=("wasm-uint8",),
        source_revision="bundle-revision-1",
        source_commit="commit-1",
        metadata={"max_input_tokens": 128},
    )

    def discover_bundles(**kwargs: Any) -> tuple[Any, ...]:
        state["discoveries"].append(kwargs)
        return (bundle,)

    module.SoproRuntime = FakePublicRuntime
    module.SynthesisRequest = FakeSynthesisRequest
    module.SynthesisResult = type("SynthesisResult", (), {})
    module.GenerationConfig = FakeGenerationConfig
    module.VoiceLevelConfig = FakeVoiceLevelConfig
    module.SoproOutputConfig = FakeOutputConfig
    module.PreparedVoice = type("PreparedVoice", (), {})
    module.RequestMeasure = type("RequestMeasure", (), {})
    module.DiscoveredBundle = type("DiscoveredBundle", (), {})
    module.SoproSynthError = SoproSynthError
    module.InvalidGenerationConfigError = type(
        "InvalidGenerationConfigError", (SoproSynthError,), {}
    )
    module.discover_bundles = discover_bundles
    module.runtime_identity = lambda bundle=None: {
        "engine_version": "0.1.0",
        "request_api_version": "1",
    }
    module.REQUEST_API_VERSION = 1
    module.request_api_contract = lambda: SimpleNamespace(
        version=1,
        supports_reference_voice=True,
        supports_request_measurement=True,
        supports_named_voices=False,
    )
    return module, state


def _install(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    from voicerender.engines.soprosynth import SoproSynthEngineAdapter

    module, state = _fake_soprosynth()
    monkeypatch.setitem(sys.modules, "soprosynth", module)
    return SoproSynthEngineAdapter(), state


def test_sopro_api_probe_discovery_and_target_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, state = _install(monkeypatch)

    probe = adapter.probe_api()
    targets = adapter.discover(CatalogRequest(engine="sopro", language="en-US"))
    capabilities = adapter.capabilities()

    assert probe.compatible, probe
    assert probe.api_version == 1
    assert len(targets) == 1
    assert targets[0].id == "model-v1"
    assert targets[0].metadata["bundle_ref"] == "sopro:model-v1"
    assert targets[0].metadata["source_revision"] == "bundle-revision-1"
    assert targets[0].voices == ()
    assert targets[0].default_voice is None
    assert {"reference_voice", "request_measurement", "voice_level"} <= targets[
        0
    ].capabilities
    assert capabilities.supports_reference_voice is True
    assert capabilities.supports_request_measurement is True
    assert capabilities.supports_named_voices is False
    assert normalize_engine_id("SoproSynth") == "sopro"
    assert state["loads"] == []


def test_sopro_preflight_measures_without_resolving_or_preparing_and_reuses_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicerender.engines import soprosynth

    adapter, state = _install(monkeypatch)
    source = _managed_source()
    resolve_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(soprosynth, "validate_voice_source", lambda *args, **kwargs: ())

    def resolve_reference(*args: Any, **kwargs: Any) -> ResolvedReferenceAudio:
        resolve_calls.append((args, kwargs))
        return ResolvedReferenceAudio(
            kind="managed_reference",
            path=Path("/cache/voice-a.wav"),
            sha256=source["sha256"],
            resolver="voicefolio",
            ref=source["value"],
            source_revision=source["source_revision"],
            source_repository=source["source_repository"],
            source_path=source["source_path"],
        )

    monkeypatch.setattr(soprosynth, "resolve_reference_audio", resolve_reference)
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="sopro",
            target_id="model-alias",
            language="en-US",
            options={
                "voice_source": source,
                "profile": "wasm-uint8",
                "voice_catalog_path": "/private/catalog",
                "voice_cache_dir": "/private/cache",
                "reference_seconds": 4.5,
                "reference_resampler": "linear",
                "temperature": 0.65,
                "top_p": 0.8,
                "top_k": 12,
                "steps": 9,
                "max_seconds": 12,
                "min_seconds": 1,
                "random_seed": 17,
                "text_chunk_size": 180,
                "output_mode": "raw",
                "voice_level": "calibrated",
                "voice_level_gain_db": 1.5,
                "providers": ("cpu",),
                "provider_options": {"arena_extend_strategy": "kSameAsRequested"},
                "session_options": {"intra_op_num_threads": 2},
            },
        )
    )
    assert diagnostics == ()
    assert adapter.validate_selection(selection) == ()
    assert resolve_calls == []
    assert state["prepared"] == []

    identity = adapter.canonical_synthesis_identity(selection)
    identity_json = json.dumps(identity, sort_keys=True)
    assert identity["voice"]["kind"] == "managed_reference"
    assert "/private/catalog" not in identity_json
    assert "/private/cache" not in identity_json
    assert "/cache/voice-a.wav" not in identity_json

    with adapter.open(selection) as session:
        measured = session.measure(
            SpeechRequest(id="measure-1", text="A measured request.", language="en-US")
        )
        assert measured.amount == 7
        assert measured.maximum == 128
        assert measured.unit == "model_tokens"
        assert measured.source == "soprosynth.measure_request"
        assert resolve_calls == []
        assert state["prepared"] == []

        first = session.synthesize(
            SpeechRequest(id="speech-1", text="First request.", language="en-US")
        )
        second = session.synthesize(
            SpeechRequest(id="speech-2", text="Second request.", language="en-GB")
        )

    assert first.id == "speech-1"
    assert second.id == "speech-2"
    assert first.sample_rate == 24000
    assert first.metadata["voice_source_identity"]["ref"] == source["value"]
    assert first.metadata["sopro_output"] == {"mode": "raw"}
    assert len(resolve_calls) == 1
    assert len(state["prepared"]) == 1
    assert state["prepared"][0] == (Path("/cache/voice-a.wav"), 4.5, "linear")
    assert len(state["syntheses"]) == 2
    assert state["syntheses"][0][0].language == "en"
    assert state["syntheses"][1][0].language == "en"
    assert state["syntheses"][0][2].temperature == 0.65
    assert state["syntheses"][0][2].seed == 17
    assert state["syntheses"][0][3].gain_db == 1.5
    assert state["syntheses"][0][4].mode == "raw"
    assert state["loads"][0][0] == "sopro:model-v1"
    assert state["loads"][0][1]["profile"] == "wasm-uint8"
    assert state["from_pretrained"] == 0
    assert state["closed"] == 1


def test_sopro_local_reference_is_rehashed_before_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, state = _install(monkeypatch)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"original local audio")
    selection, _ = adapter.resolve(
        EngineRequest(
            engine="sopro",
            target_id="model-v1",
            language="de-DE",
            options={"voice_source": local_reference_source(reference)},
        )
    )
    assert adapter.validate_selection(selection) == ()
    reference.write_bytes(b"changed local audio")

    with (
        adapter.open(selection) as session,
        pytest.raises(InvalidEngineVoiceError) as error,
    ):
        session.synthesize(
            SpeechRequest(id="local-1", text="Use this reference.", language="de-DE")
        )

    assert error.value.code == "voice_source.reference_changed"
    assert state["prepared"] == []
    assert state["closed"] == 1


def test_sopro_selection_requires_reference_and_rejects_named_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _install(monkeypatch)
    with pytest.raises(EngineSelectionError) as missing:
        adapter.resolve(
            EngineRequest(engine="sopro", target_id="model-v1", language="en-US")
        )
    assert missing.value.code == "sopro.reference_required"

    with pytest.raises(EngineSelectionError) as named:
        adapter.resolve(
            EngineRequest(
                engine="sopro",
                target_id="model-v1",
                language="en-US",
                voice="speaker-a",
                options={"voice_source": _managed_source()},
            )
        )
    assert named.value.code == "sopro.named_voice_unsupported"


def test_sopro_rejects_prepared_voice_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _ = _install(monkeypatch)
    artifact = tmp_path / "voice.qvoice"
    artifact.write_bytes(b"prepared voice artifact")
    source = local_prepared_voice_source(artifact, format="qwen3.qvoice.v2")

    with pytest.raises(EngineSelectionError) as prepared:
        adapter.resolve(
            EngineRequest(
                engine="sopro",
                target_id="model-v1",
                language="en-US",
                options={"voice_source": source},
            )
        )

    assert prepared.value.code == "sopro.reference_required"


def test_sopro_api_probe_rejects_missing_reference_measurement_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _install(monkeypatch)
    module = sys.modules["soprosynth"]
    module.request_api_contract = lambda: SimpleNamespace(
        version=1,
        supports_reference_voice=False,
        supports_request_measurement=True,
        supports_named_voices=False,
    )

    probe = adapter.probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert "capability_mismatches" in probe.details
