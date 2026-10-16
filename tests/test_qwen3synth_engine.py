from __future__ import annotations

import hashlib
import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from voicerender.engines.base import SpeechRequest
from voicerender.engines.catalog import CatalogRequest
from voicerender.engines.qwen3synth import (
    QWEN3_QVOICE_FORMAT,
    Qwen3SynthEngineAdapter,
)
from voicerender.engines.registry import normalize_engine_id
from voicerender.engines.selection import EngineRequest
from voicerender.errors import (
    EngineSelectionError,
    InvalidEngineVoiceError,
)
from voicerender.voice_sources import (
    local_prepared_voice_source,
    local_reference_source,
)


def _fake_package() -> ModuleType:
    module = ModuleType("qwen3synth")
    module.__version__ = "0.1.0"
    module.__file__ = "/fake/qwen3synth/__init__.py"

    class FakeRuntime:
        @classmethod
        def from_installed(cls, *args: Any, **kwargs: Any) -> Any:
            return cls()

        def measure_request(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def prepare_voice(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def synthesize(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def close(self) -> None:
            return None

        def design_voice(self, *args: Any, **kwargs: Any) -> Any:
            return None

    class FakePreparedVoice:
        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> Any:
            return cls()

    class FakeGenerationConfig:
        def __init__(
            self,
            temperature: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.05,
            max_new_tokens: int = 2048,
            seed: int | None = None,
        ) -> None:
            self.temperature = temperature
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty
            self.max_new_tokens = max_new_tokens
            self.seed = seed

    module.Qwen3Runtime = FakeRuntime
    module.SynthesisRequest = type("SynthesisRequest", (), {})
    module.SynthesisResult = type("SynthesisResult", (), {})
    module.GenerationConfig = FakeGenerationConfig
    module.ReferenceVoice = type("ReferenceVoice", (), {})
    module.PreparedVoice = FakePreparedVoice
    module.RequestMeasure = type("RequestMeasure", (), {})
    module.VoiceDesignRequest = type("VoiceDesignRequest", (), {})
    module.inspect_voice_profile = lambda *args, **kwargs: None
    module.discover_bundles = lambda **kwargs: ()
    module.runtime_identity = lambda *args, **kwargs: {}
    module.REQUEST_API_VERSION = 1
    module.request_api_contract = lambda: SimpleNamespace(
        version=1,
        supports_reference_voice=True,
        supports_request_measurement=True,
        supports_named_voices=False,
        supports_voice_design=True,
    )
    return module


def _bundle(
    *,
    model_type: str = "base",
    bundle_id: str = "base-0.6b",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bundle_id,
        ref=f"qwen3:{bundle_id}",
        display_name=f"Qwen3 {model_type}",
        model_type=model_type,
        layout="qwen3-wavekat-v1",
        sample_rate=24000,
        languages=("en", "de"),
        precisions=("int4", "fp32"),
        source_revision="revision-1",
        runtime_available=True,
        metadata={"model_type": model_type},
    )


def test_qwen3synth_alias_and_capabilities() -> None:
    adapter = Qwen3SynthEngineAdapter()

    assert normalize_engine_id("qwen3synth") == "qwen3"
    assert adapter.capabilities().id == "qwen3"
    assert adapter.capabilities().supports_reference_voice is True
    assert adapter.capabilities().supports_request_measurement is True
    assert adapter.capabilities().supports_named_voices is False
    assert "voice_source" in adapter.capabilities().option_names


def test_probe_accepts_public_api_v1_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    calls: list[dict[str, Any]] = []
    fake.discover_bundles = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert probe.compatible, probe
    assert probe.status == "ready"
    assert probe.api_version == 1
    assert probe.details["supports_reference_voice"] is True
    assert probe.details["supports_request_measurement"] is True
    assert probe.details["supports_voice_design"] is True
    assert calls == []


def test_probe_rejects_incompatible_request_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    fake.REQUEST_API_VERSION = 2
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "api_version_incompatible"
    assert probe.api_version == 2


def test_probe_requires_request_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    del fake.Qwen3Runtime.measure_request
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert "Qwen3Runtime.measure_request" in probe.missing_methods


def test_probe_requires_prepared_voice_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    del fake.PreparedVoice.load
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert "PreparedVoice.load" in probe.missing_methods


def test_probe_allows_base_without_voice_design_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    fake.request_api_contract = lambda: SimpleNamespace(
        supports_reference_voice=True,
        supports_request_measurement=True,
        supports_voice_design=False,
    )
    del fake.VoiceDesignRequest
    del fake.Qwen3Runtime.design_voice
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert probe.compatible, probe
    assert probe.details["supports_voice_design"] is False


def test_probe_rejects_missing_required_contract_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    fake.request_api_contract = lambda: SimpleNamespace(
        supports_reference_voice=True,
        supports_request_measurement=False,
        supports_voice_design=True,
    )
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.failed_stage == "request_capabilities"
    assert probe.details["capability_mismatches"]["supports_request_measurement"] == {
        "expected": True,
        "actual": False,
    }


def test_discovery_maps_base_and_voice_design_catalog_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    calls: list[dict[str, Any]] = []

    def discover_bundles(**kwargs: Any) -> tuple[SimpleNamespace, ...]:
        calls.append(kwargs)
        return (
            _bundle(),
            _bundle(model_type="voice_design", bundle_id="voice-design-1.7b"),
        )

    fake.discover_bundles = discover_bundles
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    targets = Qwen3SynthEngineAdapter().discover(
        CatalogRequest(engine="qwen3", language="en-US", offline=True, refresh=True)
    )

    assert calls == [{"offline": True, "refresh": True}]
    assert [target.id for target in targets] == ["base-0.6b", "voice-design-1.7b"]
    base, design = targets
    assert base.engine == "qwen3"
    assert base.display_name == "Qwen3 base"
    assert base.runtime_available is True
    assert base.sample_rate == 24000
    assert base.qualities == ("int4", "fp32")
    assert base.metadata["ref"] == "qwen3:base-0.6b"
    assert base.metadata["source_revision"] == "revision-1"
    assert base.capabilities == frozenset(
        {"reference_voice", "prepared_voice", "request_measurement"}
    )
    assert design.capabilities == frozenset({"voice_design"})


def test_resolve_rejects_voice_design_bundle_as_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package()
    fake.discover_bundles = lambda **kwargs: (
        _bundle(model_type="voice_design", bundle_id="voice-design-1.7b"),
    )
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    with pytest.raises(EngineSelectionError) as exc_info:
        Qwen3SynthEngineAdapter().resolve(
            EngineRequest(
                engine="qwen3",
                target_id="voice-design-1.7b",
                language="en-US",
            )
        )

    assert exc_info.value.code == "qwen3.voice_design_target_not_renderable"


def test_resolve_selects_base_bundle_and_preserves_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    fake = _fake_package()
    fake.discover_bundles = lambda **kwargs: (_bundle(),)
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)

    selection, diagnostics = Qwen3SynthEngineAdapter().resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={
                "voice_source": {
                    "kind": "reference",
                    "value": str(tmp_path / "narrator.wav"),
                    "sha256": "a" * 64,
                },
                "reference_text": "A reference utterance.",
            },
        )
    )

    assert diagnostics == ()
    assert selection.target_id == "base-0.6b"
    assert selection.language == "en-us"
    assert selection.metadata["source_revision"] == "revision-1"
    assert selection.options["quality"] == "int4"
    assert selection.options["provider"] == "cpu"


def test_missing_optional_package_is_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.import_module

    def import_without_qwen(name: str, package: str | None = None) -> Any:
        if name == "qwen3synth":
            raise ModuleNotFoundError("No module named 'qwen3synth'", name=name)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_qwen)

    probe = Qwen3SynthEngineAdapter().probe_api()

    assert not probe.compatible
    assert probe.status == "package_missing"


def _install_session_fake(
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
    *,
    profile: Any | None = None,
) -> ModuleType:
    fake = _fake_package()
    fake.discover_bundles = lambda **kwargs: (_bundle(),)
    fake.SynthesisRequest = lambda **values: SimpleNamespace(**values)
    fake.ReferenceVoice = lambda **values: SimpleNamespace(**values)
    fake.Qwen3Runtime.from_installed = classmethod(lambda cls, *args, **kwargs: runtime)
    fake.inspect_voice_profile = lambda path: profile
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)
    return fake


def _request(request_id: str, text: str = "A sentence.") -> SpeechRequest:
    return SpeechRequest(id=request_id, text=text, language="en-US")


def test_session_measures_without_synthesis_and_reuses_reference_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference audio")
    source = local_reference_source(reference)
    calls: dict[str, list[Any]] = {
        "measure": [],
        "prepare": [],
        "synthesize": [],
        "open": [],
    }

    class Runtime:
        def measure_request(self, request: Any) -> Any:
            calls["measure"].append(request)
            return SimpleNamespace(fits=True, amount=5, maximum=100)

        def prepare_voice(self, voice: Any) -> object:
            calls["prepare"].append(voice)
            return object()

        def synthesize(self, request: Any, *, voice: Any, config: Any) -> Any:
            calls["synthesize"].append((request, voice, config))
            return SimpleNamespace(
                request_id=request.id,
                samples=[0.0, 0.25, -0.25],
                sample_rate=24_000,
                metadata={},
            )

        def close(self) -> None:
            calls["close"] = [True]

    runtime = Runtime()
    fake = _install_session_fake(monkeypatch, runtime)
    original_open = fake.Qwen3Runtime.from_installed

    def record_open(cls: type, *args: Any, **kwargs: Any) -> Any:
        calls["open"].append((args, kwargs))
        return original_open(*args, **kwargs)

    fake.Qwen3Runtime.from_installed = classmethod(record_open)
    adapter = Qwen3SynthEngineAdapter()
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={
                "voice_source": source,
                "reference_text": "  exact reference transcript  ",
                "temperature": 0.7,
                "top_k": 24,
                "repetition_penalty": 1.2,
                "max_new_tokens": 128,
                "seed": 11,
                "catalog_path": "/catalog/qwen.json",
                "cache_dir": "/cache/qwen",
            },
        )
    )
    assert diagnostics == ()
    identity = adapter.canonical_synthesis_identity(selection)
    assert identity["voice"]["sha256"] == source["sha256"]
    assert (
        identity["voice"]["reference_text_sha256"]
        == hashlib.sha256(b"  exact reference transcript  ").hexdigest()
    )
    assert str(reference) not in repr(identity)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    duplicate = elsewhere / "voice.wav"
    duplicate.write_bytes(reference.read_bytes())
    same_selection, _ = adapter.resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={
                "voice_source": local_reference_source(duplicate),
                "reference_text": "  exact reference transcript  ",
                "temperature": 0.7,
                "top_k": 24,
                "repetition_penalty": 1.2,
                "max_new_tokens": 128,
                "seed": 11,
            },
        )
    )
    assert adapter.canonical_synthesis_identity(same_selection) == identity

    different_selection, _ = adapter.resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={
                **selection.options,
                "reference_text": "Different exact transcript.",
            },
        )
    )
    assert adapter.canonical_synthesis_identity(different_selection) != identity

    with adapter.open(selection) as session:
        measured = session.measure(_request("measure"))
        assert measured.fits is True
        assert measured.amount == 5
        assert measured.maximum == 100
        assert measured.unit == "model_tokens"
        assert calls["prepare"] == []
        assert calls["synthesize"] == []

        first = session.synthesize(_request("first"))
        second = session.synthesize(_request("second"))

    assert len(calls["open"]) == 1
    assert calls["open"][0][0] == ("base-0.6b",)
    assert calls["open"][0][1] == {
        "quality": "int4",
        "provider": "cpu",
        "catalog": "/catalog/qwen.json",
        "cache_dir": "/cache/qwen",
    }
    assert calls["measure"][0].language == "en"
    assert calls["prepare"][0].transcript == "  exact reference transcript  "
    assert calls["prepare"][0].language == "en"
    assert len(calls["prepare"]) == 1
    assert len(calls["synthesize"]) == 2
    assert calls["synthesize"][0][0].language == "en"
    assert calls["synthesize"][0][2].temperature == 0.7
    assert calls["synthesize"][0][2].top_k == 24
    assert calls["synthesize"][0][2].repetition_penalty == 1.2
    assert calls["synthesize"][0][2].max_new_tokens == 128
    assert calls["synthesize"][0][2].seed == 11
    assert first.audio.tolist() == [0.0, 0.25, -0.25]
    assert second.id == "second"
    assert "exact reference transcript" not in repr(first.metadata)
    assert str(reference) not in repr(first.metadata)
    assert calls["close"] == [True]


def test_prepared_voice_is_inspected_and_loaded_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    artifact = tmp_path / "narrator.qvoice"
    artifact.write_bytes(b"prepared voice profile")
    source = local_prepared_voice_source(
        artifact,
        format=QWEN3_QVOICE_FORMAT,
        metadata={
            "model_id": "base-0.6b",
            "model_revision": "revision-1",
            "layout": "qwen3-wavekat-v1",
        },
    )
    profile = SimpleNamespace(
        model_id="base-0.6b",
        model_revision="revision-1",
        layout="qwen3-wavekat-v1",
    )
    calls: dict[str, list[Any]] = {"load": [], "synthesize": [], "prepare": []}
    prepared_voice = object()

    class Runtime:
        def prepare_voice(self, reference: Any) -> Any:
            calls["prepare"].append(reference)
            raise AssertionError("prepared qvoice should not be encoded again")

        def synthesize(self, request: Any, *, voice: Any, config: Any) -> Any:
            calls["synthesize"].append((request, voice, config))
            return SimpleNamespace(
                request_id=request.id,
                samples=[0.0, 0.1],
                sample_rate=24_000,
                metadata={},
            )

        def close(self) -> None:
            return None

    runtime = Runtime()
    fake = _install_session_fake(monkeypatch, runtime, profile=profile)
    fake.PreparedVoice.load = classmethod(
        lambda cls, path: calls["load"].append(path) or prepared_voice
    )
    adapter = Qwen3SynthEngineAdapter()
    selection, _ = adapter.resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={"voice_source": source},
        )
    )

    assert adapter.validate_selection(selection) == ()
    identity = adapter.canonical_synthesis_identity(selection)
    assert identity["voice"] == {
        "kind": "prepared",
        "format": QWEN3_QVOICE_FORMAT,
        "sha256": source["sha256"],
    }
    assert str(artifact) not in repr(identity)
    with adapter.open(selection) as session:
        session.synthesize(_request("first"))
        session.synthesize(_request("second"))

    assert calls["load"] == [artifact.resolve()]
    assert calls["prepare"] == []
    assert len(calls["synthesize"]) == 2
    assert calls["synthesize"][0][1] is prepared_voice


def test_native_reference_errors_are_typed_and_keep_the_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference audio")
    source = local_reference_source(reference)

    class NativeReferenceError(Exception):
        pass

    class Runtime:
        def prepare_voice(self, voice: Any) -> Any:
            raise NativeReferenceError("reference cannot be encoded")

        def synthesize(self, request: Any, *, voice: Any, config: Any) -> Any:
            raise AssertionError("native voice preparation should fail first")

        def close(self) -> None:
            return None

    runtime = Runtime()
    fake = _install_session_fake(monkeypatch, runtime)
    fake.InvalidReferenceAudioError = NativeReferenceError
    selection, _ = Qwen3SynthEngineAdapter().resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={
                "voice_source": source,
                "reference_text": "An exact transcript.",
            },
        )
    )

    with (
        pytest.raises(InvalidEngineVoiceError) as error,
        Qwen3SynthEngineAdapter().open(selection) as session,
    ):
        session.synthesize(_request("request"))

    assert error.value.code == "qwen3.reference_invalid"
    assert error.value.native_error_type == "NativeReferenceError"
    assert isinstance(error.value.__cause__, NativeReferenceError)


def test_resolve_uses_origin_transcript_and_rejects_missing_or_wrong_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    fake = _fake_package()
    fake.discover_bundles = lambda **kwargs: (_bundle(),)
    monkeypatch.setitem(sys.modules, "qwen3synth", fake)
    adapter = Qwen3SynthEngineAdapter()
    origin = {
        "schema": "voicerender.voice-origin.qwen3-design.v1",
        "generator": "qwen3",
        "target_id": "voice-design-1.7b",
        "model_revision": "design-revision",
        "qwen3synth_version": "0.1.0",
        "onnxvoice_version": "0.1.0",
        "prompt": "A warm speaking voice.",
        "reference_text": "This exact generated sample.",
        "language": "en",
        "quality": "int4",
        "provider": "cpu",
        "temperature": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 128,
        "seed": 42,
    }
    source = {
        "kind": "reference",
        "value": str(tmp_path / "designed.wav"),
        "sha256": "a" * 64,
        "origin": origin,
    }
    selection, _ = adapter.resolve(
        EngineRequest(
            engine="qwen3",
            target_id="base-0.6b",
            language="en-US",
            options={"voice_source": source},
        )
    )
    assert selection.options["reference_text"] == origin["reference_text"]

    with pytest.raises(EngineSelectionError) as missing_transcript:
        adapter.resolve(
            EngineRequest(
                engine="qwen3",
                target_id="base-0.6b",
                language="en-US",
                options={
                    "voice_source": {
                        key: value for key, value in source.items() if key != "origin"
                    }
                },
            )
        )
    assert missing_transcript.value.code == "qwen3.reference_text_required"

    with pytest.raises(EngineSelectionError) as wrong_format:
        adapter.resolve(
            EngineRequest(
                engine="qwen3",
                target_id="base-0.6b",
                language="en-US",
                options={
                    "voice_source": {
                        "kind": "prepared",
                        "format": "unsupported.qvoice.v1",
                        "value": str(tmp_path / "profile.qvoice"),
                        "sha256": "b" * 64,
                    }
                },
            )
        )
    assert wrong_format.value.code == "qwen3.prepared_voice_invalid"
