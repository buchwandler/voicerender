from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from renderplan import (
    RenderPlan,
    SourceSpan,
    SpeechItem,
    load_manifest,
)

from voicerender.engines import chatterboxsynth as chatterbox_adapter_module
from voicerender.engines.base import EngineSelection, SpeechRequest
from voicerender.engines.catalog import CatalogRequest
from voicerender.engines.registry import _ADAPTERS, normalize_engine_id
from voicerender.engines.selection import EngineRequest
from voicerender.errors import (
    EngineSelectionError,
    EngineUnavailableError,
    InvalidEngineVoiceError,
)
from voicerender.preflight import preflight_render_plan
from voicerender.renderplan_input import PreflightDisposition, classify_measurement
from voicerender.renderplan_render import render_renderplan
from voicerender.voice_sources import (
    ResolvedReferenceAudio,
    local_reference_source,
)


def _bundle(
    bundle_id: str,
    *,
    aliases: tuple[str, ...],
    profiles: tuple[str, ...],
    default_profile: str,
    installed_profiles: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bundle_id,
        ref=f"chatterbox:{bundle_id}",
        display_name=f"Chatterbox {bundle_id}",
        aliases=aliases,
        languages=("en",),
        profiles=profiles,
        default_profile=default_profile,
        installed_profiles=installed_profiles,
        sample_rate=24_000,
        source_revision=("a" * 39) + ("1" if bundle_id == "nano-v2" else "2"),
        runtime_available=bool(installed_profiles),
        metadata={"source_repository": "chatterbox-bundles"},
    )


def _managed_source(sha256: str = "a" * 64) -> dict[str, str]:
    return {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:voice-a",
        "sha256": sha256,
        "source_revision": "voice-revision-1",
        "source_repository": "voice-donations",
        "source_path": "voices/a.wav",
    }


def _fake_chatterboxsynth() -> tuple[ModuleType, dict[str, Any]]:
    module = ModuleType("chatterboxsynth")
    module.__version__ = "0.1.1"
    module.__file__ = "/fake/chatterboxsynth/__init__.py"
    bundles = (
        _bundle(
            "nano-v2",
            aliases=("nano", "nano-110m"),
            profiles=("cpu", "fp32"),
            default_profile="cpu",
            installed_profiles=("cpu",),
        ),
        _bundle(
            "turbo",
            aliases=("turbo-350m",),
            profiles=("fp32", "q4f16"),
            default_profile="fp32",
        ),
    )
    state: dict[str, Any] = {
        "bundles": bundles,
        "discoveries": [],
        "opens": [],
        "from_pretrained": 0,
        "measurements": [],
        "prepared": [],
        "syntheses": [],
        "closed": 0,
        "open_error": None,
        "measurement": SimpleNamespace(
            amount=19, fits=True, maximum=None, unit="model_tokens"
        ),
        "contract": SimpleNamespace(
            version=1,
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_named_voices=False,
        ),
    }

    class ChatterboxSynthError(Exception):
        pass

    class OptionalDependencyError(ChatterboxSynthError):
        pass

    class ModelLoadError(ChatterboxSynthError):
        pass

    class ModelNotInstalledError(ModelLoadError):
        pass

    class AudioPromptError(ChatterboxSynthError):
        pass

    class IncompatibleVoiceError(ChatterboxSynthError):
        pass

    class InvalidGenerationConfigError(ChatterboxSynthError):
        pass

    class EmptyTextError(ChatterboxSynthError):
        pass

    class RuntimeClosedError(ChatterboxSynthError):
        pass

    class SynthesisError(ChatterboxSynthError):
        pass

    class FakeSynthesisRequest:
        def __init__(self, *, id: str, text: str, language: str) -> None:
            self.id = id
            self.text = text
            self.language = language

    class FakeGenerationConfig:
        def __init__(
            self,
            *,
            max_new_tokens: int | None = None,
            repetition_penalty: float | None = None,
        ) -> None:
            if max_new_tokens is not None and (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int)
                or max_new_tokens <= 0
            ):
                raise InvalidGenerationConfigError(
                    "max_new_tokens must be a positive integer or None"
                )
            if repetition_penalty is not None and (
                isinstance(repetition_penalty, bool)
                or not isinstance(repetition_penalty, (int, float))
                or repetition_penalty <= 0
            ):
                raise InvalidGenerationConfigError(
                    "repetition_penalty must be positive or None"
                )
            self.max_new_tokens = max_new_tokens
            self.repetition_penalty = repetition_penalty

    class FakeRuntime:
        def measure_request(self, request: Any) -> Any:
            state["measurements"].append(request)
            return state["measurement"]

        def prepare_voice(self, path: str | Path) -> object:
            state["prepared"].append(Path(path))
            return object()

        def synthesize(self, text: str, *, voice: Any, config: Any) -> Any:
            state["syntheses"].append((text, voice, config))
            return SimpleNamespace(
                audio=np.linspace(-0.2, 0.2, 64, dtype=np.float32),
                sample_rate=24_000,
                model="nano-v2",
                profile="cpu",
                generated_tokens=11,
                stopped_on_eos=True,
            )

        def close(self) -> None:
            state["closed"] += 1

    class ChatterboxRuntime:
        @classmethod
        def from_installed(cls, ref: str, **kwargs: Any) -> FakeRuntime:
            state["opens"].append((ref, kwargs))
            if state["open_error"] is not None:
                raise state["open_error"]
            return FakeRuntime()

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            state["from_pretrained"] += 1
            raise AssertionError("VoiceRender must not acquire Chatterbox models")

        def prepare_voice(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

        def measure_request(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

        def synthesize(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

        def close(self) -> None:
            raise NotImplementedError

    def discover_bundles(
        *, offline: bool = False, refresh: bool = False
    ) -> tuple[Any, ...]:
        state["discoveries"].append({"offline": offline, "refresh": refresh})
        return state["bundles"]

    module.ChatterboxRuntime = ChatterboxRuntime
    module.ChatterboxSynthError = ChatterboxSynthError
    module.OptionalDependencyError = OptionalDependencyError
    module.ModelLoadError = ModelLoadError
    module.ModelNotInstalledError = ModelNotInstalledError
    module.AudioPromptError = AudioPromptError
    module.IncompatibleVoiceError = IncompatibleVoiceError
    module.InvalidGenerationConfigError = InvalidGenerationConfigError
    module.EmptyTextError = EmptyTextError
    module.RuntimeClosedError = RuntimeClosedError
    module.SynthesisError = SynthesisError
    module.SynthesisRequest = FakeSynthesisRequest
    module.SynthesisResult = type("SynthesisResult", (), {})
    module.GenerationConfig = FakeGenerationConfig
    module.PreparedVoice = type("PreparedVoice", (), {})
    module.RequestMeasure = type("RequestMeasure", (), {})
    module.DiscoveredBundle = type("DiscoveredBundle", (), {})
    module.discover_bundles = discover_bundles
    module.runtime_identity = lambda: {
        "engine": "chatterboxsynth",
        "engine_version": "0.1.1",
        "request_api_version": 1,
    }
    module.REQUEST_API_VERSION = 1
    module.request_api_contract = lambda: state["contract"]
    return module, state


def _install(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, ModuleType, dict[str, Any]]:
    from voicerender.engines.chatterboxsynth import ChatterboxSynthEngineAdapter

    module, state = _fake_chatterboxsynth()
    monkeypatch.setitem(sys.modules, "chatterboxsynth", module)
    return ChatterboxSynthEngineAdapter(), module, state


def _resolve(
    adapter: Any,
    *,
    target_id: str = "nano-v2",
    language: str = "en-US",
    source: dict[str, Any] | None = None,
    voice: str | None = None,
    speaker: str | int | None = None,
    options: dict[str, Any] | None = None,
    engine_options: dict[str, Any] | None = None,
) -> EngineSelection:
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="chatterbox",
            target_id=target_id,
            language=language,
            voice=voice,
            speaker=speaker,
            options={
                **(options or {}),
                **({"voice_source": source} if source is not None else {}),
            },
            engine_options=engine_options or {},
        )
    )
    assert diagnostics == ()
    return selection


def test_chatterbox_api_probe_and_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, module, state = _install(monkeypatch)
    probe = adapter.probe_api()
    assert probe.compatible
    assert probe.api_version == 1
    assert probe.contract_source == "explicit"

    del module.REQUEST_API_VERSION
    missing = adapter.probe_api()
    assert not missing.compatible
    assert "REQUEST_API_VERSION" in missing.missing_symbols

    module.REQUEST_API_VERSION = 2
    wrong_version = adapter.probe_api()
    assert not wrong_version.compatible
    assert wrong_version.status == "api_version_incompatible"

    module.REQUEST_API_VERSION = 1
    for name, value in (
        ("supports_reference_voice", False),
        ("supports_request_measurement", False),
        ("supports_named_voices", True),
    ):
        state["contract"] = SimpleNamespace(
            version=1,
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_named_voices=False,
        )
        setattr(state["contract"], name, value)
        incompatible = adapter.probe_api()
        assert not incompatible.compatible
        assert incompatible.status == "api_incompatible"
        assert name in incompatible.details["capability_mismatches"]


def test_discovery_maps_nano_turbo_aliases_and_capacity_free_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, state = _install(monkeypatch)
    targets = adapter.discover(
        CatalogRequest(engine="chatterbox", offline=True, refresh=True)
    )
    assert [target.id for target in targets] == ["nano-v2", "turbo"]
    nano, turbo = targets
    assert chatterbox_adapter_module._find_target(targets, "nano-110m") is nano
    assert chatterbox_adapter_module._find_target(targets, "turbo-350m") is turbo
    assert nano.languages == ("en",)
    assert nano.qualities == ("cpu", "fp32")
    assert nano.runtime_available is True
    assert nano.metadata["ref"] == "chatterbox:nano-v2"
    assert (
        nano.metadata["source_revision"]
        == _bundle(
            "nano-v2", aliases=(), profiles=(), default_profile=""
        ).source_revision
    )
    assert nano.metadata["installed_profiles"] == ("cpu",)
    assert nano.capabilities == frozenset({"reference_voice", "request_measurement"})
    assert nano.voices == () and nano.default_voice is None
    assert state["discoveries"] == [{"offline": True, "refresh": True}]

    capabilities = adapter.capabilities()
    assert capabilities.id == "chatterbox"
    assert capabilities.supports_reference_voice
    assert capabilities.supports_request_measurement
    assert not capabilities.supports_named_voices
    assert normalize_engine_id("ChatterboxSynth") == "chatterbox"

    selection = _resolve(adapter, source=_managed_source())
    fragment = adapter.renderprofile_fragment(selection)
    assert fragment.frontend == "chatterboxsynth"
    assert fragment.renderer_capacity is None
    assert fragment.planner_capacity is None
    assert "capacity" not in fragment.binding["renderer_metadata"]


def test_selection_requires_english_reference_and_supported_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _, _ = _install(monkeypatch)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"pinned local reference")
    local = local_reference_source(reference)
    selection = _resolve(adapter, target_id="nano", source=local)
    assert selection.target_id == "nano-v2"
    assert selection.language == "en-us"
    assert selection.options["quality"] == "cpu"
    assert adapter.validate_selection(selection) == ()

    managed = _resolve(adapter, target_id="turbo-350m", source=_managed_source())
    assert managed.target_id == "turbo"
    assert managed.options["quality"] == "fp32"

    with pytest.raises(EngineSelectionError) as missing:
        _resolve(adapter)
    assert missing.value.code == "chatterbox.reference_required"

    with pytest.raises(EngineSelectionError) as named:
        _resolve(adapter, source=_managed_source(), voice="narrator")
    assert named.value.code == "chatterbox.named_voice_unsupported"

    with pytest.raises(EngineSelectionError) as speaker:
        _resolve(adapter, source=_managed_source(), speaker="speaker-a")
    assert speaker.value.code == "chatterbox.speaker_unsupported"

    prepared = {
        "kind": "prepared",
        "value": str((tmp_path / "voice.qvoice").resolve()),
        "sha256": "b" * 64,
        "format": "qwen3.qvoice.v2",
    }
    with pytest.raises(EngineSelectionError) as prepared_error:
        _resolve(adapter, source=prepared)
    assert prepared_error.value.code == "chatterbox.reference_required"

    with pytest.raises(EngineSelectionError) as language:
        _resolve(adapter, source=_managed_source(), language="fr-FR")
    assert language.value.code == "chatterbox.language_incompatible"

    with pytest.raises(EngineSelectionError) as profile:
        _resolve(
            adapter,
            target_id="nano-v2",
            source=_managed_source(),
            options={"quality": "q4f16"},
        )
    assert profile.value.code == "chatterbox.quality_incompatible"

    with pytest.raises(EngineSelectionError) as unknown:
        _resolve(adapter, source=_managed_source(), engine_options={"unknown": True})
    assert unknown.value.code == "chatterbox.invalid_option"

    with pytest.raises(EngineSelectionError) as generation:
        _resolve(
            adapter, source=_managed_source(), engine_options={"max_new_tokens": 0}
        )
    assert generation.value.code == "chatterbox.generation_config_invalid"


def test_measurement_is_exact_and_does_not_resolve_prepare_or_synthesize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voicerender.engines import chatterboxsynth as adapter_module

    adapter, _, state = _install(monkeypatch)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"pinned local reference")
    selection = _resolve(
        adapter,
        source=local_reference_source(reference),
        engine_options={"max_new_tokens": 128, "repetition_penalty": 1.2},
    )
    resolution_calls: list[Any] = []
    original_resolve = adapter_module.resolve_reference_audio

    def resolve_spy(*args: Any, **kwargs: Any) -> ResolvedReferenceAudio:
        resolution_calls.append((args, kwargs))
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "resolve_reference_audio", resolve_spy)
    with adapter.open(selection) as session:
        measured = session.measure(
            SpeechRequest(id="measure-1", text="Count these tokens.", language="en-US")
        )
        assert measured.amount == 19
        assert measured.maximum is None
        assert measured.fits is True
        assert measured.unit == "model_tokens"
        assert measured.source == "chatterboxsynth.measure_request"
        assert classify_measurement(measured).disposition is PreflightDisposition.OK
        assert resolution_calls == []
        assert state["prepared"] == []
        assert state["syntheses"] == []


def test_synthesis_resolves_and_prepares_reference_once_per_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _, state = _install(monkeypatch)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"pinned local reference")
    source = local_reference_source(reference)
    selection = _resolve(
        adapter,
        source=source,
        engine_options={"max_new_tokens": 128, "repetition_penalty": 1.15},
    )
    with adapter.open(selection) as session:
        first = session.synthesize(
            SpeechRequest(id="speech-1", text="First request.", language="en-US")
        )
        second = session.synthesize(
            SpeechRequest(id="speech-2", text="Second request.", language="en-GB")
        )

    assert first.id == "speech-1" and second.id == "speech-2"
    assert first.sample_rate == second.sample_rate == 24_000
    assert first.metadata["generated_tokens"] == 11
    assert first.metadata["stopped_on_eos"] is True
    assert first.metadata["voice_source_identity"] == {
        "kind": "reference",
        "sha256": source["sha256"],
    }
    assert len(state["opens"]) == 1
    assert state["opens"][0][0] == "chatterbox:nano-v2"
    assert state["from_pretrained"] == 0
    assert len(state["prepared"]) == 1
    assert state["prepared"][0] == reference.resolve()
    assert len(state["syntheses"]) == 2
    assert state["syntheses"][0][2].max_new_tokens == 128
    assert state["syntheses"][0][2].repetition_penalty == 1.15
    assert state["closed"] == 1


def test_reference_integrity_failure_prevents_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _, state = _install(monkeypatch)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"original pinned reference")
    selection = _resolve(adapter, source=local_reference_source(reference))
    reference.write_bytes(b"changed reference bytes")

    with (
        adapter.open(selection) as session,
        pytest.raises(InvalidEngineVoiceError) as changed,
    ):
        session.synthesize(
            SpeechRequest(id="changed", text="Must not prepare", language="en")
        )
    assert changed.value.code == "voice_source.reference_changed"
    assert state["prepared"] == []
    assert state["syntheses"] == []


def test_managed_pin_mismatch_from_generic_resolver_prevents_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicerender.engines import chatterboxsynth as adapter_module

    adapter, _, state = _install(monkeypatch)
    selection = _resolve(adapter, source=_managed_source())

    def reject_pin(*args: Any, **kwargs: Any) -> Any:
        raise InvalidEngineVoiceError(
            "Voicefolio reference pin mismatch",
            engine="chatterbox",
            code="voice_source.managed_pin_mismatch",
        )

    monkeypatch.setattr(adapter_module, "resolve_reference_audio", reject_pin)
    with (
        adapter.open(selection) as session,
        pytest.raises(InvalidEngineVoiceError) as mismatch,
    ):
        session.synthesize(
            SpeechRequest(id="managed", text="Must not prepare", language="en")
        )
    assert mismatch.value.code == "voice_source.managed_pin_mismatch"
    assert state["prepared"] == []
    assert state["syntheses"] == []


def test_open_translates_missing_installed_model_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, module, state = _install(monkeypatch)
    selection = _resolve(adapter, source=_managed_source())
    state["open_error"] = module.ModelNotInstalledError("not installed")

    with pytest.raises(EngineUnavailableError) as unavailable:
        adapter.open(selection)
    assert unavailable.value.code == "chatterbox.model_not_installed"
    assert unavailable.value.details["engine"] == "chatterbox"
    assert unavailable.value.details["target_id"] == "nano-v2"
    assert unavailable.value.details["native_error_type"] == "ModelNotInstalledError"
    assert unavailable.value.details["operation"] == "open"


def test_canonical_identity_pins_model_profile_voice_and_generation_without_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = _install(monkeypatch)
    first = _resolve(
        adapter,
        source=_managed_source("a" * 64),
        options={
            "voice_source": _managed_source("a" * 64),
            "voice_cache_dir": "/private/voice-cache",
            "voice_catalog_path": "/private/voice-catalog",
            "cache_dir": "/private/model-cache",
        },
        engine_options={"max_new_tokens": 100},
    )
    changed_voice = _resolve(
        adapter,
        source=_managed_source("b" * 64),
        engine_options={"max_new_tokens": 100},
    )
    changed_generation = _resolve(
        adapter,
        source=_managed_source("a" * 64),
        engine_options={"max_new_tokens": 101},
    )

    first_identity = adapter.canonical_synthesis_identity(first)
    encoded = json.dumps(first_identity, sort_keys=True)
    assert first_identity["model_revision"] == first.metadata["source_revision"]
    assert first_identity["quality"] == "cpu"
    assert first_identity["generation"]["max_new_tokens"] == 100
    assert first_identity["voice"]["sha256"] == "a" * 64
    assert "/private/" not in encoded
    assert first_identity != adapter.canonical_synthesis_identity(changed_voice)
    assert first_identity != adapter.canonical_synthesis_identity(changed_generation)


def test_renderplan_preflight_render_writes_common_wav_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, _, state = _install(monkeypatch)
    monkeypatch.setitem(_ADAPTERS, "chatterbox", adapter)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"RenderPlan pinned reference bytes")
    source = local_reference_source(reference)
    text = "A request measured and rendered by ChatterboxSynth."
    binding = {
        "renderer": "chatterbox",
        "target_id": "nano-v2",
        "options": {"voice_source": source, "quality": "cpu"},
    }
    plan = RenderPlan(
        source_plan_id="sha256:" + "3" * 64,
        render_profile_id="sha256:" + "4" * 64,
        flow=(
            SpeechItem(
                number=1,
                text=text,
                language="en-US",
                role="narrator",
                binding=binding,
                source=(SourceSpan(locator="f000001.s000", start=0, end=len(text)),),
                planning_level="sentence",
                preflight_required=True,
            ),
        ),
    ).with_identity()

    report = preflight_render_plan(plan, tmp_path / "preflight.toml")
    assert report.status == "ok"
    assert state["measurements"]
    assert state["measurements"][0].language == "en"
    assert state["prepared"] == []

    output_dir = tmp_path / "render"
    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )
    assert not result.failed
    assert result.manifest.complete
    artifact = result.manifest.speech[0]
    assert artifact.sample_rate == 24_000
    assert (output_dir / artifact.wav).is_file()
    loaded = load_manifest(result.manifest_path)
    loaded.validate_against(plan)
    assert loaded.manifest_id == result.manifest.manifest_id
    assert len(state["syntheses"]) == 1
    assert state["from_pretrained"] == 0
