"""Registry and public API-probe tests; optional SDKs remain mocked and lazy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from voicerender.engines.registry import (
    discover_targets,
    engine_status,
    get_adapter,
    list_engines,
    normalize_engine_id,
)

ENGINE_IDS = {
    "inflect",
    "kitten",
    "kokoro",
    "piper",
    "pocket",
    "sopro",
    "supertonic",
    "qwen3",
}
PACKAGES = {
    "inflect": "inflectsynth",
    "kitten": "kittensynth",
    "kokoro": "pykokoro",
    "piper": "pipersynth",
    "pocket": "pocketsynth",
    "sopro": "soprosynth",
    "supertonic": "supertonicsynth",
    "qwen3": "qwen3synth",
}

ADAPTER_MODULES = {
    "inflect": "voicerender.engines.inflectsynth",
    "kitten": "voicerender.engines.kittensynth",
    "kokoro": "voicerender.engines.kokoro",
    "piper": "voicerender.engines.pipersynth",
    "pocket": "voicerender.engines.pocketsynth",
    "sopro": "voicerender.engines.soprosynth",
    "supertonic": "voicerender.engines.supertonicsynth",
    "qwen3": "voicerender.engines.qwen3synth",
}


def test_import_and_engine_listing_do_not_import_optional_sdks() -> None:
    script = r"""
import builtins
blocked = {"inflectsynth", "kittensynth", "pykokoro", "pipersynth", "pocketsynth", "soprosynth", "supertonicsynth", "qwen3synth"}
seen = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        seen.append(name)
        raise AssertionError(f"optional SDK imported: {name}")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import voicerender
from voicerender.engines import list_engines
assert {item["id"] for item in list_engines()} == blocked_to_ids
assert not seen
"""
    # Keep the subprocess isolated from SDKs that may already be imported by pytest.
    script = script.replace(
        "blocked_to_ids",
        "{'inflect', 'kitten', 'kokoro', 'piper', 'pocket', 'sopro', 'supertonic', 'qwen3'}",
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _fake_package(package: str) -> ModuleType:
    module = ModuleType(package)
    module.__version__ = "0.1.0"
    module.__file__ = f"/fake/{package}/__init__.py"

    owner_methods = {
        "PiperVoice": ("from_pretrained", "measure", "synthesize", "close"),
        "VoiceAssetManager": ("list_voices", "get_voice_metadata"),
        "KokoroSynthesizer": (
            "prepare",
            "measure_request",
            "enroll_voice",
            "synthesize",
            "close",
        ),
        "PocketRuntime": (
            "from_resolved",
            "prepare_voice",
            "synthesize",
            "close",
            "measure_request",
        ),
        "BundleAssetManager": ("resolve_bundle", "list_bundles"),
        "KittenVoice": (
            "synthesize_prepared",
            "measure_prepared",
            "from_pretrained",
            "from_local",
            "close",
        ),
        "SoproRuntime": (
            "load",
            "prepare_voice",
            "measure_request",
            "synthesize",
            "close",
        ),
        "SupertonicRuntime": (
            "from_pretrained",
            "measure_request",
            "synthesize",
            "close",
        ),
        "InflectVoice": (
            "from_pretrained",
            "from_local",
            "synthesize_prepared",
            "measure_prepared",
            "close",
        ),
        "Qwen3Runtime": (
            "from_installed",
            "measure_request",
            "prepare_voice",
            "synthesize",
            "close",
            "design_voice",
        ),
        "PreparedVoice": ("load",),
    }
    symbols = {
        "PiperVoice",
        "VoiceAssetManager",
        "SynthesisRequest",
        "SynthesisResult",
        "SynthesisConfig",
        "LinguisticToken",
        "PronunciationOverride",
        "VoiceLevelConfig",
        "SynthesisInputTooLongError",
        "KokoroSynthesizer",
        "ShortSentenceConfig",
        "ReferenceVoice",
        "KokoroVoicePack",
        "InnoEnrollmentOptions",
        "PocketRuntime",
        "GenerationConfig",
        "BundleAssetManager",
        "PreparedVoice",
        "RequestMeasure",
        "discover_bundles",
        "KittenVoice",
        "SynthesisMeasure",
        "discover_models",
        "runtime_identity",
        "SoproSynthError",
        "SoproRuntime",
        "SoproOutputConfig",
        "DiscoveredBundle",
        "SupertonicRuntime",
        "AtomicSynthesisResult",
        "DiscoveredModel",
        "InflectVoice",
        "DescribedVoice",
        "request_api_contract",
        "capacity_api_contract",
        "Qwen3Runtime",
        "VoiceDesignRequest",
        "inspect_voice_profile",
    }
    for name in symbols:
        if name in owner_methods:
            value = type(
                name,
                (),
                {
                    method: staticmethod(lambda *args, **kwargs: None)
                    for method in owner_methods[name]
                },
            )
        elif name.endswith("Error"):
            value = type(name, (Exception,), {})
        else:
            value = lambda *args, **kwargs: None
        setattr(module, name, value)

    module._probe_discovery_calls = []

    def forbidden_discovery(*args, **kwargs):
        module._probe_discovery_calls.append((args, kwargs))
        raise AssertionError("API probes must not discover or load models")

    for name in (
        "discover_models",
        "discover_bundles",
    ):
        if name in symbols:
            setattr(module, name, forbidden_discovery)
    module.REQUEST_API_VERSION = 1
    module.CAPACITY_API_VERSION = 1
    module.request_api_contract = lambda: {"version": 1}
    module.capacity_api_contract = lambda: {"supports_known_maximum": True}
    return module


def test_registry_lists_nine_engines_without_probing() -> None:
    listed = list_engines()
    assert {item["id"] for item in listed} == ENGINE_IDS
    assert {item["package"] for item in listed} == set(PACKAGES.values())
    assert all("installed_version" in item for item in listed)
    by_id = {item["id"]: item for item in listed}
    assert by_id["sopro"]["supports_reference_voice"] is True
    assert by_id["pocket"]["supports_reference_voice"] is True
    assert by_id["kokoro"]["supports_reference_voice"] is True
    assert by_id["piper"]["supports_reference_voice"] is False


def test_soprosynth_alias_normalizes_without_loading_sdk() -> None:
    assert normalize_engine_id("SoproSynth") == "sopro"


@pytest.mark.parametrize(("engine", "package"), sorted(PACKAGES.items()))
def test_all_engine_probes_use_public_api_without_loading_models(
    monkeypatch: pytest.MonkeyPatch, engine: str, package: str
) -> None:
    import importlib

    fake = _fake_package(package)
    monkeypatch.setitem(sys.modules, package, fake)
    if engine == "inflect":
        fake.request_api_contract = lambda: {
            "version": 1,
            "supports_request_measurement": True,
            "measurement_entrypoint": "InflectVoice.measure_prepared",
            "measurement_metric": "model_tokens",
        }
    elif engine == "kitten":
        fake.request_api_contract = lambda: {
            "version": 1,
            "supports_request_measurement": True,
            "measurement_entrypoint": "KittenVoice.measure_prepared",
            "measurement_metric": "model_tokens",
        }
    if engine == "qwen3":
        fake.request_api_contract = lambda: SimpleNamespace(
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_voice_design=True,
        )
    elif engine == "sopro":
        fake.request_api_contract = lambda: SimpleNamespace(
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_named_voices=False,
        )
    elif engine == "kokoro":
        fake.request_api_contract = lambda: SimpleNamespace(
            supports_request_measurement=True,
            supports_reference_voice=True,
            supports_voice_enrollment=True,
        )
    original = importlib.import_module
    calls: list[str] = []

    def fake_import(name: str, package_arg: str | None = None):
        if name == package:
            calls.append(name)
            return fake
        return original(name, package_arg)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    adapter = get_adapter(engine)
    probe = adapter.probe_api()
    assert probe.compatible, probe
    assert probe.status == "ready"
    assert calls
    status = engine_status(engine)
    assert status["compatible"] is True
    assert fake._probe_discovery_calls == []


def test_inflect_probe_accepts_common_measurement_contract_without_capacity_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("inflectsynth")
    fake.request_api_contract = lambda: {
        "version": 1,
        "supports_request_measurement": True,
        "measurement_entrypoint": "InflectVoice.measure_prepared",
        "measurement_metric": "model_tokens",
    }
    del fake.CAPACITY_API_VERSION
    del fake.capacity_api_contract
    monkeypatch.setitem(sys.modules, "inflectsynth", fake)

    adapter = get_adapter("inflect")
    probe = adapter.probe_api()

    assert probe.compatible, probe
    assert probe.details["supports_request_measurement"] is True
    assert probe.details["measurement_entrypoint"] == "InflectVoice.measure_prepared"
    assert probe.details["measurement_metric"] == "model_tokens"
    assert probe.details["capacity_api_compatible"] is False
    assert adapter.capabilities().supports_request_measurement is True
    assert fake._probe_discovery_calls == []


def test_inflect_probe_rejects_measurement_without_declared_api_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("inflectsynth")
    fake.request_api_contract = lambda: {
        "version": 1,
        "supports_request_measurement": True,
        "measurement_entrypoint": "InflectVoice.measure_prepared",
    }
    monkeypatch.setitem(sys.modules, "inflectsynth", fake)

    probe = get_adapter("inflect").probe_api()

    assert not probe.compatible
    assert probe.failed_stage == "request_measurement_contract"
    assert probe.details["measurement_metric"] is None


def test_inflect_probe_rejects_declared_measurement_without_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("inflectsynth")
    fake.request_api_contract = lambda: {
        "version": 1,
        "supports_request_measurement": True,
        "measurement_entrypoint": "InflectVoice.measure_prepared",
        "measurement_metric": "model_tokens",
    }
    delattr(fake.InflectVoice, "measure_prepared")
    monkeypatch.setitem(sys.modules, "inflectsynth", fake)

    probe = get_adapter("inflect").probe_api()

    assert not probe.compatible
    assert "InflectVoice.measure_prepared" in probe.missing_methods


def test_piper_probe_rejects_measurement_disabled_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("pipersynth")
    fake.request_api_contract = lambda: {
        "version": 1,
        "supports_request_measurement": False,
    }
    monkeypatch.setitem(sys.modules, "pipersynth", fake)

    probe = get_adapter("piper").probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert probe.details["supports_request_measurement"] is False


def test_piper_probe_requires_measure_method(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_package("pipersynth")
    delattr(fake.PiperVoice, "measure")
    monkeypatch.setitem(sys.modules, "pipersynth", fake)

    probe = get_adapter("piper").probe_api()

    assert not probe.compatible
    assert "PiperVoice.measure" in probe.missing_methods


def test_kitten_probe_requires_measure_prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("kittensynth")
    delattr(fake.KittenVoice, "measure_prepared")
    monkeypatch.setitem(sys.modules, "kittensynth", fake)

    probe = get_adapter("kitten").probe_api()

    assert not probe.compatible
    assert "KittenVoice.measure_prepared" in probe.missing_methods


def test_kitten_probe_rejects_measurement_disabled_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("kittensynth")
    fake.request_api_contract = lambda: {
        "version": 1,
        "supports_request_measurement": False,
    }
    monkeypatch.setitem(sys.modules, "kittensynth", fake)

    probe = get_adapter("kitten").probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert probe.failed_stage == "request_measurement_contract"
    assert probe.details["supports_request_measurement"] is False
    assert any(
        "does not support request measurement" in warning for warning in probe.warnings
    )


def test_kitten_probe_fails_closed_when_measurement_contract_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("kittensynth")
    calls = 0

    def request_api_contract() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"version": 1, "supports_request_measurement": True}
        raise RuntimeError("contract unavailable")

    fake.request_api_contract = request_api_contract
    monkeypatch.setitem(sys.modules, "kittensynth", fake)

    probe = get_adapter("kitten").probe_api()

    assert not probe.compatible
    assert probe.status == "api_probe_failed"
    assert probe.failed_stage == "request_measurement_contract"
    assert probe.error_type == "RuntimeError"


def test_kitten_probe_reports_measurement_stage_for_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("kittensynth")

    def request_api_contract() -> dict[str, object]:
        raise RuntimeError("contract unavailable")

    fake.request_api_contract = request_api_contract
    monkeypatch.setitem(sys.modules, "kittensynth", fake)

    probe = get_adapter("kitten").probe_api()

    assert not probe.compatible
    assert probe.status == "api_probe_failed"
    assert probe.failed_stage == "request_measurement_contract"
    assert probe.error_type == "RuntimeError"


def test_kitten_capabilities_advertise_request_measurement() -> None:
    assert get_adapter("kitten").capabilities().supports_request_measurement is True


def test_kitten_probe_requires_synthesis_measure_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("kittensynth")
    delattr(fake, "SynthesisMeasure")
    monkeypatch.setitem(sys.modules, "kittensynth", fake)

    probe = get_adapter("kitten").probe_api()

    assert not probe.compatible
    assert "SynthesisMeasure" in probe.missing_symbols


def test_kitten_probe_accepts_attribute_measurement_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_package("kittensynth")
    fake.request_api_contract = lambda: SimpleNamespace(
        supports_request_measurement=True
    )
    monkeypatch.setitem(sys.modules, "kittensynth", fake)

    probe = get_adapter("kitten").probe_api()

    assert probe.compatible, probe


@pytest.mark.parametrize("engine", sorted(ENGINE_IDS))
def test_all_adapters_discover_fake_catalogs_without_model_downloads(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    from importlib import import_module

    from voicerender.engines.catalog import SynthesisTarget

    adapter_module = import_module(ADAPTER_MODULES[engine])
    if engine == "piper":
        fake = _fake_package(PACKAGES[engine])

        class FakeVoiceAssetManager:
            def __init__(self, *, offline: bool = False) -> None:
                self.offline = offline

            def list_voices(self, *, language: str | None, refresh: bool):
                return (
                    SimpleNamespace(
                        id="piper-demo",
                        name="Piper demo",
                        language_code=language or "en-US",
                        language_family="en",
                        sample_rate=22050,
                        quality="medium",
                    ),
                )

        fake.VoiceAssetManager = FakeVoiceAssetManager
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
    elif engine == "pocket":
        fake = _fake_package(PACKAGES[engine])
        fake.discover_bundles = lambda **kwargs: (
            SimpleNamespace(
                id="pocket-demo",
                sample_rate=24000,
                metadata={
                    "display_name": "Pocket demo",
                    "language": "en-US",
                    "profiles": {"int8": {}},
                    "predefined_voice_names": ["speaker-a"],
                    "default_voice": "speaker-a",
                },
            ),
        )
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
    elif engine == "sopro":
        fake = _fake_package(PACKAGES[engine])
        fake.discover_bundles = lambda **kwargs: (
            SimpleNamespace(
                id="sopro-demo",
                ref="sopro:model",
                display_name="Sopro demo",
                languages=("en",),
                sample_rate=24000,
                profiles=("wasm-uint8",),
                source_revision="revision-1",
                source_commit="commit-1",
                metadata={},
            ),
        )
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
    elif engine == "qwen3":
        fake = _fake_package(PACKAGES[engine])
        fake.discover_bundles = lambda **kwargs: (
            SimpleNamespace(
                id="base-0.6b",
                ref="qwen3:base-0.6b",
                display_name="Qwen Base",
                model_type="base",
                layout="qwen3-wavekat-v1",
                sample_rate=24000,
                languages=("en",),
                precisions=("int4", "fp32"),
                source_revision="revision-1",
                runtime_available=True,
                metadata={},
            ),
        )
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
    else:
        target = SynthesisTarget(
            engine=engine,
            id="demo-model",
            display_name="Demo model",
            languages=("en-us",),
        )
        helper_name = {
            "kitten": "_catalog_targets",
            "kokoro": "_discover_targets",
            "supertonic": "_target_catalog",
            "inflect": "_target_catalog",
        }[engine]
        monkeypatch.setattr(
            adapter_module, helper_name, lambda *args, **kwargs: (target,)
        )

    targets = discover_targets(engine, language="en-US")
    assert len(targets) == 1
    assert targets[0].engine == engine
    assert targets[0].id in {
        "base-0.6b",
        "piper-demo",
        "pocket-demo",
        "sopro-demo",
        "demo-model",
    }


@pytest.mark.parametrize("engine", sorted(ENGINE_IDS))
def test_all_adapters_resolve_requests_with_fake_optional_sdks(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    from importlib import import_module

    from voicerender.engines.selection import EngineRequest

    if engine == "qwen3":
        fake = _fake_package(PACKAGES[engine])
        fake.request_api_contract = lambda: SimpleNamespace(
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_voice_design=True,
        )
        fake.discover_bundles = lambda **kwargs: (
            SimpleNamespace(
                id="demo-model",
                ref="qwen3:demo-model",
                display_name="Qwen Base",
                model_type="base",
                layout="qwen3-wavekat-v1",
                sample_rate=24000,
                languages=("en",),
                precisions=("int4",),
                source_revision="revision-1",
                runtime_available=True,
                metadata={},
            ),
        )
        fake.GenerationConfig = lambda **values: SimpleNamespace(**values)
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
    if engine == "sopro":
        fake = _fake_package(PACKAGES[engine])
        fake.request_api_contract = lambda: SimpleNamespace(
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_named_voices=False,
        )
        fake.discover_bundles = lambda **kwargs: (
            SimpleNamespace(
                id="demo-model",
                ref="sopro:demo-model",
                display_name="Sopro demo",
                languages=("en",),
                sample_rate=24000,
                profiles=("wasm-uint8",),
                source_revision="revision-1",
                source_commit="commit-1",
                metadata={},
            ),
        )

        class FakeGenerationConfig:
            def __init__(self, **values) -> None:
                self.__dict__.update(values)

        class FakeOutputConfig:
            def __init__(self, mode: str = "native") -> None:
                self.mode = mode

        class FakeVoiceLevelConfig:
            def __init__(self, mode: str = "off", gain_db=None) -> None:
                self.mode = mode
                self.gain_db = gain_db

        fake.GenerationConfig = FakeGenerationConfig
        fake.SoproOutputConfig = FakeOutputConfig
        fake.VoiceLevelConfig = FakeVoiceLevelConfig
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
    if engine in {"inflect", "supertonic"}:
        fake = _fake_package(PACKAGES[engine])

        class FakeConfig:
            def __init__(self, **values) -> None:
                defaults = {
                    "speed": 1.0,
                    "steps": 5,
                    "seed": 0,
                    "variation": 0.667,
                    "mode": "off",
                }
                defaults.update(values)
                self.__dict__.update(defaults)

            def validated(self):
                return self

        fake.DEFAULT_MODEL = "demo-model"
        fake.DEFAULT_VOICE = "demo-voice"
        fake.GenerationConfig = FakeConfig
        fake.SynthesisConfig = FakeConfig
        fake.VoiceLevelConfig = FakeConfig
        monkeypatch.setitem(sys.modules, PACKAGES[engine], fake)
        adapter_module = import_module(ADAPTER_MODULES[engine])
        monkeypatch.setattr(adapter_module, "_target_catalog", lambda **kwargs: ())

    request = EngineRequest(
        engine=engine,
        target_id="demo-model",
        language="en-US",
        voice=None if engine in {"sopro", "qwen3"} else "demo-voice",
        options=(
            {
                "voice_source": {
                    "kind": "reference",
                    "value": str((Path.cwd() / "demo-reference.wav").resolve()),
                    "sha256": "a" * 64,
                },
                **({"reference_text": "A reference."} if engine == "qwen3" else {}),
            }
            if engine in {"sopro", "qwen3"}
            else {"speed": 2.0}
            if engine == "piper"
            else {}
        ),
    )
    selection, diagnostics = get_adapter(engine).resolve(request)

    assert not diagnostics
    assert selection.engine == engine
    assert selection.target_id == "demo-model"
    if engine == "piper":
        assert selection.options["length_scale"] == 0.5
