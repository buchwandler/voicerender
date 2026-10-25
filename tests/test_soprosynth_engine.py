from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from renderplan import RenderPlan, SourceSpan, SpeechItem, load_preflight, save_plan

from voicerender.engines.base import SpeechRequest
from voicerender.engines.catalog import CatalogRequest
from voicerender.engines.registry import _ADAPTERS, normalize_engine_id
from voicerender.engines.selection import EngineRequest
from voicerender.errors import (
    EngineBackendError,
    EngineSelectionError,
    InvalidEngineVoiceError,
)
from voicerender.preflight import preflight_render_plan
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


def _fake_soprosynth(
    *, max_input_tokens: int | None = 128
) -> tuple[ModuleType, dict[str, Any]]:
    module = ModuleType("soprosynth")
    module.__version__ = "0.1.0"
    module.__file__ = "/fake/soprosynth/__init__.py"
    state: dict[str, Any] = {
        "discoveries": [],
        "loads": [],
        "prepared": [],
        "measurements": [],
        "measurement_result": None,
        "measurement_results": {},
        "syntheses": [],
        "closed": 0,
        "from_pretrained": 0,
        "contract_calls": 0,
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
            result = state["measurement_results"].get(request.text)
            if result is not None:
                return result
            result = state["measurement_result"]
            if result is not None:
                return result
            return SimpleNamespace(amount=7, maximum=maximum, fits=None, unit="tokens")

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
        metadata=MappingProxyType(
            {
                "max_input_tokens": max_input_tokens,
                "catalog": MappingProxyType(
                    {
                        "profile_metadata": MappingProxyType(
                            {"wasm-uint8": MappingProxyType({"precision": "uint8"})}
                        )
                    }
                ),
            }
        ),
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

    def request_api_contract() -> Any:
        state["contract_calls"] += 1
        return SimpleNamespace(
            version=1,
            supports_reference_voice=True,
            supports_request_measurement=True,
            supports_named_voices=False,
            caller_owns_text_boundaries=False,
        )

    module.request_api_contract = request_api_contract
    return module, state


def _install(
    monkeypatch: pytest.MonkeyPatch, *, max_input_tokens: int | None = 128
) -> tuple[Any, dict[str, Any]]:
    from voicerender.engines.soprosynth import SoproSynthEngineAdapter

    module, state = _fake_soprosynth(max_input_tokens=max_input_tokens)
    monkeypatch.setitem(sys.modules, "soprosynth", module)
    return SoproSynthEngineAdapter(), state


def _measure_once(
    monkeypatch: pytest.MonkeyPatch,
    measurement: Any,
    *,
    max_input_tokens: int | None = 128,
    package_version: str | None = None,
) -> Any:
    adapter, state = _install(monkeypatch, max_input_tokens=max_input_tokens)
    module = sys.modules["soprosynth"]
    if package_version is not None:
        module.__version__ = package_version
    state["measurement_result"] = measurement
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="sopro",
            target_id="model-v1",
            language="en-US",
            options={"voice_source": _managed_source()},
        )
    )
    assert diagnostics == ()
    with adapter.open(selection) as session:
        return session.measure(
            SpeechRequest(id="measure-test", text="Measured text.", language="en-US")
        )


def _sopro_renderplan(texts: tuple[str, ...]) -> RenderPlan:
    binding = {
        "renderer": "sopro",
        "target_id": "model-v1",
        "options": {"voice_source": _managed_source()},
    }
    items = tuple(
        SpeechItem(
            number=index,
            text=text,
            language="en-US",
            role="narrator",
            binding=binding,
            source=(
                SourceSpan(
                    locator=f"source-{index}",
                    start=0,
                    end=len(text),
                ),
            ),
            planning_level="sentence",
            preflight_required=True,
        )
        for index, text in enumerate(texts, start=1)
    )
    return RenderPlan(
        source_plan_id="sha256:" + "1" * 64,
        render_profile_id="sha256:" + "2" * 64,
        flow=items,
    ).with_identity()


def test_sopro_api_probe_discovery_and_target_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, state = _install(monkeypatch)

    probe = adapter.probe_api()
    targets = adapter.discover(CatalogRequest(engine="sopro", language="en-US"))
    capabilities = adapter.capabilities()

    assert probe.compatible, probe
    assert probe.api_version == 1
    assert probe.details["request_api_contract"]["caller_owns_text_boundaries"] is False
    assert adapter.probe_api() is probe
    assert state["contract_calls"] == 1
    assert len(targets) == 1
    assert targets[0].id == "model-v1"
    assert targets[0].metadata["bundle_ref"] == "sopro:model-v1"
    assert targets[0].metadata["source_revision"] == "bundle-revision-1"
    catalog = targets[0].metadata["catalog"]
    assert isinstance(catalog, Mapping)
    profile_metadata = catalog["profile_metadata"]
    assert isinstance(profile_metadata, Mapping)
    profile = profile_metadata["wasm-uint8"]
    assert isinstance(profile, Mapping)
    assert profile["precision"] == "uint8"
    with pytest.raises(TypeError):
        profile["precision"] = "fp16"
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
        assert session._request_contract is not None
        assert session._request_contract.caller_owns_text_boundaries is False
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
    assert state["contract_calls"] == 1


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
        caller_owns_text_boundaries=False,
    )

    probe = adapter.probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    assert "capability_mismatches" in probe.details


@pytest.mark.parametrize("boundary_value", (True, None))
def test_sopro_api_probe_requires_engine_owned_text_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    boundary_value: bool | None,
) -> None:
    adapter, _ = _install(monkeypatch)
    module = sys.modules["soprosynth"]
    contract = {
        "version": 1,
        "supports_reference_voice": True,
        "supports_request_measurement": True,
        "supports_named_voices": False,
    }
    if boundary_value is not None:
        contract["caller_owns_text_boundaries"] = boundary_value
    module.request_api_contract = lambda: SimpleNamespace(**contract)

    probe = adapter.probe_api()

    assert not probe.compatible
    assert probe.status == "api_incompatible"
    mismatch = probe.details["capability_mismatches"]["caller_owns_text_boundaries"]
    assert mismatch == {"expected": False, "actual": boundary_value}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("amount", True),
        ("amount", -1),
        ("maximum", True),
        ("maximum", 0),
        ("maximum", -1),
        ("fits", "yes"),
        ("fits", 1),
        ("fits", 0),
        ("amount", "7"),
        ("amount", 7.0),
        ("maximum", "128"),
        ("unit", "characters"),
    ),
)
def test_sopro_rejects_malformed_producer_measurements(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    measurement = {"amount": 7, "maximum": None, "fits": True, "unit": "tokens"}
    measurement[field] = value

    with pytest.raises(EngineBackendError) as error:
        _measure_once(
            monkeypatch, SimpleNamespace(**measurement), max_input_tokens=None
        )
    assert error.value.code == "sopro.measurement_invalid"


@pytest.mark.parametrize(("fits", "expected"), ((True, True), (False, False)))
@pytest.mark.parametrize("unit", (None, "tokens", "model_tokens"))
def test_sopro_forwards_authoritative_producer_fit_without_a_maximum(
    monkeypatch: pytest.MonkeyPatch,
    fits: bool,
    expected: bool,
    unit: str | None,
) -> None:
    measured = _measure_once(
        monkeypatch,
        SimpleNamespace(amount=31, maximum=None, fits=fits, unit=unit),
        max_input_tokens=None,
    )

    assert measured.amount == 31
    assert measured.maximum is None
    assert measured.fits is expected
    assert measured.unit == "model_tokens"
    assert measured.details["producer_unit"] == unit
    assert measured.details["producer_maximum"] is None
    assert measured.details["fit_source"] == "producer"


@pytest.mark.parametrize("package_version", ("0.1.0", "0.1.1"))
def test_sopro_legacy_auto_boundary_contract_supplies_fit_authority(
    monkeypatch: pytest.MonkeyPatch,
    package_version: str,
) -> None:
    measured = _measure_once(
        monkeypatch,
        SimpleNamespace(amount=13, maximum=None, fits=None, unit="tokens"),
        max_input_tokens=None,
        package_version=package_version,
    )

    assert measured.amount == 13
    assert measured.maximum is None
    assert measured.fits is True
    assert measured.details["fit_source"] == "legacy_contract_auto_boundaries"


@pytest.mark.parametrize(("amount", "expected_fits"), ((20, True), (129, False)))
def test_sopro_finite_maximum_remains_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    amount: int,
    expected_fits: bool,
) -> None:
    measured = _measure_once(
        monkeypatch,
        SimpleNamespace(amount=amount, maximum=128, fits=None, unit="tokens"),
    )

    assert measured.maximum == 128
    assert measured.fits is expected_fits
    assert measured.details["fit_source"] == "finite_maximum"


def test_sopro_optional_capacity_metadata_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _install(monkeypatch, max_input_tokens=None)
    selection, diagnostics = adapter.resolve(
        EngineRequest(
            engine="sopro",
            target_id="model-v1",
            language="en-US",
            options={"voice_source": _managed_source()},
        )
    )

    assert diagnostics == ()
    assert selection.metadata["max_input_tokens"] is None
    assert adapter.renderprofile_fragment(selection).renderer_capacity is None


def test_sopro_preflight_accepts_authoritative_fits_without_a_maximum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter, state = _install(monkeypatch, max_input_tokens=None)
    from voicerender.engines import soprosynth

    monkeypatch.setattr(soprosynth, "validate_voice_source", lambda *args, **kwargs: ())
    monkeypatch.setitem(_ADAPTERS, "sopro", adapter)
    amounts = (6, 13, 31, 47)
    texts = tuple(f"request-{amount}" for amount in amounts)
    state["measurement_results"] = {
        text: SimpleNamespace(
            amount=amount,
            maximum=None,
            fits=True,
            unit="tokens",
        )
        for amount, text in zip(amounts, texts, strict=True)
    }

    report = preflight_render_plan(
        _sopro_renderplan(texts), tmp_path / "sopro-preflight.toml"
    )

    assert report.status == "ok"
    assert [check.status for check in report.checks] == ["ok"] * len(amounts)
    assert [check.measured for check in report.checks] == list(amounts)
    assert all(check.maximum is None for check in report.checks)
    assert [maximum for _, maximum in state["measurements"]] == [None] * len(amounts)
    assert state["syntheses"] == []
    assert state["contract_calls"] == 1


def test_unsupported_sopro_version_keeps_unknown_fit_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = _measure_once(
        monkeypatch,
        SimpleNamespace(amount=13, maximum=None, fits=None, unit="tokens"),
        max_input_tokens=None,
        package_version="0.1.2",
    )

    assert measured.fits is None
    assert measured.details["fit_source"] == "unknown"


def test_installed_sopro_measurement_type_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soprosynth = pytest.importorskip("soprosynth")
    native = soprosynth.RequestMeasure(
        amount=31,
        maximum=None,
        unit="tokens",
        fits=True,
    )

    measured = _measure_once(monkeypatch, native, max_input_tokens=None)

    assert measured.amount == 31
    assert measured.maximum is None
    assert measured.fits is True
    assert measured.unit == "model_tokens"
    assert measured.details["fit_source"] == "producer"


def test_sopro_unknown_fit_remains_rejected_by_generic_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter, state = _install(monkeypatch, max_input_tokens=None)
    module = sys.modules["soprosynth"]
    module.__version__ = "0.1.2"
    from voicerender.engines import soprosynth

    monkeypatch.setattr(soprosynth, "validate_voice_source", lambda *args, **kwargs: ())
    monkeypatch.setitem(_ADAPTERS, "sopro", adapter)
    state["measurement_result"] = SimpleNamespace(
        amount=13,
        maximum=None,
        fits=None,
        unit="tokens",
    )

    report = preflight_render_plan(
        _sopro_renderplan(("unknown fit",)), tmp_path / "sopro-unknown-fit.toml"
    )

    assert report.status == "reject"
    assert report.checks[0].status == "error"
    assert report.checks[0].detail == (
        "Engine measured the exact request size but did not provide an "
        "authoritative maximum/fit result"
    )
    assert state["syntheses"] == []


def test_sopro_cli_aggregates_repeated_unknown_fit_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from voicerender.cli import main
    from voicerender.engines import soprosynth

    adapter, state = _install(monkeypatch, max_input_tokens=None)
    module = sys.modules["soprosynth"]
    module.__version__ = "0.1.2"
    monkeypatch.setattr(soprosynth, "validate_voice_source", lambda *args, **kwargs: ())
    monkeypatch.setitem(_ADAPTERS, "sopro", adapter)
    texts = ("unknown one", "unknown two", "unknown three")
    state["measurement_results"] = {
        text: SimpleNamespace(
            amount=13,
            maximum=None,
            fits=None,
            unit="tokens",
        )
        for text in texts
    }
    plan_path = tmp_path / "sopro.renderplan.toml"
    report_path = tmp_path / "sopro.preflight.toml"
    save_plan(_sopro_renderplan(texts), plan_path)

    exit_code = main(["preflight", str(plan_path), "--output", str(report_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "sopro measurement: 3/3 requests have exact model-token counts" in output
    assert "fit authority: missing; maximum: not reported" in output
    assert "source: soprosynth.measure_request" in output
    assert (
        "action: update SoproSynth or verify its request-measurement contract" in output
    )
    persisted = load_preflight(report_path)
    assert len(persisted.checks) == len(texts)
    assert all(check.status == "error" for check in persisted.checks)
    assert all(
        check.detail == "Engine measured the exact request size but did not provide an "
        "authoritative maximum/fit result"
        for check in persisted.checks
    )
    assert state["syntheses"] == []
