from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import voicerender.engines.supertonicsynth as supertonic_engine
from voicerender.engines.base import EngineSelection
from voicerender.engines.catalog import SynthesisTarget
from voicerender.engines.registry import _ADAPTERS
from voicerender.engines.selection import EngineRequest
from voicerender.renderprofile import (
    create_render_profile,
    load_render_profile,
    save_render_profile,
)


class FakeGenerationConfig:
    def __init__(
        self,
        *,
        steps: int = 5,
        speed: float = 1.05,
        seed: int | None = None,
    ) -> None:
        self.steps = steps
        self.speed = speed
        self.seed = seed


class FakeVoiceLevelConfig:
    def __init__(self, *, mode: str = "off") -> None:
        self.mode = mode


class FakeRuntime:
    def __init__(self, target_id: str) -> None:
        self.bundle = SimpleNamespace(bundle_id=target_id, source_revision=None)

    @classmethod
    def from_pretrained(cls, target_id: str, **kwargs: Any) -> FakeRuntime:
        return cls(target_id)

    def close(self) -> None:
        pass


def _install(
    monkeypatch: pytest.MonkeyPatch,
    targets: tuple[SynthesisTarget, ...] = (),
) -> tuple[ModuleType, supertonic_engine.SupertonicSynthEngineAdapter]:
    fake = ModuleType("supertonicsynth")
    fake.DEFAULT_MODEL = "supertonic-3"
    fake.DEFAULT_VOICE = "F1"
    fake.GenerationConfig = FakeGenerationConfig
    fake.VoiceLevelConfig = FakeVoiceLevelConfig

    fake.SupertonicRuntime = FakeRuntime
    monkeypatch.setitem(sys.modules, "supertonicsynth", fake)
    monkeypatch.setattr(supertonic_engine, "_target_catalog", lambda **kwargs: targets)

    adapter = supertonic_engine.SupertonicSynthEngineAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    monkeypatch.setattr(
        adapter,
        "probe_api",
        lambda: SimpleNamespace(compatible=True, status="ready"),
    )
    return fake, adapter


def _resolve(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any] | None = None,
) -> EngineSelection:
    _, adapter = _install(monkeypatch)
    request = EngineRequest(
        engine="supertonic",
        target_id="supertonic-3",
        language="en-US",
        voice="F1",
        options=options or {},
    )
    selection, diagnostics = adapter.resolve(request)
    assert diagnostics == ()
    return selection


def test_resolve_default_omits_unset_seed_and_persists_normalized_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _resolve(monkeypatch)

    assert selection.options["steps"] == 5
    assert selection.options["speed"] == 1.05
    assert selection.options["voice_level"] == "off"
    assert "seed" not in selection.options


def test_resolve_explicit_none_seed_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _resolve(monkeypatch, {"seed": None})

    assert "seed" not in selection.options


def test_resolve_preserves_explicit_integer_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _resolve(monkeypatch, {"seed": 17})

    assert selection.options["seed"] == 17


def test_resolve_omits_none_passthrough_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _resolve(
        monkeypatch,
        {
            "cache_dir": None,
            "providers": None,
            "provider_options": None,
            "session_options": None,
        },
    )

    assert (
        not {
            "cache_dir",
            "providers",
            "provider_options",
            "session_options",
        }
        & selection.options.keys()
    )


def test_resolve_preserves_false_force_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _resolve(monkeypatch, {"force_download": False})

    assert selection.options["force_download"] is False


def test_open_without_seed_constructs_generation_config_with_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, adapter = _install(monkeypatch)
    configured_seeds: list[int | None] = []

    class RecordingGenerationConfig:
        def __init__(
            self,
            *,
            steps: int = 5,
            speed: float = 1.05,
            seed: int | None = None,
        ) -> None:
            self.steps = steps
            self.speed = speed
            self.seed = seed
            configured_seeds.append(seed)

    fake.GenerationConfig = RecordingGenerationConfig
    request = EngineRequest(
        engine="supertonic",
        target_id="supertonic-3",
        language="en-US",
        voice="F1",
    )
    selection, diagnostics = adapter.resolve(request)
    assert diagnostics == ()
    assert "seed" not in selection.options

    with adapter.open(selection):
        pass

    assert configured_seeds[-1] is None


def test_renderprofile_round_trip_omits_default_seed_and_preserves_explicit_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    target = SynthesisTarget(
        engine="supertonic",
        id="supertonic-3",
        display_name="Supertonic 3",
        languages=("en",),
        voices=("F1",),
        default_voice="F1",
    )
    _install(monkeypatch, (target,))

    default_profile = create_render_profile(
        engine="supertonic",
        target_id="supertonic-3",
        language="en-US",
        voice="F1",
    )
    default_options = default_profile.roles["narrator"].binding["options"]
    assert "seed" not in default_options
    default_path = tmp_path / "default.renderprofile.toml"
    save_render_profile(default_profile, default_path)
    loaded_default = load_render_profile(default_path)
    assert loaded_default == default_profile
    assert "seed" not in loaded_default.roles["narrator"].binding["options"]
    assert "seed =" not in default_path.read_text(encoding="utf-8")

    explicit_profile = create_render_profile(
        engine="supertonic",
        target_id="supertonic-3",
        language="en-US",
        voice="F1",
        options={"seed": 17},
    )
    explicit_path = tmp_path / "explicit.renderprofile.toml"
    save_render_profile(explicit_profile, explicit_path)
    loaded_explicit = load_render_profile(explicit_path)
    assert loaded_explicit == explicit_profile
    assert loaded_explicit.roles["narrator"].binding["options"]["seed"] == 17
    assert "seed = 17" in explicit_path.read_text(encoding="utf-8")
