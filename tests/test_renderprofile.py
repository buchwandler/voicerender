from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from renderplan import (
    CapacityConstraint,
    PausePolicy,
    PlanningPolicy,
    RenderPlan,
    RenderProfile,
    RoleProfile,
    RoleRequirements,
    SourceSpan,
    SpeechItem,
    load_plan,
    save_plan,
)

from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineSelection,
    RenderProfileFragment,
)
from voicerender.engines.kittensynth import KittenSynthEngineAdapter
from voicerender.engines.profiles import make_renderprofile_fragment
from voicerender.engines.registry import _ADAPTERS
from voicerender.engines.selection import EngineRequest
from voicerender.errors import ProfileError
from voicerender.renderprofile import (
    create_render_profile,
    load_render_profile,
    refresh_render_profile,
    save_render_profile,
    validate_render_profile,
)
from voicerender.voice_sources import local_reference_source


class _ProfileAdapter:
    id = "profile-test"
    package_name = "profile-test"

    def __init__(self) -> None:
        self.maximum = 96
        self.source_revision = "model-r1"

    def probe_api(self) -> EngineApiProbe:
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            distribution_version="1.0",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(id=self.id, supports_reference_voice=True)

    def version(self) -> str:
        return "1.0"

    def resolve(self, request: EngineRequest):
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or "target-v1",
                language=request.language or "en-US",
                voice=request.voice
                or ("voice-a" if "voice_source" not in request.options else None),
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection):
        return ()

    def target_metadata(self, selection: EngineSelection) -> dict[str, Any]:
        return {
            "max_tokens": self.maximum,
            "source_revision": self.source_revision,
            "capacity_policy_revision": "budget.v1",
        }

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            token_fields=("span", "pos"),
            frontend="profile-test.frontend",
            pronunciation="profile-test.pronunciation",
            capacity_key="max_tokens",
            capacity_metric="model_tokens",
        )

    @contextmanager
    def open(self, selection: EngineSelection):
        yield object()


class _KokoroProfileAdapter(_ProfileAdapter):
    id = "kokoro"
    package_name = "kokoro"


def _install(monkeypatch: pytest.MonkeyPatch) -> _ProfileAdapter:
    adapter = _ProfileAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    return adapter


def test_profile_create_uses_adapter_facts_and_shared_codec(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install(monkeypatch)
    profile = create_render_profile(
        engine="profile-test",
        target_id="target-v1",
        language="en-US",
        role="narrator",
        voice="voice-a",
    )

    assert profile.profile_id == profile.computed_id()
    assert profile.default_role == "narrator"
    role = profile.roles["narrator"]
    assert role.binding["renderer"] == "profile-test"
    assert role.binding["target_id"] == "target-v1"
    assert role.binding["voice"] == "voice-a"
    assert role.binding["renderer_metadata"]["source_revision"] == "model-r1"
    assert role.binding["renderer_metadata"]["capacity_policy_revision"] == "budget.v1"
    assert role.requirements == RoleRequirements(
        token_fields=("span", "pos"),
        frontend="profile-test.frontend",
        pronunciation="profile-test.pronunciation",
    )
    assert role.renderer_capacity == CapacityConstraint(
        metric="model_tokens", maximum=96, guarantee="hard"
    )
    assert profile.planning == PlanningPolicy(
        split="sentence", capacity_fallback=("clause", "token")
    )
    assert profile.pauses == PausePolicy(
        mode="manual",
        durations_ms={
            "clause": 180,
            "parenthetical": 180,
            "sentence": 500,
            "paragraph": 800,
        },
    )

    path = tmp_path / "profile.renderprofile.toml"
    save_render_profile(profile, path)
    assert load_render_profile(path) == profile
    assert validate_render_profile(path) == profile


@pytest.mark.parametrize("engine", ("kokoro", "pykokoro"))
def test_new_profiles_default_to_parenthetical_pause(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
    adapter = _KokoroProfileAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)

    profile = create_render_profile(
        engine=engine,
        target_id="target-v1",
        language="en-US",
        voice="voice-a",
    )

    assert profile.pauses == PausePolicy(
        mode="manual",
        durations_ms={
            "clause": 180,
            "parenthetical": 180,
            "sentence": 500,
            "paragraph": 800,
        },
    )


def test_refresh_updates_only_engine_facts_and_preserves_user_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install(monkeypatch)
    generated = create_render_profile(
        engine=adapter.id,
        target_id="target-v1",
        language="en-US",
        role="narrator",
        voice="voice-a",
    )
    authored = replace(
        generated,
        planning=PlanningPolicy(split="clause", capacity_fallback=("token",)),
        pauses=PausePolicy(mode="none"),
        profile_id=None,
    ).with_identity()
    adapter.maximum = 48
    adapter.source_revision = "model-r2"
    with pytest.raises(
        ProfileError, match="engine-derived requirements/capacity are stale"
    ):
        validate_render_profile(authored)

    proposal = refresh_render_profile(authored)
    updated = proposal.roles["narrator"]
    assert proposal.planning == authored.planning
    assert proposal.pauses == authored.pauses
    assert proposal.default_role == authored.default_role
    assert updated.binding["renderer"] == "profile-test"
    assert updated.binding["target_id"] == "target-v1"
    assert updated.binding["voice"] == "voice-a"
    assert updated.binding["options"] == authored.roles["narrator"].binding["options"]
    assert updated.binding["renderer_metadata"]["source_revision"] == "model-r2"
    assert updated.renderer_capacity == CapacityConstraint(
        metric="model_tokens", maximum=48, guarantee="hard"
    )
    assert authored.roles["narrator"].renderer_capacity.maximum == 96
    assert validate_render_profile(proposal) == proposal


def test_profile_creation_accepts_content_pinned_local_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install(monkeypatch)
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"reference wav")
    source = local_reference_source(reference)

    profile = create_render_profile(
        engine="profile-test",
        target_id="target-v1",
        language="en-US",
        voice_source=source,
    )

    role = profile.roles["narrator"]
    assert role.binding["options"]["voice_source"] == source
    assert "voice" not in role.binding
    assert validate_render_profile(profile) == profile
    refreshed = refresh_render_profile(profile)
    assert refreshed.roles["narrator"].binding["options"]["voice_source"] == source

    with pytest.raises(ProfileError, match="voice/speaker and voice_source"):
        create_render_profile(
            engine="profile-test",
            target_id="target-v1",
            language="en-US",
            voice="voice-a",
            voice_source=source,
        )


def test_qwen_origin_survives_profile_and_renderplan_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install(monkeypatch)
    reference = tmp_path / "designed.wav"
    reference.write_bytes(b"materialized voice design")
    source = local_reference_source(reference)
    source["origin"] = {
        "schema": "voicerender.voice-origin.qwen3-design.v1",
        "generator": "qwen3",
        "target_id": "voice-design-1.7b",
        "model_revision": "revision-1",
        "qwen3synth_version": "0.1.0",
        "onnxvoice_version": "0.1.0",
        "prompt": "A calm speaking voice.",
        "reference_text": "This exact generated sample.",
        "language": "en",
        "quality": "int4",
        "provider": "cpu",
        "temperature": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 128,
        "seed": 11,
        "model_id": "wavekat/voice-design-1.7b",
        "runtime_identity": {
            "engine": "qwen3synth",
            "engine_version": "0.1.0",
            "request_api_version": 1,
            "onnxvoice_version": "0.1.0",
            "layouts": ["qwen3-wavekat-v1"],
            "sample_rate": 24000,
        },
    }

    profile = create_render_profile(
        engine="profile-test",
        target_id="target-v1",
        language="en-US",
        voice_source=source,
    )
    profile_path = tmp_path / "narrator.renderprofile.toml"
    save_render_profile(profile, profile_path)
    loaded_profile = load_render_profile(profile_path)
    assert loaded_profile.roles["narrator"].binding["options"]["voice_source"] == source

    text = "A sentence using the designed voice."
    plan = RenderPlan(
        source_plan_id="sha256:" + "1" * 64,
        render_profile_id=profile.profile_id,
        flow=(
            SpeechItem(
                number=1,
                text=text,
                language="en-US",
                role="narrator",
                binding=loaded_profile.roles["narrator"].binding,
                source=(SourceSpan(locator="source", start=0, end=len(text)),),
                planning_level="sentence",
                preflight_required=True,
            ),
        ),
    ).with_identity()
    plan_path = tmp_path / "narration.renderplan.toml"
    save_plan(plan, plan_path)
    loaded_plan = load_plan(plan_path)
    assert loaded_plan.speech[0].binding["options"]["voice_source"] == source


def test_profile_validation_rejects_reference_audio_from_imported_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    role = RoleProfile(
        language="en-US",
        binding={
            "renderer": "profile-test",
            "target_id": "target-v1",
            "voice_source": {"kind": "reference", "value": "old-profile-relative.wav"},
            "options": {},
        },
        requirements=RoleRequirements(),
    )
    profile = RenderProfile(roles={"narrator": role}, default_role="narrator")
    with pytest.raises(
        ProfileError, match="voice_source must be located in binding.options"
    ):
        validate_render_profile(profile)


def test_kokoro_fragment_uses_discovered_capacity_and_exact_token_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicerender.engines.kokoro import PyKokoroEngineAdapter

    adapter = PyKokoroEngineAdapter()
    monkeypatch.setattr(
        adapter,
        "target_metadata",
        lambda selection: {"max_tokens": 510, "source_revision": "kokoro-r4"},
    )
    selection = EngineSelection(
        engine="kokoro",
        target_id="kokoro-v1",
        language="en-US",
        voice="af_heart",
    )

    fragment = adapter.renderprofile_fragment(selection)

    assert fragment.token_fields == ("span", "pos", "tag")
    assert fragment.renderer_capacity == CapacityConstraint(
        metric="model_tokens", maximum=510, guarantee="hard"
    )
    assert fragment.planner_capacity is None
    assert fragment.binding["renderer"] == "kokoro"
    assert fragment.binding["voice"] == "af_heart"
    assert fragment.binding["renderer_metadata"]["source_revision"] == "kokoro-r4"


def test_kittensynth_authoritative_capacity_survives_profile_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = KittenSynthEngineAdapter()
    metadata = {
        "max_tokens": 256,
        "capacity_policy_revision": "kitten.capacity.v1",
    }
    probe = EngineApiProbe(
        engine=adapter.id,
        package=adapter.package_name,
        compatible=True,
        status="ready",
        expected_api_version=1,
        api_version=1,
        contract_source="explicit",
    )
    monkeypatch.setattr(adapter, "probe_api", lambda: probe)
    monkeypatch.setattr(adapter, "validate_selection", lambda selection: ())
    monkeypatch.setattr(adapter, "target_metadata", lambda selection: dict(metadata))
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)

    profile = create_render_profile(
        engine="kitten",
        target_id="nano-0.8-int8",
        language="en-US",
        role="narrator",
        voice="Jasper",
    )
    assert adapter.capabilities().supports_linguistic_tokens is True
    assert profile.roles["narrator"].requirements.token_fields == ()
    new_profile_path = tmp_path / "kitten-new.renderprofile.toml"
    save_render_profile(profile, new_profile_path)
    assert "token_fields = []" in new_profile_path.read_text(encoding="utf-8")

    old_role = replace(
        profile.roles["narrator"],
        requirements=replace(
            profile.roles["narrator"].requirements,
            token_fields=("span", "pos", "tag", "lemma", "morph"),
        ),
    )
    old_profile = replace(
        profile,
        roles={"narrator": old_role},
        profile_id=None,
    ).with_identity()
    metadata.update(
        max_tokens=128,
        capacity_policy_revision="kitten.capacity.v2",
    )
    refreshed = refresh_render_profile(old_profile)

    role = refreshed.roles["narrator"]
    assert refreshed.profile_id != old_profile.profile_id
    assert role.requirements.token_fields == ()
    refreshed_path = tmp_path / "kitten-refreshed.renderprofile.toml"
    save_render_profile(refreshed, refreshed_path)
    assert "token_fields = []" in refreshed_path.read_text(encoding="utf-8")
    assert role.renderer_capacity == CapacityConstraint(
        metric="model_tokens", maximum=128, guarantee="hard"
    )
    assert role.binding["renderer_metadata"]["max_tokens"] == 128
    assert (
        role.binding["renderer_metadata"]["capacity_policy_revision"]
        == "kitten.capacity.v2"
    )
