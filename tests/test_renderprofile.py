from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
from renderplan import (
    CapacityConstraint,
    PausePolicy,
    PlanningPolicy,
    RenderProfile,
    RoleProfile,
    RoleRequirements,
)

from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineSelection,
    RenderProfileFragment,
)
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
        return EngineCapabilities(id=self.id)

    def version(self) -> str:
        return "1.0"

    def resolve(self, request: EngineRequest):
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or "target-v1",
                language=request.language or "en-US",
                voice=request.voice or "voice-a",
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
        durations_ms={"clause": 180, "sentence": 500, "paragraph": 800},
    )

    path = tmp_path / "profile.renderprofile.toml"
    save_render_profile(profile, path)
    assert load_render_profile(path) == profile
    assert validate_render_profile(path) == profile


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


def test_reference_audio_bindings_are_rejected_precisely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    with pytest.raises(
        ProfileError, match="Reference-audio bindings are not supported"
    ):
        create_render_profile(
            engine="profile-test",
            target_id="target-v1",
            language="en-US",
            options={
                "voice_source": {
                    "kind": "reference",
                    "value": "/tmp/profile-relative.wav",
                }
            },
        )


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
        ProfileError, match="Reference-audio bindings are not supported"
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
