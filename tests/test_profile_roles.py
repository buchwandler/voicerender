from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from renderplan import (
    PausePolicy,
    PlanningPolicy,
    RenderProfile,
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
from voicerender.profile_roles import (
    list_profile_roles,
    remove_profile_role,
    set_default_profile_role,
    set_profile_role,
)
from voicerender.renderprofile import create_render_profile


class _RoleAdapter:
    id = "role-test"
    package_name = "role-test"

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
        if selection.options.get("bad_option"):
            return (
                SimpleNamespace(
                    severity="error",
                    code="option.invalid",
                    message="invalid option value",
                ),
            )
        return ()

    def target_metadata(self, selection: EngineSelection) -> dict[str, Any]:
        return {"max_tokens": 96, "source_revision": "r1"}

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            token_fields=("span",),
            frontend="role-test.frontend",
            pronunciation="role-test.pronunciation",
            capacity_key="max_tokens",
            capacity_metric="model_tokens",
        )

    @contextmanager
    def open(self, selection: EngineSelection):
        yield object()


def _install(monkeypatch: pytest.MonkeyPatch) -> _RoleAdapter:
    adapter = _RoleAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    return adapter


def _profile(monkeypatch: pytest.MonkeyPatch) -> RenderProfile:
    _install(monkeypatch)
    narrator = create_render_profile(
        engine="role-test",
        target_id="target-v1",
        language="en-US",
        role="narrator",
        voice="voice-a",
    )
    guest = create_render_profile(
        engine="role-test",
        target_id="target-v1",
        language="en-GB",
        role="guest",
        voice="voice-b",
    )
    return replace(
        narrator,
        roles={"narrator": narrator.roles["narrator"], "guest": guest.roles["guest"]},
        planning=PlanningPolicy(split="clause", capacity_fallback=("token",)),
        pauses=PausePolicy(mode="none"),
        profile_id=None,
    ).with_identity()


def test_list_roles_and_set_new_role_preserve_policy_and_existing_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch)
    original_narrator = profile.roles["narrator"]

    updated = set_profile_role(
        profile,
        "announcer",
        engine="role-test",
        target_id="target-v1",
        language="en-US",
        voice="voice-c",
        options={"style": "warm"},
    )

    assert list_profile_roles(profile) == profile.roles
    assert updated.profile_id == updated.computed_id()
    assert updated.profile_id != profile.profile_id
    assert updated.planning == profile.planning
    assert updated.pauses == profile.pauses
    assert updated.default_role == profile.default_role
    assert updated.roles["narrator"] is original_narrator
    assert updated.roles["announcer"].binding["voice"] == "voice-c"
    assert updated.roles["announcer"].binding["options"] == {"style": "warm"}
    assert (
        updated.roles["announcer"].renderer_capacity
        == original_narrator.renderer_capacity
    )


def test_set_existing_role_can_update_default_and_recomputes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch)

    updated = set_profile_role(
        profile,
        "guest",
        engine="role-test",
        target_id="target-v1",
        language="en-US",
        voice="voice-new",
        make_default=True,
    )

    assert updated.default_role == "guest"
    assert updated.profile_id == updated.computed_id()
    assert updated.profile_id != profile.profile_id
    assert updated.roles["guest"].binding["voice"] == "voice-new"
    assert updated.planning == profile.planning
    assert updated.pauses == profile.pauses


def test_role_mutation_rejects_invalid_engine_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch)

    with pytest.raises(ProfileError, match="option.invalid: invalid option value"):
        set_profile_role(
            profile,
            "guest",
            engine="role-test",
            target_id="target-v1",
            language="en-US",
            options={"bad_option": True},
        )


def test_remove_role_requires_valid_default_and_recomputes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch)

    with pytest.raises(ProfileError, match="requires --new-default-role"):
        remove_profile_role(profile, "narrator")
    with pytest.raises(ProfileError, match="not present after removal"):
        remove_profile_role(profile, "narrator", new_default_role="missing")

    updated = remove_profile_role(profile, "narrator", new_default_role="guest")
    assert set(updated.roles) == {"guest"}
    assert updated.default_role == "guest"
    assert updated.profile_id == updated.computed_id()
    assert updated.profile_id != profile.profile_id
    assert updated.planning == profile.planning
    assert updated.pauses == profile.pauses


def test_remove_final_or_unknown_role_and_set_unknown_default_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch)
    single = RenderProfile(
        roles={"narrator": profile.roles["narrator"]}, default_role="narrator"
    ).with_identity()

    with pytest.raises(ProfileError, match="final RenderProfile role"):
        remove_profile_role(single, "narrator")
    with pytest.raises(ProfileError, match="no role 'missing'"):
        remove_profile_role(profile, "missing")
    with pytest.raises(ProfileError, match="no role 'missing'"):
        set_default_profile_role(profile, "missing")


def test_default_role_mutation_preserves_roles_and_refreshes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch)

    updated = set_default_profile_role(profile, "guest")

    assert updated.default_role == "guest"
    assert updated.profile_id == updated.computed_id()
    assert updated.profile_id != profile.profile_id
    assert updated.roles == profile.roles
    assert updated.planning == profile.planning
    assert updated.pauses == profile.pauses
    assert set_default_profile_role(updated, "guest") is updated
