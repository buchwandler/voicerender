from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import voicerender
from voicerender.errors import RenderEventCallbackError
from voicerender.events import RenderEvent, RenderEventKind, emit_render_event


def test_control_surface_apis_are_exported_from_package_root() -> None:
    expected = {
        "EngineAssetManager",
        "EngineOption",
        "ModelArtifact",
        "ModelInstallRequest",
        "ModelInstallResult",
        "ModelManagementCapabilities",
        "ModelRemoveRequest",
        "ModelRemoveResult",
        "ModelVerifyResult",
        "ParsedVoiceSelector",
        "RoleProfile",
        "VoiceCloneRequest",
        "VoiceDescriptor",
        "VoiceKind",
        "BundleContractError",
        "BundleRenderResult",
        "clone_voice_profile",
        "create_voice_source",
        "engine_options",
        "inspect_voice",
        "install_model",
        "list_models",
        "list_profile_roles",
        "list_voices",
        "model_status",
        "parse_voice_selector",
        "remove_model",
        "remove_profile_role",
        "set_default_profile_role",
        "set_profile_role",
        "verify_model",
        "EtaConfidence",
        "RenderProgress",
        "RenderEstimate",
        "estimate_render_book",
        "estimate_renderplan",
        "RenderScope",
    }

    assert expected.issubset(set(voicerender.__all__))
    assert all(hasattr(voicerender, name) for name in expected)

    assert callable(voicerender.render_bundle)
    assert not hasattr(voicerender, "render" + "_book")
    assert not hasattr(voicerender, "render" + "_document")


def test_progress_types_and_event_dispatch_are_public_and_typed() -> None:
    assert hasattr(voicerender, "RenderProgress")
    snapshot = voicerender.RenderProgress(
        completed_count=1,
        total_count=2,
        progress=0.5,
        rendered_count=1,
        reused_count=0,
        failed_count=0,
        elapsed_seconds=1.25,
        eta_seconds=1.25,
        eta_confidence="low",
    )
    event = RenderEvent(
        kind=RenderEventKind.SEGMENT_RENDERED,
        scope="render",
        progress_snapshot=snapshot,
    )
    assert event.progress_snapshot is snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.progress = 0.75  # type: ignore[misc]


def test_event_dispatch_wraps_callback_errors() -> None:
    event = RenderEvent(kind=RenderEventKind.RENDER_STARTED)

    def broken_callback(_event: RenderEvent) -> None:
        raise ValueError("callback failed")

    with pytest.raises(RenderEventCallbackError) as error:
        emit_render_event(broken_callback, event)
    assert error.value.event_kind == "render_started"
    assert error.value.native_error_type == "ValueError"


def test_event_dispatch_is_noop_without_callback() -> None:
    emit_render_event(None, RenderEvent(kind=RenderEventKind.RENDER_STARTED))
