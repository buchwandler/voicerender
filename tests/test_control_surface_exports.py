from __future__ import annotations

import voicerender


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
    }

    assert expected.issubset(set(voicerender.__all__))
    assert all(hasattr(voicerender, name) for name in expected)
