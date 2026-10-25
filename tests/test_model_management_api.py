from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from voicerender.engines.assets import (
    ModelArtifact,
    ModelInstallRequest,
    ModelInstallResult,
    ModelManagementCapabilities,
    ModelRemoveRequest,
    ModelRemoveResult,
    ModelVerifyResult,
)
from voicerender.models_api import (
    install_model,
    list_models,
    model_artifact_to_dict,
    model_result_to_dict,
    model_status,
    remove_model,
    verify_model,
)


def test_model_listing_projects_discovery_without_opening_synthesis_runtime(
    monkeypatch,
) -> None:
    target = SimpleNamespace(
        engine="kokoro",
        id="v1.0",
        display_name="Kokoro v1",
        runtime_available=True,
        status="ready",
        languages=("en-US",),
        qualities=("fp32",),
        metadata={},
    )
    monkeypatch.setattr("voicerender.models_api.get_asset_manager", lambda engine: None)
    monkeypatch.setattr(
        "voicerender.models_api.discover_targets", lambda *a, **k: (target,)
    )
    monkeypatch.setattr(
        "voicerender.engines.registry.open_session",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("synthesis session opened")
        ),
    )

    models = list_models("kokoro", installed=True)

    assert models == (
        ModelArtifact(
            engine="kokoro",
            target_id="v1.0",
            display_name="Kokoro v1",
            installed=True,
            status="ready",
            languages=("en-US",),
            qualities=("fp32",),
            installed_qualities=("fp32",),
        ),
    )


def test_installed_profiles_project_exactly_for_listing_and_inspection(
    monkeypatch,
) -> None:
    qualities = ("fp32", "q4f16")
    inventories = ((), ("q4f16",), ("fp32",), ("fp32", "q4f16"))
    monkeypatch.setattr(
        "voicerender.models_api.get_asset_manager", lambda _engine: None
    )

    for installed_profiles in inventories:
        target = SimpleNamespace(
            engine="chatterbox",
            id="turbo",
            display_name="Chatterbox Turbo",
            runtime_available=bool(installed_profiles),
            status="ready",
            languages=("en",),
            qualities=qualities,
            metadata={"installed_profiles": installed_profiles},
        )
        monkeypatch.setattr(
            "voicerender.models_api.discover_targets",
            lambda *_args, target=target, **_kwargs: (target,),
        )

        listed = list_models("chatterbox")
        expected = tuple(
            quality for quality in qualities if quality in installed_profiles
        )
        assert listed[0].qualities == qualities
        assert listed[0].installed_qualities == expected
        assert listed[0].installed is bool(expected)
        for quality in qualities:
            inspected = model_status("chatterbox", "turbo", quality=quality)
            assert inspected.installed is (quality in expected)
            assert inspected.installed_qualities == expected


def test_unsupported_install_has_stable_engine_operation_diagnostic(
    monkeypatch,
) -> None:
    monkeypatch.setattr("voicerender.models_api.get_asset_manager", lambda engine: None)

    result = install_model(ModelInstallRequest("kokoro", "v1.0"))

    assert result.status == "unsupported"
    assert result.changed is False
    assert result.diagnostics == (
        "Engine 'kokoro' does not support model install for target 'v1.0'",
    )


def test_lazy_chatterbox_manager_installs_through_public_download_api(
    monkeypatch, tmp_path: Path
) -> None:
    import sys
    from types import ModuleType

    from voicerender.engines import registry
    from voicerender.engines.assets import ModelInstallRequest, ModelRemoveRequest
    from voicerender.models_api import (
        install_model,
        list_models,
        model_status,
        remove_model,
        verify_model,
    )

    installed_profiles: list[str] = []
    download_calls: list[tuple[str, str, Path | None, bool]] = []

    def target() -> SimpleNamespace:
        return SimpleNamespace(
            engine="chatterbox",
            id="turbo",
            display_name="Chatterbox Turbo",
            languages=("en",),
            qualities=("fp32", "q4f16"),
            status="ready",
            runtime_available=bool(installed_profiles),
            metadata={
                "default_profile": "fp32",
                "profiles": ("fp32", "q4f16"),
                "installed_profiles": tuple(installed_profiles),
                "source_revision": "catalog-rev",
            },
        )

    def discover(_engine: str, **_kwargs: object) -> tuple[SimpleNamespace, ...]:
        return (target(),)

    monkeypatch.setattr(registry, "_ASSET_MANAGERS", {})
    monkeypatch.delitem(sys.modules, "chatterboxsynth", raising=False)
    registry.list_engines()
    assert "chatterboxsynth" not in sys.modules

    manager = registry.get_asset_manager("chatterbox")
    assert manager is not None
    assert manager.capabilities().can_install
    assert "chatterboxsynth" not in sys.modules

    from voicerender.engines import chatterbox_assets

    monkeypatch.setattr(chatterbox_assets, "discover_targets", discover)

    class FakeRuntime:
        @staticmethod
        def download(
            target_id: str,
            *,
            quality: str,
            cache_dir: Path | None,
            force_download: bool,
        ) -> None:
            download_calls.append((target_id, quality, cache_dir, force_download))
            if quality == "fp32":
                raise RuntimeError("backend download failed")
            installed_profiles.append(quality)

    backend = ModuleType("chatterboxsynth")
    backend.ChatterboxRuntime = FakeRuntime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chatterboxsynth", backend)

    listed = list_models("chatterbox")
    assert len(listed) == 1
    assert listed[0].installed is False
    assert listed[0].qualities == ("fp32", "q4f16")

    result = install_model(
        ModelInstallRequest(
            "chatterbox", "turbo", "q4f16", cache_dir=tmp_path, force=True
        )
    )
    assert result.status == "installed"
    assert result.changed is True
    assert result.artifact is not None
    assert result.artifact.installed_qualities == ("q4f16",)
    assert download_calls == [("turbo", "q4f16", tmp_path, True)]

    repeated = install_model(
        ModelInstallRequest("chatterbox", "turbo", "q4f16", cache_dir=tmp_path)
    )
    assert repeated.status == "installed"
    assert repeated.changed is False
    assert repeated.artifact is not None
    assert repeated.artifact.installed_qualities == ("q4f16",)
    assert download_calls == [
        ("turbo", "q4f16", tmp_path, True),
        ("turbo", "q4f16", tmp_path, False),
    ]

    assert model_status("chatterbox", "turbo", quality="fp32").installed is False
    assert model_status("chatterbox", "turbo", quality="q4f16").installed is True
    assert model_status("chatterbox", "turbo").installed_qualities == ("q4f16",)

    invalid_target = install_model(
        ModelInstallRequest("chatterbox", "not-canonical", "q4f16")
    )
    invalid_quality = install_model(
        ModelInstallRequest("chatterbox", "turbo", "unknown")
    )
    empty_quality = install_model(ModelInstallRequest("chatterbox", "turbo", ""))
    assert (
        invalid_target.status
        == invalid_quality.status
        == empty_quality.status
        == "failed"
    )
    assert download_calls == [
        ("turbo", "q4f16", tmp_path, True),
        ("turbo", "q4f16", tmp_path, False),
    ]

    failed = install_model(ModelInstallRequest("chatterbox", "turbo", "fp32"))
    assert failed.status == "failed"
    assert "backend download failed" in failed.diagnostics[0]
    sourced = install_model(
        ModelInstallRequest("chatterbox", "turbo", source="alternate")
    )
    removed = remove_model(ModelRemoveRequest("chatterbox", "turbo"))
    verified = verify_model("chatterbox", "turbo")
    assert sourced.status == removed.status == verified.status == "unsupported"


def test_install_delegates_to_registered_public_asset_manager(
    monkeypatch, tmp_path: Path
) -> None:
    class Manager:
        called = None

        def capabilities(self):
            return ModelManagementCapabilities(can_install=True, supports_quality=True)

        def install_model(self, request):
            self.called = request
            return ModelInstallResult("kokoro", request.target_id, "installed", True)

    manager = Manager()
    monkeypatch.setattr(
        "voicerender.models_api.get_asset_manager", lambda engine: manager
    )

    result = install_model(
        ModelInstallRequest("kokoro", "v1.0", "fp32", cache_dir=tmp_path, force=True)
    )

    assert result.status == "installed"
    assert manager.called == ModelInstallRequest(
        "kokoro", "v1.0", "fp32", cache_dir=tmp_path, force=True
    )


def test_remove_and_verify_delegate_to_public_asset_manager(
    monkeypatch, tmp_path: Path
) -> None:
    class Manager:
        def capabilities(self):
            return ModelManagementCapabilities(can_remove=True, can_verify=True)

        def remove_model(self, request):
            return ModelRemoveResult("kokoro", request.target_id, "removed", True)

        def verify_model(self, target_id, *, quality=None, cache_dir=None):
            assert cache_dir == tmp_path
            return ModelVerifyResult("kokoro", target_id, "verified", True)

    monkeypatch.setattr(
        "voicerender.models_api.get_asset_manager", lambda engine: Manager()
    )

    removed = remove_model(ModelRemoveRequest("kokoro", "v1.0", cache_dir=tmp_path))
    verified = verify_model("kokoro", "v1.0", cache_dir=tmp_path)

    assert removed.status == "removed" and removed.changed
    assert verified.status == "verified" and verified.verified


def test_model_artifact_serialization_does_not_use_deepcopy_asdict_semantics() -> None:
    model = ModelArtifact(
        engine="sopro",
        target_id="model-v1",
        display_name="Sopro model v1",
        installed=True,
        status="ready",
        languages=("en", "de"),
        path=Path("/models/model-v1"),
        metadata=MappingProxyType(
            {
                "backend": "onnx",
                "catalog": MappingProxyType(
                    {
                        "profile": MappingProxyType(
                            {
                                "providers": ("CPUExecutionProvider",),
                                "cache_path": Path("/models/cache"),
                            }
                        )
                    }
                ),
            }
        ),
    )

    payload = model_artifact_to_dict(model)

    assert payload["path"] == "/models/model-v1"
    assert payload["metadata"] == {
        "backend": "onnx",
        "catalog": {
            "profile": {
                "providers": ["CPUExecutionProvider"],
                "cache_path": "/models/cache",
            }
        },
    }
    json.dumps(payload)


def test_model_result_to_dict_serializes_nested_read_only_artifact_metadata() -> None:
    model = ModelArtifact(
        engine="sopro",
        target_id="model-v1",
        display_name="Sopro model v1",
        installed=True,
        status="ready",
        metadata=MappingProxyType(
            {"catalog": MappingProxyType({"profile": ("uint8",)})}
        ),
    )
    result = ModelInstallResult("sopro", "model-v1", "installed", True, artifact=model)

    payload = model_result_to_dict(result)

    assert payload["artifact"]["metadata"] == {"catalog": {"profile": ["uint8"]}}
    json.dumps(payload)


def test_model_artifact_serializer_error_identifies_metadata_path() -> None:
    model = ModelArtifact(
        engine="sopro",
        target_id="model-v1",
        display_name="Sopro model v1",
        installed=True,
        status="ready",
        metadata={"catalog": {"profile": {"session": object()}}},
    )

    with pytest.raises(
        TypeError,
        match=(
            r"model 'sopro' target 'model-v1' metadata at catalog\.profile\.session "
            r"contains unsupported JSON value object"
        ),
    ):
        model_artifact_to_dict(model)
