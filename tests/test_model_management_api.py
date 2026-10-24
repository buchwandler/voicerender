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
