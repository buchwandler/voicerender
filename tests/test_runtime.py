from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any

import pytest

import voicerender
from voicerender import runtime
from voicerender.errors import RuntimeControlError


def _status(**changes: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "importable": True,
        "module_version": "1.20.0",
        "module_path": Path("/venv/lib/python/site-packages/onnxruntime/__init__.py"),
        "build_device": "CUDA",
        "build_info": "test build",
        "distributions": (
            SimpleNamespace(
                name="onnxruntime-gpu",
                version="1.20.0",
                location=Path("/venv/lib/python/site-packages"),
                owns_import=True,
            ),
        ),
        "imported_distribution": "onnxruntime-gpu",
        "distribution_state": "resolved",
        "available_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "configured_providers": ("cuda", "cpu"),
        "provider_source": "user_config",
        "resolved_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "import_error": None,
        "resolution_error": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_runtime_status_copies_every_field_to_voicerender_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _status()
    monkeypatch.setattr(
        runtime,
        "_onnxvoice",
        lambda: SimpleNamespace(runtime_status=lambda: source),
    )

    result = voicerender.onnx_runtime_status()

    assert type(result) is voicerender.OnnxRuntimeStatus
    assert dataclasses.asdict(result) == {
        "importable": True,
        "module_version": "1.20.0",
        "module_path": Path("/venv/lib/python/site-packages/onnxruntime/__init__.py"),
        "build_device": "CUDA",
        "build_info": "test build",
        "distributions": (
            {
                "name": "onnxruntime-gpu",
                "version": "1.20.0",
                "location": Path("/venv/lib/python/site-packages"),
                "owns_import": True,
            },
        ),
        "imported_distribution": "onnxruntime-gpu",
        "distribution_state": "resolved",
        "available_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "configured_providers": ("cuda", "cpu"),
        "provider_source": "user_config",
        "resolved_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "import_error": None,
        "resolution_error": None,
    }
    assert type(result.distributions[0]) is voicerender.RuntimeDistribution
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.provider_source = "explicit"  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("distribution_state", "importable", "imported_distribution", "resolution_error"),
    [
        ("ambiguous", True, None, None),
        ("vendor", True, None, None),
        ("not_installed", False, None, None),
        ("import_failed", False, None, "ImportError: failed to import"),
    ],
)
def test_runtime_status_preserves_distribution_and_error_states(
    monkeypatch: pytest.MonkeyPatch,
    distribution_state: str,
    importable: bool,
    imported_distribution: str | None,
    resolution_error: str | None,
) -> None:
    source = _status(
        distribution_state=distribution_state,
        importable=importable,
        imported_distribution=imported_distribution,
        resolution_error=resolution_error,
        import_error=resolution_error
        if distribution_state == "import_failed"
        else None,
        distributions=(
            SimpleNamespace(
                name="ort-a",
                version=None,
                location=None,
                owns_import=None,
            ),
            SimpleNamespace(
                name="ort-b",
                version="1.0",
                location=Path("/vendor"),
                owns_import=True,
            ),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_onnxvoice",
        lambda: SimpleNamespace(runtime_status=lambda: source),
    )

    result = runtime.onnx_runtime_status()

    assert result.distribution_state == distribution_state
    assert result.importable is importable
    assert result.imported_distribution == imported_distribution
    assert result.resolution_error == resolution_error
    assert result.distributions == (
        voicerender.RuntimeDistribution("ort-a", None, None, None),
        voicerender.RuntimeDistribution("ort-b", "1.0", Path("/vendor"), True),
    )


@pytest.mark.parametrize("providers", ["cpu", "cuda", "auto", "cuda,cpu"])
def test_set_providers_passes_request_unchanged(
    monkeypatch: pytest.MonkeyPatch, providers: str
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        runtime,
        "_onnxvoice",
        lambda: SimpleNamespace(
            set_default_providers=lambda value: calls.append(value)
        ),
    )

    assert voicerender.set_onnx_runtime_providers(providers) is None
    assert calls == [providers]


def test_set_providers_preserves_ordered_sequence_without_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers = ["cuda", "cpu"]
    calls: list[object] = []
    monkeypatch.setattr(
        runtime,
        "_onnxvoice",
        lambda: SimpleNamespace(
            set_default_providers=lambda value: calls.append(value)
        ),
    )

    voicerender.set_onnx_runtime_providers(providers)

    assert calls[0] is providers


def test_clear_providers_delegates_to_onnxvoice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_onnxvoice",
        lambda: SimpleNamespace(clear_default_providers=lambda: calls.append("clear")),
    )

    voicerender.clear_onnx_runtime_providers()

    assert calls == ["clear"]


def test_runtime_errors_are_stable_and_keep_provider_validation_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(_providers: object) -> None:
        raise ValueError("CUDAExecutionProvider is unavailable")

    monkeypatch.setattr(
        runtime,
        "_onnxvoice",
        lambda: SimpleNamespace(set_default_providers=reject),
    )

    with pytest.raises(RuntimeControlError) as caught:
        voicerender.set_onnx_runtime_providers("cuda")

    assert caught.value.code == "runtime.control_failed"
    assert caught.value.details == {
        "operation": "set_default_providers",
        "native_error_type": "ValueError",
        "provider_validation_message": "CUDAExecutionProvider is unavailable",
    }
    assert "CUDAExecutionProvider is unavailable" in str(caught.value)
    assert caught.value.__cause__ is None


def test_missing_onnxvoice_is_reported_as_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise ModuleNotFoundError("onnxvoice")

    monkeypatch.setattr(runtime, "_onnxvoice", missing)

    with pytest.raises(RuntimeControlError) as caught:
        voicerender.onnx_runtime_status()

    assert caught.value.code == "runtime.onnxvoice_unavailable"
    assert caught.value.details["operation"] == "runtime_status"
    assert "install a compatible onnxvoice" in str(caught.value)


def test_missing_onnxvoice_api_is_reported_as_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_onnxvoice", lambda: SimpleNamespace())

    with pytest.raises(RuntimeControlError) as caught:
        voicerender.clear_onnx_runtime_providers()

    assert caught.value.code == "runtime.onnxvoice_incompatible"
    assert caught.value.details["native_error_type"] == "AttributeError"


def test_importing_voicerender_does_not_import_onnxvoice_or_onnxruntime() -> None:
    script = dedent(
        """
        import importlib.abc
        import sys

        class RejectOnnxRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] == "onnxruntime":
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, RejectOnnxRuntime())
        import voicerender

        assert "onnxvoice" not in sys.modules
        assert "onnxruntime" not in sys.modules
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_project_requires_lightweight_onnxvoice_without_ort_distribution() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]

    assert "onnxvoice>=0.2.12,<0.3" in dependencies
    assert not any(
        "onnxruntime" in dependency.casefold() for dependency in dependencies
    )
