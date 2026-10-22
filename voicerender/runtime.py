"""Stable VoiceRender controls for ONNXVoice-managed ONNX Runtime policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import RuntimeControlError


@dataclass(frozen=True, slots=True)
class RuntimeDistribution:
    """A Python distribution that may provide the imported ONNX Runtime module."""

    name: str
    version: str | None
    location: Path | None
    owns_import: bool | None


@dataclass(frozen=True, slots=True)
class OnnxRuntimeStatus:
    """Snapshot of ONNX Runtime availability and configured provider policy."""

    importable: bool
    module_version: str | None
    module_path: Path | None
    build_device: str | None
    build_info: str | None
    distributions: tuple[RuntimeDistribution, ...]
    imported_distribution: str | None
    distribution_state: str
    available_providers: tuple[str, ...]
    configured_providers: tuple[str, ...]
    provider_source: str
    resolved_providers: tuple[str, ...]
    import_error: str | None = None
    resolution_error: str | None = None


def _onnxvoice() -> Any:
    """Import ONNXVoice only when a runtime-control operation is requested."""
    import onnxvoice

    return onnxvoice


class _RuntimeApiUnavailableError(RuntimeError):
    """ONNXVoice does not expose one of the documented runtime functions."""


def _runtime_error(operation: str, error: Exception) -> RuntimeControlError:
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        code = "runtime.onnxvoice_unavailable"
        message = "ONNXVoice is unavailable; install a compatible onnxvoice package."
    elif isinstance(error, (AttributeError, _RuntimeApiUnavailableError)):
        code = "runtime.onnxvoice_incompatible"
        message = (
            "The installed ONNXVoice package lacks a required runtime-control API."
        )
    else:
        code = "runtime.control_failed"
        detail = str(error).strip() or type(error).__name__
        message = f"ONNXVoice runtime operation '{operation}' failed: {detail}"

    details: dict[str, object] = {
        "operation": operation,
        "native_error_type": type(error).__name__,
    }
    if operation == "set_default_providers":
        details["provider_validation_message"] = str(error)
    return RuntimeControlError(message, code=code, details=details)


def _invoke(name: str, *args: object) -> Any:
    try:
        api = getattr(_onnxvoice(), name)
        if not callable(api):
            raise _RuntimeApiUnavailableError(
                f"ONNXVoice public API {name!r} is not callable"
            )
        return api(*args)
    except RuntimeControlError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _runtime_error(name, exc) from None


def _as_path(value: Any) -> Path | None:
    return None if value is None else Path(value)


def _convert_status(status: Any) -> OnnxRuntimeStatus:
    distributions = tuple(
        RuntimeDistribution(
            name=item.name,
            version=item.version,
            location=_as_path(item.location),
            owns_import=item.owns_import,
        )
        for item in status.distributions
    )
    return OnnxRuntimeStatus(
        importable=status.importable,
        module_version=status.module_version,
        module_path=_as_path(status.module_path),
        build_device=status.build_device,
        build_info=status.build_info,
        distributions=distributions,
        imported_distribution=status.imported_distribution,
        distribution_state=status.distribution_state,
        available_providers=tuple(status.available_providers),
        configured_providers=tuple(status.configured_providers),
        provider_source=status.provider_source,
        resolved_providers=tuple(status.resolved_providers),
        import_error=status.import_error,
        resolution_error=status.resolution_error,
    )


def onnx_runtime_status() -> OnnxRuntimeStatus:
    """Inspect ONNX Runtime and ONNXVoice provider policy without loading a TTS model."""
    status = _invoke("runtime_status")
    try:
        return _convert_status(status)
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        raise _runtime_error("runtime_status", exc) from None


def set_onnx_runtime_providers(providers: str | Sequence[str]) -> None:
    """Persist an ONNXVoice provider preference for future compatible sessions."""
    _invoke("set_default_providers", providers)


def clear_onnx_runtime_providers() -> None:
    """Clear the persisted ONNXVoice provider preference."""
    _invoke("clear_default_providers")
