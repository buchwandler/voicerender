"""Structured diagnostics for public synthesis-engine API compatibility."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)
SUPPORTED_REQUEST_API_VERSION = 1

EngineApiStatus = Literal[
    "ready",
    "package_missing",
    "api_incompatible",
    "api_version_incompatible",
    "api_probe_failed",
]
ContractSource = Literal["explicit", "legacy_symbols"]


@dataclass(frozen=True, slots=True)
class EngineApiProbe:
    """Serializable result of checking one engine's documented public API."""

    engine: str
    package: str
    compatible: bool
    status: EngineApiStatus
    distribution_version: str | None = None
    module_version: str | None = None
    module_path: str | None = None
    expected_api_version: int | None = None
    api_version: int | None = None
    contract_source: ContractSource | None = None
    missing_symbols: tuple[str, ...] = ()
    missing_methods: tuple[str, ...] = ()
    failed_stage: str | None = None
    failed_symbol: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    warnings: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


def _probe_public_api(
    *,
    engine: str,
    package: str,
    required_symbols: tuple[str, ...],
    required_methods: Mapping[str, tuple[str, ...]],
    expected_api_version: int | None = None,
    require_explicit_contract: bool = False,
    contract_required_symbols: tuple[str, ...] = (),
    contract_required_methods: Mapping[str, tuple[str, ...]] | None = None,
    legacy_required_methods: Mapping[str, tuple[str, ...]] | None = None,
    contract_fields: tuple[str, ...] = (),
) -> EngineApiProbe:
    """Probe a package root and concrete public entry points without opening a runtime."""
    metadata_warning: str | None = None
    try:
        distribution_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    except Exception as exc:
        logger.debug(
            "Engine %s distribution metadata probe failed", engine, exc_info=True
        )
        distribution_version = None
        metadata_warning = (
            f"Distribution metadata could not be read: {type(exc).__name__}: {exc}"
        )
    else:
        metadata_warning = None

    common: dict[str, Any] = {
        "engine": engine,
        "package": package,
        "expected_api_version": expected_api_version,
        "distribution_version": distribution_version,
    }
    warnings: list[str] = []
    if metadata_warning is not None:
        warnings.append(metadata_warning)

    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError as exc:
        logger.debug("Engine %s package import failed", engine, exc_info=True)
        if exc.name == package:
            return EngineApiProbe(
                **common,
                compatible=False,
                status="package_missing",
                failed_stage="module_import",
                error_type=type(exc).__name__,
                error_message=str(exc),
                warnings=tuple(warnings),
            )
        return _probe_failed(
            common,
            warnings,
            stage="module_import",
            error=exc,
        )
    except Exception as exc:
        logger.debug("Engine %s package import failed", engine, exc_info=True)
        return _probe_failed(common, warnings, stage="module_import", error=exc)

    try:
        module_version_value = getattr(module, "__version__", None)
        module_path_value = getattr(module, "__file__", None)
        module_version = (
            str(module_version_value) if module_version_value is not None else None
        )
        module_path = str(module_path_value) if module_path_value is not None else None
    except Exception as exc:
        logger.debug(
            "Engine %s module metadata resolution failed", engine, exc_info=True
        )
        return _probe_failed(
            common,
            warnings,
            stage="module_metadata",
            error=exc,
        )

    if (
        distribution_version is not None
        and module_version is not None
        and distribution_version != module_version
    ):
        warnings.append("Imported module version differs from distribution metadata.")

    common.update(module_version=module_version, module_path=module_path)
    resolved: dict[str, Any] = {}
    missing_symbols: list[str] = []
    for name in required_symbols:
        try:
            resolved[name] = getattr(module, name)
        except AttributeError:
            missing_symbols.append(name)
        except Exception as exc:
            logger.debug(
                "Engine %s public symbol %s resolution failed",
                engine,
                name,
                exc_info=True,
            )
            return _probe_failed(
                common,
                warnings,
                stage="symbol_resolution",
                error=exc,
                failed_symbol=name,
                missing_symbols=tuple(missing_symbols),
            )

    missing_methods: list[str] = []
    for owner_name, methods in required_methods.items():
        owner = (
            module
            if owner_name in {"__module__", "module"}
            else resolved.get(owner_name)
        )
        if (
            owner is None
            and owner_name not in {"__module__", "module"}
            and owner_name not in resolved
        ):
            # A missing class is already reported as a missing public symbol.
            continue
        for method_name in methods:
            qualified_name = (
                method_name
                if owner_name in {"__module__", "module"}
                else f"{owner_name}.{method_name}"
            )
            try:
                method = getattr(owner, method_name)
            except AttributeError:
                missing_methods.append(qualified_name)
                continue
            except Exception as exc:
                logger.debug(
                    "Engine %s public method %s resolution failed",
                    engine,
                    qualified_name,
                    exc_info=True,
                )
                return _probe_failed(
                    common,
                    warnings,
                    stage="method_resolution",
                    error=exc,
                    failed_symbol=qualified_name,
                    missing_symbols=tuple(missing_symbols),
                    missing_methods=tuple(missing_methods),
                )
            if not callable(method):
                missing_methods.append(qualified_name)

    api_version: int | None = None
    contract_source: ContractSource | None = None
    contract_names = ("REQUEST_API_VERSION", "request_api_contract")
    contract_values: dict[str, Any] = {}
    contract_snapshot: dict[str, Any] = {}
    for name in contract_names:
        try:
            contract_values[name] = getattr(module, name)
        except AttributeError:
            continue
        except Exception as exc:
            logger.debug(
                "Engine %s request API marker %s resolution failed",
                engine,
                name,
                exc_info=True,
            )
            return _probe_failed(
                common,
                warnings,
                stage="contract_resolution",
                error=exc,
                failed_symbol=name,
                missing_symbols=tuple(missing_symbols),
                missing_methods=tuple(missing_methods),
            )

    has_version = "REQUEST_API_VERSION" in contract_values
    has_contract = "request_api_contract" in contract_values
    if has_version and has_contract:
        contract_source = "explicit"
        version_value = contract_values["REQUEST_API_VERSION"]
        if isinstance(version_value, bool) or not isinstance(version_value, int):
            error = TypeError("REQUEST_API_VERSION must be an integer")
            logger.debug("Engine %s request API version is invalid: %s", engine, error)
            return _probe_failed(
                common,
                warnings,
                stage="contract_version",
                error=error,
                failed_symbol="REQUEST_API_VERSION",
                missing_symbols=tuple(missing_symbols),
                missing_methods=tuple(missing_methods),
            )
        api_version = version_value
        contract_function = contract_values["request_api_contract"]
        if not callable(contract_function):
            error = TypeError("request_api_contract is not callable")
            logger.debug("Engine %s request API contract is invalid: %s", engine, error)
            return _probe_failed(
                common,
                warnings,
                stage="contract_inspection",
                error=error,
                failed_symbol="request_api_contract",
                missing_symbols=tuple(missing_symbols),
                missing_methods=tuple(missing_methods),
            )
        try:
            contract_result = contract_function()
            for field_name in contract_fields:
                contract_snapshot[field_name] = (
                    contract_result.get(field_name)
                    if isinstance(contract_result, Mapping)
                    else getattr(contract_result, field_name, None)
                )
        except Exception as exc:
            logger.debug(
                "Engine %s request API contract inspection failed",
                engine,
                exc_info=True,
            )
            return _probe_failed(
                common,
                warnings,
                stage="contract_inspection",
                error=exc,
                failed_symbol="request_api_contract",
                missing_symbols=tuple(missing_symbols),
                missing_methods=tuple(missing_methods),
                api_version=api_version,
                contract_source=contract_source,
            )
    elif has_version or has_contract:
        if require_explicit_contract:
            contract_source = "explicit"
            absent_name = (
                "request_api_contract" if has_version else "REQUEST_API_VERSION"
            )
            missing_symbols.append(absent_name)
        else:
            contract_source = "legacy_symbols"
            warnings.append(
                "Incomplete explicit request API contract; using legacy symbol checks."
            )
    elif required_symbols or required_methods:
        if require_explicit_contract:
            contract_source = "explicit"
            missing_symbols.extend(contract_names)
        else:
            contract_source = "legacy_symbols"

    common.update(api_version=api_version, contract_source=contract_source)

    conditional_symbols = (
        contract_required_symbols if contract_source == "explicit" else ()
    )
    conditional_methods = (
        contract_required_methods or {}
        if contract_source == "explicit"
        else legacy_required_methods or {}
    )
    for name in conditional_symbols:
        if name in resolved:
            continue
        try:
            resolved[name] = getattr(module, name)
        except AttributeError:
            missing_symbols.append(name)
        except Exception as exc:
            logger.debug(
                "Engine %s public symbol %s resolution failed",
                engine,
                name,
                exc_info=True,
            )
            return _probe_failed(
                common,
                warnings,
                stage="symbol_resolution",
                error=exc,
                failed_symbol=name,
                missing_symbols=tuple(missing_symbols),
                missing_methods=tuple(missing_methods),
                api_version=api_version,
                contract_source=contract_source,
            )

    for owner_name, methods in conditional_methods.items():
        owner = (
            module
            if owner_name in {"__module__", "module"}
            else resolved.get(owner_name)
        )
        if (
            owner is None
            and owner_name not in {"__module__", "module"}
            and owner_name not in resolved
        ):
            continue
        for method_name in methods:
            qualified_name = (
                method_name
                if owner_name in {"__module__", "module"}
                else f"{owner_name}.{method_name}"
            )
            try:
                method = getattr(owner, method_name)
            except AttributeError:
                missing_methods.append(qualified_name)
            except Exception as exc:
                logger.debug(
                    "Engine %s public method %s resolution failed",
                    engine,
                    qualified_name,
                    exc_info=True,
                )
                return _probe_failed(
                    common,
                    warnings,
                    stage="method_resolution",
                    error=exc,
                    failed_symbol=qualified_name,
                    missing_symbols=tuple(missing_symbols),
                    missing_methods=tuple(missing_methods),
                    api_version=api_version,
                    contract_source=contract_source,
                )
            else:
                if not callable(method):
                    missing_methods.append(qualified_name)

    if (
        api_version is not None
        and expected_api_version is not None
        and api_version != expected_api_version
    ):
        return EngineApiProbe(
            **common,
            compatible=False,
            status="api_version_incompatible",
            missing_symbols=tuple(dict.fromkeys(missing_symbols)),
            missing_methods=tuple(dict.fromkeys(missing_methods)),
            warnings=tuple(warnings),
        )

    if missing_symbols or missing_methods:
        return EngineApiProbe(
            **common,
            compatible=False,
            status="api_incompatible",
            missing_symbols=tuple(dict.fromkeys(missing_symbols)),
            missing_methods=tuple(dict.fromkeys(missing_methods)),
            warnings=tuple(warnings),
        )

    return EngineApiProbe(
        **common,
        compatible=True,
        status="ready",
        warnings=tuple(warnings),
        details={"request_api_contract": contract_snapshot} if contract_fields else {},
    )


def probe_public_api(
    *,
    engine: str,
    package: str,
    required_symbols: tuple[str, ...],
    required_methods: Mapping[str, tuple[str, ...]],
    expected_api_version: int | None = None,
    require_explicit_contract: bool = False,
    contract_required_symbols: tuple[str, ...] = (),
    contract_required_methods: Mapping[str, tuple[str, ...]] | None = None,
    legacy_required_methods: Mapping[str, tuple[str, ...]] | None = None,
    contract_fields: tuple[str, ...] = (),
) -> EngineApiProbe:
    """Probe a public package API and emit one compact structured debug event."""
    probe = _probe_public_api(
        engine=engine,
        package=package,
        required_symbols=required_symbols,
        required_methods=required_methods,
        expected_api_version=expected_api_version,
        require_explicit_contract=require_explicit_contract,
        contract_required_symbols=contract_required_symbols,
        contract_required_methods=contract_required_methods,
        legacy_required_methods=legacy_required_methods,
        contract_fields=contract_fields,
    )
    event = {
        "engine": probe.engine,
        "package": probe.package,
        "distribution_version": probe.distribution_version,
        "module_version": probe.module_version,
        "module_path": probe.module_path,
        "api_version": probe.api_version,
        "expected_api_version": probe.expected_api_version,
        "contract_source": probe.contract_source,
        "compatible": probe.compatible,
        "status": probe.status,
        "missing_symbols": probe.missing_symbols,
        "missing_methods": probe.missing_methods,
        "failed_stage": probe.failed_stage,
        "failed_symbol": probe.failed_symbol,
        "error_type": probe.error_type,
        "error_message": probe.error_message,
        "warnings": probe.warnings,
    }
    logger.debug(
        "engine.api_probe %s", json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    return probe


def _probe_failed(
    common: Mapping[str, Any],
    warnings: list[str],
    *,
    stage: str,
    error: Exception,
    failed_symbol: str | None = None,
    missing_symbols: tuple[str, ...] = (),
    missing_methods: tuple[str, ...] = (),
    api_version: int | None = None,
    contract_source: ContractSource | None = None,
) -> EngineApiProbe:
    return EngineApiProbe(
        **common,
        compatible=False,
        status="api_probe_failed",
        api_version=api_version,
        contract_source=contract_source,
        missing_symbols=missing_symbols,
        missing_methods=missing_methods,
        failed_stage=stage,
        failed_symbol=failed_symbol,
        error_type=type(error).__name__,
        error_message=str(error),
        warnings=tuple(warnings),
    )


__all__ = [
    "ContractSource",
    "EngineApiProbe",
    "EngineApiStatus",
    "probe_public_api",
]
