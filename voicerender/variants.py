"""Read-only discovery and status inspection for SSMD bundle RenderPlan variants."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from renderplan import RenderPlan, RenderProfile, load_manifest, load_plan
from renderplan import codec as renderplan_codec

from .book_contract import (
    load_validated_renderplan_set,
    validate_chapter_plan,
    validate_profile_snapshot,
)
from .bundle_contract import (
    BundleContractError,
    BundleIdentity,
    BundleKind,
    load_bundle_identity,
)
from .document_contract import load_document_workspace
from .renderplan_render import RENDER_MANIFEST_FILENAME

PlanStatus = Literal["valid", "invalid"]
RenderStatus = Literal["not_rendered", "complete", "partial", "failed", "unknown"]
VariantState = Literal[
    "invalid", "not_rendered", "complete", "partial", "failed", "unknown"
]
UnitKind = Literal["chapters", "documents"]
REPORT_FILENAME = "voicerender-report.toml"
_MAX_REPORT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BundleVariant:
    """Plan and output status for one direct child of a RenderPlan root."""

    name: str
    profile_id: str | None
    plan_status: PlanStatus
    validation_error: str | None
    render_status: RenderStatus
    state: VariantState
    unit_kind: UnitKind
    completed_units: int
    total_units: int
    planned_speech: int
    available_speech: int
    pending_composition: bool | None
    report_path: Path | None
    error_code: str | None = None
    error_message: str | None = None
    output_error: str | None = None
    render_diagnostics: Path | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_root_cause_type: str | None = None
    error_root_cause_message: str | None = None

    @property
    def completed_chapters(self) -> int | None:
        return self.completed_units if self.unit_kind == "chapters" else None

    @property
    def total_chapters(self) -> int | None:
        return self.total_units if self.unit_kind == "chapters" else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "profile_id": self.profile_id,
            "plan_status": self.plan_status,
            "validation_error": self.validation_error,
            "render_status": self.render_status,
            "state": self.state,
            "unit_kind": self.unit_kind,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "planned_speech": self.planned_speech,
            "available_speech": self.available_speech,
            "pending_composition": self.pending_composition,
            "report_path": str(self.report_path) if self.report_path else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "output_error": self.output_error,
            "render_diagnostics": str(self.render_diagnostics)
            if self.render_diagnostics
            else None,
            "error_type": self.error_type,
            "error_stage": self.error_stage,
            "error_root_cause_type": self.error_root_cause_type,
            "error_root_cause_message": self.error_root_cause_message,
        }
        if self.unit_kind == "chapters":
            result["completed_chapters"] = self.completed_units
            result["total_chapters"] = self.total_units
        else:
            result["completed_documents"] = self.completed_units
            result["total_documents"] = self.total_units
        return result


@dataclass(frozen=True, slots=True)
class BundleVariantsResult:
    """A stable bundle-level listing suitable for API and CLI consumers."""

    bundle: Path
    kind: BundleKind
    renderplan_root: Path
    output_root: Path
    variants: tuple[BundleVariant, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-variants",
            "version": 1,
            "bundle": str(self.bundle),
            "kind": self.kind,
            "renderplan_root": str(self.renderplan_root),
            "output_root": str(self.output_root),
            "count": len(self.variants),
            "variants": [variant.to_dict() for variant in self.variants],
        }


@dataclass(frozen=True, slots=True)
class BundleVariantProfile:
    """The validated immutable RenderProfile snapshot in one bundle variant."""

    name: str
    path: Path
    profile: RenderProfile


@dataclass(frozen=True, slots=True)
class _PlanUnit:
    name: str
    plan: RenderPlan


@dataclass(frozen=True, slots=True)
class _ValidatedVariant:
    profile_id: str
    units: tuple[_PlanUnit, ...]
    profile_path: Path
    profile: RenderProfile


@dataclass(frozen=True, slots=True)
class _Report:
    status: Literal["failed", "partial", "complete"] | None
    pending_composition: bool | None
    available_speech: int | None
    error_code: str | None
    error_message: str | None
    issue: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_root_cause_type: str | None = None
    error_root_cause_message: str | None = None
    render_diagnostics: Path | None = None


class _PlanInvalid(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "plan.invalid",
        profile_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.profile_id = profile_id


def list_bundle_variants(
    bundle: str | Path,
    *,
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> BundleVariantsResult:
    """List every direct RenderPlan variant candidate without creating or modifying files."""
    identity = load_bundle_identity(bundle)
    rp_root = _resolve_renderplan_root(identity, renderplan_root)
    out_input = (
        Path(output_root).expanduser()
        if output_root is not None
        else identity.root / "voicerender"
    )
    out_root = _resolve_root(out_input, "VoiceRender")
    if renderplan_root is None and not _within(rp_root, identity.root):
        raise BundleContractError("default RenderPlan root escapes the bundle")
    output_root_error: str | None = None
    if output_root is None and not _within(out_root, identity.root):
        output_root_error = "default VoiceRender output root escapes the bundle"

    if rp_root.exists() and not rp_root.is_dir():
        raise BundleContractError(f"RenderPlan root is not a directory: {rp_root}")
    if out_root.exists() and not out_root.is_dir():
        output_root_error = f"VoiceRender output root is not a directory: {out_root}"

    if not rp_root.exists():
        candidates: list[Path] = []
    else:
        try:
            candidates = sorted(
                (
                    child
                    for child in rp_root.iterdir()
                    if child.is_dir() or child.is_symlink()
                ),
                key=lambda child: child.name,
            )
        except OSError as exc:
            raise BundleContractError(
                f"cannot enumerate RenderPlan root {rp_root}: {exc}"
            ) from exc

    unit_kind: UnitKind = "chapters" if identity.kind == "book" else "documents"
    results = tuple(
        _inspect_candidate(
            identity.root,
            identity.kind,
            rp_root,
            out_root,
            candidate,
            unit_kind,
            output_root_error,
        )
        for candidate in candidates
    )
    return BundleVariantsResult(
        bundle=identity.root,
        kind=identity.kind,
        renderplan_root=rp_root,
        output_root=out_root,
        variants=results,
    )


def load_bundle_variant_profile(
    bundle: str | Path,
    variant: str,
    *,
    renderplan_root: str | Path | None = None,
) -> BundleVariantProfile:
    """Load the validated immutable RenderProfile snapshot for an exact variant."""
    identity = load_bundle_identity(bundle)
    rp_root = _resolve_renderplan_root(identity, renderplan_root)
    if (
        not isinstance(variant, str)
        or not variant
        or variant in {".", ".."}
        or "/" in variant
        or "\\" in variant
        or "\x00" in variant
    ):
        raise BundleContractError(
            "variant must name one direct child of the RenderPlan root",
            code="path.unsafe",
        )
    candidate = rp_root / variant
    if candidate.is_symlink():
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise BundleContractError(
                f"cannot resolve bundle variant {variant!r}: {exc}",
                code="variant.invalid",
            ) from exc
        if not _within(resolved, rp_root):
            raise BundleContractError(
                "variant directory symlink escapes the RenderPlan root",
                code="path.unsafe",
            )
        raise BundleContractError(
            "variant directory must not be a symlink", code="path.unsafe"
        )
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleContractError(
            f"cannot resolve bundle variant {variant!r}: {exc}", code="variant.invalid"
        ) from exc
    if not resolved_candidate.is_dir() or not _within(resolved_candidate, rp_root):
        raise BundleContractError(
            f"bundle variant is not a direct safe directory: {variant!r}",
            code="path.unsafe",
        )
    try:
        validated = (
            _validate_book_variant(resolved_candidate, variant)
            if identity.kind == "book"
            else _validate_document_variant(identity.root, resolved_candidate, variant)
        )
    except _PlanInvalid as exc:
        raise BundleContractError(str(exc), code=exc.code) from exc
    return BundleVariantProfile(variant, validated.profile_path, validated.profile)


def _inspect_candidate(
    bundle: Path,
    kind: BundleKind,
    renderplan_root: Path,
    output_root: Path,
    candidate: Path,
    unit_kind: UnitKind,
    output_root_error: str | None,
) -> BundleVariant:
    profile_id: str | None = None
    units: tuple[_PlanUnit, ...] = ()
    validation_error: str | None = None
    try:
        if candidate.is_symlink():
            resolved = candidate.resolve(strict=True)
            if not _within(resolved, renderplan_root):
                raise _PlanInvalid(
                    "variant directory symlink escapes the RenderPlan root",
                    code="path.unsafe",
                )
            raise _PlanInvalid(
                "variant directory must not be a symlink", code="path.unsafe"
            )
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_dir() or not _within(
            resolved_candidate, renderplan_root
        ):
            raise _PlanInvalid(
                "variant directory escapes the RenderPlan root", code="path.unsafe"
            )
        validated = (
            _validate_book_variant(resolved_candidate, candidate.name)
            if kind == "book"
            else _validate_document_variant(bundle, resolved_candidate, candidate.name)
        )
        profile_id = validated.profile_id
        units = validated.units
    except _PlanInvalid as exc:
        profile_id = exc.profile_id
        validation_error = str(exc) or "invalid RenderPlan metadata"
    except Exception as exc:  # noqa: BLE001 - Isolate failures to this candidate.
        validation_error = (
            str(exc) or f"invalid RenderPlan metadata ({type(exc).__name__})"
        )

    output = _inspect_output(
        output_root,
        candidate.name,
        units,
        kind,
        output_root_error,
    )
    plan_status: PlanStatus = "valid" if validation_error is None else "invalid"
    state: VariantState
    if plan_status == "invalid":
        state = "invalid"
    else:
        state = output.render_status
    return BundleVariant(
        name=candidate.name,
        profile_id=profile_id,
        plan_status=plan_status,
        validation_error=validation_error,
        render_status=output.render_status,
        state=state,
        unit_kind=unit_kind,
        completed_units=output.completed_units,
        total_units=len(units),
        planned_speech=sum(len(unit.plan.speech) for unit in units),
        available_speech=output.available_speech,
        pending_composition=output.pending_composition,
        report_path=output.report_path,
        error_code=output.error_code,
        error_message=output.error_message,
        render_diagnostics=output.render_diagnostics,
        error_type=output.error_type,
        error_stage=output.error_stage,
        error_root_cause_type=output.error_root_cause_type,
        error_root_cause_message=output.error_root_cause_message,
        output_error=output.output_error,
    )


def _validate_book_variant(variant_dir: Path, name: str) -> _ValidatedVariant:
    profile_id: str | None = None
    try:
        set_path = _contained_regular(
            variant_dir / "renderplan-set.toml", variant_dir, "RenderPlan set"
        )
        renderplan_set = load_validated_renderplan_set(set_path)
        profile_path = _contained_regular(
            variant_dir / renderplan_set.snapshot, variant_dir, "RenderProfile snapshot"
        )
        profile = validate_profile_snapshot(renderplan_set, profile_path)
        profile_id = profile.profile_id or profile.computed_id()
        expected_name = (
            f"{renderplan_set.variant}--{renderplan_set.profile_fingerprint}"
        )
        if name != expected_name:
            raise _PlanInvalid(
                f"variant directory {name!r} does not match set identity {expected_name!r}",
                code="variant.identity_mismatch",
                profile_id=profile_id,
            )
        units: list[_PlanUnit] = []
        for entry in renderplan_set.chapters:
            plan_path = _contained_relative_file(
                variant_dir, entry.renderplan, f"RenderPlan for {entry.id}"
            )
            plan = load_plan(plan_path)
            if not isinstance(plan, RenderPlan):
                raise TypeError(f"RenderPlan for {entry.id} has an invalid type")
            validate_chapter_plan(renderplan_set, entry, plan)
            units.append(_PlanUnit(entry.id, plan))
        return _ValidatedVariant(
            profile_id=profile_id,
            units=tuple(units),
            profile_path=profile_path,
            profile=profile,
        )
    except _PlanInvalid:
        raise
    except Exception as exc:
        raise _PlanInvalid(
            str(exc) or f"invalid book RenderPlan metadata ({type(exc).__name__})",
            code=getattr(exc, "code", "plan.invalid"),
            profile_id=profile_id,
        ) from exc


def _validate_document_variant(
    bundle: Path, variant_dir: Path, name: str
) -> _ValidatedVariant:
    profile_id: str | None = None
    try:
        # Validate canonical document identity as well as the bundle-level kind.
        load_document_workspace(bundle)
        profile_path = _contained_regular(
            variant_dir / "renderprofile.toml", variant_dir, "RenderProfile snapshot"
        )
        profile = renderplan_codec.load_profile(profile_path)
        profile_id = profile.profile_id or profile.computed_id()
        fingerprint = profile_id.removeprefix("sha256:")[:12]
        if not name.endswith("--" + fingerprint):
            raise _PlanInvalid(
                "variant fingerprint does not match RenderProfile snapshot",
                code="variant.identity_mismatch",
                profile_id=profile_id,
            )
        plan_path = _contained_regular(
            variant_dir / "document.renderplan.toml", variant_dir, "document RenderPlan"
        )
        plan = load_plan(plan_path)
        if not isinstance(plan, RenderPlan) or plan.render_profile_id != profile_id:
            raise _PlanInvalid(
                "document RenderPlan profile identity does not match profile snapshot",
                code="plan.identity_mismatch",
                profile_id=profile_id,
            )
        return _ValidatedVariant(
            profile_id=profile_id,
            units=(_PlanUnit("document-0001", plan),),
            profile_path=profile_path,
            profile=profile,
        )
    except _PlanInvalid:
        raise
    except Exception as exc:
        raise _PlanInvalid(
            str(exc) or f"invalid document RenderPlan metadata ({type(exc).__name__})",
            code=getattr(exc, "code", "plan.invalid"),
            profile_id=profile_id,
        ) from exc


@dataclass(frozen=True, slots=True)
class _OutputStatus:
    render_status: RenderStatus
    completed_units: int
    available_speech: int
    pending_composition: bool | None
    report_path: Path | None
    error_code: str | None
    error_message: str | None
    output_error: str | None
    render_diagnostics: Path | None
    error_type: str | None
    error_stage: str | None
    error_root_cause_type: str | None
    error_root_cause_message: str | None


def _inspect_output(
    output_root: Path,
    variant_name: str,
    units: tuple[_PlanUnit, ...],
    kind: BundleKind,
    root_error: str | None,
) -> _OutputStatus:
    output_variant = output_root / variant_name
    report_path = output_variant / REPORT_FILENAME
    report_present = (
        not root_error
        and not output_variant.is_symlink()
        and (report_path.exists() or report_path.is_symlink())
    )
    visible_report_path = report_path if report_present else None
    report: _Report | None = None
    errors: list[str] = [root_error] if root_error else []
    malformed_manifest = False
    completed = 0
    available = 0
    valid_manifest_count = 0
    report_available: int | None = None
    output_accessible = False
    if not root_error and (output_variant.exists() or output_variant.is_symlink()):
        try:
            output_variant = _safe_directory(
                output_variant, output_root, "output variant"
            )
            output_accessible = True
            if report_present:
                report_file = _contained_regular(
                    report_path, output_variant, "render report"
                )
                report = _load_report(report_file, kind)
                if report.issue:
                    errors.append(report.issue)
        except Exception as exc:  # noqa: BLE001 - Output metadata errors are candidate-local.
            errors.append(str(exc) or "unsafe VoiceRender output metadata")
            report = None

    if output_accessible:
        for unit in units:
            output_dir = output_variant / unit.name
            manifest_path = output_dir / RENDER_MANIFEST_FILENAME
            if not (output_dir.exists() or output_dir.is_symlink()):
                continue
            try:
                safe_output_dir = _safe_directory(
                    output_dir, output_variant, "unit output"
                )
                if not (manifest_path.exists() or manifest_path.is_symlink()):
                    continue
                safe_manifest = _contained_regular(
                    manifest_path, safe_output_dir, "render manifest"
                )
                manifest = load_manifest(safe_manifest)
                manifest.validate_against(unit.plan)
                if type(manifest.complete) is not bool:
                    raise ValueError("render manifest complete flag is not boolean")
                valid_manifest_count += 1
                available += len(manifest.speech)
                if manifest.complete:
                    completed += 1
            except Exception as exc:  # noqa: BLE001 - Invalid manifests must not stop discovery.
                malformed_manifest = True
                errors.append(str(exc) or "invalid render manifest")

    if report is not None:
        report_available = report.available_speech
        if report.issue:
            errors.append(report.issue)
    if valid_manifest_count == 0 and report_available is not None:
        available = report_available

    latest_failed = report is not None and report.status == "failed"
    all_complete = bool(units) and completed == len(units)
    usable_output = valid_manifest_count > 0
    malformed_output = (
        bool(errors) or malformed_manifest or (report_present and report is None)
    )

    if latest_failed:
        render_status: RenderStatus = "failed"
    elif all_complete:
        render_status = "complete"
    elif malformed_output:
        render_status = "unknown"
    elif usable_output or report is not None and report.status == "partial":
        render_status = "partial"
    elif report is not None and report.status == "complete":
        # A successful report without the manifests that prove completion is stale or incomplete.
        render_status = "unknown"
        errors.append(
            "render report indicates completion but no complete manifest is available"
        )
    elif report_present:
        render_status = "unknown"
    else:
        render_status = "not_rendered"

    pending = report.pending_composition if report is not None else None
    code = report.error_code if report is not None else None
    message = report.error_message if report is not None else None
    if malformed_manifest and message is None:
        message = errors[-1]
    return _OutputStatus(
        render_status=render_status,
        completed_units=completed,
        available_speech=available,
        pending_composition=pending,
        report_path=visible_report_path,
        error_code=code,
        error_message=message,
        render_diagnostics=report.render_diagnostics if report is not None else None,
        error_type=report.error_type if report is not None else None,
        error_stage=report.error_stage if report is not None else None,
        error_root_cause_type=(
            report.error_root_cause_type if report is not None else None
        ),
        error_root_cause_message=(
            report.error_root_cause_message if report is not None else None
        ),
        output_error="; ".join(dict.fromkeys(errors)) or None,
    )


def _load_report(path: Path, kind: BundleKind) -> _Report:
    try:
        if path.stat().st_size > _MAX_REPORT_BYTES:
            raise ValueError("render report exceeds the 2 MiB size limit")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("render report root must be a TOML table")
        expected_format = (
            "voicerender-bundle-book-report"
            if kind == "book"
            else "voicerender-bundle-document-report"
        )
        if (
            data.get("format") != expected_format
            or type(data.get("version")) is not int
            or data.get("version") != 1
        ):
            raise ValueError("render report format or version is unsupported")
        pending = data.get("pending_composition")
        if pending is not None and type(pending) is not bool:
            raise ValueError("render report pending_composition must be boolean")
        code = data.get("error_code")
        message = data.get("error_message")
        if code is not None and not isinstance(code, str):
            raise ValueError("render report error_code must be text")
        if message is not None and not isinstance(message, str):
            raise ValueError("render report error_message must be text")
        error_type = data.get("error_type")
        error_stage = data.get("error_stage")
        error_root_cause_type = data.get("error_root_cause_type")
        error_root_cause_message = data.get("error_root_cause_message")
        for label, value in (
            ("error_type", error_type),
            ("error_stage", error_stage),
            ("error_root_cause_type", error_root_cause_type),
            ("error_root_cause_message", error_root_cause_message),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"render report {label} must be text")
        raw_diagnostics = data.get("render_diagnostics")
        if raw_diagnostics is not None and not isinstance(raw_diagnostics, str):
            raise ValueError("render report render_diagnostics must be text")
        render_diagnostics = None
        if raw_diagnostics is not None:
            diagnostics_path = Path(raw_diagnostics)
            render_diagnostics = (
                diagnostics_path
                if diagnostics_path.is_absolute()
                else path.parent / diagnostics_path
            )
        available: int | None = None
        status: Literal["failed", "partial", "complete"] | None
        if kind == "document":
            raw_status = data.get("status")
            if raw_status not in {"failed", "partial", "complete", "reused"}:
                raise ValueError("document render report has an invalid status")
            status = "complete" if raw_status in {"complete", "reused"} else raw_status
            value = data.get("available_speech")
            if value is not None:
                available = _nonnegative_int(value, "available_speech")
        else:
            failed = _nonnegative_int(data.get("failed"), "failed")
            partial = _nonnegative_int(data.get("partial_chapters"), "partial_chapters")
            skipped = _nonnegative_int(data.get("skipped"), "skipped")
            status = "failed" if failed else "partial" if partial or skipped else None
            items = data.get("item", [])
            if not isinstance(items, list):
                raise ValueError("book render report item entries must be a list")
            available_values: list[int] = []
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError("book render report item must be a table")
                item_status = item.get("status")
                item_code = item.get("error_code")
                item_message = item.get("error_message")
                item_error_fields = {
                    "error_type": item.get("error_type"),
                    "error_stage": item.get("error_stage"),
                    "error_root_cause_type": item.get("error_root_cause_type"),
                    "error_root_cause_message": item.get("error_root_cause_message"),
                }
                for label, value in item_error_fields.items():
                    if value is not None and not isinstance(value, str):
                        raise ValueError(
                            f"book render report item {label} must be text"
                        )
                item_diagnostics = item.get("render_diagnostics")
                if item_diagnostics is not None and not isinstance(
                    item_diagnostics, str
                ):
                    raise ValueError(
                        "book render report item render_diagnostics must be text"
                    )
                if item_status is not None and not isinstance(item_status, str):
                    raise ValueError("book render report item status must be text")
                if item_code is not None and not isinstance(item_code, str):
                    raise ValueError("book render report item error_code must be text")
                if item_message is not None and not isinstance(item_message, str):
                    raise ValueError(
                        "book render report item error_message must be text"
                    )
                if item_status == "failed":
                    if code is None:
                        code = item_code
                    if message is None:
                        message = item_message
                    if error_type is None:
                        error_type = item_error_fields["error_type"]
                    if error_stage is None:
                        error_stage = item_error_fields["error_stage"]
                    if error_root_cause_type is None:
                        error_root_cause_type = item_error_fields[
                            "error_root_cause_type"
                        ]
                    if error_root_cause_message is None:
                        error_root_cause_message = item_error_fields[
                            "error_root_cause_message"
                        ]
                    if render_diagnostics is None and item_diagnostics is not None:
                        diagnostics_path = Path(item_diagnostics)
                        render_diagnostics = (
                            diagnostics_path
                            if diagnostics_path.is_absolute()
                            else path.parent / diagnostics_path
                        )
                if "available_speech" in item:
                    available_values.append(
                        _nonnegative_int(
                            item["available_speech"], "item.available_speech"
                        )
                    )
            if available_values:
                available = sum(available_values)
        return _Report(
            status,
            pending,
            available,
            code,
            message,
            error_type=error_type,
            error_stage=error_stage,
            error_root_cause_type=error_root_cause_type,
            error_root_cause_message=error_root_cause_message,
            render_diagnostics=render_diagnostics,
        )
    except Exception as exc:  # noqa: BLE001 - Malformed reports become conservative status.
        return _Report(
            None,
            None,
            None,
            None,
            None,
            f"cannot interpret render report: {exc}",
        )


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"render report {label} must be a non-negative integer")
    return value


def _resolve_root(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise BundleContractError(f"cannot resolve {label} root {path}: {exc}") from exc
    return resolved


def _resolve_renderplan_root(
    identity: BundleIdentity,
    renderplan_root: str | Path | None,
) -> Path:
    path = (
        Path(renderplan_root).expanduser()
        if renderplan_root is not None
        else identity.root / "renderplan"
    )
    resolved = _resolve_root(path, "RenderPlan")
    if renderplan_root is None and not _within(resolved, identity.root):
        raise BundleContractError("default RenderPlan root escapes the bundle")
    if resolved.exists() and not resolved.is_dir():
        raise BundleContractError(f"RenderPlan root is not a directory: {resolved}")
    return resolved


def _safe_directory(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or not _within(resolved, root):
        raise ValueError(f"{label} escapes its containing root: {path}")
    return resolved


def _contained_regular(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} is missing or unreadable: {path}") from exc
    if not _within(resolved, root.resolve(strict=True)):
        raise ValueError(f"{label} resolves outside its containing directory: {path}")
    try:
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    return resolved


def _contained_relative_file(root: Path, relative: str, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError(f"{label} path is unsafe")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        raise ValueError(f"{label} path must be normalized and relative")
    return _contained_regular(root.joinpath(*posix.parts), root, label)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "BundleVariant",
    "BundleVariantProfile",
    "BundleVariantsResult",
    "list_bundle_variants",
    "load_bundle_variant_profile",
]
