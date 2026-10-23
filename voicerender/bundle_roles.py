"""Read and verify the renderer-neutral role inventory in an SSMD bundle.

This module intentionally implements the serialized UtterPlan contract locally;
VoiceRender does not parse SSMD or import UtterPlan.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

from renderplan import RenderProfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from .bundle_contract import BundleIdentity, BundleKind, load_bundle_identity
from .errors import VoiceRenderError
from .renderprofile import load_render_profile
from .role_bindings import (
    DuplicateVoiceGroup,
    RoleVoiceIdentity,
    duplicate_roles_by_role,
    duplicate_voice_groups,
    role_voice_identity,
    role_voice_selector,
)
from .variants import load_bundle_variant_profile

BUNDLE_ROLE_INVENTORY_FORMAT = "utterplan.bundle-role-inventory"
BUNDLE_ROLE_INVENTORY_VERSION = 1
ROLE_INVENTORY_FILENAME = "role-inventory.toml"
MAX_ROLE_INVENTORY_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class BundleRolesError(VoiceRenderError, ValueError):
    """A bundle role inventory is missing, stale, unsafe, or invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "bundle.roles_invalid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class BundleRequiredRole:
    """An explicit logical role and its bundle-wide occurrence count."""

    name: str
    occurrences: int


@dataclass(frozen=True, slots=True)
class BundleRoleInventoryItem:
    """Role and freshness metadata for one compiled bundle item."""

    id: str
    source: str
    source_sha256: str
    plan: str
    plan_id: str
    explicit_roles: tuple[str, ...]
    explicit_occurrences: Mapping[str, int]
    uses_default_role: bool
    implicit_occurrences: int


@dataclass(frozen=True, slots=True)
class BundleRoleInventory:
    """Validated aggregate role inventory for a book or document bundle."""

    bundle_kind: BundleKind
    manifest_sha256: str
    roles: tuple[BundleRequiredRole, ...]
    uses_default_role: bool
    implicit_occurrences: int
    items: tuple[BundleRoleInventoryItem, ...]


@dataclass(frozen=True, slots=True)
class LoadedBundleRoleInventory:
    """Canonical bundle identity and its fresh, validated role inventory."""

    bundle: BundleIdentity
    path: Path
    inventory: BundleRoleInventory


def role_inventory_path(
    bundle: str | Path,
    *,
    utterplan_root: str | Path | None = None,
) -> tuple[BundleIdentity, Path]:
    """Resolve the canonical bundle and selected directory's inventory path.

    ``utterplan_root`` means the directory containing ``role-inventory.toml``;
    it does not replace or relax canonical bundle root detection.
    """
    identity = load_bundle_identity(bundle)
    directory = (
        Path(utterplan_root).expanduser()
        if utterplan_root is not None
        else identity.root / "utterplan"
    )
    try:
        directory = directory.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise BundleRolesError(
            f"cannot resolve UtterPlan inventory directory {directory}: {exc}",
            code="path.unsafe",
        ) from exc
    return identity, directory / ROLE_INVENTORY_FILENAME


def read_bundle_role_inventory(
    bundle: str | Path,
    *,
    utterplan_root: str | Path | None = None,
) -> LoadedBundleRoleInventory:
    """Read a role inventory and verify it still describes the current bundle."""
    identity, path = role_inventory_path(bundle, utterplan_root=utterplan_root)
    if path.is_symlink():
        raise BundleRolesError(
            f"bundle role inventory must not be a symlink: {path}",
            code="path.unsafe",
        )
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise BundleRolesError(
            "bundle role inventory is missing:\n"
            f"  {path}\n"
            f"run `utterplan bundle compile {identity.root}` first",
            code="bundle.roles_missing",
        ) from exc
    except OSError as exc:
        raise BundleRolesError(
            f"cannot inspect bundle role inventory {path}: {exc}",
            code="bundle.roles_invalid",
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise BundleRolesError(
            f"bundle role inventory must be a regular file: {path}",
            code="bundle.roles_invalid",
        )
    if info.st_size > MAX_ROLE_INVENTORY_BYTES:
        raise BundleRolesError(
            f"bundle role inventory exceeds the {MAX_ROLE_INVENTORY_BYTES}-byte size limit: {path}",
            code="bundle.roles_invalid",
        )
    try:
        with path.open("rb") as stream:
            opened_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_info.st_mode):
                raise BundleRolesError(
                    f"bundle role inventory must be a regular file: {path}",
                    code="bundle.roles_invalid",
                )
            raw = stream.read(MAX_ROLE_INVENTORY_BYTES + 1)
    except BundleRolesError:
        raise
    except OSError as exc:
        raise BundleRolesError(
            f"cannot read bundle role inventory {path}: {exc}",
            code="bundle.roles_invalid",
        ) from exc
    if len(raw) > MAX_ROLE_INVENTORY_BYTES:
        raise BundleRolesError(
            f"bundle role inventory exceeds the {MAX_ROLE_INVENTORY_BYTES}-byte size limit: {path}",
            code="bundle.roles_invalid",
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BundleRolesError(
            f"bundle role inventory is not valid UTF-8: {path}",
            code="bundle.roles_invalid",
        ) from exc
    try:
        data = tomllib.loads(text)
    except (ValueError, TypeError) as exc:
        raise BundleRolesError(
            f"invalid bundle role inventory TOML in {path}: {exc}",
            code="bundle.roles_invalid",
        ) from exc
    inventory = _parse_inventory(data)
    _verify_inventory_freshness(identity, inventory)
    return LoadedBundleRoleInventory(identity, path, inventory)


def load_bundle_role_inventory(
    bundle: str | Path,
    *,
    utterplan_root: str | Path | None = None,
) -> BundleRoleInventory:
    """Convenience API returning the validated inventory model."""
    return read_bundle_role_inventory(bundle, utterplan_root=utterplan_root).inventory


def sha256_file(path: str | Path) -> str:
    """Hash a file incrementally, without loading its contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_inventory(data: object) -> BundleRoleInventory:
    root = _mapping(data, "inventory")
    _exact_keys(
        root,
        {
            "format",
            "version",
            "bundle_kind",
            "manifest_sha256",
            "uses_default_role",
            "implicit_occurrences",
            "roles",
            "items",
        },
        "inventory",
    )
    if root["format"] != BUNDLE_ROLE_INVENTORY_FORMAT:
        raise _invalid("unsupported bundle role inventory format")
    version = root["version"]
    if type(version) is not int or version != BUNDLE_ROLE_INVENTORY_VERSION:
        raise _invalid(f"unsupported bundle role inventory version {version!r}")
    kind = root["bundle_kind"]
    if not isinstance(kind, str) or kind not in {"book", "document"}:
        raise _invalid("bundle_kind must be 'book' or 'document'")
    manifest_sha256 = _sha256(root["manifest_sha256"], "manifest_sha256")
    uses_default = _boolean(root["uses_default_role"], "uses_default_role")
    implicit_count = _count(root["implicit_occurrences"], "implicit_occurrences")
    if uses_default != (implicit_count > 0):
        raise _invalid(
            "uses_default_role must match whether implicit_occurrences is positive"
        )

    roles_data = _array(root["roles"], "roles")
    roles: list[BundleRequiredRole] = []
    for index, value in enumerate(roles_data):
        location = f"roles[{index}]"
        role_record = _mapping(value, location)
        _exact_keys(role_record, {"name", "occurrences"}, location)
        roles.append(
            BundleRequiredRole(
                name=_nonempty_string(role_record["name"], f"{location}.name"),
                occurrences=_count(
                    role_record["occurrences"], f"{location}.occurrences"
                ),
            )
        )
    if len({role.name for role in roles}) != len(roles):
        raise _invalid("duplicate aggregate role name")

    items_data = _array(root["items"], "items")
    items: list[BundleRoleInventoryItem] = []
    for index, value in enumerate(items_data):
        location = f"items[{index}]"
        item = _mapping(value, location)
        _exact_keys(
            item,
            {
                "id",
                "source",
                "source_sha256",
                "plan",
                "plan_id",
                "explicit_roles",
                "uses_default_role",
                "implicit_occurrences",
                "explicit_occurrences",
            },
            location,
        )
        explicit_roles_data = _array(
            item["explicit_roles"], f"{location}.explicit_roles"
        )
        explicit_roles = tuple(
            _nonempty_string(role, f"{location}.explicit_roles[{role_index}]")
            for role_index, role in enumerate(explicit_roles_data)
        )
        if len(set(explicit_roles)) != len(explicit_roles):
            raise _invalid(f"duplicate explicit role name within {location}")
        counts_data = _mapping(
            item["explicit_occurrences"], f"{location}.explicit_occurrences"
        )
        if set(counts_data) != set(explicit_roles):
            raise _invalid(
                f"{location}.explicit_occurrences must contain exactly the roles in explicit_roles"
            )
        counts: dict[str, int] = {}
        for role in explicit_roles:
            counts[role] = _count(
                counts_data[role], f"{location}.explicit_occurrences[{role!r}]"
            )
        item_uses_default = _boolean(
            item["uses_default_role"], f"{location}.uses_default_role"
        )
        item_implicit_count = _count(
            item["implicit_occurrences"], f"{location}.implicit_occurrences"
        )
        if item_uses_default != (item_implicit_count > 0):
            raise _invalid(
                f"{location}.uses_default_role must match whether implicit_occurrences is positive"
            )
        source = _bundle_relative_path(item["source"], f"{location}.source")
        plan = _bundle_relative_path(item["plan"], f"{location}.plan")
        items.append(
            BundleRoleInventoryItem(
                id=_nonempty_string(item["id"], f"{location}.id"),
                source=source,
                source_sha256=_sha256(
                    item["source_sha256"], f"{location}.source_sha256"
                ),
                plan=plan,
                plan_id=_nonempty_string(item["plan_id"], f"{location}.plan_id"),
                explicit_roles=explicit_roles,
                explicit_occurrences=MappingProxyType(counts),
                uses_default_role=item_uses_default,
                implicit_occurrences=item_implicit_count,
            )
        )
    if len({item.id for item in items}) != len(items):
        raise _invalid("duplicate bundle item id")

    inventory = BundleRoleInventory(
        bundle_kind=cast(BundleKind, kind),
        manifest_sha256=manifest_sha256,
        roles=tuple(roles),
        uses_default_role=uses_default,
        implicit_occurrences=implicit_count,
        items=tuple(items),
    )
    _validate_aggregate(inventory)
    return inventory


def _validate_aggregate(inventory: BundleRoleInventory) -> None:
    expected_order: list[str] = []
    expected_counts: dict[str, int] = {}
    expected_implicit = 0
    for item in inventory.items:
        for role in item.explicit_roles:
            if role not in expected_counts:
                expected_order.append(role)
                expected_counts[role] = 0
            expected_counts[role] += item.explicit_occurrences[role]
        expected_implicit += item.implicit_occurrences
    actual_counts = {role.name: role.occurrences for role in inventory.roles}
    if tuple(role.name for role in inventory.roles) != tuple(expected_order):
        raise _invalid("aggregate role order does not match bundle items")
    if actual_counts != expected_counts:
        raise _invalid("aggregate role counts do not match bundle items")
    if inventory.implicit_occurrences != expected_implicit:
        raise _invalid("aggregate implicit_occurrences do not match bundle items")


def _verify_inventory_freshness(
    identity: BundleIdentity, inventory: BundleRoleInventory
) -> None:
    if inventory.bundle_kind != identity.kind:
        raise _invalid(
            f"role inventory bundle_kind {inventory.bundle_kind!r} does not match "
            f"bundle kind {identity.kind!r}"
        )
    try:
        current_manifest_sha = sha256_file(identity.manifest_path)
    except OSError as exc:
        raise _invalid(
            f"cannot hash bundle manifest {identity.manifest_path}: {exc}"
        ) from exc
    if current_manifest_sha != inventory.manifest_sha256:
        raise _stale(identity.root, "bundle manifest changed")

    for item in inventory.items:
        source_path = _resolve_source_path(identity.root, item.source)
        try:
            source_info = source_path.stat()
        except FileNotFoundError as exc:
            raise _stale(
                identity.root, f"bundle source is missing: {item.source}"
            ) from exc
        except OSError as exc:
            raise _invalid(
                f"cannot inspect bundle source {item.source!r}: {exc}"
            ) from exc
        if not stat.S_ISREG(source_info.st_mode):
            raise _invalid(f"bundle source is not a regular file: {item.source}")
        try:
            current_source_sha = sha256_file(source_path)
        except OSError as exc:
            if isinstance(exc, FileNotFoundError):
                raise _stale(
                    identity.root, f"bundle source is missing: {item.source}"
                ) from exc
            raise _invalid(f"cannot hash bundle source {item.source!r}: {exc}") from exc
        if current_source_sha != item.source_sha256:
            raise _stale(identity.root, f"bundle source changed: {item.source}")


def _resolve_source_path(bundle_root: Path, relative: str) -> Path:
    candidate = bundle_root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise BundleRolesError(
            f"cannot safely resolve bundle source {relative!r}: {exc}",
            code="path.unsafe",
        ) from exc
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise BundleRolesError(
            f"bundle source path escapes the canonical bundle: {relative!r}",
            code="path.unsafe",
        ) from exc
    return resolved


def _bundle_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(f"{field} must be a non-empty bundle-relative POSIX path")
    if "\\" in value or "\x00" in value:
        raise BundleRolesError(
            f"{field} is not a safe POSIX bundle-relative path: {value!r}",
            code="path.unsafe",
        )
    path = PurePosixPath(value)
    parts = value.split("/")
    if (
        path.is_absolute()
        or _WINDOWS_DRIVE_RE.match(value) is not None
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise BundleRolesError(
            f"{field} is not a safe bundle-relative path: {value!r}",
            code="path.unsafe",
        )
    return path.as_posix()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise _invalid(f"{field} must be a TOML table")
    return cast(Mapping[str, Any], value)


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid(f"{field} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise _invalid(f"{field} has {' and '.join(details)} fields")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{field} must be a non-empty string")
    return value


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise _invalid(f"{field} must be a non-negative integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise _invalid(f"{field} must be a boolean")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _invalid(f"{field} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _invalid(message: str) -> BundleRolesError:
    return BundleRolesError(message, code="bundle.roles_invalid")


def _stale(bundle: Path, reason: str) -> BundleRolesError:
    return BundleRolesError(
        "bundle role inventory is stale; rerun\n"
        f"`utterplan bundle compile {bundle}`\n"
        f"reason: {reason}",
        code="bundle.roles_stale",
    )


@dataclass(frozen=True, slots=True)
class BundleRoleCoverage:
    """Profile role coverage against source requirements."""

    required: tuple[str, ...]
    available: tuple[str, ...]
    missing: tuple[str, ...]
    unused: tuple[str, ...]
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": list(self.required),
            "available": list(self.available),
            "missing": list(self.missing),
            "unused": list(self.unused),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class BundleRoleDistinctness:
    """Distinctness findings for all required bound roles."""

    distinct: bool
    duplicate_groups: tuple[DuplicateVoiceGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct": self.distinct,
            "duplicate_groups": [
                {
                    "roles": list(group.roles),
                    "identity": _identity_to_dict(group.identity),
                }
                for group in self.duplicate_groups
            ],
        }


@dataclass(frozen=True, slots=True)
class BundleRoleBinding:
    """One source-required logical role and its optional concrete profile binding."""

    role: str
    occurrences: int
    required_by_default: bool
    profile_default: bool | None
    bound: bool | None
    language: str | None
    renderer: str | None
    target_id: str | None
    selector: str | None
    binding: Mapping[str, Any] | None
    voice_identity: RoleVoiceIdentity | None
    duplicate_roles: tuple[str, ...]
    binding_valid: bool | None
    binding_error: str | None
    explicit_occurrences: int
    implicit_occurrences: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "occurrences": self.occurrences,
            "required_by_default": self.required_by_default,
            "profile_default": self.profile_default,
            "bound": self.bound,
            "language": self.language,
            "renderer": self.renderer,
            "target_id": self.target_id,
            "selector": self.selector,
            "binding": _json_value(self.binding),
            "voice_identity": (
                _identity_to_dict(self.voice_identity) if self.voice_identity else None
            ),
            "duplicate_roles": list(self.duplicate_roles),
            "binding_valid": self.binding_valid,
            "binding_error": self.binding_error,
            "explicit_occurrences": self.explicit_occurrences,
            "implicit_occurrences": self.implicit_occurrences,
        }


@dataclass(frozen=True, slots=True)
class BundleRolesResult:
    """Source role requirements and optional cast coverage/readiness."""

    bundle: Path
    kind: BundleKind
    inventory_path: Path
    inventory: BundleRoleInventory
    profile_path: Path | None
    profile_id: str | None
    profile_default_role: str | None
    profile_source: Literal["profile", "variant"] | None
    variant: str | None
    coverage: BundleRoleCoverage | None
    distinctness: BundleRoleDistinctness | None
    roles: tuple[BundleRoleBinding, ...]
    ready: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-roles",
            "version": 1,
            "bundle": str(self.bundle),
            "kind": self.kind,
            "inventory": {
                "path": str(self.inventory_path),
                "fresh": True,
                "explicit_roles": [role.name for role in self.inventory.roles],
                "uses_default_role": self.inventory.uses_default_role,
                "implicit_occurrences": self.inventory.implicit_occurrences,
                "items": len(self.inventory.items),
            },
            "profile": (
                {
                    "path": str(self.profile_path) if self.profile_path else None,
                    "profile_id": self.profile_id,
                    "default_role": self.profile_default_role,
                    "source": self.profile_source,
                    "variant": self.variant,
                }
                if self.profile_source is not None
                else None
            ),
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "distinctness": self.distinctness.to_dict() if self.distinctness else None,
            "roles": [role.to_dict() for role in self.roles],
            "ready": self.ready,
        }


def inspect_bundle_roles(
    bundle: str | Path,
    *,
    profile: str | Path | None = None,
    variant: str | None = None,
    utterplan_root: str | Path | None = None,
    renderplan_root: str | Path | None = None,
) -> BundleRolesResult:
    """Inspect source roles and optionally compare an authored profile or variant."""
    if profile is not None and variant is not None:
        raise BundleRolesError(
            "--profile and --variant are mutually exclusive",
            code="bundle.roles_invalid",
        )
    loaded = read_bundle_role_inventory(bundle, utterplan_root=utterplan_root)
    selected_profile: RenderProfile | None = None
    profile_path: Path | None = None
    profile_source: Literal["profile", "variant"] | None = None
    selected_variant: str | None = None
    if profile is not None:
        profile_path = Path(profile).expanduser().resolve(strict=False)
        selected_profile = load_render_profile(profile_path)
        profile_source = "profile"
    elif variant is not None:
        selected = load_bundle_variant_profile(
            loaded.bundle.root, variant, renderplan_root=renderplan_root
        )
        profile_path = selected.path
        selected_profile = selected.profile
        profile_source = "variant"
        selected_variant = selected.name

    explicit_counts = {role.name: role.occurrences for role in loaded.inventory.roles}
    required = list(explicit_counts)
    default_required = False
    if selected_profile is not None and loaded.inventory.uses_default_role:
        default_required = True
        if selected_profile.default_role not in required:
            required.append(selected_profile.default_role)
    required_tuple = tuple(required)

    coverage: BundleRoleCoverage | None = None
    distinctness: BundleRoleDistinctness | None = None
    role_rows: list[BundleRoleBinding] = []
    if selected_profile is None:
        role_rows.extend(
            BundleRoleBinding(
                role=name,
                occurrences=explicit_counts[name],
                required_by_default=False,
                profile_default=None,
                bound=None,
                language=None,
                renderer=None,
                target_id=None,
                selector=None,
                binding=None,
                voice_identity=None,
                duplicate_roles=(),
                binding_valid=None,
                binding_error=None,
                explicit_occurrences=explicit_counts[name],
                implicit_occurrences=0,
            )
            for name in required_tuple
        )
        ready: bool | None = None
    else:
        available = tuple(selected_profile.roles)
        available_set = set(available)
        missing = tuple(name for name in required_tuple if name not in available_set)
        required_set = set(required_tuple)
        unused = tuple(name for name in available if name not in required_set)
        coverage = BundleRoleCoverage(
            required=required_tuple,
            available=available,
            missing=missing,
            unused=unused,
            complete=not missing,
        )
        identities: list[tuple[str, RoleVoiceIdentity]] = []
        for name in required_tuple:
            is_default_required = (
                default_required and name == selected_profile.default_role
            )
            explicit_occurrences = explicit_counts.get(name, 0)
            implicit_occurrences = (
                loaded.inventory.implicit_occurrences if is_default_required else 0
            )
            role_profile = selected_profile.roles.get(name)
            if role_profile is None:
                role_rows.append(
                    BundleRoleBinding(
                        role=name,
                        occurrences=explicit_occurrences or implicit_occurrences,
                        required_by_default=is_default_required,
                        profile_default=name == selected_profile.default_role,
                        bound=False,
                        language=None,
                        renderer=None,
                        target_id=None,
                        selector=None,
                        binding=None,
                        voice_identity=None,
                        duplicate_roles=(),
                        binding_valid=None,
                        binding_error=None,
                        explicit_occurrences=explicit_occurrences,
                        implicit_occurrences=implicit_occurrences,
                    )
                )
                continue
            binding = MappingProxyType(dict(role_profile.binding))
            renderer = binding.get("renderer")
            target_id = binding.get("target_id")
            voice_identity: RoleVoiceIdentity | None = None
            binding_error: str | None = None
            try:
                voice_identity = role_voice_identity(role_profile)
            except (ValueError, TypeError) as exc:
                binding_error = str(exc)
            if voice_identity is not None:
                identities.append((name, voice_identity))
            role_rows.append(
                BundleRoleBinding(
                    role=name,
                    occurrences=explicit_occurrences or implicit_occurrences,
                    required_by_default=is_default_required,
                    profile_default=name == selected_profile.default_role,
                    bound=True,
                    language=role_profile.language,
                    renderer=renderer if isinstance(renderer, str) else None,
                    target_id=target_id if isinstance(target_id, str) else None,
                    selector=role_voice_selector(role_profile),
                    binding=binding,
                    voice_identity=voice_identity,
                    duplicate_roles=(),
                    binding_valid=voice_identity is not None,
                    binding_error=binding_error,
                    explicit_occurrences=explicit_occurrences,
                    implicit_occurrences=implicit_occurrences,
                )
            )
        duplicate_groups = duplicate_voice_groups(identities)
        duplicate_map = duplicate_roles_by_role(duplicate_groups)
        role_rows = [
            BundleRoleBinding(
                role=row.role,
                occurrences=row.occurrences,
                required_by_default=row.required_by_default,
                profile_default=row.profile_default,
                bound=row.bound,
                language=row.language,
                renderer=row.renderer,
                target_id=row.target_id,
                selector=row.selector,
                binding=row.binding,
                voice_identity=row.voice_identity,
                duplicate_roles=duplicate_map.get(row.role, ()),
                binding_valid=row.binding_valid,
                binding_error=row.binding_error,
                explicit_occurrences=row.explicit_occurrences,
                implicit_occurrences=row.implicit_occurrences,
            )
            for row in role_rows
        ]
        distinctness = BundleRoleDistinctness(
            distinct=not duplicate_groups, duplicate_groups=duplicate_groups
        )
        ready = (
            coverage.complete
            and all(
                row.bound is True and row.binding_valid is True for row in role_rows
            )
            and distinctness.distinct
        )

    return BundleRolesResult(
        bundle=loaded.bundle.root,
        kind=loaded.bundle.kind,
        inventory_path=loaded.path,
        inventory=loaded.inventory,
        profile_path=profile_path,
        profile_id=(
            selected_profile.profile_id or selected_profile.computed_id()
            if selected_profile is not None
            else None
        ),
        profile_default_role=(
            selected_profile.default_role if selected_profile is not None else None
        ),
        profile_source=profile_source,
        variant=selected_variant,
        coverage=coverage,
        distinctness=distinctness,
        roles=tuple(role_rows),
        ready=ready,
    )


def _identity_to_dict(identity: RoleVoiceIdentity) -> dict[str, Any]:
    return {
        "renderer": identity.renderer,
        "target_id": identity.target_id,
        "voice": identity.voice,
        "speaker": identity.speaker,
        "source": dict(identity.source) if identity.source is not None else None,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "BUNDLE_ROLE_INVENTORY_FORMAT",
    "BUNDLE_ROLE_INVENTORY_VERSION",
    "MAX_ROLE_INVENTORY_BYTES",
    "ROLE_INVENTORY_FILENAME",
    "BundleRequiredRole",
    "BundleRoleBinding",
    "BundleRoleCoverage",
    "BundleRoleDistinctness",
    "BundleRoleInventory",
    "BundleRoleInventoryItem",
    "BundleRolesError",
    "BundleRolesResult",
    "DuplicateVoiceGroup",
    "LoadedBundleRoleInventory",
    "RoleVoiceIdentity",
    "inspect_bundle_roles",
    "load_bundle_role_inventory",
    "read_bundle_role_inventory",
    "role_inventory_path",
    "sha256_file",
]
