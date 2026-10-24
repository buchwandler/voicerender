"""Crash-safe exclusive locks for render output directories."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import socket
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except (
    ImportError
):  # pragma: no cover - non-POSIX hosts cannot safely reclaim stale locks.
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX hosts do not provide msvcrt.
    msvcrt = None  # type: ignore[assignment]
from .errors import RenderLockedError, RenderLockInvalidError

_LOCK_FILENAME = ".voicerender.lock"
_LOCK_GUARD_FILENAME = ".voicerender.lock.guard"
_LOCK_SCHEMA = "voicerender.render-lock.v1"
_MAX_RECLAIM_ATTEMPTS = 8
_TEMP_COPY_PATTERN = re.compile(r"^\.[A-Za-z0-9._-]+\.wav\.tmp-(\d+)$")


@dataclass(frozen=True, slots=True)
class RenderLockInfo:
    pid: int
    hostname: str | None
    created_at: str | None
    process_start: str | None
    legacy: bool = False


@dataclass(frozen=True, slots=True)
class RenderLockAcquisition:
    path: Path
    recovered_stale: bool
    previous_owner: RenderLockInfo | None


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mtime_ns: int
    size: int


def _lock_file_identity(path: Path) -> _FileIdentity:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise OSError(errno.EINVAL, "render lock is not a regular file", str(path))
    return _FileIdentity(info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)


def _process_start_token(pid: int) -> str | None:
    """Return Linux procfs starttime (stat field 22), if available."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing_paren = stat.rfind(")")
        if closing_paren < 0:
            return None
        fields_from_state = stat[closing_paren + 1 :].split()
        return fields_from_state[19]
    except (OSError, UnicodeError, IndexError):
        return None


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_lock_payload() -> bytes:
    payload = {
        "schema": _LOCK_SCHEMA,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": _created_at(),
        "process_start": _process_start_token(os.getpid()),
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _parse_lock(text: str) -> RenderLockInfo:
    stripped = text.strip()
    if re.fullmatch(r"[0-9]+", stripped):
        pid = int(stripped)
        if pid < 1:
            raise ValueError("PID must be positive")
        return RenderLockInfo(pid, None, None, None, legacy=True)

    try:
        payload = json.loads(stripped)
    except (ValueError, TypeError) as exc:
        raise ValueError("contents are neither a legacy PID nor valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _LOCK_SCHEMA:
        raise ValueError("unsupported lock schema")

    raw_pid = payload.get("pid")
    hostname = payload.get("hostname")
    created_at = payload.get("created_at")
    process_start = payload.get("process_start")
    if isinstance(raw_pid, bool) or not isinstance(raw_pid, int) or raw_pid < 1:
        raise ValueError("lock PID must be a positive integer")
    pid = raw_pid
    if hostname is not None and not isinstance(hostname, str):
        raise ValueError("lock hostname must be a string or null")
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError("lock created_at must be a string or null")
    if process_start is not None and not isinstance(process_start, str):
        raise ValueError("lock process_start must be a string or null")
    return RenderLockInfo(pid, hostname, created_at, process_start)


def _owner_is_definitely_dead(owner: RenderLockInfo) -> bool:
    hostname = socket.gethostname()
    if owner.legacy:
        pass  # Legacy locks were always produced on this host's POSIX process space.
    elif owner.hostname != hostname:
        return False

    try:
        os.kill(owner.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as exc:
        return exc.errno == errno.ESRCH

    if owner.legacy or owner.process_start is None:
        return False
    current_start = _process_start_token(owner.pid)
    return current_start is not None and current_start != owner.process_start


@contextmanager
def _directory_guard(output_dir: Path) -> Iterator[bool]:
    """Serialize lock inspection/reclamation among VoiceRender processes."""
    if fcntl is not None:
        descriptor = os.open(output_dir, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield True
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        return

    if msvcrt is not None:
        # Windows cannot open a directory as a file descriptor.  Keep a
        # sidecar guard file and lock one byte in it instead.  The guard is
        # deliberately retained so concurrent processes always coordinate on
        # the same inode/file, including after stale-lock recovery.
        guard = output_dir / _LOCK_GUARD_FILENAME
        descriptor = os.open(
            guard,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            try:
                yield True
            finally:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(descriptor)
        return

    # A platform without either primitive cannot prove that stale-lock
    # reclamation is serialized, so callers must fail closed.
    yield False

def _install_no_replace(source: Path, destination: Path) -> None:
    """Atomically move a complete temp file into place without replacement."""
    if os.name == "nt":
        # Windows rename fails when the destination already exists, while
        # removing the temporary name after creating a hard link can fail
        # with a still-open handle (WinError 32).
        os.rename(source, destination)
        return

    renameat2 = None
    native_library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(native_library, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number, os.strerror(error_number), str(destination)
            )
        if error_number not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise OSError(error_number, os.strerror(error_number), str(destination))

    python_link = getattr(os, "link", None)
    if python_link is not None:
        python_link(source, destination)
        return
    raise OSError(errno.ENOSYS, "atomic no-replace install is unavailable")


def _install_lock(path: Path, payload: bytes) -> bool:
    """Atomically install a complete lock without replacing an existing path."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".voicerender-lock-tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _install_no_replace(temporary, path)
        except FileExistsError:
            return False
        except OSError as exc:
            raise RenderLockInvalidError(
                f"Cannot safely install render lock {path}: {exc}"
            ) from exc
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _remove_owned_output_temps(output_dir: Path, dead_pid: int) -> None:
    try:
        for candidate in output_dir.iterdir():
            match = _TEMP_COPY_PATTERN.fullmatch(candidate.name)
            if match is None or int(match.group(1)) != dead_pid:
                continue
            try:
                if candidate.is_file() and not candidate.is_symlink():
                    candidate.unlink()
            except OSError:
                # A leftover temp is harmless; never widen cleanup or mask recovery.
                continue
    except OSError:
        # Directory enumeration is best-effort; recovery must remain possible.
        return


def _read_existing_lock(path: Path) -> tuple[RenderLockInfo, _FileIdentity]:
    try:
        before = _lock_file_identity(path)
        text = path.read_text(encoding="utf-8")
        after = _lock_file_identity(path)
    except FileNotFoundError as exc:
        raise _LockChanged from exc
    except (OSError, UnicodeError) as exc:
        raise RenderLockInvalidError(
            f"Cannot determine lock owner safely from {path}: {exc}"
        ) from exc
    if before != after:
        raise _LockChanged
    try:
        owner = _parse_lock(text)
    except ValueError as exc:
        raise RenderLockInvalidError(
            f"Cannot determine lock owner safely from {path}: {exc}"
        ) from exc
    return owner, after


class _LockChanged(Exception):
    """The canonical lock changed while it was being inspected."""


def _unlink_if_unchanged(path: Path, identity: _FileIdentity) -> bool:
    try:
        current = _lock_file_identity(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise RenderLockInvalidError(
            f"Cannot safely inspect render lock {path}: {exc}"
        ) from exc
    if current != identity:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    return True


@contextmanager
def exclusive_render_lock(
    output_dir: Path,
    *,
    recover_stale: bool = True,
) -> Iterator[RenderLockAcquisition]:
    """Hold the output-directory render lock, reclaiming only proven-dead owners."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _LOCK_FILENAME
    payload = _make_lock_payload()
    recovered_stale = False
    previous_owner: RenderLockInfo | None = None
    installed_identity: _FileIdentity | None = None

    for _attempt in range(_MAX_RECLAIM_ATTEMPTS):
        should_yield = False
        with _directory_guard(output_dir) as serialized:
            if _install_lock(path, payload):
                installed_identity = _lock_file_identity(path)
                should_yield = True
            else:
                try:
                    owner, identity = _read_existing_lock(path)
                except _LockChanged:
                    continue
                if not recover_stale or not _owner_is_definitely_dead(owner):
                    raise RenderLockedError(
                        f"Render directory is already being rendered by pid {owner.pid}: "
                        f"{output_dir}"
                    )
                if not serialized:
                    raise RenderLockInvalidError(
                        f"Cannot safely reclaim stale render lock without process-safe "
                        f"directory locking: {path}"
                    )
                try:
                    unchanged = _unlink_if_unchanged(path, identity)
                except _LockChanged:
                    continue
                if not unchanged:
                    continue
                previous_owner = owner
                recovered_stale = True
                _remove_owned_output_temps(output_dir, owner.pid)
                _fsync_directory(output_dir)
                continue
        if should_yield:
            break
    else:
        raise RenderLockedError(
            f"Could not acquire render lock after concurrent lock changes: {path}"
        )

    acquisition = RenderLockAcquisition(path, recovered_stale, previous_owner)
    try:
        yield acquisition
    finally:
        if installed_identity is not None:
            try:
                _unlink_if_unchanged(path, installed_identity)
                _fsync_directory(output_dir)
            except RenderLockInvalidError:
                # Preserve the original render result if lock cleanup cannot be proven safe.
                pass


__all__ = [
    "RenderLockAcquisition",
    "RenderLockInfo",
    "exclusive_render_lock",
]
