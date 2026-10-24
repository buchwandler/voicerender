from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from voicerender import render_lock
from voicerender.errors import RenderLockedError, RenderLockInvalidError
from voicerender.render_lock import exclusive_render_lock


def _json_lock(
    *,
    pid: int,
    hostname: str | None = None,
    process_start: str | None = "old-start",
) -> str:
    return json.dumps(
        {
            "schema": "voicerender.render-lock.v1",
            "pid": pid,
            "hostname": hostname or render_lock.socket.gethostname(),
            "created_at": "2026-10-23T17:42:12Z",
            "process_start": process_start,
        }
    )


def _raise_process_lookup(_pid: int, _signal: int) -> None:
    raise ProcessLookupError


def test_lock_is_complete_json_during_render_and_removed_afterward(tmp_path):
    lock_path = tmp_path / ".voicerender.lock"
    with exclusive_render_lock(tmp_path) as acquisition:
        assert acquisition.path == lock_path
        assert acquisition.recovered_stale is False
        assert acquisition.previous_owner is None
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["schema"] == "voicerender.render-lock.v1"
        assert payload["pid"] == os.getpid()
        assert payload["hostname"]
        assert "created_at" in payload
        assert "process_start" in payload
    assert not lock_path.exists()


def test_lock_install_uses_atomic_rename_on_windows(tmp_path, monkeypatch):
    source = tmp_path / "lock.tmp"
    destination = tmp_path / ".voicerender.lock"
    source.write_bytes(b"lock")

    monkeypatch.setattr(render_lock.os, "name", "nt")

    real_rename = os.rename

    def rename_without_replacement(source_path: Path, destination_path: Path) -> None:
        if destination_path.exists():
            raise FileExistsError(destination_path)
        real_rename(source_path, destination_path)

    monkeypatch.setattr(render_lock.os, "rename", rename_without_replacement)
    monkeypatch.setattr(
        render_lock.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: pytest.fail("Windows must not load the process library"),
    )

    render_lock._install_no_replace(source, destination)

    assert destination.read_bytes() == b"lock"
    assert not source.exists()



def test_active_legacy_pid_lock_is_preserved(tmp_path):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(RenderLockedError) as caught, exclusive_render_lock(tmp_path):
        pytest.fail("active lock was stolen")

    assert caught.value.code == "render.locked"
    assert lock_path.read_text(encoding="utf-8") == str(os.getpid())


def test_dead_legacy_pid_lock_is_replaced_and_only_its_temp_is_removed(
    tmp_path, monkeypatch
):
    dead_pid = 987654321
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(str(dead_pid), encoding="utf-8")
    owned_temp = tmp_path / f".000001-a1b2c3d4e5f6.wav.tmp-{dead_pid}"
    other_temp = tmp_path / ".000002-a1b2c3d4e5f6.wav.tmp-111"
    unrelated = tmp_path / "keep.tmp"
    for path in (owned_temp, other_temp, unrelated):
        path.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(render_lock.os, "kill", _raise_process_lookup)

    with exclusive_render_lock(tmp_path) as acquisition:
        assert acquisition.recovered_stale is True
        assert acquisition.previous_owner is not None
        assert acquisition.previous_owner.pid == dead_pid
        assert acquisition.previous_owner.legacy is True
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
        assert not owned_temp.exists()
        assert other_temp.exists()
        assert unrelated.exists()

    assert not lock_path.exists()


def test_dead_new_format_lock_is_reclaimed(tmp_path, monkeypatch):
    dead_pid = 987654321
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(_json_lock(pid=dead_pid), encoding="utf-8")
    monkeypatch.setattr(render_lock.os, "kill", _raise_process_lookup)

    with exclusive_render_lock(tmp_path) as acquisition:
        assert acquisition.recovered_stale is True
        assert acquisition.previous_owner is not None
        assert acquisition.previous_owner.legacy is False
        assert acquisition.previous_owner.pid == dead_pid


def test_pid_reuse_is_reclaimed_when_start_token_differs(tmp_path, monkeypatch):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(
        _json_lock(pid=123, process_start="old-start"), encoding="utf-8"
    )
    monkeypatch.setattr(render_lock.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(render_lock, "_process_start_token", lambda _pid: "new-start")

    with exclusive_render_lock(tmp_path) as acquisition:
        assert acquisition.recovered_stale is True
        assert acquisition.previous_owner is not None
        assert acquisition.previous_owner.pid == 123


def test_active_matching_process_is_never_stolen(tmp_path, monkeypatch):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(
        _json_lock(pid=123, process_start="same-start"), encoding="utf-8"
    )
    monkeypatch.setattr(render_lock.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(render_lock, "_process_start_token", lambda _pid: "same-start")

    with pytest.raises(RenderLockedError) as caught, exclusive_render_lock(tmp_path):
        pytest.fail("matching owner was stolen")

    assert caught.value.code == "render.locked"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == 123


def test_foreign_host_lock_is_not_reclaimed(tmp_path, monkeypatch):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(_json_lock(pid=123, hostname="another-host"), encoding="utf-8")
    monkeypatch.setattr(render_lock.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(render_lock.os, "kill", _raise_process_lookup)

    with pytest.raises(RenderLockedError) as caught, exclusive_render_lock(tmp_path):
        pytest.fail("foreign-host lock was stolen")

    assert caught.value.code == "render.locked"
    assert lock_path.exists()


def test_malformed_lock_fails_closed(tmp_path):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text("not a pid or lock document", encoding="utf-8")

    with (
        pytest.raises(RenderLockInvalidError) as caught,
        exclusive_render_lock(tmp_path),
    ):
        pytest.fail("malformed lock was ignored")

    assert caught.value.code == "render.lock_invalid"
    assert lock_path.exists()


def test_lock_replacement_during_inspection_is_not_deleted(tmp_path, monkeypatch):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(
        _json_lock(pid=123, process_start="dead-start"), encoding="utf-8"
    )
    monkeypatch.setattr(render_lock.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        render_lock, "_process_start_token", lambda _pid: "active-start"
    )
    real_identity = render_lock._lock_file_identity
    identity_calls = 0
    active_lock = _json_lock(pid=456, process_start="active-start")

    def replace_after_read(path):
        nonlocal identity_calls
        if path == lock_path:
            identity_calls += 1
            if identity_calls == 2:
                replacement = tmp_path / ".replacement-lock"
                replacement.write_text(active_lock, encoding="utf-8")
                os.replace(replacement, lock_path)
        return real_identity(path)

    monkeypatch.setattr(render_lock, "_lock_file_identity", replace_after_read)

    with pytest.raises(RenderLockedError) as caught, exclusive_render_lock(tmp_path):
        pytest.fail("changed lock was stolen")

    assert caught.value.code == "render.locked"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == 456
