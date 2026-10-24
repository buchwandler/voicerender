from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import textwrap
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


class _FakeWin32Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeWin32API:
    def __init__(
        self,
        *,
        open_handle=123,
        wait_result=render_lock._WAIT_TIMEOUT,
        creation_high=0x01234567,
        creation_low=0x89ABCDEF,
        get_times=True,
        close_result=True,
    ):
        self.close_calls = []
        self.creation_high = creation_high
        self.creation_low = creation_low
        self.get_times = get_times
        self.close_result = close_result
        self.OpenProcess = _FakeWin32Function(lambda *_args: open_handle)
        self.WaitForSingleObject = _FakeWin32Function(lambda *_args: wait_result)
        self.GetProcessTimes = _FakeWin32Function(self._get_process_times)
        self.CloseHandle = _FakeWin32Function(self._close_handle)

    def _get_process_times(
        self, _handle, creation, _exit_time, _kernel_time, _user_time
    ):
        if self.get_times:
            filetime = render_lock.ctypes.cast(
                creation, render_lock.ctypes.POINTER(render_lock._FILETIME)
            ).contents
            filetime.dwHighDateTime = self.creation_high
            filetime.dwLowDateTime = self.creation_low
        return self.get_times

    def _close_handle(self, handle):
        self.close_calls.append(handle)
        return self.close_result


def _probe_windows_api(api, *, last_error=0):
    return render_lock._probe_process_windows_api(123, api, lambda: last_error)


def test_windows_probe_reports_alive_and_creation_token_and_closes_handle():
    api = _FakeWin32API()

    probe = _probe_windows_api(api)

    assert probe == render_lock._ProcessProbe(
        "alive", "win32-filetime:0123456789abcdef"
    )
    assert api.close_calls == [123]
    assert api.OpenProcess.argtypes is not None
    assert api.GetProcessTimes.argtypes is not None


def test_windows_probe_reports_signaled_process_dead_and_closes_handle():
    api = _FakeWin32API(wait_result=render_lock._WAIT_OBJECT_0)

    probe = _probe_windows_api(api)

    assert probe == render_lock._ProcessProbe("dead", None)
    assert api.close_calls == [123]


def test_windows_probe_reports_wait_failure_unknown_and_closes_handle():
    api = _FakeWin32API(wait_result=0xFFFFFFFF)

    probe = _probe_windows_api(api)

    assert probe == render_lock._ProcessProbe("unknown", None)
    assert api.close_calls == [123]


def test_windows_probe_keeps_alive_state_when_creation_time_is_unavailable():
    api = _FakeWin32API(get_times=False)

    probe = _probe_windows_api(api)

    assert probe == render_lock._ProcessProbe("alive", None)
    assert api.close_calls == [123]


def test_windows_probe_treats_invalid_pid_as_dead_and_denial_as_unknown():
    missing = _FakeWin32API(open_handle=None)
    denied = _FakeWin32API(open_handle=None)

    assert _probe_windows_api(missing, last_error=87).state == "dead"
    assert _probe_windows_api(denied, last_error=5).state == "unknown"
    assert missing.close_calls == []
    assert denied.close_calls == []


def test_windows_probe_close_failure_is_unknown():
    api = _FakeWin32API(close_result=False)

    assert _probe_windows_api(api).state == "unknown"
    assert api.close_calls == [123]


def test_windows_probe_closes_handle_after_api_exceptions():
    def fail_api(*_args):
        raise RuntimeError("native wrapper failure")

    wait_failure = _FakeWin32API()
    wait_failure.WaitForSingleObject = _FakeWin32Function(fail_api)
    assert _probe_windows_api(wait_failure) == render_lock._ProcessProbe(
        "unknown", None
    )
    assert wait_failure.close_calls == [123]

    token_failure = _FakeWin32API()
    token_failure.GetProcessTimes = _FakeWin32Function(fail_api)
    assert _probe_windows_api(token_failure) == render_lock._ProcessProbe("alive", None)
    assert token_failure.close_calls == [123]


def test_posix_probe_classifies_lookup_permission_and_other_errors(monkeypatch):
    def missing(_pid, _signal):
        raise ProcessLookupError

    def denied(_pid, _signal):
        raise PermissionError

    def unexpected(_pid, _signal):
        raise OSError(errno.EPERM, "not permitted")

    monkeypatch.setattr(render_lock.os, "kill", missing)
    assert render_lock._probe_process_posix(123).state == "dead"
    monkeypatch.setattr(render_lock.os, "kill", denied)
    assert render_lock._probe_process_posix(123).state == "unknown"
    monkeypatch.setattr(render_lock.os, "kill", unexpected)
    assert render_lock._probe_process_posix(123).state == "unknown"


def test_posix_probe_live_process_includes_linux_token_when_available(monkeypatch):
    monkeypatch.setattr(render_lock.os, "kill", lambda *_args: None)
    monkeypatch.setattr(render_lock, "_linux_process_start_token", lambda _pid: "start")

    assert render_lock._probe_process_posix(123) == render_lock._ProcessProbe(
        "alive", "start"
    )


def test_windows_dispatch_does_not_call_os_kill(monkeypatch):
    expected = render_lock._ProcessProbe("alive", "win32-filetime:1")
    seen = []

    def forbidden_kill(*_args):
        pytest.fail("Windows liveness probe must not call os.kill")

    def windows_probe(pid):
        seen.append(pid)
        return expected

    monkeypatch.setattr(render_lock.os, "name", "nt")
    monkeypatch.setattr(render_lock.os, "kill", forbidden_kill)
    monkeypatch.setattr(render_lock, "_probe_process_windows", windows_probe)

    assert render_lock._probe_process(456) == expected
    assert seen == [456]


@pytest.mark.parametrize(
    ("state", "token", "legacy", "saved_token", "expected"),
    [
        ("dead", None, False, "old", True),
        ("unknown", None, False, "old", False),
        ("alive", "same", True, "same", False),
        ("alive", "same", False, None, False),
        ("alive", None, False, "same", False),
        ("alive", "same", False, "same", False),
        ("alive", "new", False, "old", True),
    ],
)
def test_owner_dead_policy_uses_probe_state_and_process_identity(
    monkeypatch, state, token, legacy, saved_token, expected
):
    owner = render_lock.RenderLockInfo(
        pid=123,
        hostname=render_lock.socket.gethostname(),
        created_at=None,
        process_start=saved_token,
        legacy=legacy,
    )
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe(state, token),
    )

    assert render_lock._owner_is_definitely_dead(owner) is expected


def test_lock_payload_uses_current_process_probe_token(monkeypatch):
    token = "win32-filetime:0123456789abcdef"
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe("alive", token),
    )

    payload = json.loads(render_lock._make_lock_payload())

    assert payload["process_start"] == token


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
        lambda *_args, **_kwargs: pytest.fail(
            "Windows must not load the process library"
        ),
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
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe("dead", None),
    )

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
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe("dead", None),
    )

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
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe("alive", "new-start"),
    )

    with exclusive_render_lock(tmp_path) as acquisition:
        assert acquisition.recovered_stale is True
        assert acquisition.previous_owner is not None
        assert acquisition.previous_owner.pid == 123


def test_active_matching_process_is_never_stolen(tmp_path, monkeypatch):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(
        _json_lock(pid=123, process_start="same-start"), encoding="utf-8"
    )
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe("alive", "same-start"),
    )

    with pytest.raises(RenderLockedError) as caught, exclusive_render_lock(tmp_path):
        pytest.fail("matching owner was stolen")

    assert caught.value.code == "render.locked"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == 123


def test_foreign_host_lock_is_not_reclaimed(tmp_path, monkeypatch):
    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(_json_lock(pid=123, hostname="another-host"), encoding="utf-8")
    monkeypatch.setattr(render_lock.socket, "gethostname", lambda: "test-host")

    def probe_unless_foreign_owner(pid):
        if pid == 123:
            pytest.fail("foreign-host locks must not be probed")
        return render_lock._ProcessProbe("alive", None)

    monkeypatch.setattr(render_lock, "_probe_process", probe_unless_foreign_owner)

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
    monkeypatch.setattr(
        render_lock,
        "_probe_process",
        lambda _pid: render_lock._ProcessProbe("alive", "active-start"),
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


@pytest.mark.skipif(os.name != "nt", reason="Windows console semantics")
def test_windows_active_legacy_lock_does_not_generate_ctrl_c(tmp_path):
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        from voicerender.errors import RenderLockedError
        from voicerender.render_lock import exclusive_render_lock

        output_dir = Path(sys.argv[1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / ".voicerender.lock").write_text(
            str(os.getpid()), encoding="utf-8"
        )
        try:
            with exclusive_render_lock(output_dir):
                raise AssertionError("active legacy lock was reclaimed")
        except RenderLockedError as exc:
            assert exc.code == "render.locked"
            print("ACTIVE_LOCK_PRESERVED", flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "child-render")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ACTIVE_LOCK_PRESERVED" in result.stdout
    assert "KeyboardInterrupt" not in result.stderr


@pytest.mark.skipif(os.name != "nt", reason="Win32 process-handle semantics")
def test_windows_dead_child_lock_is_reclaimed(tmp_path):
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(0.5)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        owner_probe = render_lock._probe_process_windows(child.pid)
        assert owner_probe.state == "alive"
        assert owner_probe.start_token is not None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)

        if child.stdout is not None:
            child.stdout.close()

    lock_path = tmp_path / ".voicerender.lock"
    lock_path.write_text(
        _json_lock(pid=child.pid, process_start=owner_probe.start_token),
        encoding="utf-8",
    )

    with exclusive_render_lock(tmp_path) as acquisition:
        assert acquisition.recovered_stale is True
        assert acquisition.previous_owner is not None
        assert acquisition.previous_owner.pid == child.pid
