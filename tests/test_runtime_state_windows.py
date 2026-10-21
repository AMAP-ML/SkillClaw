from __future__ import annotations

import ctypes
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skillclaw import runtime_state

native_windows = pytest.mark.skipif(sys.platform != "win32", reason="native Windows process handles")


@native_windows
@pytest.mark.parametrize("exit_code", [0, 259])
def test_exited_process_is_not_alive_with_retained_handle(exit_code: int) -> None:
    # Popen keeps its Windows process handle open after wait(). The kernel object
    # therefore still exists, even though the process has finished executing.
    with subprocess.Popen([sys.executable, "-c", f"raise SystemExit({exit_code})"]) as child:
        assert child.wait(timeout=10) == exit_code
        assert runtime_state.process_alive(child.pid) is False


@pytest.fixture
def kernel32(monkeypatch):
    # API failures cannot safely be reproduced with foreign/protected processes.
    kernel = SimpleNamespace(
        OpenProcess=Mock(return_value=0),
        WaitForSingleObject=Mock(return_value=258),
        CloseHandle=Mock(return_value=1),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    return kernel


def test_access_denied_does_not_mark_process_dead(kernel32, monkeypatch) -> None:
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    assert runtime_state._process_alive_windows(12345) is True
    kernel32.WaitForSingleObject.assert_not_called()
    kernel32.CloseHandle.assert_not_called()


def test_unexpected_open_process_error_does_not_mark_process_dead(kernel32, monkeypatch) -> None:
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 8, raising=False)
    assert runtime_state._process_alive_windows(12345) is True
    kernel32.WaitForSingleObject.assert_not_called()
    kernel32.CloseHandle.assert_not_called()


@native_windows
def test_running_process_is_alive(monkeypatch) -> None:
    import os
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel32)
    assert runtime_state.process_alive(os.getpid()) is True
    assert kernel32.OpenProcess.argtypes == [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    assert kernel32.OpenProcess.restype is wintypes.HANDLE
    assert kernel32.WaitForSingleObject.argtypes == [wintypes.HANDLE, wintypes.DWORD]
    assert kernel32.WaitForSingleObject.restype is wintypes.DWORD
    assert kernel32.CloseHandle.argtypes == [wintypes.HANDLE]
    assert kernel32.CloseHandle.restype is wintypes.BOOL


@native_windows
def test_exited_process_does_not_block_daemon_start_lock(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runtime_state, "_state_dir", lambda: tmp_path)
    with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
        child.wait(timeout=10)
        lock = runtime_state.daemon_start_lock_path()
        lock.write_text(str(child.pid), encoding="utf-8")
        with runtime_state.daemon_start_lock():
            assert runtime_state.read_pid() is None
            assert lock.exists()
        assert not lock.exists()


def test_missing_process_is_not_alive(kernel32, monkeypatch) -> None:
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87, raising=False)
    assert runtime_state._process_alive_windows(12345) is False
    kernel32.CloseHandle.assert_not_called()


@pytest.mark.parametrize("wait_result, expected", [(0, False), (258, True), (0xFFFFFFFF, True)])
def test_open_handle_is_closed_after_wait(kernel32, wait_result: int, expected: bool) -> None:
    handle = 0x100001234  # Exercises passing a pointer-width value unmodified.
    kernel32.OpenProcess.return_value = handle
    kernel32.WaitForSingleObject.return_value = wait_result
    assert runtime_state._process_alive_windows(12345) is expected
    kernel32.OpenProcess.assert_called_once_with(0x00100000, False, 12345)
    kernel32.WaitForSingleObject.assert_called_once_with(handle, 0)
    kernel32.CloseHandle.assert_called_once_with(handle)


def test_open_handle_is_closed_if_wait_raises(kernel32) -> None:
    kernel32.OpenProcess.return_value = 123
    kernel32.WaitForSingleObject.side_effect = OSError("wait failed")
    with pytest.raises(OSError, match="wait failed"):
        runtime_state._process_alive_windows(12345)
    kernel32.CloseHandle.assert_called_once_with(123)
