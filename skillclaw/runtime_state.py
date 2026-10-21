# Adapted from MetaClaw
from __future__ import annotations

import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _state_dir() -> Path:
    return Path.home() / ".skillclaw"


def pid_file_path() -> Path:
    return _state_dir() / "skillclaw.pid"


def daemon_start_lock_path() -> Path:
    return _state_dir() / "daemon_start.lock"


def _process_alive_unix(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _process_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0
    ERROR_INVALID_PARAMETER = 87
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        # Only a nonexistent PID proves death; other inspection failures must
        # not permit lock reclamation.
        return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
    try:
        # An exited process can still be opened while another handle retains its
        # kernel object. Only a signaled process handle proves termination; a
        # timeout or failed wait must not allow a live daemon's lock to be stolen.
        return kernel32.WaitForSingleObject(handle, 0) != WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(handle)


_USE_WINDOWS = platform.system() == "Windows"


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _USE_WINDOWS:
        return _process_alive_windows(pid)
    return _process_alive_unix(pid)


def _coerce_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def read_pid() -> int | None:
    try:
        pid = _coerce_pid(pid_file_path().read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None
    if pid is None:
        pid_file_path().unlink(missing_ok=True)
        return None
    return pid


def clear_pid():
    pid_file_path().unlink(missing_ok=True)


def clear_pid_if_matches(pid: int):
    if read_pid() == pid:
        clear_pid()


def _write_text_atomic(path: Path, text: str):
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@contextmanager
def daemon_start_lock():
    path = daemon_start_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    owner_pid = os.getpid()

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = _coerce_pid(path.read_text(encoding="utf-8").strip())
            except FileNotFoundError:
                continue

            if holder is None or not process_alive(holder):
                path.unlink(missing_ok=True)
                continue

            raise RuntimeError(holder)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(owner_pid))
                handle.flush()
                os.fsync(handle.fileno())
            break
        except Exception:
            path.unlink(missing_ok=True)
            raise

    try:
        yield
    finally:
        try:
            holder = _coerce_pid(path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            return
        except Exception:
            path.unlink(missing_ok=True)
            return

        if holder == owner_pid or holder is None:
            path.unlink(missing_ok=True)
