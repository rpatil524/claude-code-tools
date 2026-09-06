"""Non-reaping child-exit observation across supported POSIX platforms."""

from __future__ import annotations

import ctypes
import os
import sys


class _DarwinSiginfo(ctypes.Structure):
    """Darwin siginfo_t from sys/signal.h (including reserved ABI fields)."""

    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", ctypes.c_void_p),
        ("si_band", ctypes.c_long),
        ("padding", ctypes.c_ulong * 7),
    ]


def child_exited_without_reaping(pid: int) -> bool:
    """Observe an exited child while keeping its PID reserved for cleanup.

    Args:
        pid: PID of a direct child that this caller has not reaped.

    Returns:
        Whether the child has exited. A running child returns False.

    Raises:
        OSError: The OS cannot observe this child, including ECHILD or EINTR.
        NotImplementedError: No supported non-reaping API is available.
    """
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    waitid = getattr(os, "waitid", None)
    if waitid is not None:
        return waitid(os.P_PID, pid, flags) is not None
    if sys.platform != "darwin":
        raise NotImplementedError("non-reaping child observation needs waitid")

    # Some macOS Python builds omit os.waitid although libSystem provides it.
    # P_PID is 1 in Darwin's sys/wait.h; Python may omit that constant too.
    library = ctypes.CDLL(None, use_errno=True)
    native_waitid = library.waitid
    native_waitid.argtypes = [
        ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_DarwinSiginfo), ctypes.c_int
    ]
    native_waitid.restype = ctypes.c_int
    info = _DarwinSiginfo()
    if native_waitid(1, pid, ctypes.byref(info), flags) == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return info.si_pid == pid
