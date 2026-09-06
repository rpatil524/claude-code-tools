"""Real process regressions for Python builds without os.waitid on macOS."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from claude_code_tools.codex_server_process import run_diagnostic
from claude_code_tools.codex_server_wait import child_exited_without_reaping

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Darwin ABI")


@pytest.fixture(autouse=True)
def missing_python_waitid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the actual native API used by affected Python builds."""
    monkeypatch.delattr(os, "waitid", raising=False)
    monkeypatch.delattr(os, "P_PID", raising=False)


def test_native_observation_preserves_exit_status() -> None:
    """Observe running/finished children without consuming their wait status."""
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(7)"],
        stdin=subprocess.PIPE,
    ) as process:
        assert not child_exited_without_reaping(process.pid)
        assert process.stdin is not None
        process.stdin.close()
        deadline = time.monotonic() + 5
        while not child_exited_without_reaping(process.pid):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert child_exited_without_reaping(process.pid)
        assert process.wait(timeout=2) == 7
        with pytest.raises(ChildProcessError):
            child_exited_without_reaping(process.pid)


def test_diagnostic_without_python_waitid() -> None:
    """A version-like diagnostic succeeds and preserves stdout and status."""
    result = run_diagnostic(
        [sys.executable, "-c", "print('codex-cli 0.153.4')"], os.environ
    )
    assert result is not None
    assert result.returncode == 0
    assert result.stdout.strip() == "codex-cli 0.153.4"


def test_closed_pipes_do_not_hide_running_process() -> None:
    """A child closing its pipes must still time out and be cleaned up."""
    started = time.monotonic()
    result = run_diagnostic(
        [sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(30)"],
        os.environ,
        timeout=0.2,
    )
    assert result is None
    assert time.monotonic() - started < 5
