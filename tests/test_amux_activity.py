"""Session input evidence with real history, metadata, and goal fixtures."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from claude_code_tools.amux import activity
from claude_code_tools.amux.model import Agent

SID = "12345678-1234-1234-1234-123456789abc"
START = "Sun Sep  6 10:00:00 2026"


def write_history(home: Path, records: list[object]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "history.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )


def claude_record(home: Path, pid: int, **extra: object) -> None:
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    (home / "sessions" / f"{pid}.json").write_text(
        json.dumps({"pid": pid, "procStart": START, "sessionId": SID, **extra})
    )


@pytest.fixture
def profile_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    return tmp_path


def test_history_units_maximum_and_invalid_rows(tmp_path: Path) -> None:
    write_history(tmp_path, [
        {"sessionId": SID, "timestamp": 5000},
        {"sessionId": SID, "timestamp": 3000},
        {"sessionId": SID, "timestamp": True},
        {"sessionId": SID, "timestamp": float("inf")},
        ["wrong shape"],
    ])
    assert activity._history(tmp_path, "claude") == ({SID: 5.0}, "complete")
    write_history(tmp_path, [{"session_id": SID, "ts": 8000}])
    assert activity._history(tmp_path, "codex") == ({SID: 8000.0}, "complete")


def test_history_tail_does_not_invent_old_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_history(tmp_path, [
        {"session_id": "old", "ts": 1},
        {"session_id": "new", "ts": 2},
    ])
    monkeypatch.setattr(activity, "_HISTORY_LIMIT", 40)
    assert activity._history(tmp_path, "codex") == ({"new": 2.0}, "tail")
    assert activity._history(tmp_path / "missing", "codex") == ({}, "unavailable")


def test_profile_discovery_custom_and_symlink(
    profile_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = profile_root / ".claude-rja"
    work.mkdir()
    (profile_root / ".claude-alias").symlink_to(work, target_is_directory=True)
    custom = profile_root / "custom"
    custom.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    assert set(activity._homes("claude")) == {work, custom}


def test_claude_pid_evidence_shared_and_runtime(
    profile_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = profile_root / ".claude-rja"
    write_history(home, [{"sessionId": SID, "timestamp": 5000}])
    claude_record(home, 101, name="current", status="shell")
    claude_record(home, 102, status="idle")
    monkeypatch.setattr(activity, "_run", lambda _: f"101 {START}\n102 {START}\n")
    agents = [Agent("work:1.1", "work", "claude", pid=101),
              Agent("work:1.2", "work", "claude", pid=102)]
    activity.enrich(agents, {})
    assert all(a.last_input_at == 5.0 for a in agents)
    assert all(a.input_scope == "shared-session" for a in agents)
    assert agents[0].state == "bg"
    assert agents[0].name == "current"
    assert agents[1].state == "idle"


def test_stale_pid_record_is_not_evidence(
    profile_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = profile_root / ".claude"
    claude_record(home, 101)
    write_history(home, [{"sessionId": SID, "timestamp": 5000}])
    monkeypatch.setattr(activity, "_run", lambda _: "101 Mon Sep 7 10:00:00 2026")
    agent = Agent("a:1.1", "a", "claude", pid=101)
    activity.enrich([agent], {})
    assert agent.last_input_at is None
    assert agent.session_id == ""


def test_codex_native_descriptor_and_active_goal(
    profile_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = profile_root / ".codex-rja"
    write_history(home, [{"session_id": SID, "ts": 5000}])
    with sqlite3.connect(home / "goals_1.sqlite") as connection:
        connection.execute("CREATE TABLE thread_goals(thread_id TEXT, status TEXT)")
        connection.execute("INSERT INTO thread_goals VALUES (?, 'active')", (SID,))
    rollout = home / "sessions" / "2026" / f"rollout-2026-09-06-{SID}.jsonl"
    monkeypatch.setattr(activity, "_run", lambda _: f"p202\nn{rollout}\n")
    agent = Agent("a:1.1", "a", "codex", pid=201)
    activity.enrich([agent], {201: [(202, "/vendor/codex --resume wrong")]})
    assert agent.session_id == SID
    assert agent.last_input_at == 5000
    assert agent.input_source == "codex-open-rollout/history-complete"
    assert agent.state == "bg"
    assert agent.extra["goal_status"] == "active"


def test_codex_argv_is_not_current_session_evidence(
    profile_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_history(profile_root / ".codex", [{"session_id": SID, "ts": 5000}])
    monkeypatch.setattr(activity, "_run", lambda _: None)
    agent = Agent("a:1.1", "a", "codex", pid=201)
    activity.enrich([agent], {201: [(202, f"/vendor/codex resume {SID}")]})
    assert agent.session_id == ""
    assert agent.last_input_at is None
    assert agent.extra["activity_probe_ok"] is False


def test_multiple_rollouts_are_ambiguous(
    profile_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = profile_root / ".codex"
    home.mkdir()
    other = "aaaaaaaa-1234-1234-1234-123456789abc"
    lines = "p201\n" + "".join(
        f"n{home}/sessions/rollout-2026-{sid}.jsonl\n" for sid in [SID, other]
    )
    monkeypatch.setattr(activity, "_run", lambda _: lines)
    agent = Agent("a:1.1", "a", "codex", pid=201)
    activity.enrich([agent], {})
    assert agent.last_input_at is None
    assert agent.extra["input_evidence"] == "ambiguous"


def test_future_and_explicit_synthetic_input_are_ignored(tmp_path: Path) -> None:
    write_history(tmp_path, [
        {"session_id": SID, "ts": 100, "text": "real input"},
        {"session_id": SID, "ts": time.time() + 1000},
        {"session_id": SID, "ts": 200, "text": "<hook_prompt>nudge"},
    ])
    assert activity._history(tmp_path, "codex")[0] == {SID: 100.0}


def test_subprocess_uses_utc_for_runtime_process_start_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TZ", "America/New_York")
    output = activity._run([sys.executable, "-c", "import os; print(os.getenv('TZ'))"])
    assert output == "UTC\n"
