"""Read bounded, session-scoped evidence of submitted agent input.

History timestamps are not pane activity: another pane can resume the same
conversation. Missing or ambiguous evidence stays unknown. No transcript mtimes,
launch arguments, or terminal output are used as input timestamps.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .model import Agent

_HISTORY_LIMIT = 16 * 1024 * 1024
_UUID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
_ROLLOUT = re.compile(rf"rollout-.*-({_UUID})\.jsonl$")


def _homes(kind: str) -> list[Path]:
    """Discover immediate profile directories and the explicit current home."""
    root = Path.home()
    paths = [root / f".{kind}"]
    try:
        paths.extend(p for p in root.iterdir() if p.name.startswith(f".{kind}"))
    except OSError:
        pass
    variable = "CLAUDE_CONFIG_DIR" if kind == "claude" else "CODEX_HOME"
    configured = os.environ.get(variable)
    if configured:
        paths.append(Path(configured).expanduser())
    result: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
            if path.is_dir() and resolved not in result:
                result.append(resolved)
        except OSError:
            continue
    return result


def _history(home: Path, kind: str) -> tuple[dict[str, float], str]:
    """Read a bounded history tail, returning timestamps and read status."""
    try:
        with (home / "history.jsonl").open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - _HISTORY_LIMIT)
            stream.seek(start)
            data = stream.read(_HISTORY_LIMIT)
        if start:
            data = data.partition(b"\n")[2]
    except OSError:
        return {}, "unavailable"
    times: dict[str, float] = {}
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        submitted = record.get("display" if kind == "claude" else "text", "")
        if isinstance(submitted, str) and submitted.lstrip().startswith(
            ("<hook_prompt", "<task-notification", "<subagent", "<system-reminder")
        ):
            continue
        sid = record.get("sessionId" if kind == "claude" else "session_id")
        timestamp = record.get("timestamp" if kind == "claude" else "ts")
        if not isinstance(sid, str) or not sid:
            continue
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            continue
        try:
            value = float(timestamp) / (1000 if kind == "claude" else 1)
        except OverflowError:
            continue
        if math.isfinite(value) and 0 < value <= time.time():
            times[sid] = max(times.get(sid, 0), value)
    return times, "tail" if start else "complete"


def _run(args: list[str]) -> str | None:
    """Run a bounded read-only subprocess; failure is distinct from no records."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5,
            env={**os.environ, "TZ": "UTC"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _starts(pids: set[int]) -> dict[int, str]:
    """Get process start identities to reject stale PID metadata."""
    if not pids:
        return {}
    output = _run(["ps", "-p", ",".join(map(str, sorted(pids))), "-o", "pid=,lstart="])
    result: dict[int, str] = {}
    for line in (output or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            result[int(parts[0])] = " ".join(parts[1].split())
    return result


def _claude_record(home: Path, pid: int, start: str) -> dict[str, Any] | None:
    """Accept only runtime metadata matching a currently observed process."""
    try:
        with (home / "sessions" / f"{pid}.json").open("rb") as stream:
            record = json.loads(stream.read(64 * 1024))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    recorded_start = record.get("procStart")
    sid = record.get("sessionId")
    if (
        record.get("pid") != pid
        or not isinstance(recorded_start, str)
        or " ".join(recorded_start.split()) != start
        or not start
        or not isinstance(sid, str)
        or not sid
    ):
        return None
    return record


def _codex_pids(agent: Agent, tree: dict[int, list[tuple[int, str]]]) -> set[int]:
    """Include native Codex children of the selected launcher, bounded in size."""
    pids = {agent.pid} if agent.pid > 0 else set()
    queue = list(pids)
    while queue and len(pids) < 32:
        for pid, command in tree.get(queue.pop(0), []):
            # Follow only the launcher/native chain, never arbitrary tool workers.
            executable = command.split()[0] if command.split() else ""
            if pid not in pids and Path(executable).name == "codex":
                pids.add(pid)
                queue.append(pid)
    return pids


def _rollouts(pids: set[int]) -> tuple[dict[int, set[Path]], bool]:
    """Identify open rollout descriptors in one lsof query, without reading them."""
    if not pids:
        return {}, True
    output = _run(["lsof", "-nP", "-a", "-p", ",".join(map(str, sorted(pids))), "-Fn"])
    if output is None:
        return {}, False
    found: dict[int, set[Path]] = {}
    pid = 0
    for line in output.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
        elif pid in pids and line.startswith("n"):
            path = Path(line[1:])
            if _ROLLOUT.fullmatch(path.name):
                found.setdefault(pid, set()).add(path)
    return found, True


def _goal_status(home: Path, sid: str) -> str:
    """Read only the matched thread's goal status, never its objective."""
    try:
        uri = (home / "goals_1.sqlite").as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=0.1)
        try:
            row = connection.execute(
                "SELECT status FROM thread_goals WHERE thread_id = ?", (sid,)
            ).fetchone()
        finally:
            connection.close()
    except (sqlite3.Error, ValueError):
        return "unknown"
    return str(row[0]) if row else "none"


def enrich(agents: list[Agent], child_map: dict[int, list[tuple[int, str]]]) -> None:
    """Add last submitted-input evidence without modifying running sessions.

    Args:
        agents: Live pane records to enrich in place.
        child_map: Process snapshot containing launcher/native parentage.
    """
    homes = {kind: _homes(kind) for kind in {a.kind for a in agents}}
    histories: dict[tuple[Path, str], tuple[dict[str, float], str]] = {}
    starts = _starts({a.pid for a in agents if a.kind == "claude" and a.pid > 0})
    pid_sets = {
        a.pane: _codex_pids(a, child_map) for a in agents if a.kind == "codex"
    }
    paths, lsof_ok = _rollouts(set().union(*pid_sets.values()) if pid_sets else set())
    identities: list[tuple[Agent, Path, str]] = []
    for agent in agents:
        agent.session_id = ""
        agent.last_input_at = None
        agent.input_source = "unknown"
        agent.input_scope = "unknown"
        matches: list[tuple[Path, str]] = []
        runtime_records: list[dict[str, Any]] = []
        if agent.kind == "claude":
            for home in homes["claude"]:
                record = _claude_record(home, agent.pid, starts.get(agent.pid, ""))
                if record:
                    matches.append((home, record["sessionId"]))
                    runtime_records.append(record)
            method = "claude-pid"
        else:
            open_paths = set().union(
                *(paths.get(p, set()) for p in pid_sets[agent.pane])
            )
            for path in open_paths:
                match = _ROLLOUT.fullmatch(path.name)
                for home in homes["codex"]:
                    if match and path.is_relative_to(home / "sessions"):
                        matches.append((home, match.group(1)))
            method = "codex-open-rollout"
            agent.extra["activity_probe_ok"] = lsof_ok
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            agent.extra["input_evidence"] = "ambiguous" if matches else "unavailable"
            continue
        home, sid = matches[0]
        agent.session_id = sid
        if agent.kind == "claude":
            record = runtime_records[0]
            status = record.get("status", "unknown")
            agent.extra["runtime_status"] = status
            if isinstance(record.get("name"), str) and record["name"]:
                agent.name = record["name"]
            if status in ("shell", "working", "running") and agent.state == "idle":
                agent.state = "bg"
        else:
            goal = _goal_status(home, sid)
            agent.extra["goal_status"] = goal
            if goal == "active" and agent.state == "idle":
                agent.state = "bg"
        agent.input_scope = "session"
        key = (home, agent.kind)
        if key not in histories:
            histories[key] = _history(home, agent.kind)
        timestamps, status = histories[key]
        agent.last_input_at = timestamps.get(sid)
        agent.input_source = f"{method}/history-{status}"
        agent.extra["activity_home"] = str(home)
        agent.extra["input_evidence"] = "found" if sid in timestamps else "unavailable"
        identities.append((agent, home, sid))
    counts = Counter((home, sid) for _, home, sid in identities)
    for agent, home, sid in identities:
        if counts[home, sid] > 1:
            agent.input_scope = "shared-session"
