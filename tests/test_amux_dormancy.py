"""Dormancy is input age plus a waiting state, never output age alone."""

from argparse import Namespace

import pytest

from claude_code_tools.amux import cli, detect, render
from claude_code_tools.amux.model import Agent


@pytest.mark.parametrize(
    "state,stamp,expected",
    [("idle", 1, "dormant"), ("input", 1, "dormant"),
     ("bg", 1, "working"), ("busy", 1, "working"),
     ("idle", None, "unknown"), ("idle", 300000, "recent")],
)
def test_classification(state: str, stamp: float | None, expected: str) -> None:
    agent = Agent("a:1.1", "a", "claude", state=state, last_input_at=stamp)
    agent.classify_inactivity(now=300001)
    assert agent.inactivity == expected


def test_threshold_boundary_and_refresh() -> None:
    agent = Agent("a:1.1", "a", "codex", last_input_at=1)
    agent.classify_inactivity(now=72 * 3600 + 1)
    assert agent.inactivity == "recent"
    agent.classify_inactivity(now=72 * 3600 + 2)
    assert agent.inactivity == "dormant"
    agent.classify_inactivity(threshold_hours=100, now=72 * 3600 + 2)
    assert agent.inactivity == "recent"


def test_display_oldest_unknown_last_and_filter() -> None:
    agents = [Agent("a:1.1", "a", "claude"),
              Agent("a:1.2", "a", "claude", last_input_at=100),
              Agent("a:1.3", "a", "codex", state="bg", last_input_at=1)]
    args = Namespace(dormant_hours=72, dormant=False, sort="oldest")
    assert [a.pane for a in cli._display(agents, args)] == [
        "a:1.3", "a:1.2", "a:1.1"]
    args.dormant = True
    assert [a.pane for a in cli._display(agents, args)] == ["a:1.2"]
    row = render.picker_row(agents[1], colour=False)
    assert "dormant" in row
    assert "INPUT AGE" in render.table(agents, colour=False)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_threshold(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["list", "--dormant-hours", value])


def test_idle_subagent_rows_do_not_mean_working() -> None:
    screen = "❯\n⏵⏵ bypass permissions on\n◯ chk4 Checking … idle  "
    assert detect.detect_state(screen, "claude") == "idle"
    assert detect.detect_state(screen + "\n◯ chk5 Grepping …", "claude") == "bg"


def test_running_shell_keeps_pane_working() -> None:
    screen = "✻ Baked for 42m · 1 shell still running\n❯\n⏵⏵ bypass permissions"
    assert detect.detect_state(screen, "claude") == "bg"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "yesterday"])
def test_invalid_cached_age_rejected(value: object) -> None:
    data = Agent("a:1.1", "a", "codex").to_dict()
    data["last_input_at"] = value
    assert Agent.from_dict(data) is None


def test_past_shell_command_is_not_background() -> None:
    screen = "Ran 1 shell command\n❯\n⏵⏵ bypass permissions on"
    assert detect.detect_state(screen, "claude") == "idle"


def test_prompt_with_active_goal_is_not_dormant() -> None:
    agent = Agent("a:1.1", "a", "codex", state="input", last_input_at=1,
                  extra={"goal_status": "active"})
    agent.classify_inactivity(now=300001)
    assert agent.inactivity == "working"


@pytest.mark.parametrize("extra", [None, [], "broken"])
def test_invalid_cached_extra_rejected(extra: object) -> None:
    data = Agent("a:1.1", "a", "codex").to_dict()
    data["extra"] = extra
    assert Agent.from_dict(data) is None


def test_cached_nonmatching_picker_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    from subprocess import CompletedProcess

    cached = Agent("a:1.1", "a", "claude", state="busy", last_input_at=1)
    fresh = Agent("a:1.1", "a", "claude", state="idle", last_input_at=1)
    calls: list[str] = []
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/fzf")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "_agents_for_display", lambda _: ([cached], True))
    monkeypatch.setattr(cli, "_fresh", lambda: [fresh])

    def run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(str(kwargs["input"]))
        return CompletedProcess(command, 1, stdout="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    args = cli.build_parser().parse_args(["pick", "--dormant", "--sort", "oldest"])
    assert cli.cmd_pick(args) == 0
    assert len(calls) == 1
    assert "dormant" in calls[0]


@pytest.mark.parametrize("arguments", [
    ["--dormant", "--max-age", "0"],
    ["--max-age", "0", "--dormant"],
    ["--dormant", "--max-age=0"],
    ["--max-age=0", "--dormant"],
    ["--max-age", "0", "pick", "--dormant"],
    ["pick", "--dormant", "--max-age", "0"],
])
def test_picker_option_order(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Namespace] = []
    monkeypatch.setattr(cli.scan, "tmux_available", lambda: True)
    monkeypatch.setattr(cli, "cmd_pick", lambda args: seen.append(args) or 0)
    assert cli.main(arguments) == 0
    assert len(seen) == 1
    assert seen[0].max_age == 0
    assert seen[0].dormant is True
