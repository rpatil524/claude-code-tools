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
