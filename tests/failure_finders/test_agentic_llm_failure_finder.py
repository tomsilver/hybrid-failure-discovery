"""Tests for AgenticLLMFailureFinder.

All tests pass ``claude_cmd="true"`` to skip the real Claude CLI session.
The ``true`` command exits immediately without writing any files, which lets
us test all of the sandbox seeding and replay infrastructure in isolation.
"""

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from gym_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from gym_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
    HoverCraftSceneSpec,
)
from gym_failure_discovery.failure_finders.agentic_llm_failure_finder import (
    AgenticLLMFailureFinder,
    _HELPER_FILENAME,
    _RESULT_FILENAME,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)

# ---------------------------------------------------------------------------
# env_setup_source strings embedded verbatim into run_episode.py
# ---------------------------------------------------------------------------

_CONVEYORBELT_SETUP = textwrap.dedent("""\
    from gym_failure_discovery.envs.conveyorbelt_env import (
        ConveyorBeltEnv, ConveyorBeltSceneSpec)
    from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
        ConveyorBeltFailureMonitor)

    def _make():
        return (
            ConveyorBeltEnv(ConveyorBeltSceneSpec()),
            ConveyorBeltFailureMonitor(),
        )
""")

_HOVERCRAFT_SETUP = textwrap.dedent("""\
    from gym_failure_discovery.envs.hovercraft_env import (
        HoverCraftEnv, HoverCraftSceneSpec)
    from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
        HoverCraftFailureMonitor)

    def _make():
        spec = HoverCraftSceneSpec()
        return HoverCraftEnv(spec), HoverCraftFailureMonitor(spec)
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finder(
    env_setup_source: str,
    sandbox_dir: Path,
    **kwargs: Any,
) -> AgenticLLMFailureFinder:
    """Return a finder that skips the real Claude CLI (uses ``true``)."""
    return AgenticLLMFailureFinder(
        env_setup_source=env_setup_source,
        seed=0,
        claude_cmd="true",  # exits immediately without writing any files
        sandbox_dir=sandbox_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sandbox_helper_seeded(tmp_path: Path) -> None:
    """run_episode.py is written to sandbox_dir before the agent session."""
    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    finder.find_failure(env, monitor)
    env.close()

    assert (tmp_path / _HELPER_FILENAME).exists(), (
        f"{_HELPER_FILENAME} was not seeded into the sandbox"
    )
    source = (tmp_path / _HELPER_FILENAME).read_text()
    assert "def run_episode(" in source
    assert "def _make(" in source


def test_setup_source_embedded_in_helper(tmp_path: Path) -> None:
    """env_setup_source is embedded verbatim inside run_episode.py."""
    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    finder.find_failure(env, monitor)
    env.close()

    helper_src = (tmp_path / _HELPER_FILENAME).read_text()
    assert "ConveyorBeltEnv" in helper_src
    assert "ConveyorBeltFailureMonitor" in helper_src


def test_no_failure_returns_none(tmp_path: Path) -> None:
    """When the agent writes no result file, find_failure returns None."""
    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    result = finder.find_failure(env, monitor)
    env.close()

    assert result is None
    assert not (tmp_path / _RESULT_FILENAME).exists()


def test_prewritten_result_triggers_replay(tmp_path: Path) -> None:
    """When failure_result.json is pre-written, find_failure replays it."""
    result_data = {
        "failed": True,
        "seed": 0,
        "actions": [3, 1, 3, 2, 3, 3, 3, 1],
        "last_obs": {},
    }
    (tmp_path / _RESULT_FILENAME).write_text(
        json.dumps(result_data), encoding="utf-8"
    )

    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    result = finder.find_failure(env, monitor)
    env.close()

    assert result is not None, "Should have replayed the pre-written sequence"
    actions_in_traj = [action for _, action in result]
    assert actions_in_traj == [3, 1, 3, 2, 3, 3, 3, 1]


def test_replay_trajectory_format(tmp_path: Path) -> None:
    """Replayed trajectory consists of (obs, action) pairs ending at failure."""
    result_data = {
        "failed": True,
        "seed": 0,
        "actions": [3, 1, 3, 2, 3, 3, 3, 1],
        "last_obs": {},
    }
    (tmp_path / _RESULT_FILENAME).write_text(
        json.dumps(result_data), encoding="utf-8"
    )

    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    result = finder.find_failure(env, monitor)
    env.close()

    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
    obs, action = result[0]
    assert isinstance(obs, dict), "Observation should be a dict"
    assert env.action_space.contains(action)


def test_empty_actions_in_result_returns_none(tmp_path: Path) -> None:
    """A failure_result.json with no actions key causes find_failure → None."""
    (tmp_path / _RESULT_FILENAME).write_text(
        json.dumps({"failed": True, "seed": 0, "last_obs": {}}),
        encoding="utf-8",
    )
    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    result = finder.find_failure(env, monitor)
    env.close()

    assert result is None


def test_sandbox_dir_files_persist_after_run(tmp_path: Path) -> None:
    """With sandbox_dir set, the helper persists after the run."""
    env = ConveyorBeltEnv(ConveyorBeltSceneSpec())
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    finder.find_failure(env, monitor)
    env.close()

    assert (tmp_path / _HELPER_FILENAME).exists()
    assert not (tmp_path / _RESULT_FILENAME).exists()


def test_different_envs_use_correct_setup_source(tmp_path: Path) -> None:
    """env_setup_source determines which env is constructed in run_episode.py."""
    hovercraft_sandbox = tmp_path / "hovercraft"
    hovercraft_sandbox.mkdir()
    spec = HoverCraftSceneSpec()
    env = HoverCraftEnv(spec)
    monitor = HoverCraftFailureMonitor(spec)
    finder = _make_finder(_HOVERCRAFT_SETUP, hovercraft_sandbox)
    finder.find_failure(env, monitor)
    env.close()

    helper_src = (hovercraft_sandbox / _HELPER_FILENAME).read_text()
    assert "HoverCraftEnv" in helper_src
    assert "HoverCraftFailureMonitor" in helper_src
    assert "ConveyorBelt" not in helper_src


@pytest.mark.make_videos
def test_prewritten_result_replays_with_video(
    tmp_path: Path,
    maybe_record: Any,
) -> None:
    """Replay from failure_result.json works with a recorded (video) env."""
    result_data = {
        "failed": True,
        "seed": 0,
        "actions": [3, 1, 3, 2, 3, 3, 3, 1],
        "last_obs": {},
    }
    (tmp_path / _RESULT_FILENAME).write_text(
        json.dumps(result_data), encoding="utf-8"
    )

    env = maybe_record(ConveyorBeltEnv(ConveyorBeltSceneSpec()))
    monitor = ConveyorBeltFailureMonitor()
    finder = _make_finder(_CONVEYORBELT_SETUP, tmp_path)
    result = finder.find_failure(env, monitor)
    env.close()

    assert result is not None
    assert len(result) > 0
