"""A failure finder that uses the Claude Code CLI as an autonomous agent.

The finder seeds a sandbox directory with a ``run_episode.py`` helper, then
launches a ``claude`` CLI session pointing at that directory.  Claude has
full tool access (Bash, Write, Read, Glob) and iterates autonomously —
writing scripts, running them against the real environment, observing
results, and refining — until it writes ``failure_result.json`` or the
session budget/timeout is exhausted.

Prerequisites
-------------
* Install the Claude Code CLI::

    npm install -g @anthropic-ai/claude-code

* Authenticate (either):
    - ``claude auth login``  (free with subscription)
    - Set ``ANTHROPIC_API_KEY`` in your environment.
"""

import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from gym_failure_discovery.failure_finders.failure_finder import FailureFinder
from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor

_RESULT_FILENAME = "failure_result.json"
_HELPER_FILENAME = "run_episode.py"

# System prompt that orients Claude as a failure-finding agent.
_SYSTEM_PROMPT = (
    "You are a failure-finding agent for a gymnasium environment. "
    "Your only goal is to find an action sequence that triggers a "
    "failure (makes the monitor return True). "
    "IMPORTANT: Interact with the environment ONLY through "
    "run_episode.py. Do NOT import or inspect gym_failure_discovery "
    "modules directly — you must treat the environment as a black box "
    "accessible only via run_episode(). "
    "Write and run test scripts using the Bash tool. Iterate based on "
    "results. When you find a failure, run_episode() will automatically "
    f"write {_RESULT_FILENAME}. Verify it exists before finishing."
)

# Template for the helper script seeded into each sandbox.
# Placeholders __SETUP_SOURCE__ and __RESULT_FILENAME__ are replaced via
# str.replace() (not .format()) to avoid escaping Python dict literals.
_HELPER_TEMPLATE = '''"""Interact with the environment via run_episode(actions, seed=0).

IMPORTANT: Do NOT import or inspect gym_failure_discovery modules
directly. Interact with the environment ONLY via run_episode().
"""
import json
import sys
import numpy as np

# Environment and monitor construction (implementation hidden).
__SETUP_SOURCE__


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def run_episode(actions, seed=0):
    """Roll out *actions* and return a result dict.

    Writes __RESULT_FILENAME__ automatically when a failure is detected.
    """
    _env, _monitor = _make()
    obs, _ = _env.reset(seed=seed)
    _monitor.reset(obs)
    failed = False
    for action in actions:
        next_obs, _, terminated, _, _ = _env.step(int(action))
        if _monitor.step(obs, action, next_obs):
            obs = next_obs
            failed = True
            break
        obs = next_obs
        if terminated:
            break
    _env.close()
    result = {
        "failed": failed,
        "seed": seed,
        "last_obs": _serialize(obs),
    }
    if failed:
        result["actions"] = [int(a) for a in actions]
        with open("__RESULT_FILENAME__", "w") as _f:
            json.dump(result, _f, indent=2)
    return result


if __name__ == "__main__":
    _actions = json.loads(sys.argv[1])
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(json.dumps(run_episode(_actions, _seed), indent=2))
'''


class AgenticLLMFailureFinder(FailureFinder):
    """Failure finder that runs the Claude Code CLI as an autonomous agent.

    Claude is given a sandbox directory containing ``run_episode.py`` —
    a helper that wraps the real environment.  It uses its built-in tools
    (Bash, Write, Read, Glob) to write scripts, execute them, observe
    results, and iterate until it writes ``failure_result.json`` or the
    session budget/timeout is exhausted.

    The environment is accessed only through the gym action/observation
    interface — Claude never sees implementation source code.

    Parameters
    ----------
    env_setup_source:
        Python source that defines ``_make() -> (gym.Env, FailureMonitor)``.
        Embedded verbatim into the sandbox ``run_episode.py`` helper; must
        be self-contained (all imports included).  Example::

            from gym_failure_discovery.envs.conveyorbelt_env import (
                ConveyorBeltEnv, ConveyorBeltSceneSpec)
            from gym_failure_discovery.failure_monitors\\
.conveyorbelt_failure_monitor import ConveyorBeltFailureMonitor

            def _make():
                return (
                    ConveyorBeltEnv(ConveyorBeltSceneSpec()),
                    ConveyorBeltFailureMonitor(),
                )

    seed:
        RNG seed for reproducibility of the replay step.
    max_trajectory_length:
        Maximum steps per rollout inside the subprocess.
    session_timeout:
        Seconds to wait for the entire Claude session before killing it.
    model:
        Claude model alias or full ID (e.g. ``"sonnet"``, ``"opus"``,
        ``"claude-opus-4-5-20251101"``).
    max_budget_tokens:
        Approximate token budget passed to the Claude CLI (used for
        cost control via ``--max-turns`` equivalent in newer CLI versions).
    claude_cmd:
        Path or name of the Claude Code CLI binary.  Defaults to
        ``"claude"`` (must be on ``PATH``).
    sandbox_dir:
        Directory for sandbox files.  A fresh temporary directory is
        created and deleted automatically when this is ``None``.
    """

    def __init__(
        self,
        env_setup_source: str,
        seed: int = 0,
        max_trajectory_length: int = 500,
        session_timeout: int = 600,
        model: str = "sonnet",
        max_budget_tokens: int = 32000,
        claude_cmd: str = "claude",
        sandbox_dir: Path | None = None,
    ) -> None:
        self._env_setup_source = env_setup_source
        self._rng = np.random.default_rng(seed)
        self._max_trajectory_length = max_trajectory_length
        self._session_timeout = session_timeout
        self._model = model
        self._max_budget_tokens = max_budget_tokens
        self._claude_cmd = claude_cmd
        self._sandbox_dir = sandbox_dir

    def find_failure(
        self,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
    ) -> list[tuple[ObsType, ActType]] | None:
        """Run the Claude Code agent and return a failure trajectory."""
        from tomsutils.utils import sample_seed_from_rng  # pylint: disable=import-outside-toplevel
        replay_seed = sample_seed_from_rng(self._rng)
        if self._sandbox_dir is not None:
            self._sandbox_dir.mkdir(parents=True, exist_ok=True)
            return self._run_session(
                self._sandbox_dir, env, monitor, replay_seed
            )
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        ) as tmp:
            return self._run_session(
                Path(tmp), env, monitor, replay_seed
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_session(
        self,
        sandbox: Path,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
        replay_seed: int,
    ) -> list[tuple[ObsType, ActType]] | None:
        """Seed the sandbox, launch claude, and replay if a result is found."""
        self._seed_sandbox(sandbox)
        prompt = self._build_prompt(env, monitor)
        self._run_claude(prompt, sandbox)
        result_path = sandbox / _RESULT_FILENAME
        if result_path.exists():
            return self._replay_failure(result_path, env, monitor, replay_seed)
        return None

    def _seed_sandbox(self, sandbox: Path) -> None:
        """Write run_episode.py into the sandbox directory."""
        content = (
            _HELPER_TEMPLATE
            .replace("__SETUP_SOURCE__", self._env_setup_source)
            .replace("__RESULT_FILENAME__", _RESULT_FILENAME)
        )
        (sandbox / _HELPER_FILENAME).write_text(content, encoding="utf-8")

    def _build_prompt(
        self,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
    ) -> str:
        """Build the task prompt describing the failure-finding objective."""
        env_block = _get_docstring(env)
        monitor_block = f"```python\n{_get_source(monitor)}\n```"
        python_exe = sys.executable
        return (
            "Your task: find an action sequence that triggers a failure"
            " in the environment.\n\n"
            f"Environment description:\n{env_block}\n\n"
            f"Failure monitor:\n{monitor_block}\n\n"
            f"Action space: {env.action_space}\n"
            f"Observation space: {env.observation_space}\n\n"
            f"Use `{python_exe}` to run scripts (it has all packages).\n\n"
            "A helper script `run_episode.py` is in your working directory.\n"
            "Use it to test action sequences:\n\n"
            "```python\n"
            "from run_episode import run_episode\n"
            "result = run_episode([0, 1, 2, 3])\n"
            "# {'failed': bool, 'seed': int, 'last_obs': {...}}\n"
            f"# Writes {_RESULT_FILENAME} automatically when"
            " failed=True.\n"
            "```\n\n"
            "Strategy:\n"
            "1. Read run_episode.py to understand the observation format.\n"
            "2. Write a script that tries candidate action sequences.\n"
            "3. Run it with Bash and study the output.\n"
            "4. Iterate — vary the sequences based on results.\n"
            f"5. When a failure is found, verify {_RESULT_FILENAME} exists.\n\n"
            "CONSTRAINT: Do NOT inspect or import from gym_failure_discovery"
            " or any environment module directly.\n"
            "Interact with the environment ONLY via run_episode().\n"
        )

    def _run_claude(self, prompt: str, sandbox: Path) -> None:
        """Invoke the Claude Code CLI in the sandbox directory."""
        cmd = [
            self._claude_cmd,
            "--dangerously-skip-permissions",
            "--model", self._model,
            "--print", prompt,
        ]
        env_vars = {
            **os.environ,
            # Ensure the subprocess uses the same Python as this process.
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
        try:
            subprocess.run(
                cmd,
                cwd=str(sandbox),
                env=env_vars,
                timeout=self._session_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Claude Code CLI not found at '{self._claude_cmd}'.\n"
                "Install it with: npm install -g @anthropic-ai/claude-code\n"
                "Then authenticate: claude auth login"
            ) from exc

    def _replay_failure(
        self,
        result_path: Path,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
        fallback_seed: int,
    ) -> list[tuple[ObsType, ActType]] | None:
        """Replay the discovered actions in the parent process."""
        data = json.loads(result_path.read_text(encoding="utf-8"))
        actions: list[Any] = data.get("actions", [])
        if not actions:
            return None
        replay_seed = int(data.get("seed", fallback_seed))
        wrapped = FailureMonitorWrapper(env, monitor)
        obs, _ = wrapped.reset(seed=replay_seed)
        trajectory: list[tuple[ObsType, ActType]] = []
        for action in actions:
            trajectory.append((obs, action))
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            if reward == 1.0:
                return trajectory
            if terminated or truncated:
                break
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _actual(obj: object) -> object:
    """Unwrap gymnasium wrappers to reach the real object."""
    return obj.unwrapped if isinstance(obj, gym.Wrapper) else obj


def _get_source(obj: object) -> str:
    """Return the full source of the module that defines obj's class."""
    module = inspect.getmodule(_actual(obj).__class__)
    assert module is not None
    return inspect.getsource(module)


def _get_docstring(obj: object) -> str:
    """Return module- and class-level docstrings for obj's class."""
    cls = _actual(obj).__class__
    module = inspect.getmodule(cls)
    assert module is not None
    parts: list[str] = []
    if module.__doc__:
        parts.append(
            f"Module description:\n{inspect.cleandoc(module.__doc__)}"
        )
    class_doc = inspect.getdoc(cls)
    if class_doc:
        parts.append(f"Class `{cls.__name__}` description:\n{class_doc}")
    return (
        "\n\n".join(parts)
        if parts
        else f"(No docstring available for {cls.__name__})"
    )
