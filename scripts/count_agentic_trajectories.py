"""Count agentic LLM sessions until failure for 15 seeds across all envs.

The AgenticLLMFailureFinder launches an LLM agent that iteratively writes
and runs Python scripts against the real environment, observing results and
refining its search across multiple turns per seed.  The environment is
accessed only through the gym action/observation interface — the agent
never sees implementation source code.
"""

import textwrap
import numpy as np

from gym_failure_discovery.envs.blocks_env import BlocksEnv, BlocksSceneSpec
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
)
from gym_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)

SEEDS: list[int] = [
    0, 79, 158, 237, 317, 396, 475, 555, 634, 713, 793, 872, 951, 1031, 1110,
]
MODEL = "sonnet"          # claude alias or full model ID
SESSION_TIMEOUT = 600    # seconds for the entire Claude session per seed

# ---------------------------------------------------------------------------
# env_setup_source strings — embedded verbatim into run_episode.py.
# Each defines a _make() function returning (env, monitor).
# ---------------------------------------------------------------------------

_HOVERCRAFT_SETUP = textwrap.dedent("""\
    from gym_failure_discovery.envs.hovercraft_env import (
        HoverCraftEnv, HoverCraftSceneSpec)
    from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor\
 import HoverCraftFailureMonitor

    def _make():
        spec = HoverCraftSceneSpec()
        return HoverCraftEnv(spec), HoverCraftFailureMonitor(spec)
""")

_BLOCKS_SETUP = textwrap.dedent("""\
    from gym_failure_discovery.envs.blocks_env import BlocksEnv, BlocksSceneSpec
    from gym_failure_discovery.failure_monitors.blocks_failure_monitor\
 import BlocksFailureMonitor

    def _make():
        spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
        env = BlocksEnv(spec)
        return env, BlocksFailureMonitor(env)
""")

_CONVEYORBELT_SETUP = textwrap.dedent("""\
    from gym_failure_discovery.envs.conveyorbelt_env import (
        ConveyorBeltEnv, ConveyorBeltSceneSpec)
    from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor\
 import ConveyorBeltFailureMonitor

    def _make():
        return (
            ConveyorBeltEnv(ConveyorBeltSceneSpec(), render_mode="rgb_array"),
            ConveyorBeltFailureMonitor(),
        )
""")


def run_env(
    label: str,
    make_env_monitor: object,
    env_setup_source: str,
    max_traj: int,
) -> None:
    """Run agentic failure finder over all seeds; print found/not-found."""
    found_count = 0
    for seed in SEEDS:
        env, monitor = make_env_monitor()  # type: ignore[operator]
        finder = AgenticLLMFailureFinder(
            env_setup_source=env_setup_source,
            seed=seed,
            max_trajectory_length=max_traj,
            session_timeout=SESSION_TIMEOUT,
            model=MODEL,
        )
        result = finder.find_failure(env, monitor)
        env.close()
        status = "FOUND" if result is not None else "NOT FOUND"
        if result is not None:
            found_count += 1
        print(f"  seed={seed:4d}  {status}")

    n_miss = len(SEEDS) - found_count
    pct = 100.0 * found_count / len(SEEDS)
    print(
        f"\n  {label}:\n"
        f"    found={found_count}/{len(SEEDS)} ({pct:.0f}%)"
        f"  no_failure={n_miss}/{len(SEEDS)}\n"
    )


def main() -> None:
    """Run agentic LLM failure finder experiments and report statistics."""
    print(f"Seeds: {SEEDS}")
    print(f"Model: {MODEL}  session_timeout: {SESSION_TIMEOUT}s\n")

    # ── Hovercraft ────────────────────────────────────────────────────────────
    print(
        "=== Agentic / Hovercraft"
        f" (max_turns={MAX_AGENT_TURNS}, max_traj=500) ==="
    )
    run_env(
        "Agentic / Hovercraft",
        lambda: (
            HoverCraftEnv(HoverCraftSceneSpec()),
            HoverCraftFailureMonitor(HoverCraftSceneSpec()),
        ),
        env_setup_source=_HOVERCRAFT_SETUP,
        max_traj=500,
    )

    # ── Blocks ────────────────────────────────────────────────────────────────
    print(
        "=== Agentic / Blocks"
        f" (max_turns={MAX_AGENT_TURNS}, max_traj=20) ==="
    )

    def _blocks() -> tuple:
        spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
        raw = BlocksEnv(spec)
        return raw, BlocksFailureMonitor(raw)

    run_env(
        "Agentic / Blocks",
        _blocks,
        env_setup_source=_BLOCKS_SETUP,
        max_traj=20,
    )

    # ── ConveyorBelt ──────────────────────────────────────────────────────────
    print(
        "=== Agentic / ConveyorBelt"
        f" (max_turns={MAX_AGENT_TURNS}, max_traj=200) ==="
    )
    run_env(
        "Agentic / ConveyorBelt",
        lambda: (
            ConveyorBeltEnv(ConveyorBeltSceneSpec(), render_mode="rgb_array"),
            ConveyorBeltFailureMonitor(),
        ),
        env_setup_source=_CONVEYORBELT_SETUP,
        max_traj=200,
    )


if __name__ == "__main__":
    main()
