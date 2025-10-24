"""Tests for conveyorbelt_failure_monitor.py."""

from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltAction,
    ConveyorBeltEnv,
)
from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


def test_conveyorbelt_failure_monitor():
    """Tests for conveyorbelt_failure_monitor.py."""
    env = ConveyorBeltEnv(seed=123)
    monitor = ConveyorBeltFailureMonitor(env.scene_spec)

    # --- Initial state ---
    state, _ = env.reset(seed=123)
    initial_score = monitor.get_robustness_score(state)
    print("Initial robustness score:", initial_score)

    # --- Case 1: Overlap (two consecutive drops) ---
    state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # t = 0
    state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # t = 1
    overlap_score = monitor.get_robustness_score(state)
    failed_overlap = getattr(monitor, "_check_failures")(state)
    print("After consecutive drops:")
    print("  Failure detected:", failed_overlap)
    print("  Robustness score:", overlap_score)

    # --- Case 2: Spaced drops (safe) ---
    env = ConveyorBeltEnv(seed=42)
    monitor = ConveyorBeltFailureMonitor(env.scene_spec)
    state, _ = env.reset(seed=42)
    state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # t = 0
    for _ in range(60):  # let it move forward for ~0.6 s
        state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=False))
    state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # second drop
    spaced_score = monitor.get_robustness_score(state)
    failed_spaced = getattr(monitor, "_check_failures")(state)
    print("After spaced drops:")
    print("  Failure detected:", failed_spaced)
    print("  Robustness score:", spaced_score)
