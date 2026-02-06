"""
Generate basic statistics and plots for the conveyor-belt random shooting
failure finder.

Outputs (under reports/conveyor_random_shooting/):
- CSV of per-run results
- PNGs for success rate heatmap, trajectories-to-failure histogram,
  and steps-to-failure histogram
- JSON summary of aggregate success rates
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / "mpl_cache"))
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

try:
    from tomsutils.utils import sample_seed_from_rng
except ImportError:
    # Lightweight fallback: sample a 32-bit seed from a Generator
    def sample_seed_from_rng(rng: np.random.Generator) -> int:  # type: ignore
        return int(rng.integers(0, 2**31 - 1))

from hybrid_failure_discovery.commander.random_commander import RandomCommander
from hybrid_failure_discovery.commander.random_initial_state_commander import (
    RandomInitialStateCommander,
)
from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from hybrid_failure_discovery.structs import Trajectory
from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


@dataclass
class TrialResult:
    seed: int
    max_num_trajectories: int
    max_trajectory_length: int
    secret_length: int
    found: bool
    traj_index_found: int | None
    steps_to_failure: int | None
    simulated_time_to_failure: float | None
    min_gap_at_failure: float | None
    boxes_at_failure: int | None
    boxes_max_seen: int
    command_counts_off: int
    command_counts_slow: int
    command_counts_mid: int
    command_counts_fast: int


def compute_min_gap(state: Any) -> float:
    """Smallest gap between box positions; inf if <=1 box."""
    if len(state.positions) <= 1:
        return math.inf
    sorted_positions = np.sort(state.positions)
    diffs = np.diff(sorted_positions)
    return float(np.min(diffs))


def simulate_single_trajectory(
    env: ConveyorBeltEnv,
    controller: ConveyorBeltController,
    monitor: ConveyorBeltFailureMonitor,
    commander: RandomCommander,
    max_len: int,
    rng: np.random.Generator,
    initial_state: Any,
) -> tuple[bool, Trajectory | None, int, float, float, int, Counter[str]]:
    """
    Run one trajectory from a provided initial state; return:
    (failure_found, trajectory, steps, min_gap, boxes_max, command_counts)
    """
    state = initial_state
    trajectory = Trajectory([state], [], [])

    min_gap = compute_min_gap(state)
    boxes_max = len(state.positions)
    command_counts: Counter[str] = Counter()

    for step in range(max_len):
        command = commander.get_command()
        mode = command.mode if isinstance(command, ConveyorBeltCommand) else str(command)
        command_counts[mode] += 1

        action = controller.step(state, command)
        next_states = env.get_next_states(state, action)
        next_states.seed(sample_seed_from_rng(rng))
        state = next_states.sample()

        min_gap = min(min_gap, compute_min_gap(state))
        boxes_max = max(boxes_max, len(state.positions))

        trajectory.actions.append(action)
        trajectory.commands.append(command)
        trajectory.observations.append(state)

        if monitor.step(command, action, state):
            return True, trajectory, step + 1, min_gap, boxes_max, command_counts

        commander.update(action, state)

    return False, None, max_len, min_gap, boxes_max, command_counts


def run_trial(
    seed: int,
    max_num_trajectories: int,
    max_trajectory_length: int,
    secret_length: int,
) -> TrialResult:
    """Run a full random-shooting search trial and collect metrics."""
    scene_spec = ConveyorBeltSceneSpec(
        box_width=0.3, conveyor_belt_velocity=2.0, dt=0.01, belt_length=3.0
    )
    # Add min_spacing dynamically as tests do
    object.__setattr__(scene_spec, "min_spacing", 0.1)

    env = ConveyorBeltEnv(scene_spec=scene_spec)
    base_seq = ["fast", "mid", "slow", "fast", "slow", "mid", "slow", "fast"]
    # Repeat and trim to desired length
    repeats = math.ceil(secret_length / len(base_seq))
    secret_mode_sequence = (base_seq * repeats)[:secret_length]
    controller = ConveyorBeltController(
        seed=seed,
        scene_spec=env.scene_spec,
        secret_failure_mode_sequence=secret_mode_sequence,
    )
    monitor = ConveyorBeltFailureMonitor(env.scene_spec)

    rng = np.random.default_rng(seed)

    # Command space and initializer
    command_space = controller.get_command_space()
    initializer = RandomInitialStateCommander(env.get_initial_states())

    best_boxes_max = 0

    for traj_idx in range(max_num_trajectories):
        init_seed = sample_seed_from_rng(rng)
        initializer.seed(init_seed)
        initial_state = initializer.initialize()

        commander_seed = sample_seed_from_rng(rng)
        commander = RandomCommander(command_space)
        commander.seed(commander_seed)

        # Set the initial state into env/controller/monitor
        controller.reset(initial_state)
        monitor.reset(initial_state)
        commander.reset(initial_state)
        env._current_state = initial_state  # type: ignore[attr-defined]

        failure_found, traj, steps, min_gap, boxes_max, command_counts = (
            simulate_single_trajectory(
                env,
                controller,
                monitor,
                commander,
                max_trajectory_length,
                rng,
                initial_state,
            )
        )
        best_boxes_max = max(best_boxes_max, boxes_max)

        if failure_found and traj is not None:
            # Metrics at failure
            final_state = traj.observations[-1]
            boxes_at_failure = len(final_state.positions)
            sim_time = steps * scene_spec.dt
            return TrialResult(
                seed=seed,
                max_num_trajectories=max_num_trajectories,
                max_trajectory_length=max_trajectory_length,
        secret_length=secret_length,
                found=True,
                traj_index_found=traj_idx + 1,
                steps_to_failure=steps,
                simulated_time_to_failure=sim_time,
                min_gap_at_failure=min_gap,
                boxes_at_failure=boxes_at_failure,
                boxes_max_seen=boxes_max,
                command_counts_off=command_counts.get("off", 0),
                command_counts_slow=command_counts.get("slow", 0),
                command_counts_mid=command_counts.get("mid", 0),
                command_counts_fast=command_counts.get("fast", 0),
            )

    # No failure found
    return TrialResult(
        seed=seed,
        max_num_trajectories=max_num_trajectories,
        max_trajectory_length=max_trajectory_length,
        secret_length=secret_length,
        found=False,
        traj_index_found=None,
        steps_to_failure=None,
        simulated_time_to_failure=None,
        min_gap_at_failure=None,
        boxes_at_failure=None,
        boxes_max_seen=best_boxes_max,
        command_counts_off=0,
        command_counts_slow=0,
        command_counts_mid=0,
        command_counts_fast=0,
    )


def aggregate_and_plot(results: list[TrialResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = out_dir / "conveyor_random_shooting_results.csv"
    fieldnames = list(asdict(results[0]).keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Aggregate success rates by params
    grid_success: dict[tuple[int, int], list[bool]] = defaultdict(list)
    success_vs_secret: dict[int, list[bool]] = defaultdict(list)
    traj_indices: list[int] = []
    steps_list: list[int] = []
    min_gap_list: list[float] = []
    boxes_max_list: list[int] = []
    boxes_at_failure_list: list[int] = []
    cmd_usage_found: list[Counter[str]] = []
    cmd_usage_not_found: list[Counter[str]] = []
    by_len_steps: dict[int, list[int]] = defaultdict(list)
    by_len_traj_idx: dict[int, list[int]] = defaultdict(list)
    by_len_mingap: dict[int, list[float]] = defaultdict(list)
    by_numtraj_success: dict[int, list[bool]] = defaultdict(list)

    for r in results:
        key = (r.max_num_trajectories, r.max_trajectory_length)
        grid_success[key].append(r.found)
        success_vs_secret[r.secret_length].append(r.found)
        by_numtraj_success[r.max_num_trajectories].append(r.found)
        if r.found and r.traj_index_found is not None:
            traj_indices.append(r.traj_index_found)
            by_len_traj_idx[r.max_trajectory_length].append(r.traj_index_found)
        if r.found and r.steps_to_failure is not None:
            steps_list.append(r.steps_to_failure)
            by_len_steps[r.max_trajectory_length].append(r.steps_to_failure)
        if r.min_gap_at_failure is not None:
            min_gap_list.append(r.min_gap_at_failure)
            by_len_mingap[r.max_trajectory_length].append(r.min_gap_at_failure)
        boxes_max_list.append(r.boxes_max_seen)
        if r.boxes_at_failure is not None:
            boxes_at_failure_list.append(r.boxes_at_failure)

        cmd_counts = Counter(
            {
                "off": r.command_counts_off,
                "slow": r.command_counts_slow,
                "mid": r.command_counts_mid,
                "fast": r.command_counts_fast,
            }
        )
        if r.found:
            cmd_usage_found.append(cmd_counts)
        else:
            cmd_usage_not_found.append(cmd_counts)

    # Success heatmap
    counts = sorted({k[0] for k in grid_success.keys()})
    lengths = sorted({k[1] for k in grid_success.keys()})
    heat = np.zeros((len(counts), len(lengths)))
    for i, c in enumerate(counts):
        for j, l in enumerate(lengths):
            vals = grid_success[(c, l)]
            heat[i, j] = np.mean(vals) if vals else 0.0

    plt.figure(figsize=(6, 4))
    im = plt.imshow(heat, origin="lower", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Success rate")
    plt.xticks(range(len(lengths)), lengths)
    plt.yticks(range(len(counts)), counts)
    plt.xlabel("max_trajectory_length")
    plt.ylabel("max_num_trajectories")
    plt.title("Conveyor Random Shooting: Success Rate")
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate_heatmap.png", dpi=200)
    plt.close()

    # Histogram: trajectory index to failure (use fixed bins for consistency)
    if traj_indices:
        plt.figure(figsize=(5, 3))
        max_traj_idx = max(traj_indices)
        bins = range(1, max_traj_idx + 2)
        plt.hist(traj_indices, bins=bins, edgecolor="k")
        plt.xlabel("Trajectory index to failure")
        plt.ylabel("Count")
        plt.title("Distribution of Trajectory Index to First Failure")
        plt.xlim(0.5, max_traj_idx + 0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "traj_index_hist.png", dpi=200)
        plt.close()

    # Histogram: steps to failure (align sizing/limits to traj_index_hist)
    if steps_list:
        plt.figure(figsize=(5, 3))
        steps_max = max(steps_list)
        bins = range(1, steps_max + 2)
        plt.hist(steps_list, bins=bins, edgecolor="k")
        plt.xlabel("Steps to failure (within trajectory)")
        plt.ylabel("Count")
        plt.title("Distribution of Steps to Failure")
        plt.xlim(0.5, steps_max + 0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "steps_to_failure_hist.png", dpi=200)
        plt.close()

    # Histogram: min gap at failure
    finite_min_gap = [v for v in min_gap_list if math.isfinite(v)]
    if finite_min_gap:
        plt.figure(figsize=(5, 3))
        plt.hist(finite_min_gap, bins=20, edgecolor="k")
        plt.xlabel("Minimum gap at failure")
        plt.ylabel("Count")
        plt.title("Distribution of Min Gap at Failure")
        plt.tight_layout()
        plt.savefig(out_dir / "min_gap_at_failure_hist.png", dpi=200)
        plt.close()

    # Histogram: boxes seen (max)
    if boxes_max_list:
        plt.figure(figsize=(5, 3))
        plt.hist(boxes_max_list, bins=range(0, max(boxes_max_list) + 2), edgecolor="k")
        plt.xlabel("Max boxes seen in a run")
        plt.ylabel("Count")
        plt.title("Distribution of Max Boxes Seen")
        plt.tight_layout()
        plt.savefig(out_dir / "boxes_max_seen_hist.png", dpi=200)
        plt.close()

    # Histogram: boxes at failure
    if boxes_at_failure_list:
        plt.figure(figsize=(5, 3))
        plt.hist(
            boxes_at_failure_list,
            bins=range(0, max(boxes_at_failure_list) + 2),
            edgecolor="k",
        )
        plt.xlabel("Boxes at failure")
        plt.ylabel("Count")
        plt.title("Distribution of Boxes at Failure")
        plt.tight_layout()
        plt.savefig(out_dir / "boxes_at_failure_hist.png", dpi=200)
        plt.close()

    # Success vs trajectory length (per max_num_trajectories)
    plt.figure(figsize=(6, 4))
    for c in sorted({k[0] for k in grid_success.keys()}):
        xs, ys = [], []
        for l in sorted({k[1] for k in grid_success.keys() if k[0] == c}):
            vals = grid_success[(c, l)]
            if vals:
                xs.append(l)
                ys.append(np.mean(vals))
        plt.plot(xs, ys, marker="o", label=f"max_num_traj={c}")
    plt.ylim(0, 1.05)
    plt.xlabel("max_trajectory_length")
    plt.ylabel("Success rate")
    plt.title("Success vs Trajectory Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_vs_length.png", dpi=200)
    plt.close()

    # Success vs secret sequence length
    plt.figure(figsize=(6, 4))
    xs, ys = [], []
    for slen in sorted(success_vs_secret.keys()):
        vals = success_vs_secret[slen]
        if vals:
            xs.append(slen)
            ys.append(np.mean(vals))
    plt.plot(xs, ys, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("secret sequence length")
    plt.ylabel("Success rate")
    plt.title("Success vs Secret Sequence Length")
    plt.tight_layout()
    plt.savefig(out_dir / "success_vs_secret_length.png", dpi=200)
    plt.close()

    # Success vs max_num_trajectories (per trajectory length)
    plt.figure(figsize=(6, 4))
    for l in sorted({k[1] for k in grid_success.keys()}):
        xs, ys = [], []
        for c in sorted({k[0] for k in grid_success.keys() if k[1] == l}):
            vals = grid_success[(c, l)]
            if vals:
                xs.append(c)
                ys.append(np.mean(vals))
        plt.plot(xs, ys, marker="o", label=f"max_len={l}")
    plt.ylim(0, 1.05)
    plt.xlabel("max_num_trajectories")
    plt.ylabel("Success rate")
    plt.title("Success vs Max # Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_vs_num_trajectories.png", dpi=200)
    plt.close()

    # Boxplots: steps to failure by trajectory length
    if any(by_len_steps.values()):
        plt.figure(figsize=(6, 4))
        labels, data = zip(*sorted(by_len_steps.items(), key=lambda kv: kv[0]))
        plt.boxplot(data, tick_labels=[str(l) for l in labels])
        plt.xlabel("max_trajectory_length")
        plt.ylabel("Steps to failure")
        plt.title("Steps to Failure by Trajectory Length")
        plt.tight_layout()
        plt.savefig(out_dir / "steps_to_failure_boxplot.png", dpi=200)
        plt.close()

    # Boxplots: min gap at failure by trajectory length
    if any(by_len_mingap.values()):
        plt.figure(figsize=(6, 4))
        labels, data = zip(*sorted(by_len_mingap.items(), key=lambda kv: kv[0]))
        plt.boxplot(data, tick_labels=[str(l) for l in labels])
        plt.xlabel("max_trajectory_length")
        plt.ylabel("Min gap at failure")
        plt.title("Min Gap at Failure by Trajectory Length")
        plt.tight_layout()
        plt.savefig(out_dir / "min_gap_at_failure_boxplot.png", dpi=200)
        plt.close()

    # Command usage comparison (found vs not found)
    def average_cmd_usage(cmd_list: list[Counter[str]]) -> dict[str, float]:
        if not cmd_list:
            return {"off": 0.0, "slow": 0.0, "mid": 0.0, "fast": 0.0}
        total = Counter()
        for c in cmd_list:
            total.update(c)
        n = sum(total.values()) or 1
        return {k: total[k] / n for k in ["off", "slow", "mid", "fast"]}

    avg_found = average_cmd_usage(cmd_usage_found)
    avg_not_found = average_cmd_usage(cmd_usage_not_found)
    labels = ["off", "slow", "mid", "fast"]
    found_vals = [avg_found[k] for k in labels]
    not_vals = [avg_not_found[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, found_vals, width, label="Found")
    plt.bar(x + width / 2, not_vals, width, label="Not found")
    plt.xticks(x, labels)
    plt.ylabel("Average command frequency")
    plt.title("Command Mix vs Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "command_mix_vs_outcome.png", dpi=200)
    plt.close()

    # JSON summary
    summary = {
        "total_runs": len(results),
        "success_rate_overall": float(
            np.mean([r.found for r in results]) if results else 0.0
        ),
        "param_grid_success": {
            f"{c}_{l}": float(np.mean(vals)) for (c, l), vals in grid_success.items()
        },
        "success_vs_secret_length": {
            str(k): float(np.mean(v)) for k, v in success_vs_secret.items()
        },
        "traj_index_mean_if_found": float(np.mean(traj_indices)) if traj_indices else None,
        "steps_to_failure_mean_if_found": float(np.mean(steps_list)) if steps_list else None,
        "min_gap_at_failure_mean": float(np.mean(finite_min_gap))
        if finite_min_gap
        else None,
        "boxes_max_seen_mean": float(np.mean(boxes_max_list)) if boxes_max_list else None,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    out_dir = Path("reports") / "conveyor_random_shooting"
    # Expanded grid over max_num_trajectories, lengths, and secret lengths; seeds trimmed
    max_traj_options = [20, 50, 100, 150, 300]
    max_len_options = [20, 50, 100]
    secret_len_options = [4, 8, 12]
    seeds = list(range(6))  # adjust for runtime

    results: list[TrialResult] = []
    for max_num_traj in max_traj_options:
        for max_len in max_len_options:
            for secret_len in secret_len_options:
                for seed in seeds:
                    result = run_trial(seed, max_num_traj, max_len, secret_len)
                    results.append(result)
                    status = "FOUND" if result.found else "none"
                    print(
                        f"seed={seed:3d} num_traj={max_num_traj:3d} "
                        f"len={max_len:3d} secret_len={secret_len:2d} -> {status}"
                    )

    aggregate_and_plot(results, out_dir)
    print(f"Done. Results written to: {out_dir}")


if __name__ == "__main__":
    main()

