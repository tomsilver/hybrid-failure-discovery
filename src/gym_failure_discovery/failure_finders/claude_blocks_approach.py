"""
Approach for BlocksEnv: Trigger the failure condition where a non-held block
is displaced during a high-level action.

GEOMETRIC ANALYSIS (qualitative, no numbers):
- Blocks are cuboid shapes arranged in a line on a flat table surface.
- Neighboring blocks are nearly adjacent, separated by small gaps.
- Observation includes: block_positions (array of 3D positions), held_block
  (index of held block, or sentinel for 'none'), robot_joints (arm config).
- The robot arm carries blocks via a planned end-effector trajectory;
  physics runs at each waypoint, so arm links or the held block can displace
  adjacent table blocks.
- Maximum collision probability: pick from FAR LEFT, stack on FAR RIGHT.
  The arm sweeps the full width of the block cluster, crossing all intermediate
  blocks.
- Also: picking a block from a SURROUNDED position (neighbors on both sides)
  forces the gripper into a tight space, risking nudging neighbors.
- When the tower grows tall (approaching IK limits), arm joints reach extreme
  configurations; their links can sweep into the workspace of table blocks.

STRATEGY:
1. Use observations to know which blocks are on the table and which are stacked.
2. Always pick from one extreme, stack on the other extreme (max sweep).
3. Use BiRRT for pick ordering, CSP for fallback actions.
4. After full tower build, cycle Unstack-Restack to generate more arm motion.
5. Adapt when IK fails (switch sweep direction).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Any

# Sentinel value for "no block held"
_NO_BLOCK_HELD = None


class GeneratedApproach:
    """Triggers arm collision failures by sweeping arm across the block cluster."""

    def __init__(
        self,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        primitives: dict[str, Any],
    ) -> None:
        self._action_space = action_space
        self._observation_space = observation_space
        self._primitives = primitives
        self._csp = primitives['csp']
        self._check_collision = primitives['check_action_collision']
        self._BiRRT = primitives['BiRRT']
        self._rng = np.random.default_rng(0)

        self._n_blocks = int(action_space['block'].n)  # type: ignore[index]
        # Sentinel: held_block = n_blocks means "none held"
        self._no_block_sentinel = self._n_blocks

        self._plan: list[dict] = []
        self._step_count: int = 0
        self._last_obs: Any = None
        self._last_info: dict = {}
        self._ik_failures: int = 0
        self._plan_phase: int = 0

        # Table-level z coordinate (blocks start here)
        self._table_z: float | None = None

    # -------------------------------------------------------------------------
    # Observation parsing
    # -------------------------------------------------------------------------

    def _parse_obs(self, obs: Any) -> tuple[np.ndarray, int]:
        """Extract block positions and held_block index from observation.

        Returns:
            positions: array of shape (n_blocks, 3) with block positions
            held_idx: index of held block, or -1 if none held
        """
        if isinstance(obs, dict):
            positions = np.asarray(obs.get('block_positions', []))
            held_raw = obs.get('held_block', self._no_block_sentinel)
        else:
            # Fallback for unknown observation formats
            return np.zeros((self._n_blocks, 3)), -1

        held_idx = int(held_raw)
        if held_idx >= self._n_blocks:
            held_idx = -1  # sentinel → none held

        return positions, held_idx

    def _get_table_blocks(self, positions: np.ndarray) -> list[int]:
        """Return indices of blocks that are at table level (not stacked)."""
        if self._table_z is None or len(positions) == 0:
            return list(range(self._n_blocks))

        # Blocks at table level have z close to the initial table height
        tol = 0.05  # small tolerance
        table_blocks = [
            i for i in range(len(positions))
            if abs(float(positions[i, 2]) - self._table_z) < tol
        ]
        return table_blocks

    def _get_tower_top(self, positions: np.ndarray, held_idx: int) -> int | None:
        """Return the index of the topmost non-held block."""
        if len(positions) == 0:
            return None

        best_z = -np.inf
        best_idx = None
        for i in range(len(positions)):
            if i == held_idx:
                continue
            z = float(positions[i, 2])
            if z > best_z:
                best_z = z
                best_idx = i
        return best_idx

    # -------------------------------------------------------------------------
    # Plan building
    # -------------------------------------------------------------------------

    def reset(self, state: Any, info: dict) -> None:
        """Called at the start of each episode."""
        self._last_obs = state
        self._last_info = info
        self._step_count = 0
        self._ik_failures = 0
        self._plan_phase = 0
        self._plan = []

        # Calibrate table-level z from initial observation
        positions, _ = self._parse_obs(state)
        if len(positions) > 0:
            self._table_z = float(np.min(positions[:, 2]))

        self._build_plan_phase_0()

    def _build_plan_phase_0(self) -> None:
        """Phase 0: Pick from left end, stack on right end (max arm sweep).

        Uses BiRRT to find the block-index traversal order.
        """
        n = self._n_blocks

        # BiRRT: plan traversal from block 0 to block n-1 through index space.
        # Each step in the path = one block to pick.
        # Longer path = more arm sweeps.
        def sample_fn(state: float) -> float:
            return float(self._rng.integers(0, n))

        def extend_fn(s1: float, s2: float) -> list[float]:
            a, b = int(round(s1)), int(round(s2))
            if a == b:
                return []
            step = 1 if b > a else -1
            result = []
            curr = a + step
            while (step > 0 and curr <= b) or (step < 0 and curr >= b):
                result.append(float(curr))
                curr += step
            return result

        def collision_fn(state: float) -> bool:
            return not (0 <= int(round(state)) < n)

        def distance_fn(s1: float, s2: float) -> float:
            return abs(s1 - s2)

        birrt = self._BiRRT(
            sample_fn=sample_fn,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            rng=self._rng,
            num_attempts=5,
            num_iters=30,
            smooth_amt=0,
        )

        path = birrt.query(0.0, float(n - 1))

        if path is not None:
            pick_order: list[int] = []
            seen: set[int] = set()
            for p in path:
                idx = int(round(p))
                if 0 <= idx < n and idx not in seen:
                    seen.add(idx)
                    pick_order.append(idx)
        else:
            pick_order = list(range(n))

        plan = self._make_sweep_plan(pick_order, tower_base=n - 1)

        # Add Unstack-Restack cycles for additional arm motion
        for _ in range(3):
            for blk in range(n - 2, 0, -1):
                plan.append({'block': blk, 'type': 1})      # Unstack
                plan.append({'block': 0, 'type': 2})        # Stack on left
                plan.append({'block': blk, 'type': 1})      # Unstack again
                plan.append({'block': n - 1, 'type': 2})   # Stack on right

        self._plan = plan

    def _make_sweep_plan(self, pick_order: list[int], tower_base: int) -> list[dict]:
        """Build action sequence: pick each block, stack on tower base.

        The arm sweeps from the pick position to the tower_base position each time.
        """
        plan = []
        prev_top = tower_base

        for block_idx in pick_order:
            if block_idx == tower_base:
                continue
            # Determine action type: 0=Pick (from table), 1=Unstack (from stack)
            # We'll use 0 (Pick) for initial picks; env will handle invalid ones.
            plan.append({'block': block_idx, 'type': 0})   # Pick from table
            plan.append({'block': prev_top, 'type': 2})     # Stack on tower
            prev_top = block_idx

        return plan

    def _build_plan_phase_1(self) -> None:
        """Phase 1: Reverse direction — pick from right, stack on leftmost."""
        n = self._n_blocks
        plan = []
        prev_top = 0  # leftmost is tower base

        for block_idx in range(n - 1, 0, -1):
            plan.append({'block': block_idx, 'type': 0})
            plan.append({'block': prev_top, 'type': 2})
            prev_top = block_idx

        # Add unstack-restack cycles
        for _ in range(3):
            for blk in range(1, n - 1):
                plan.append({'block': blk, 'type': 1})
                plan.append({'block': n - 1, 'type': 2})
                plan.append({'block': blk, 'type': 1})
                plan.append({'block': 0, 'type': 2})

        self._plan = plan

    def _build_plan_phase_2(self) -> None:
        """Phase 2: Aggressive Unstack-Restack cycling."""
        n = self._n_blocks
        plan = []

        # Repeatedly unstack top blocks and restack on opposite end
        for cycle in range(5):
            for blk in range(n - 1, -1, -1):
                plan.append({'block': blk, 'type': 1})              # Unstack
                target = 0 if cycle % 2 == 0 else n - 1
                plan.append({'block': target, 'type': 2})           # Restack

        self._plan = plan

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def get_action(self, obs: Any) -> dict:
        """Return the next action, using state awareness to guide decisions."""
        self._last_obs = obs

        # Parse current state from observation
        positions, held_idx = self._parse_obs(obs)

        # If holding a block, ALWAYS try to stack it (don't pick again)
        if held_idx >= 0:
            tower_top = self._get_tower_top(positions, held_idx)
            if tower_top is not None:
                # Stack on the topmost block that's not held
                return {'block': tower_top, 'type': 2}
            else:
                # Stack on whatever block index we can
                for blk in range(self._n_blocks):
                    if blk != held_idx:
                        return {'block': blk, 'type': 2}

        # Not holding anything — execute next planned pick
        if not self._plan:
            return self._get_csp_action(obs)

        action = self._plan.pop(0)
        self._step_count += 1

        # Use check_action_collision to skip no-ops
        try:
            if self._check_collision(obs, action):
                if self._plan:
                    action = self._plan.pop(0)
                    self._step_count += 1
                else:
                    return self._get_csp_action(obs)
        except Exception:
            pass

        # If the planned action is a Stack but we're not holding anything,
        # skip it and take the next Pick action
        if action.get('type') == 2 and held_idx < 0:
            # Skip stack actions when not holding
            while self._plan and self._plan[0].get('type') == 2:
                self._plan.pop(0)
            if self._plan:
                action = self._plan.pop(0)
                self._step_count += 1
            else:
                return self._get_csp_action(obs)

        return action

    def _get_csp_action(self, obs: Any) -> dict:
        """Use CSP to find a non-trivial action."""
        csp_mod = self._csp

        _, held_idx = self._parse_obs(obs)

        block_var = csp_mod.CSPVariable(
            'block', gym.spaces.Discrete(self._n_blocks)
        )
        type_var = csp_mod.CSPVariable(
            'type', gym.spaces.Discrete(3)
        )

        def non_trivial(block_val: int, type_val: int) -> bool:
            action = {'block': int(block_val), 'type': int(type_val)}
            # Semantic constraints based on held state
            if held_idx >= 0 and int(type_val) != 2:
                return False  # if holding, only Stack is valid
            if held_idx < 0 and int(type_val) == 2:
                return False  # if not holding, Stack is invalid
            if int(block_val) == held_idx:
                return False  # can't target the held block
            try:
                return not self._check_collision(obs, action)
            except Exception:
                return True

        constraint = csp_mod.FunctionalCSPConstraint(
            'non_trivial', [block_var, type_var], non_trivial
        )

        csp_problem = csp_mod.CSP(
            variables=[block_var, type_var],
            constraints=[constraint]
        )

        def sampler_fn(current_vals, rng):
            if held_idx >= 0:
                type_choice = 2  # Stack
            else:
                type_choice = int(rng.integers(0, 2))  # Pick or Unstack
            blk = int(rng.integers(0, self._n_blocks))
            while blk == held_idx:
                blk = int(rng.integers(0, self._n_blocks))
            return {block_var: blk, type_var: type_choice}

        sampler = csp_mod.FunctionalCSPSampler(
            sampler_fn, csp_problem, {block_var, type_var}
        )

        solver = csp_mod.RandomWalkCSPSolver(
            seed=int(self._rng.integers(0, 10000)),
            max_iters=30,
            show_progress_bar=False
        )

        init = {
            block_var: int(self._rng.integers(0, self._n_blocks)),
            type_var: 2 if held_idx >= 0 else int(self._rng.integers(0, 2))
        }

        solution = solver.solve(csp_problem, init, [sampler])
        if solution is not None:
            return {
                'block': int(solution[block_var]),
                'type': int(solution[type_var])
            }

        return self._action_space.sample()

    def step(self) -> dict:
        """Called by the episode runner."""
        return self.get_action(self._last_obs)

    def update(self, state: Any, reward: float, done: bool, info: dict) -> None:
        """Record state and adapt plan based on outcomes."""
        self._last_obs = state
        self._last_info = info

        if info.get('ik_failed', False):
            self._ik_failures += 1
            if self._ik_failures >= 2:
                self._plan_phase += 1
                self._ik_failures = 0
                if self._plan_phase == 1:
                    self._build_plan_phase_1()
                elif self._plan_phase == 2:
                    self._build_plan_phase_2()
                else:
                    self._plan = []  # CSP takes over
