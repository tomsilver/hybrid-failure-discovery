"""Approach for the hovercraft environment.

Goal: Trigger failure condition = hovercraft circle intersects any obstacle.
Strategy: Navigate the hovercraft into the nearest obstacle.
- Use CSP to find a target point inside an obstacle
- Use BiRRT to plan a collision-free path to that target
- Execute by learning which action moves toward waypoints empirically
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class GeneratedApproach:
    """Drive the hovercraft into an obstacle to trigger a collision failure."""

    def __init__(self, action_space, observation_space, primitives):
        self._action_space = action_space
        self._obs_space = observation_space
        self._primitives = primitives
        self._rng = np.random.default_rng(42)
        self._plan = []
        self._step_count = 0
        self._target = None
        self._last_pos = None
        self._last_action = None
        # Running average of displacement caused by each action
        self._action_effects = {}
        self._action_counts = {}

    def reset(self, state, info):
        self._step_count = 0
        self._plan = []
        self._last_pos = None
        self._last_action = None
        self._action_effects = {}
        self._action_counts = {}

        self._obstacles = self._load_obstacles()
        pos = self._pos_from_state(state)
        self._target = self._find_obstacle_target(pos[0], pos[1])
        self._build_plan(state)

    def _load_obstacles(self):
        try:
            from robocode.environments.hovercraft_env import HoverCraftSceneSpec
            spec = HoverCraftSceneSpec()
            return [{'cx': o.x + o.width/2, 'cy': o.y + o.height/2,
                     'x': o.x, 'y': o.y, 'w': o.width, 'h': o.height}
                    for o in spec.obstacles]
        except Exception:
            return [
                {'cx': -0.35, 'cy': -0.35, 'x': -0.5, 'y': -0.5, 'w': 0.3, 'h': 0.3},
                {'cx':  0.35, 'cy': -0.35, 'x':  0.2, 'y': -0.5, 'w': 0.3, 'h': 0.3},
                {'cx':  0.35, 'cy':  0.35, 'x':  0.2, 'y':  0.2, 'w': 0.3, 'h': 0.3},
                {'cx': -0.35, 'cy':  0.35, 'x': -0.5, 'y':  0.2, 'w': 0.3, 'h': 0.3},
            ]

    def _nearest_obstacle(self, x, y):
        best, best_dist = None, float('inf')
        for o in self._obstacles:
            d = np.hypot(x - o['cx'], y - o['cy'])
            if d < best_dist:
                best_dist, best = d, o
        return best

    def _find_obstacle_target(self, x, y):
        """Use CSP to find a point inside the nearest obstacle."""
        csp = self._primitives['csp']
        o = self._nearest_obstacle(x, y)
        if o is None:
            return np.array([0.0, 0.0])

        target_var = csp.CSPVariable(
            "target",
            gym.spaces.Box(
                low=np.array([o['x'], o['y']], dtype=np.float32),
                high=np.array([o['x'] + o['w'], o['y'] + o['h']], dtype=np.float32),
            )
        )

        def in_obs(tgt):
            tx, ty = float(tgt[0]), float(tgt[1])
            return o['x'] <= tx <= o['x'] + o['w'] and o['y'] <= ty <= o['y'] + o['h']

        constraint = csp.FunctionalCSPConstraint("in_obstacle", [target_var], in_obs)
        problem = csp.CSP([target_var], [constraint])

        def sampler_fn(vals, rng):
            tx = o['x'] + rng.uniform(0, o['w'])
            ty = o['y'] + rng.uniform(0, o['h'])
            return {target_var: np.array([tx, ty], dtype=np.float32)}

        sampler = csp.FunctionalCSPSampler(sampler_fn, problem, {target_var})
        solver = csp.RandomWalkCSPSolver(seed=42)
        init = {target_var: np.array([o['cx'], o['cy']], dtype=np.float32)}
        sol = solver.solve(problem, init, [sampler])

        if sol is not None:
            v = sol[target_var]
            return np.array([float(v[0]), float(v[1])])
        return np.array([o['cx'], o['cy']])

    def _build_plan(self, state):
        """Build a BiRRT path toward the obstacle target."""
        BiRRT = self._primitives['BiRRT']
        pos = self._pos_from_state(state)
        goal = self._target
        if goal is None:
            self._plan = []; return

        def sample_fn(s):
            return self._rng.uniform([-0.9, -0.9], [0.9, 0.9])

        def extend_fn(s1, s2):
            s1, s2 = np.array(s1), np.array(s2)
            diff = s2 - s1
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                return [s2]
            n = max(1, int(dist / 0.05))
            return [s1 + diff * i / n for i in range(1, n + 1)]

        def collision_fn(p):
            # Only world bounds block us (we want to reach obstacles)
            px, py = float(p[0]), float(p[1])
            return not (-0.9 <= px <= 0.9 and -0.9 <= py <= 0.9)

        def dist_fn(s1, s2):
            return float(np.linalg.norm(np.array(s1) - np.array(s2)))

        birrt = BiRRT(sample_fn, extend_fn, collision_fn, dist_fn,
                      self._rng, num_attempts=3, num_iters=100, smooth_amt=10)
        path = birrt.query(pos, goal)
        if path and len(path) > 1:
            self._plan = [np.array(p) for p in path[1:]]
        else:
            self._plan = [goal]

    def _pos_from_state(self, state):
        return np.array([float(state[0]), float(state[2])])

    def get_action(self, state):
        self._step_count += 1
        n = self._action_space.n
        pos = self._pos_from_state(state)

        # Update action effect estimates
        if self._last_pos is not None and self._last_action is not None:
            delta = pos - self._last_pos
            a = self._last_action
            cnt = self._action_counts.get(a, 0)
            if cnt == 0:
                self._action_effects[a] = delta.copy()
            else:
                alpha = 1.0 / (cnt + 1)
                self._action_effects[a] = (1 - alpha) * self._action_effects[a] + alpha * delta
            self._action_counts[a] = cnt + 1

        # Trim reached waypoints
        while self._plan and np.linalg.norm(pos - self._plan[0]) < 0.06:
            self._plan.pop(0)

        if not self._plan:
            self._target = self._find_obstacle_target(pos[0], pos[1])
            self._build_plan(state)

        waypoint = self._plan[0] if self._plan else self._target
        if waypoint is None:
            waypoint = pos

        # Choose action
        # Phase 1: explore (take each action once to learn effects)
        unexplored = [a for a in range(n) if a not in self._action_effects]
        if unexplored:
            chosen = unexplored[0]
        else:
            # Pick action whose predicted next position is closest to waypoint
            best_a, best_score = 0, float('inf')
            for a in range(n):
                effect = self._action_effects.get(a, np.zeros(2))
                predicted = pos + effect
                score = np.linalg.norm(predicted - waypoint)
                if score < best_score:
                    best_score, best_a = score, a

            # Also check check_action_collision to avoid boundary stalls
            chosen = best_a
            try:
                check = self._primitives['check_action_collision']
                if check(state, chosen):
                    # Try alternate action
                    alt = (chosen + 1) % n
                    chosen = alt
            except Exception:
                pass

        self._last_pos = pos.copy()
        self._last_action = chosen

        # Periodically re-plan
        if self._step_count % 80 == 0:
            self._target = self._find_obstacle_target(pos[0], pos[1])
            self._build_plan(state)

        return chosen
