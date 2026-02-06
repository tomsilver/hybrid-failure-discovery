# Interim Report: Conveyor-Belt Failure Discovery
Abigail Girma  
January 2026

## 1 Introduction
The conveyor-belt environment is a fast, controllable testbed for studying rare-but-real failures in hybrid failure discovery. By combining a simple 1D physics model, an intentionally faulty controller, and a random-shooting failure finder, we can tune how often unsafe drops or collisions occur and benchmark failure-detection methods. This outline mirrors the structure of the provided interim report template, adapted to the conveyor-belt domain.

## 2 Motivation
- Pure neural search (random shooting) can miss low-probability unsafe sequences; structured faults and monitors expose such events.
- Humans can encode simple, interpretable safety logic, but the current controller is deliberately flawed (lenient spacing, lenient falling-height checks, secret mode sequence).
- Goal: surface true physical failures (collisions/unsafe drops) in a lightweight environment to drive the design of stronger failure finders and monitors.

## 3 Background: Hybrid Systems & Benchmarks
Hybrid autonomous systems combine continuous physical evolution and discrete decision logic. Viewing them as hybrid automata highlights how mode-specific dynamics and discrete transitions interact—small timing or state perturbations can trigger rare but damaging failures, making the failure set measure-zero and hard to find via naive simulation. Real robotic systems (household robots, AVs, industrial automation) share this structure: discrete safety/logic layered over physics.

Our benchmarks reflect this spectrum:
- **Conveyor belt (this work):** 1D deterministic physics; discrete mode commands with injected faults; failures can hinge on spacing/landing height and discrete sequences.
- **Hovercraft:** continuous planar dynamics with discrete goal-switch commands; tests interaction of navigation and mode toggling.
- **Blocks (PyBullet):** contact-rich 3D physics coupled with discrete grasp/release logic; illustrates hybrid failures in manipulation.

These simplified domains let us study rare hybrid failures in isolation and control their rarity/structure for testing search and monitoring methods.

## 4 Approach
The aim is to expose rare-but-real unsafe drops/collisions by running a random-shooting loop over a structured (but faulty) controller and a simple 1D conveyor environment. The loop mirrors a residual-style architecture: state → programmatic controller → action; (optional) learned finder/monitor inspects outcomes; repeat across steps and trajectories until failure.

### 4.1 Architecture (Conveyor Random Shooting)
- At each step, the environment emits state \(s_t\).
- The random commander samples a mode command.
- The conveyor controller maps \((s_t,\ \text{mode}) \to a^{ctrl}_t\), possibly emitting `explode=True`.
- The environment applies the action, advancing physics to \(s_{t+1}\) (gravity, belt motion, drop/removal, explosion propagation).
- The failure monitor checks \(s_{t+1}\); if failure, trajectory ends; otherwise continue. Only the monitor/finder “learns” (the controller is fixed and faulty by design).

### 4.2 Environment Setup (Conveyor-Belt)
- State: positions (left edge), falling_heights (bottom offset), step_count, exploded flag.
- Scene spec: conveyor_belt_velocity, dt, belt_length, box_width/height, gravity, drop_start_height, initial_drop_position, min_spacing (injected).
- Dynamics per step: update falling_heights via gravity; landed boxes move with belt; new drops at x=0 when commanded; remove off-belt boxes; propagate exploded.

### 4.3 Programmatic Controller & Monitor (Priors)
- Controller modes off/slow/mid/fast with timing map; `_steps_since_last_drop` initialized large (immediate first drop if allowed). Safety faults: height gate uses >0.05 (lenient), spacing ignores box_width and uses 0.8× min_spacing. Secret mode sequence triggers `explode=True`.
- Failure monitor (current): explosion-only (`state.exploded`). Desired: collision-based (gaps < box_width/min_spacing, landing-collision using heights).
- Commander: RandomCommander sampling modes uniformly each step (seedable).

### 4.4 Failure Finder (Random Shooting) and Parameters
- Parameters: max_num_trajectories, max_trajectory_length, seed.
- Loop over trajectories: initialize state; per step sample mode → controller → env → monitor; stop on failure or length; stop overall on first failure or trajectory budget.
- Scene/controller levers: box_width, min_spacing, belt_velocity, dt, drop height; secret sequence on/off/length; safety thresholds; initial states (currently fixed empty belt).

## 5 Experiments
- Grids run: num_trajectories ∈ {10…500}, trajectory_length ∈ {20,50,100}, seeds 0–7.
- Metrics: success (found), trajectory index to failure, steps to failure, min gap, boxes at failure/max seen, command mix.
- Outputs: success heatmap, success vs length, success vs num_trajectories, traj-index hist, steps-to-failure hist/boxplot, min-gap hist/boxplot, boxes hist, command mix vs outcome.

## 6 Results
- Success patterns: mostly “none”; sparse FOUND at higher trajectory counts/lengths.
- Collision-based behavior from earlier versions is absent under explosion-only monitor; short lengths (20) yield none; long lengths require many trajectories for rare hits.
- Quantitative snapshot: successes are sparse across the expanded grid; most configurations have zero success; few seeds succeed at high num_trajectories/lengths.

## 7 Discussion
- Explosion-only monitor + secret sequence makes failures seed-sensitive and rare; random commands seldom hit the sequence within short trajectories.
- Safety faults remain, but without collision monitoring, they don’t register as failures.
- Current setup is good for testing rare-event search breadth but not for collision fidelity.

## 8 Next Steps
- Restore collision-based monitor (edge overlap/landing collisions); make secret sequence optional/longer.
- Tune controller timing/safety thresholds to control rarity; vary initial states; adjust length/num_trajectories.
- Rerun reporting after monitor/controller changes to observe success-rate shifts.

## 9 Risks & Mitigations
- Too rare: increase breadth (num_trajectories), loosen safety, enable collision monitor.
- Too frequent: tighten safety/monitor, slow drops, lengthen secret sequence.

## 10 Timeline
- Phase 1: Reintroduce collision monitor; disable/lengthen secret sequence.
- Phase 2: Parameter/initial-state tuning for target rarity; rerun reporting.
- Phase 3: Analyze updated plots; iterate failure-finder/monitor refinements.

