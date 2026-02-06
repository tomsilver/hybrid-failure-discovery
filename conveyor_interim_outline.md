# Interim Report Outline — Conveyor Belt Focus

## 1) Context & Goals
- Hybrid failure discovery repo with fast, tunable conveyor-belt environment to surface rare but real failures (unsafe drops/collisions).
- Objective: benchmark and improve failure-finders by creating controllable failure modes with adjustable rarity.

## 2) System Overview
- Components: environment (physics), controller (faulty safety + secret sequence), commander (random), failure finder (random shooting), failure monitor (currently explosion-only; previously collision-based), reporting scripts.
- Data flow (per step): command → controller → action → env step → monitor; failure finder orchestrates trajectories.

## 3) Conveyor Belt Environment
- State: positions (left edge), falling_heights (bottom offset from belt), step_count, exploded flag.
- Scene spec: conveyor_belt_velocity, dt, belt_length, box_width/height, gravity, drop_start_height, initial_drop_position, min_spacing (added dynamically).
- Dynamics: gravity lowers falling_heights; landed boxes move with belt; drops occur at x=0; boxes removed at belt end; explosion flag propagates.

## 4) Controller & Faults
- Modes timing (off/slow/mid/fast); `_steps_since_last_drop` starts at 1e9 allowing immediate first drop if mode permits.
- Secret_failure_mode_sequence triggers explode=True when matched.
- Safety faults: height check uses >0.05 (allows almost-landed boxes), spacing ignores box_width and uses 0.8× min_spacing margin.

## 5) Failure Monitor
- Current: returns True if `state.exploded`.
- Prior/desired: collision checks on gaps < box_width and min_spacing; potential landing-collision using heights.

## 6) Failure Finder (Random Shooting)
- Runs many trajectories; each step: commander samples command → controller produces action → env steps → monitor checks.
- Params: max_num_trajectories, max_trajectory_length, seed flow for commander/initial states/env transitions.

## 7) Recent Results (Reporting Script)
- Grid: num_trajectories ∈ {10…500}, lengths ∈ {20,50,100}, seeds 0–7.
- Success patterns: mostly “none” under explosion-only monitor; sporadic FOUND at higher lengths/trajectory counts.
- Plots generated: success heatmap, success vs length, success vs num trajectories, traj-index hist, steps-to-failure hist/boxplot, min-gap hist/boxplot, boxes-at-failure/max-seen, command mix vs outcome.
- Quantitative snapshot (latest run): successes are sparse (e.g., only a few seeds at higher num_trajectories/lengths hit failure), most configurations yield zero success rate; trajectory index/steps hists reflect very few failures overall.

## 8) Observations & Issues
- Explosion-only monitor makes failures rare/seed-sensitive; secret sequence rarely hit in short runs.
- Collision-based behavior from earlier versions is absent; short lengths (20) yield none; long lengths need many trajectories for occasional hits.

## 9) Next Steps
- Reintroduce true collision monitor (edge overlap, landing collisions).
- Disable/lengthen secret sequence; tune controller timing/safety to modulate rarity.
- Vary initial states; adjust trajectory length/num to target desired rarity.

## 10) Risks & Mitigations
- Too rare: increase breadth (more trajectories), loosen safety/monitor.
- Too frequent: tighten safety/monitor, reduce drop rate.

## 11) Timeline (lightweight)
- Phase 1: Restore collision monitor.
- Phase 2: Tune parameters/initial states for target rarity.
- Phase 3: Rerun reporting and update plots.

