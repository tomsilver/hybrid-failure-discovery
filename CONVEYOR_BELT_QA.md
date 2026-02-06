# Conveyor Belt System: Questions & Answers

## Table of Contents
1. [Failure Detection & Collision Physics](#failure-detection--collision-physics)
2. [Ordering & Sorting](#ordering--sorting)
3. [Time & Step Count](#time--step-count)
4. [Safety Checks & Drop Logic](#safety-checks--drop-logic)
5. [Scene Parameters](#scene-parameters)
6. [Controller Modes & State](#controller-modes--state)
7. [State Management](#state-management)
8. [Failure Monitor Mechanics](#failure-monitor-mechanics)
9. [Failure Finder & Trajectory Sampling](#failure-finder--trajectory-sampling)
10. [Initial States & Randomness](#initial-states--randomness)
11. [Making Failures Rarer](#making-failures-rarer)

---

## Failure Detection & Collision Physics

### Q: Right now the failure finder only uses positions not height. Could we trigger failure only at actual collision?

**A:** You're absolutely right! The current implementation is overly conservative. Here's what's happening:

**Current behavior (in `ConveyorBeltFailureMonitor._check_failures`)**:
```python
sorted_positions = np.sort(state.positions)
diffs = np.diff(sorted_positions)

# Fails if boxes TOO CLOSE (not necessarily colliding)
if np.any(diffs < min_spacing) or np.any(diffs < box_width):
    return True  # FAILURE
```

**Problems with this**:
1. **Fails before actual collision**: If boxes are 0.2 units apart with `box_width=0.3`, it fails even though they're not touching
2. **Ignores height**: Two boxes could be at same x-position but at different heights (one falling, one landed) - no collision!
3. **Uses center-to-center distance**: Not accurate for actual collision detection

**What actual collision means**:
```
Box A: position=0.5, width=0.3, height=0.0 (landed)
  ├─────────┤
  0.5      0.8

Box B: position=0.7, width=0.3, height=0.0 (landed)
         ├─────────┤
         0.7      1.0

Overlap region: 0.7 to 0.8 (0.1 units)
This is ACTUAL collision!
```

**Better failure detection (checking actual collision)**:
```python
def _check_actual_collisions(self, state: ConveyorBeltState) -> bool:
    """Check for ACTUAL physical collisions, not just proximity."""
    if len(state.positions) <= 1:
        return False
    
    box_width = self._scene_spec.box_width
    
    # Check each pair of boxes
    for i in range(len(state.positions)):
        for j in range(i + 1, len(state.positions)):
            pos_i, height_i = state.positions[i], state.falling_heights[i]
            pos_j, height_j = state.positions[j], state.falling_heights[j]
            
            # Only check boxes that are both on belt (height = 0)
            # Falling boxes can't collide with landed boxes yet
            if height_i > 0.0 or height_j > 0.0:
                continue
            
            # Check if boxes overlap in x-direction
            left_i = pos_i
            right_i = pos_i + box_width
            left_j = pos_j
            right_j = pos_j + box_width
            
            # Collision if ranges overlap
            if not (right_i <= left_j or right_j <= left_i):
                return True  # ACTUAL COLLISION
    
    return False
```

**Key improvements**:
1. Only checks boxes that are BOTH landed (height=0)
2. Computes actual bounding boxes (left/right edges)
3. Detects true overlap, not just proximity
4. Makes failures MUCH rarer (only on actual contact)

### Q: Why does box_width matter if min_spacing accounts for distance between centers?

**A:** This is a key confusion! Let me clarify:

**Box width and spacing are DIFFERENT concepts**:

```
Scenario 1: Checking box_width
├─────────┤         ├─────────┤
Box A (w=0.3)       Box B (w=0.3)
Center: 0.5         Center: 0.7
Distance between centers: 0.2

If distance < 0.3 → OVERLAP (collision)
```

```
Scenario 2: Checking min_spacing
├─────────┤             ├─────────┤
Box A               gap    Box B
Center: 0.5               Center: 1.0
Distance between centers: 0.5
Gap between edges: 0.5 - 0.3 = 0.2

If gap < min_spacing (0.1) → TOO CLOSE (but not colliding)
```

**What they check**:
- **`box_width`**: Prevents OVERLAP (physical collision)
- **`min_spacing`**: Enforces SAFETY MARGIN (buffer zone)

**Your concern is valid**: If you only care about actual collisions, remove min_spacing checks!

```python
# ONLY check for actual overlap:
if np.any(diffs < box_width):
    return True  # Boxes are overlapping
```

**Better yet**: Use the edge-based collision detection above.

### Q: Should we only fail when a falling box hits a landed box?

**A:** YES! This would make failures much more realistic. Here's how:

```python
def _check_landing_collisions(self, state: ConveyorBeltState) -> bool:
    """Check if any falling box lands ON a landed box."""
    box_width = self._scene_spec.box_width
    
    for i in range(len(state.positions)):
        # Is this box just landing? (height very close to 0)
        if state.falling_heights[i] <= 0.01 and state.falling_heights[i] >= 0.0:
            # Check if it's landing on top of another box
            for j in range(len(state.positions)):
                if i == j:
                    continue
                
                # Is the other box already landed?
                if state.falling_heights[j] == 0.0:
                    # Check x-position overlap
                    left_i = state.positions[i]
                    right_i = state.positions[i] + box_width
                    left_j = state.positions[j]
                    right_j = state.positions[j] + box_width
                    
                    if not (right_i <= left_j or right_j <= left_i):
                        return True  # LANDING COLLISION!
    
    return False
```

This checks for the exact moment when a falling box (height ≈ 0) lands on a spot occupied by another box (height = 0).

---

## Ordering & Sorting

### Q: "Failure detection doesn't care about order (sorts positions)" - so how is it really ordered?

**A:** Great question! There are TWO different orderings:

**1. Storage order (in arrays)**:
```python
positions = [1.5, 0.3, 2.1]  # Order boxes were added
# Box 0 (oldest): position=1.5
# Box 1 (middle): position=0.3  
# Box 2 (newest): position=2.1
```
- Boxes stored in **temporal order** (when they were dropped)
- Oldest box at index 0, newest at end
- This order preserved throughout trajectory

**2. Spatial order (for failure detection)**:
```python
sorted_positions = np.sort([1.5, 0.3, 2.1])  # [0.3, 1.5, 2.1]
# Box at x=0.3 (leftmost)
# Box at x=1.5 (middle)
# Box at x=2.1 (rightmost)
```
- Boxes sorted by **spatial location** (left to right on belt)
- Only used for computing gaps between adjacent boxes
- Original storage order unchanged

**Why sorting is needed**:
```python
# Without sorting:
positions = [1.5, 0.3, 2.1]
diffs = np.diff(positions)  # [0.3-1.5, 2.1-0.3] = [-1.2, 1.8]
# Negative diff! Meaningless!

# With sorting:
sorted_positions = [0.3, 1.5, 2.1]
diffs = np.diff(sorted_positions)  # [1.5-0.3, 2.1-1.5] = [1.2, 0.6]
# Actual gaps between adjacent boxes!
```

**Summary**:
- **Storage**: Temporal order (when boxes added)
- **Collision detection**: Spatial order (where boxes are on belt)
- Sorting is LOCAL to failure detection, doesn't affect storage

### Q: Is position measured from center, left edge, or right edge?

**A:** **Position is the LEFT edge** (where box starts), not the center!

**Evidence from code**:
```python
# When checking if box is off belt:
if (p >= belt_length - box_width) or (p < 0):
    continue  # Remove box
```

This checks:
- `p < 0`: Left edge before belt start → off belt
- `p >= belt_length - box_width`: Left edge would make right edge exceed belt → off belt

**Visualization**:
```
Belt: [0 ────────────────────── 5.0]

Box at position=0.5, width=0.3:
      ├─────────┤
      0.5      0.8
      ↑         ↑
   position  right edge
   (left edge)
```

**For collision detection**:
```python
# Box A
left_a = position_a        # 0.5
right_a = position_a + width  # 0.8

# Box B  
left_b = position_b        # 0.7
right_b = position_b + width  # 1.0

# Overlap check:
if not (right_a <= left_b or right_b <= left_a):
    # Collision!
```

---

## Time & Step Count

### Q: How does step_count correlate to actual seconds?

**A:** Direct relationship through `dt`:

```
step_count = number of simulation steps
dt = time per step (in seconds)
elapsed_time = step_count * dt
```

**Example**:
```python
scene_spec = ConveyorBeltSceneSpec(dt=0.01)  # 0.01 seconds per step

# After 100 steps:
step_count = 100
elapsed_time = 100 * 0.01 = 1.0 seconds

# After 1500 steps:
step_count = 1500
elapsed_time = 1500 * 0.01 = 15.0 seconds
```

**Rendering FPS relationship**:
```python
metadata = {"render_fps": 50}  # 50 frames per second in video

# If dt = 0.01 seconds per step:
# Real time: 1 second = 100 steps
# Video time: 1 second = 50 frames
# Video plays at: 50 frames/sec ÷ 100 steps/sec = 0.5x speed (slow motion)

# If dt = 0.02 seconds per step:
# Real time: 1 second = 50 steps  
# Video time: 1 second = 50 frames
# Video plays at: 50 frames/sec ÷ 50 steps/sec = 1.0x speed (real time)
```

**What's actually going on**:
1. Physics simulation runs in discrete steps
2. Each step advances time by `dt` seconds
3. `step_count` tracks how many steps occurred
4. `step_count * dt` = simulated elapsed time
5. Rendering converts steps to video frames (may speed up/slow down)

---

## Safety Checks & Drop Logic

### Q: What safety checks happen before drop_package=True?

**A:** The controller goes through multiple checks:

**Check sequence**:
```python
def step_action_space(state, command):
    # 1. Increment timer
    _steps_since_last_drop += 1
    
    # 2. Get timing requirement
    steps_required = _mode_to_steps[command.mode]
    
    # 3. Check if mode is "off"
    if steps_required is None:
        return ConveyorBeltAction(drop_package=False)  # ✗ Never drop
    
    # 4. Check if enough time passed
    if _steps_since_last_drop < steps_required:
        return ConveyorBeltAction(drop_package=False)  # ✗ Too soon
    
    # 5. Check safety (falling boxes, spacing)
    if not _safe_to_drop(state):
        return ConveyorBeltAction(drop_package=False)  # ✗ Unsafe
    
    # 6. All checks passed
    _steps_since_last_drop = 0  # Reset timer
    return ConveyorBeltAction(drop_package=True)  # ✓ DROP!
```

**Inside `_safe_to_drop(state)`**:
```python
def _safe_to_drop(state):
    # Check 1: Any boxes falling?
    for h in state.falling_heights:
        if h > 0.05:  # FAULT: should be > 0.0
            return False  # ✗ Box still falling
    
    # Check 2: Nearest box too close?
    if len(state.positions) > 0:
        nearest = min(state.positions)
        if nearest < min_spacing * 0.8:  # FAULT: should check box_width too
            return False  # ✗ Too close
    
    # All safety checks passed
    return True  # ✓ Safe to drop
```

**When it rightfully doesn't drop**:
1. **Mode = "off"**: Never drops (intended behavior)
2. **Timer not ready**: Drops too frequent (rate limiting)
3. **Box still falling**: Collision risk (safety check)
4. **Box too close**: Collision risk (safety check)

**When faults allow wrong drops**:
1. **Falling height = 0.03**: Passes check (h > 0.05 is False), but box almost landed
2. **Nearest box = 0.09**: Passes check (0.09 < 0.08 is False), but should check box_width

### Q: I'm confused about h > 0.05 being False. What does this mean?

**A:** Let me clarify the logic:

```python
# CORRECT check (what it SHOULD be):
if h > 0.0:
    return False  # ANY falling box → not safe to drop

# FAULTY check (what it IS):
if h > 0.05:
    return False  # Only if h > 0.05 → not safe
```

**Scenario breakdown**:

```
Box height = 0.10:
  h > 0.05 → True → return False → ✓ Correctly prevents drop

Box height = 0.03:
  h > 0.05 → False → doesn't return → ✗ FAULT! Allows drop
  (Box almost landed, should prevent drop)

Box height = 0.00:
  h > 0.05 → False → doesn't return → ✓ Correctly allows drop
  (Box fully landed, safe to drop)
```

**You said**: "It's more preventive if it was > 0.0"

**You're CORRECT!** Checking `h > 0.0` is MORE restrictive:
- Prevents drops whenever ANY box is falling
- Even if height = 0.001, prevents drop
- Safer, but makes failures RARER

**Checking `h > 0.05` is LESS restrictive** (current):
- Allows drops when box almost landed (0.0 < h ≤ 0.05)
- Creates window for collisions
- Makes failures MORE common (which is what the test wants to demonstrate)

**Your preference**: Check actual collision (when box lands on another)
- Don't prevent drops based on height alone
- Only fail when falling box physically hits landed box
- Most realistic, makes failures RAREST

---

## Scene Parameters

### Q: Is conveyor_belt_velocity ever variable?

**A:** No, it's constant per trajectory!

```python
@dataclass(frozen=True)
class ConveyorBeltSceneSpec:
    conveyor_belt_velocity: float = 1.0  # Constant
```

**Why frozen**:
- `frozen=True` makes it immutable
- Can't change during simulation
- Set once at environment creation

**Could vary between trajectories**:
```python
# Trajectory 1:
scene_spec_1 = ConveyorBeltSceneSpec(conveyor_belt_velocity=1.0)

# Trajectory 2:
scene_spec_2 = ConveyorBeltSceneSpec(conveyor_belt_velocity=2.0)
```

But within a trajectory, it never changes.

### Q: Would varying box_width make rare failures emerge?

**A:** YES! Box width directly affects collision probability:

**Smaller box_width** → **Rarer failures**:
```
box_width = 0.1 (small boxes)
├─┤       ├─┤       ├─┤
More space between boxes → harder to collide
```

**Larger box_width** → **More failures**:
```
box_width = 0.5 (large boxes)
├──────┤  ├──────┤  ├──────┤
Less space between boxes → easier to collide
```

**How to make failures rarer**:
```python
scene_spec = ConveyorBeltSceneSpec(
    box_width=0.2,  # Smaller boxes (was 0.3)
    min_spacing=0.05,  # Tighter spacing requirement (was 0.1)
)
```

**Trade-offs**:
- Smaller boxes: Rarer failures, need more trajectories to find
- Larger boxes: More failures, but less realistic

---

## Controller Modes & State

### Q: Why does ConveyorBeltCommand only contain mode: str = "off"?

**A:** This is a **default value**, not the only option!

```python
@dataclass(frozen=True)
class ConveyorBeltCommand:
    mode: str = "off"  # DEFAULT value
```

**All valid commands**:
```python
command1 = ConveyorBeltCommand(mode="off")   # Uses default
command2 = ConveyorBeltCommand(mode="slow")
command3 = ConveyorBeltCommand(mode="mid")
command4 = ConveyorBeltCommand(mode="fast")
```

**Command space contains all options**:
```python
def get_command_space(self):
    return EnumSpace([
        ConveyorBeltCommand(mode="off"),
        ConveyorBeltCommand(mode="slow"),
        ConveyorBeltCommand(mode="mid"),
        ConveyorBeltCommand(mode="fast"),
    ])
```

**Random commander samples from all 4**:
```python
command = command_space.sample()  # Randomly picks one of 4 modes
```

### Q: How does mode_to_steps convert timing to steps?

**A:** Simple division:

```python
dt = 0.01  # seconds per step

_mode_to_steps = {
    "off": None,
    "slow": int(2.0 / dt),   # int(2.0 / 0.01) = int(200.0) = 200
    "mid": int(1.2 / dt),    # int(1.2 / 0.01) = int(120.0) = 120
    "fast": int(0.2 / dt),   # int(0.2 / 0.01) = int(20.0) = 20
}
```

**What this means**:
- "slow": Drop every 200 steps = 200 * 0.01 = 2.0 seconds
- "mid": Drop every 120 steps = 120 * 0.01 = 1.2 seconds
- "fast": Drop every 20 steps = 20 * 0.01 = 0.2 seconds

**How it ensures rate**:
```python
_steps_since_last_drop += 1  # Increments every step

if _steps_since_last_drop >= steps_required:
    # Enough time passed, can drop again
```

**Example timeline (mode="fast", requires 20 steps)**:
```
Step  _steps_since_last_drop  Action
0     0                        Drop! (reset counter)
1     1                        No drop (1 < 20)
2     2                        No drop (2 < 20)
...
19    19                       No drop (19 < 20)
20    20                       Drop! (20 >= 20, reset counter)
21    1                        No drop (1 < 20)
```

### Q: When is _steps_since_last_drop set to 10^9 vs 0?

**A:** Clear rules:

**Set to 10^9** (once only):
```python
def __init__(self, seed, scene_spec):
    self._steps_since_last_drop = 10**9  # At initialization
```
- Happens ONCE when controller created
- Prevents immediate drop on first step
- "Effectively infinite" means no drop until explicitly allowed

**Set to 0** (every drop):
```python
if drop:
    self._steps_since_last_drop = 0  # After each successful drop
```
- Resets timer after each drop
- Starts counting up again from 0

**Incremented every step**:
```python
def step_action_space(self, state, command):
    self._steps_since_last_drop += 1  # ALWAYS, every step
```

**Full lifecycle**:
```
Initialization:  _steps_since_last_drop = 10^9

Step 1:  10^9 + 1 = 10^9 + 1 (still huge)
  Mode="fast" requires 20 → 10^9 >= 20 → Check safety → Drop!
  Reset: _steps_since_last_drop = 0

Step 2:  0 + 1 = 1
  Mode="fast" requires 20 → 1 < 20 → No drop

Step 3:  1 + 1 = 2
  Mode="fast" requires 20 → 2 < 20 → No drop

...

Step 21: 19 + 1 = 20
  Mode="fast" requires 20 → 20 >= 20 → Check safety → Drop!
  Reset: _steps_since_last_drop = 0

Step 22: 0 + 1 = 1
  Mode="slow" requires 200 → 1 < 200 → No drop

...

Step 221: 199 + 1 = 200
  Mode="slow" requires 200 → 200 >= 200 → Check safety → Drop!
  Reset: _steps_since_last_drop = 0
```

**Mode switches**:
```python
# Mode doesn't affect counter directly!
Step 50: _steps_since_last_drop = 5, mode="fast" (requires 20)
Step 51: _steps_since_last_drop = 6, mode="slow" (requires 200)
  6 < 200 → No drop (timer continues from where it was)
```

The counter is INDEPENDENT of mode. Mode only affects the threshold.

### Q: Is off an option if drop_package=True?

**A:** No, they're OPPOSITE!

**"off" mode**:
```python
command = ConveyorBeltCommand(mode="off")
steps_required = _mode_to_steps["off"]  # None

if steps_required is None:
    drop = False  # NEVER drop
```
Result: `ConveyorBeltAction(drop_package=False)`

**Other modes** (slow/mid/fast):
```python
command = ConveyorBeltCommand(mode="fast")
steps_required = _mode_to_steps["fast"]  # 20

if _steps_since_last_drop >= 20:
    if _safe_to_drop(state):
        drop = True  # CAN drop (if safe)
```
Result: `ConveyorBeltAction(drop_package=True)` (if checks pass)

**Relationship**:
```
Command        →  Controller Logic  →  Action
mode="off"     →  Never drop         →  drop_package=False
mode="slow"    →  Drop every 2.0s    →  drop_package=True (if safe & ready)
mode="mid"     →  Drop every 1.2s    →  drop_package=True (if safe & ready)
mode="fast"    →  Drop every 0.2s    →  drop_package=True (if safe & ready)
```

---

## State Management

### Q: What is _initial_state and why is it stored?

**A:** It's for trajectory reconstruction (debugging/verification).

**Stored at reset**:
```python
def reset(self, initial_state):
    self._initial_state = initial_state  # Save it
    self._steps_since_last_drop = 10**9
```

**Where it could be used** (though not currently in conveyor belt):
```python
def reconstruct_trajectory(self, actions):
    """Replay trajectory from initial state."""
    state = self._initial_state  # Start from saved initial state
    for action in actions:
        state = env.step(state, action)
    return state
```

**In trajectory extension**:
```python
# Fast-forward verifies determinism:
for t in range(len(actions)):
    recovered_action = controller.step(states[t], commands[t])
    assert env.actions_are_equal(recovered_action, actions[t])
```

If controller needs to know initial state for this verification, it's available in `_initial_state`.

**Practical use**: Minimal in current code, mostly for completeness/debugging.

### Q: What does "deterministic action sampling" mean?

**A:** Controller must produce SAME action given SAME inputs:

```python
# Deterministic:
state = ConveyorBeltState(positions=[0.5], heights=[0.0])
command = ConveyorBeltCommand(mode="fast")

action1 = controller.step(state, command)  # drop_package=True
action2 = controller.step(state, command)  # drop_package=True (SAME)
```

**If controller used randomness** (non-deterministic):
```python
def step(self, state, command):
    if random.random() < 0.5:
        return ConveyorBeltAction(drop_package=True)
    else:
        return ConveyorBeltAction(drop_package=False)

# Non-deterministic:
action1 = controller.step(state, command)  # drop_package=True
action2 = controller.step(state, command)  # drop_package=False (DIFFERENT!)
```

**Why determinism matters**:
- Trajectory replay must produce same results
- Verification checks rely on it
- Debugging requires reproducibility

**When RNG is needed**:
```python
class ConstraintBasedController:
    def __init__(self, seed):
        self._np_random = np.random.default_rng(seed)  # Seeded RNG
    
    def step(self, state, command):
        action_space = self.step_action_space(state, command)
        action_space.seed(sample_seed_from_rng(self._np_random))
        return action_space.sample()  # Deterministic (seeded)
```

Even with randomness, SEEDED RNG makes it deterministic!

### Q: In the example, why is middle box (1.2) not shown moving to 1.22?

**A:** Because it's FALLING (height > 0)!

**Example revisited**:
```python
# BEFORE:
positions       = [0.5,  1.2,  2.8]
falling_heights = [0.0,  0.3,  0.0]
# Box 0: landed
# Box 1: falling (height=0.3)
# Box 2: landed

# Physics simulation:
for p, h in zip(positions, falling_heights):
    if h > 0.0:
        predicted_positions.append(p)  # Box 1: 1.2 (unchanged)
    else:
        predicted_positions.append(p + 2.0*0.01)  # Boxes 0,2: +0.02

# AFTER:
predicted_positions = [0.52, 1.2, 2.82]
#                       ^^^  ^^^  ^^^
#                       moved  stationary  moved
```

**Box 1 doesn't move** because:
- `height = 0.3 > 0.0` → falling
- Falling boxes don't move horizontally
- Only landed boxes move with belt

**Full explanation added to docs!**

### Q: What happens with continue in the loop?

**A:** `continue` skips to next iteration:

```python
keep_pos, keep_fall = [], []

for p, h in zip(predicted_positions, falling_heights):
    if (p >= belt_length - box_width) or (p < 0):
        continue  # SKIP this box, go to next iteration
        # (Never reaches append statements below)
    
    keep_pos.append(p)
    keep_fall.append(h)
```

**Example**:
```python
predicted_positions = [0.5, 2.8, 5.0]  # Third box off belt (>= 2.7)
falling_heights = [0.0, 0.0, 0.0]
belt_length = 3.0
box_width = 0.3

# Iteration 1: p=0.5
#   0.5 >= 2.7? No
#   0.5 < 0? No
#   → Keep: keep_pos=[0.5], keep_fall=[0.0]

# Iteration 2: p=2.8
#   2.8 >= 2.7? YES!
#   → continue (skip append)
#   → keep_pos=[0.5], keep_fall=[0.0]

# Iteration 3: p=5.0
#   5.0 >= 2.7? YES!
#   → continue (skip append)
#   → keep_pos=[0.5], keep_fall=[0.0]

# Final: keep_pos=[0.5], keep_fall=[0.0]
```

---

## Failure Monitor Mechanics

### Q: How does state_check from MemorylessFailureMonitor work?

**A:** It's a function pointer:

```python
class MemorylessStateFailureMonitor:
    def __init__(self, state_check: Callable[[ObsType], bool]):
        self._state_check = state_check  # Store function
    
    def step(self, command, action, state):
        return self._state_check(state)  # Call stored function
```

**For ConveyorBeltFailureMonitor**:
```python
class ConveyorBeltFailureMonitor(MemorylessStateFailureMonitor):
    def __init__(self, scene_spec):
        super().__init__(self._check_failures)  # Pass method as function
        self._scene_spec = scene_spec
    
    def _check_failures(self, state):
        # Actual failure logic here
        if len(state.positions) <= 1:
            return False
        # ... compute diffs, check violations ...
```

**Flow**:
```
1. Create monitor: ConveyorBeltFailureMonitor(scene_spec)
   → Calls super().__init__(self._check_failures)
   → Stores self._check_failures in self._state_check

2. Call monitor.step(cmd, act, state):
   → Calls self._state_check(state)
   → Which calls self._check_failures(state)
   → Returns True/False
```

**Why this design?**
- Separates interface (MemorylessStateFailureMonitor) from implementation (_check_failures)
- Allows different monitors to plug in different check functions
- Follows strategy pattern

### Q: Does diffs < box_width auto-trigger failure?

**A:** Yes, currently:

```python
diffs = np.diff(sorted_positions)

if np.any(diffs < box_width):  # ANY gap < 0.3?
    return True  # FAILURE
```

**But this is WRONG because positions are left edges!**

```
Box A: position=0.5, width=0.3
├─────────┤
0.5      0.8

Box B: position=0.8, width=0.3
        ├─────────┤
        0.8      1.1

Center-to-center distance: 0.8 - 0.5 = 0.3
diffs = [0.3]
0.3 < 0.3? NO → No failure

BUT THEY'RE TOUCHING! Right edge of A = left edge of B = 0.8
```

**Correct check**:
```python
# Overlap if gap between EDGES < 0
for i in range(len(positions) - 1):
    right_edge_i = positions[i] + box_width
    left_edge_j = positions[i+1]
    gap = left_edge_j - right_edge_i  # Gap between boxes
    
    if gap < 0:  # Negative gap = overlap
        return True  # COLLISION
```

### Q: What is overlap_margin calculating?

**A:** It's measuring "safety margin" from collision:

```python
sorted_positions = [0.5, 1.0, 2.3]
diffs = [0.5, 1.3]  # Gaps between centers
min_gap = 0.5  # Smallest gap

# Margins:
box_width = 0.4
min_spacing = 0.1

spacing_margin = min_gap - min_spacing  # 0.5 - 0.1 = 0.4
overlap_margin = min_gap - box_width    # 0.5 - 0.4 = 0.1

robustness = min(0.4, 0.1) = 0.1
```

**Interpretation**:
- **Positive margin**: Safe (distance above threshold)
- **Zero margin**: At threshold (borderline)
- **Negative margin**: Violated (collision/too close)

```
min_gap = 0.5 (actual distance)
box_width = 0.4 (collision threshold)
overlap_margin = 0.5 - 0.4 = 0.1 (safety margin)

Meaning: "We're 0.1 units away from collision"
```

**But again, this uses center-to-center distance, which is wrong for collision detection!**

---

## Failure Finder & Trajectory Sampling

### Q: Which component controls when failure is found on trajectory 15 vs 1?

**A:** Combination of all components:

**Primary factors**:

1. **`max_trajectory_length`** (in `CommanderFailureFinder`):
   - **Short** (15 steps): Each trajectory explores less → might not reach failure
   - **Long** (100 steps): Each trajectory explores more → likely finds failure quickly

2. **`_seed` and `_rng`** (in `CommanderFailureFinder`):
   - Fixed seed → deterministic trajectory sequence
   - Trajectory 1 always same commands/states
   - If trajectory 1 doesn't find failure with length=15, it NEVER will

3. **Commander** (`RandomCommander`):
   - Samples commands randomly
   - Different trajectories get different command sequences
   - Some sequences lead to failures faster than others

4. **Controller faults**:
   - How lenient safety checks are
   - More lenient → failures happen sooner

5. **Scene parameters**:
   - `box_width`, `min_spacing`, `velocity`
   - Affect how quickly failures develop

**To find failure on trajectory 15-30 instead of 1**:

**Option 1: Shorter trajectories**
```python
failure_finder = RandomShootingFailureFinder(
    seed=42,
    max_num_trajectories=50,
    max_trajectory_length=15  # Short! Each traj less likely to find failure
)
```

**Option 2: Different seed**
```python
# Try seeds until you find one that fails on trajectory 15-30
for test_seed in range(1000):
    failure_finder = RandomShootingFailureFinder(
        seed=test_seed,
        max_num_trajectories=50,
        max_trajectory_length=25
    )
    result = failure_finder.run(env, controller, monitor)
    if result is not None:
        num_trajs = getattr(result, '_found_after_trajectories', 0)
        if 15 <= num_trajs <= 30:
            print(f"Found! seed={test_seed}, trajectories={num_trajs}")
            break
```

**Option 3: Stricter safety checks**
```python
# Make controller MORE restrictive
def _safe_to_drop(self, state):
    # Check h > 0.0 instead of h > 0.05
    for h in state.falling_heights:
        if h > 0.0:  # Stricter!
            return False
    
    # Check full box_width + min_spacing
    if len(state.positions) > 0:
        nearest = min(state.positions)
        if nearest < (self._scene_spec.box_width + min_spacing):  # Stricter!
            return False
    
    return True
```

More restrictive → fewer unsafe drops → rarer failures → need more trajectories

### Q: Does this mean initial state doesn't vary?

**A:** For conveyor belt, YES! It's deterministic:

```python
def get_initial_states(self):
    return EnumSpace([
        ConveyorBeltState(
            positions=np.array([]),  # ALWAYS empty
            falling_heights=np.array([]),
            step_count=0
        )
    ])
```

**Single initial state** → `initial_space.sample()` always returns same thing

**For hovercraft**, it CAN vary:
```python
# Hypothetical hovercraft initial states:
def get_initial_states(self):
    return BoxSpace(
        low=np.array([0.0, 0.0, -np.pi]),   # (x, y, theta) ranges
        high=np.array([10.0, 10.0, np.pi])
    )

# Different samples:
state1 = initial_space.sample()  # (2.3, 5.1, 0.5)
state2 = initial_space.sample()  # (7.8, 1.2, -1.2)
```

**How to make conveyor belt vary**:
```python
def get_initial_states(self):
    return EnumSpace([
        # Empty belt
        ConveyorBeltState(positions=[], heights=[], step_count=0),
        
        # One box already on belt
        ConveyorBeltState(positions=[1.0], heights=[0.0], step_count=0),
        
        # Two boxes on belt
        ConveyorBeltState(positions=[0.5, 2.0], heights=[0.0, 0.0], step_count=0),
    ])
```

Now `initial_space.sample()` randomly picks one of 3 starting configurations!

### Q: Why is hovercraft not passed max_num_trajectories?

**A:** It uses defaults:

```python
# In test:
failure_finder = RandomShootingFailureFinder(seed=123)
# No max_num_trajectories or max_trajectory_length specified

# In __init__:
def __init__(
    self,
    max_num_trajectories: int = 1000,  # DEFAULT
    max_trajectory_length: int = 100,  # DEFAULT
    seed: int = 0,
):
```

Hovercraft test uses:
- `max_num_trajectories=1000` (default)
- `max_trajectory_length=100` (default)

Conveyor belt test explicitly overrides:
- `max_num_trajectories=50`
- `max_trajectory_length=20`

---

## Initial States & Randomness

### Q: What does seeding a RandomCommander or RNG mean?

**A:** Sets initial state for deterministic randomness:

**RNG (Random Number Generator)**:
```python
# No seed → different every time
rng = np.random.default_rng()
print(rng.random())  # 0.234 (run 1)
print(rng.random())  # 0.876 (run 2)

# With seed → same sequence every time
rng = np.random.default_rng(seed=42)
print(rng.random())  # 0.374 (run 1)
rng = np.random.default_rng(seed=42)
print(rng.random())  # 0.374 (run 2, SAME!)
```

**RandomCommander**:
```python
commander = RandomCommander(command_space)
commander.seed(42)  # Set seed

# Now sampling is deterministic:
cmd1 = commander.get_command()  # mode="fast"
cmd2 = commander.get_command()  # mode="slow"

# Reset with same seed:
commander.seed(42)
cmd3 = commander.get_command()  # mode="fast" (SAME as cmd1!)
cmd4 = commander.get_command()  # mode="slow" (SAME as cmd2!)
```

**Why seed**:
- Reproducibility: Same seed → same behavior
- Debugging: Can replay exact scenario
- Testing: Consistent results

### Q: Does commander depend on state or is it random?

**A:** For `RandomCommander`, it's **independent** of state:

```python
class RandomCommander:
    def get_command(self):
        return self.command_space.sample()  # Ignores state!
```

Purely random sampling, doesn't look at current state.

**Other commanders COULD use state**:
```python
class GreedyCommander:
    def get_command(self, state):  # Takes state
        if len(state.positions) > 5:
            return ConveyorBeltCommand(mode="off")  # Too many boxes
        else:
            return ConveyorBeltCommand(mode="fast")  # Keep dropping
```

But RandomCommander doesn't. That's why it's "naive" / "random shooting".

### Q: What does a large seed number vs low seed mean?

**A:** **Nothing significant!** Seeds are just identifiers:

```python
rng1 = np.random.default_rng(seed=1)
rng2 = np.random.default_rng(seed=999999)
```

Both generate equally "random" sequences. The number itself doesn't matter.

**What DOES matter**:
- **Same seed** → same sequence
- **Different seeds** → different sequences

```python
# seed=5 vs seed=5 → SAME
rng_a = np.random.default_rng(5)
rng_b = np.random.default_rng(5)
rng_a.random()  # 0.223
rng_b.random()  # 0.223 (SAME)

# seed=5 vs seed=6 → DIFFERENT  
rng_c = np.random.default_rng(5)
rng_d = np.random.default_rng(6)
rng_c.random()  # 0.223
rng_d.random()  # 0.891 (DIFFERENT)
```

Magnitude doesn't matter. Just distinctness.

---

## Making Failures Rarer

### Summary: How to Make Rare Failures Actually Rare

**1. Use Actual Collision Detection**:
```python
def _check_actual_collisions(self, state):
    # Only fail when boxes PHYSICALLY overlap
    # Ignore "too close" spacing violations
    # Check if falling box lands ON landed box
```

**2. Remove min_spacing checks**:
```python
# DON'T check spacing violations:
# if np.any(diffs < min_spacing):
#     return True

# ONLY check actual overlaps:
if np.any(diffs < box_width):
    return True
```

**3. Stricter controller safety checks**:
```python
def _safe_to_drop(self, state):
    # Check h > 0.0 (not > 0.05)
    for h in state.falling_heights:
        if h > 0.0:
            return False
    
    # Check full box_width + min_spacing
    nearest = min(state.positions)
    if nearest < (box_width + min_spacing):
        return False
```

**4. Adjust scene parameters**:
```python
scene_spec = ConveyorBeltSceneSpec(
    box_width=0.2,  # Smaller boxes (harder to collide)
    conveyor_belt_velocity=1.5,  # Slower belt (more spacing)
    dt=0.01,  # Smaller timestep (more accurate)
)
```

**5. Shorter trajectories, more samples**:
```python
failure_finder = RandomShootingFailureFinder(
    seed=random.randint(0, 1000),  # Random seed
    max_num_trajectories=200,  # More trajectories
    max_trajectory_length=20,  # Shorter each (less likely to fail)
)
```

**Result**: Failures only occur on ACTUAL collisions, making them genuinely rare while still findable with enough exploration!











