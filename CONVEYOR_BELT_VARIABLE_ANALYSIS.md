# Conveyor Belt System: Complete Variable Analysis

## System Overview
The conveyor belt system simulates boxes dropping onto a moving belt. The system consists of:
1. **Environment** (`ConveyorBeltEnv`) - Physics simulation
2. **Controller** (`ConveyorBeltController`) - Decision making (with intentional faults)
3. **Failure Monitor** (`ConveyorBeltFailureMonitor`) - Failure detection
4. **Scene Specification** (`ConveyorBeltSceneSpec`) - Configuration parameters

---

## 1. CONVEYOR BELT STATE (`ConveyorBeltState`)

### `positions: NDArray[np.float32]`
- **What it is**: X-coordinates of all boxes on the belt
- **How it works**: 
  - Each element represents one box's position along the belt (0 to belt_length)
  - Empty array `[]` = no boxes on belt
  - Array length = number of boxes currently on belt
- **Updates**: 
  - When box lands: position stays at drop point initially
  - When box is on belt: `position += conveyor_belt_velocity * dt` each step
  - When box falls off: removed from array
- **Example**: `[0.5, 1.2, 2.8]` = 3 boxes at positions 0.5, 1.2, and 2.8

### `falling_heights: NDArray[np.float32]`
- **What it is**: Vertical distance above belt for each box
- **How it works**:
  - `0.0` = box is on the belt (landed)
  - `> 0.0` = box is falling (in air)
  - Parallel array to `positions` (same length, index-matched)
- **Updates**:
  - Each step: `height -= gravity * dt` (gravity pulls down)
  - When `height <= 0`: clamped to `0.0` (box has landed)
- **Example**: `[0.0, 0.3, 0.0]` = first box landed, second falling (0.3m high), third landed

### Why `positions` and `falling_heights` are Parallel Arrays

**Design Choice**: These arrays are parallel (index-matched) rather than using a single array of objects.

**What "Parallel" Means**:
```python
positions       = [0.5,  1.2,  2.8,  0.0]
falling_heights = [0.0,  0.3,  0.0,  0.9]
                   ↓     ↓     ↓     ↓
Box 0: position=0.5, height=0.0 (on belt)
Box 1: position=1.2, height=0.3 (falling)
Box 2: position=2.8, height=0.0 (on belt)
Box 3: position=0.0, height=0.9 (falling)
```

**Invariant**: `len(positions) == len(falling_heights)` ALWAYS

**Why This Design**:

1. **Each Box Needs Both Properties**:
   - Can't represent a box with just position (don't know if falling or landed)
   - Can't represent a box with just height (don't know where it is horizontally)
   - `positions[i]` and `falling_heights[i]` together describe box `i` completely

2. **Efficient Numpy Operations**:
   ```python
   # Update all falling heights at once (vectorized)
   falling_heights -= gravity * dt
   falling_heights = np.maximum(falling_heights, 0.0)  # Clamp to 0
   
   # Update all positions at once (vectorized)
   positions += conveyor_belt_velocity * dt
   ```
   - Parallel arrays enable fast numpy broadcasting
   - Alternative (list of objects) would require Python loops (slower)

3. **Conditional Updates Based on State**:
   ```python
   for p, h in zip(positions, falling_heights, strict=True):
       if h > 0.0:
           predicted_positions.append(p)  # Falling: position unchanged
       else:
           predicted_positions.append(p + v * dt)  # On belt: move
   ```
   - Need to check `falling_heights[i]` to decide how to update `positions[i]`
   - `zip()` naturally pairs corresponding elements from parallel arrays

4. **Synchronized Operations**:
   ```python
   # When dropping new box, both arrays extended together:
   if action.drop_package:
       predicted_positions.append(initial_drop_position)  # x-position
       falling_heights.append(drop_start_height)         # height
   
   # When removing off-belt boxes, both arrays filtered together:
   keep_pos, keep_fall = [], []
   for p, h in zip(predicted_positions, falling_heights, strict=True):
       if not ((p >= belt_length - box_width) or (p < 0)):
           keep_pos.append(p)
           keep_fall.append(h)
   ```
   - Operations that add/remove boxes must maintain parallelism
   - `strict=True` in `zip()` enforces equal length (catches bugs)

5. **Dataclass Immutability**:
   ```python
   @dataclass(frozen=True)
   class ConveyorBeltState:
       positions: NDArray[np.float32]
       falling_heights: NDArray[np.float32]
   ```
   - State is frozen (immutable)
   - Can't modify individual box properties in-place
   - Must create new arrays for new state

**Alternative Design (Not Used)**:
```python
# Could use array of objects:
boxes = [
    {"position": 0.5, "height": 0.0},
    {"position": 1.2, "height": 0.3},
    {"position": 2.8, "height": 0.0},
]

# Why NOT used:
# 1. Slower (Python objects vs numpy arrays)
# 2. Can't use numpy vectorization
# 3. More complex state representation
# 4. Harder to integrate with numerical algorithms
```

**How Parallelism is Maintained**:

Every operation that modifies one array must modify the other correspondingly:

| Operation | Effect on `positions` | Effect on `falling_heights` | Parallelism Maintained? |
|-----------|----------------------|----------------------------|-------------------------|
| Drop box | Append `0.0` | Append `1.0` | ✓ Both grow by 1 |
| Box lands | No change | `h` → `0.0` | ✓ Same indices |
| Box moves | `p` → `p + v*dt` | No change | ✓ Same indices |
| Box falls | No change | `h` → `h - g*dt` | ✓ Same indices |
| Remove box | Delete `positions[i]` | Delete `falling_heights[i]` | ✓ Both shrink by 1 |

**Accessing Box Properties**:
```python
# To get state of box i:
box_position = positions[i]
box_height = falling_heights[i]
is_falling = falling_heights[i] > 0.0
is_landed = falling_heights[i] == 0.0

# To iterate over all boxes:
for pos, height in zip(positions, falling_heights, strict=True):
    print(f"Box at position {pos} with height {height}")

# To find falling boxes:
falling_mask = falling_heights > 0.0
falling_positions = positions[falling_mask]
falling_heights_subset = falling_heights[falling_mask]
```

**Failure Detection Uses Positions Only**:
```python
# Interestingly, failure detection ONLY uses positions:
sorted_positions = np.sort(state.positions)
diffs = np.diff(sorted_positions)
if np.any(diffs < min_spacing) or np.any(diffs < box_width):
    return True  # FAILURE

# Heights don't matter for collisions (only x-position matters)
# But heights affect WHEN failures occur (boxes must land to collide)
```

**Example Execution Showing Parallelism**:

```python
# Initial state (empty)
positions       = []
falling_heights = []

# Step 1: Drop box
positions       = [0.0]
falling_heights = [1.0]
# Box 0: x=0.0, h=1.0 (falling)

# Step 2: Gravity applied, box still falling
positions       = [0.0]
falling_heights = [0.9019]
# Box 0: x=0.0, h=0.9019 (still falling)

# Step 10: Box lands (h → 0)
positions       = [0.0]
falling_heights = [0.0]
# Box 0: x=0.0, h=0.0 (landed)

# Step 11: Box moves with belt
positions       = [0.02]
falling_heights = [0.0]
# Box 0: x=0.02, h=0.0 (landed, moving)

# Step 20: Drop second box
positions       = [0.22, 0.0]
falling_heights = [0.0,  1.0]
# Box 0: x=0.22, h=0.0 (landed, moving)
# Box 1: x=0.0, h=1.0 (falling)

# Note: Box 1 added at END of arrays (FIFO)
```

**Why Order Matters**:
- Boxes are added to END of arrays (append)
- Oldest boxes at front, newest at back
- But failure detection doesn't care about order (sorts positions)
- Order is primarily for tracking individual boxes over time

### `step_count: int`
- **What it is**: Number of simulation steps elapsed
- **How it works**: Increments by 1 each time step is called
- **Purpose**: Used for visualization (rotating rollers) and debugging
- **Example**: `step_count=150` means 150 time steps have passed

---

## 2. CONVEYOR BELT ACTION (`ConveyorBeltAction`)

### `drop_package: bool`
- **What it is**: Whether to drop a new box this time step
- **How it works**:
  - `True` = create new box at `initial_drop_position` with `drop_start_height`
  - `False` = no new box this step
- **Decision**: Made by controller based on command mode and safety checks
- **Example**: `drop_package=True` → new box appears at position 0.0, height 1.0m

---

## 3. SCENE SPECIFICATION (`ConveyorBeltSceneSpec`)

### `conveyor_belt_velocity: float = 1.0`
- **What it is**: Speed of belt movement (units per second)
- **How it works**: 
  - Each step, landed boxes move: `position += velocity * dt`
  - Higher = boxes move faster, less time between drops
- **Impact**: Affects collision probability (faster = more spacing needed)

### `dt: float = 0.01`
- **What it is**: Time step size (seconds per simulation step)
- **How it works**: 
  - Smaller = more accurate but slower simulation
  - Used in: `position += velocity * dt`, `height -= gravity * dt`
- **Impact**: Affects all time-based calculations

### `belt_length: float = 5.0`
- **What it is**: Total length of the conveyor belt
- **How it works**:
  - Boxes removed when `position >= belt_length - box_width` or `position < 0`
  - Defines valid position range: `[0, belt_length]`
- **Impact**: Longer belt = more boxes can fit, more time before removal

### `box_width: float = 0.4`
- **What it is**: Width of each box (in x-direction)
- **How it works**:
  - Used to detect overlaps: if `|pos1 - pos2| < box_width` → collision
  - Used to remove boxes: `position >= belt_length - box_width` → off belt
- **Impact**: Larger boxes = easier to collide, fewer fit on belt

### `box_height: float = 0.5`
- **What it is**: Height of each box (for visualization only)
- **How it works**: Only affects rendering, not physics

### `gravity: float = 9.81`
- **What it is**: Acceleration due to gravity (m/s²)
- **How it works**: 
  - Each step: `falling_heights[i] -= gravity * dt`
  - Determines how fast boxes fall
- **Impact**: Higher gravity = faster falling, less time in air

### `drop_start_height: float = 1.0`
- **What it is**: Initial height when box is dropped
- **How it works**: New boxes start at this height, then fall
- **Impact**: Higher = more time to fall, more time before landing

### `initial_drop_position: float = 0.0`
- **What it is**: X-position where new boxes appear
- **How it works**: All new boxes start at this position
- **Impact**: Usually 0.0 (left end of belt)

### `min_spacing: float` (added dynamically)
- **What it is**: Minimum required distance between box centers
- **How it works**: 
  - Not in base spec, added via `object.__setattr__(scene_spec, "min_spacing", 0.1)`
  - Used by controller (safety check) and failure monitor (violation detection)
- **Impact**: Larger = stricter spacing requirement, harder to satisfy

---

## 4. CONTROLLER VARIABLES (`ConveyorBeltController`)

### `_scene_spec: ConveyorBeltSceneSpec`
- **What it is**: Reference to scene configuration
- **How it works**: Provides access to `box_width`, `min_spacing`, `dt` for safety checks

### `_initial_state: Optional[ConveyorBeltState]`
- **What it is**: First state seen (stored on reset)
- **How it works**: Set in `reset()`, used for trajectory reconstruction

### `_mode_to_steps: dict[str, int | None]`
- **What it is**: Maps command modes to required steps between drops
- **How it works**:
  - `"off"`: `None` (never drop)
  - `"slow"`: `int(2.0 / dt)` = 200 steps (if dt=0.01)
  - `"mid"`: `int(1.2 / dt)` = 120 steps
  - `"fast"`: `int(0.2 / dt)` = 20 steps
- **Purpose**: Enforces timing between drops based on mode

### `_steps_since_last_drop: int`
- **What it is**: Counter tracking steps since last successful drop
- **How it works**:
  - Starts at `10**9` (effectively infinite, prevents immediate drop)
  - Increments each step: `_steps_since_last_drop += 1`
  - Resets to `0` when drop occurs
  - Drop allowed when `_steps_since_last_drop >= steps_required`
- **Purpose**: Enforces minimum time between drops

### `_np_random: np.random.Generator`
- **What it is**: Random number generator (seeded)
- **How it works**: Used for deterministic action sampling if needed

---

## 5. CONTROLLER COMMAND (`ConveyorBeltCommand`)

### `mode: str`
- **What it is**: Drop rate mode
- **Values**: `"off"`, `"slow"`, `"mid"`, `"fast"`
- **How it works**: 
  - Selected by commander (failure finder)
  - Controller uses it to determine timing requirement
  - `"off"` = never drop, others = drop at specified intervals

---

## 6. CONTROLLER SAFETY CHECK (`_safe_to_drop`)

### Input: `state: ConveyorBeltState`
- **What it is**: Current system state
- **How it works**: Controller checks if dropping now would be safe

### Check 1: Falling boxes
- **Logic**: `for h in state.falling_heights: if h > 0.05: return False`
- **FAULT**: Should check `h > 0.0`, but checks `h > 0.05`
- **Impact**: Allows drops when boxes are almost landed (height 0.0-0.05)
- **Result**: Can cause collisions with nearly-landed boxes

### Check 2: Nearest box spacing
- **Logic**: `nearest = min(state.positions)`, then `if nearest < min_spacing * 0.8: return False`
- **FAULT 1**: Should check `nearest < (box_width + min_spacing)` but only checks `min_spacing`
- **FAULT 2**: Uses 80% margin (`* 0.8`) instead of full requirement
- **Impact**: 
  - Ignores box width (allows overlaps if spacing satisfied)
  - Allows drops when 20% too close
- **Result**: Boxes can overlap or be too close

### Return: `bool`
- **True**: Safe to drop (all checks passed)
- **False**: Not safe (some check failed)

---

## 7. CONTROLLER ACTION GENERATION (`step_action_space`)

### Input: `state: ConveyorBeltState`, `command: ConveyorBeltCommand`
- **What it is**: Current state and desired mode
- **How it works**: Controller decides whether to drop

### Process:
1. **Increment counter**: `_steps_since_last_drop += 1`
2. **Get timing requirement**: `steps_required = _mode_to_steps[command.mode]`
3. **Check timing**: 
   - If `"off"` → `drop = False`
   - If `_steps_since_last_drop >= steps_required` → check safety
   - Else → `drop = False`
4. **Check safety**: `drop = _safe_to_drop(state)` (if timing satisfied)
5. **Reset counter**: If `drop == True`, set `_steps_since_last_drop = 0`
6. **Return**: `ConveyorBeltAction(drop_package=drop)`

---

## 8. ENVIRONMENT STATE TRANSITION (`get_next_states`)

### Input: `state: ConveyorBeltState`, `action: ConveyorBeltAction`
- **What it is**: Current state and action to apply
- **How it works**: Physics simulation for one time step

### Key Variables in State Transition

#### `positions` (from input state)
- **What it is**: X-coordinates from the PREVIOUS time step (current state)
- **Source**: `positions = list(state.positions)`
- **Mutability**: Converted to mutable Python list from immutable numpy array
- **Purpose**: Starting point for computing next positions

#### `predicted_positions` (computed during transition)
- **What it is**: X-coordinates for the NEXT time step (after physics simulation)
- **Source**: Computed fresh, starts as empty list `predicted_positions = []`
- **Purpose**: Build up new positions by:
  1. Moving landed boxes forward with belt
  2. Keeping falling boxes stationary
  3. Adding new dropped boxes
  4. Filtering out off-belt boxes

#### The Difference:
```
positions          = [0.5,  1.2,  2.8]      # BEFORE this step (input)
                           ↓ physics simulation
predicted_positions = [0.52, 1.2,  2.82]    # AFTER this step (output)
```

**Why separate variables?**
- **Immutability**: Input state is frozen, can't modify in-place
- **Clarity**: Distinguish between current vs. next positions
- **Multi-step process**: Need intermediate variable while building next state
- **Filtering**: May add/remove boxes, can't update array in-place

### Process:

#### Step 1: Update Falling Heights
```python
for i in range(len(falling_heights)):
    if falling_heights[i] > 0:
        falling_heights[i] -= gravity * dt  # Apply gravity
        if falling_heights[i] <= 0:
            falling_heights[i] = 0.0  # Clamp to 0 (landed)
```
- **Effect**: All falling boxes move down by `gravity * dt`
- **Result**: Boxes eventually land (height → 0)

#### Step 2: Predict Positions
```python
for p, h in zip(positions, falling_heights):
    if h > 0.0:
        predicted_positions.append(p)  # Falling: position unchanged
    else:
        predicted_positions.append(p + v * dt)  # On belt: move with belt
```
- **Effect**: 
  - Falling boxes: position stays same (only moving vertically)
  - Landed boxes: position increases by `velocity * dt` (moving with belt)

**Detailed Explanation of `positions` vs `predicted_positions`:**

This step transforms the input `positions` array into the output `predicted_positions` array:

```
INPUT (current state):
positions       = [0.5,  1.2,  2.8]
falling_heights = [0.0,  0.3,  0.0]  # (already updated in Step 1)

PROCESSING:
Box 0: h=0.0 (landed) → p_new = 0.5 + 2.0*0.01 = 0.52
Box 1: h=0.3 (falling) → p_new = 1.2 (unchanged)
Box 2: h=0.0 (landed) → p_new = 2.8 + 2.0*0.01 = 2.82

OUTPUT:
predicted_positions = [0.52, 1.2, 2.82]
```

**Why "predicted"?**
- Name comes from physics simulation terminology
- We're "predicting" where boxes will be after one time step
- Not truly a prediction (deterministic), but conventional naming
- Emphasizes this is FUTURE state, not current state

**Key Insight**: The physics happens in stages:
1. Heights updated (gravity) → boxes may land
2. Positions updated (belt movement) → but only for LANDED boxes
3. New boxes added (if dropping)
4. Off-belt boxes removed

**Why falling boxes don't move horizontally**:
- In reality, boxes fall straight down (no horizontal velocity)
- Belt moves beneath them while they're in air
- Only when landed (h=0.0) do they "stick" to belt and move with it
- This creates realistic physics behavior

**Memory flow**:
```
state.positions (immutable numpy array)
      ↓ copy to mutable list
positions (Python list)
      ↓ transform element-by-element
predicted_positions (Python list, initially empty)
      ↓ filter and convert
keep_pos (Python list, filtered)
      ↓ convert to immutable numpy array
next_state.positions (immutable numpy array)
```

#### Step 3: Drop New Package (if action says so)
```python
if action.drop_package:
    predicted_positions.append(initial_drop_position)  # Usually 0.0
    falling_heights.append(drop_start_height)  # Usually 1.0
```
- **Effect**: Adds new box to arrays

#### Step 4: Remove Off-Belt Boxes
```python
for p, h in zip(predicted_positions, falling_heights):
    if (p >= belt_length - box_width) or (p < 0):
        continue  # Skip (removed)
    keep_pos.append(p)
    keep_fall.append(h)
```
- **Effect**: Removes boxes that moved off either end of belt
- **Condition**: `position >= belt_length - box_width` (right end) or `position < 0` (left end)

#### Step 5: Create New State
```python
ConveyorBeltState(
    positions=np.array(keep_pos),
    falling_heights=np.array(keep_fall),
    step_count=state.step_count + 1
)
```

### Complete Example: `positions` vs `predicted_positions` Evolution

Let's trace a complete state transition to see how these variables differ:

#### Initial State (t=0):
```python
state.positions       = [0.5,  1.5]
state.falling_heights = [0.0,  1.0]
action.drop_package   = True

# Box 0: x=0.5, h=0.0 (landed, on belt)
# Box 1: x=1.5, h=1.0 (falling in air)
```

#### Step 1: Copy to Mutable Lists
```python
positions       = [0.5,  1.5]      # Copy from state
falling_heights = [0.0,  1.0]      # Copy from state
```
- These are now MUTABLE Python lists
- Can be modified without affecting input state

#### Step 2: Update Heights (Gravity)
```python
# Box 0: 0.0 stays 0.0 (already landed)
# Box 1: 1.0 - 9.81*0.01 = 0.9019

falling_heights = [0.0,  0.9019]   # MODIFIED in-place
```
- Heights updated
- `positions` still unchanged: `[0.5, 1.5]`

#### Step 3: Compute Predicted Positions
```python
predicted_positions = []  # Start empty

# Box 0: h=0.0 (landed) → move with belt
predicted_positions.append(0.5 + 2.0*0.01)
# predicted_positions = [0.52]

# Box 1: h=0.9019 (falling) → stays in place
predicted_positions.append(1.5)
# predicted_positions = [0.52, 1.5]
```

**At this point**:
```python
positions           = [0.5,  1.5]    # OLD (input state)
predicted_positions = [0.52, 1.5]    # NEW (after physics)
falling_heights     = [0.0,  0.9019] # NEW (after gravity)
```

#### Step 4: Add New Box (action.drop_package=True)
```python
predicted_positions.append(0.0)   # New box at drop position
falling_heights.append(1.0)        # New box at drop height

predicted_positions = [0.52, 1.5, 0.0]
falling_heights     = [0.0, 0.9019, 1.0]
```

**At this point**:
```python
# OLD positions (2 boxes):
positions = [0.5, 1.5]

# NEW predicted_positions (3 boxes):
predicted_positions = [0.52, 1.5, 0.0]

# Box count increased from 2 → 3
```

#### Step 5: Filter Off-Belt Boxes
```python
keep_pos, keep_fall = [], []

# Check each box:
# Box 0: p=0.52, in range [0, 3.0-0.3] → KEEP
keep_pos.append(0.52)
keep_fall.append(0.0)

# Box 1: p=1.5, in range [0, 2.7] → KEEP
keep_pos.append(1.5)
keep_fall.append(0.9019)

# Box 2: p=0.0, in range [0, 2.7] → KEEP
keep_pos.append(0.0)
keep_fall.append(1.0)

keep_pos = [0.52, 1.5, 0.0]
keep_fall = [0.0, 0.9019, 1.0]
```

#### Step 6: Create Output State
```python
next_state = ConveyorBeltState(
    positions       = np.array([0.52, 1.5, 0.0]),
    falling_heights = np.array([0.0, 0.9019, 1.0]),
    step_count      = 1
)
```

#### Summary of Variable Transformations:
```
INPUT STATE (t=0):
  state.positions = [0.5, 1.5]
        ↓ copy
  positions = [0.5, 1.5]
        ↓ physics simulation (belt movement for landed boxes)
  predicted_positions = [0.52, 1.5]
        ↓ add new box
  predicted_positions = [0.52, 1.5, 0.0]
        ↓ filter off-belt boxes
  keep_pos = [0.52, 1.5, 0.0]
        ↓ convert to numpy array
OUTPUT STATE (t=1):
  next_state.positions = [0.52, 1.5, 0.0]
```

**Key Observations**:
1. `positions` (input) never changes: `[0.5, 1.5]`
2. `predicted_positions` built incrementally: `[] → [0.52, 1.5] → [0.52, 1.5, 0.0]`
3. Box count can change: 2 boxes → 3 boxes
4. Only landed boxes moved: Box 0 moved 0.02 units, Box 1 stayed at 1.5
5. New box added at end: Box 2 at position 0.0

**Why This Matters for Failure Detection**:
```python
# Failure monitor receives next_state with NEW positions:
failure_monitor.step(command, action, next_state)

# Inside monitor:
sorted_positions = np.sort(next_state.positions)  # [0.0, 0.52, 1.5]
diffs = np.diff(sorted_positions)                  # [0.52, 0.98]

# Checks:
# Gap between Box 2 and Box 0: 0.52 > 0.3 (box_width) ✓
# Gap between Box 0 and Box 1: 0.98 > 0.3 (box_width) ✓
# No failure
```

**Common Confusion**:
❌ "Why not update `positions` in-place?"
- State is IMMUTABLE (frozen dataclass)
- Must create NEW state with NEW arrays
- Can't modify `state.positions[i] = new_value`

✓ "Use intermediate `predicted_positions` variable"
- Build up new positions step by step
- Allows adding/removing boxes
- Clear separation between input and output

---

## 9. FAILURE MONITOR VARIABLES (`ConveyorBeltFailureMonitor`)

### `_scene_spec: ConveyorBeltSceneSpec`
- **What it is**: Reference to scene configuration
- **How it works**: Provides `box_width` and `min_spacing` for failure detection

### `_state_check: Callable[[ConveyorBeltState], bool]`
- **What it is**: Function that checks if state represents a failure
- **How it works**: Points to `_check_failures` method

---

## 10. FAILURE DETECTION (`_check_failures`)

### Input: `state: ConveyorBeltState`
- **What it is**: State to check for failures
- **How it works**: Detects spacing violations and overlaps

### Process:

#### Step 1: Early Exit
```python
if len(state.positions) <= 1:
    return False  # Can't fail with 0 or 1 boxes
```

#### Step 2: Sort and Compute Differences
```python
sorted_positions = np.sort(state.positions)  # Sort by x-position
diffs = np.diff(sorted_positions)  # Compute gaps between adjacent boxes
```
- **Effect**: 
  - `sorted_positions = [0.5, 1.2, 2.8]` → `diffs = [0.7, 1.6]`
  - `diffs[i]` = gap between box `i` and box `i+1`

#### Step 3: Check Violations
```python
min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)
box_width = self._scene_spec.box_width

if np.any(diffs < min_spacing) or np.any(diffs < box_width):
    return True  # FAILURE DETECTED
```
- **Failure conditions**:
  - `diffs < min_spacing`: Boxes too close (spacing violation)
  - `diffs < box_width`: Boxes overlapping (collision)
- **Example**: 
  - If `min_spacing=0.1`, `box_width=0.4`
  - `diffs = [0.05, 0.3]` → `0.05 < 0.1` → **FAILURE** (spacing violation)
  - `diffs = [0.2, 0.3]` → `0.2 < 0.4` → **FAILURE** (overlap)

---

## 11. ROBUSTNESS SCORE (`get_robustness_score`)

### Input: `state: ConveyorBeltState`
- **What it is**: State to evaluate
- **How it works**: Computes how close state is to failure

### Process:

#### Step 1: Find Minimum Gap
```python
sorted_positions = np.sort(state.positions)
diffs = np.diff(sorted_positions)
min_gap = np.min(diffs)  # Smallest gap between any two boxes
```

#### Step 2: Compute Margins
```python
spacing_margin = min_gap - min_spacing  # How much above spacing requirement
overlap_margin = min_gap - box_width    # How much above overlap threshold
```

#### Step 3: Return Minimum Margin
```python
return min(spacing_margin, overlap_margin)
```
- **Positive**: Safe (gap larger than both requirements)
- **Zero or negative**: Already failed (gap smaller than requirement)
- **Example**: 
  - `min_gap=0.5`, `min_spacing=0.1`, `box_width=0.4`
  - `spacing_margin = 0.4`, `overlap_margin = 0.1`
  - Returns `0.1` (closer to overlap than spacing violation)

---

## 12. VARIABLE FLOW THROUGH SYSTEM

### Execution Flow:

1. **Initialization**:
   - `ConveyorBeltEnv` created with `scene_spec`
   - `ConveyorBeltController` created with `seed` and `scene_spec`
   - `ConveyorBeltFailureMonitor` created with `scene_spec`

2. **Reset**:
   - `env.reset()` → `state = ConveyorBeltState(positions=[], falling_heights=[], step_count=0)`
   - `controller.reset(state)` → `_steps_since_last_drop = 10**9`
   - `failure_monitor.reset(state)` → (no-op for memoryless monitor)

3. **Each Step**:
   - **Commander** selects `command: ConveyorBeltCommand(mode="fast")`
   - **Controller** processes:
     - `_steps_since_last_drop += 1`
     - Checks timing: `_steps_since_last_drop >= steps_required?`
     - If yes: `drop = _safe_to_drop(state)` (checks falling heights, spacing)
     - `action = ConveyorBeltAction(drop_package=drop)`
   - **Environment** processes:
     - Updates `falling_heights` (gravity)
     - Updates `positions` (belt movement)
     - If `action.drop_package`: adds new box
     - Removes off-belt boxes
     - Creates new `state` with `step_count + 1`
   - **Failure Monitor** checks:
     - `failure = _check_failures(new_state)`
     - If `failure == True`: trajectory is a failure

---

## 13. KEY INTERACTIONS

### Controller Faults → Failures:

1. **Fault 1** (falling height check):
   - **Should**: `if h > 0.0: return False`
   - **Actually**: `if h > 0.05: return False`
   - **Result**: Drops allowed when box is 0.0-0.05m high (almost landed)
   - **Failure**: New box can collide with nearly-landed box

2. **Fault 2** (spacing check):
   - **Should**: `if nearest < (box_width + min_spacing): return False`
   - **Actually**: `if nearest < min_spacing * 0.8: return False`
   - **Result**: 
     - Ignores `box_width` (allows overlaps if spacing OK)
     - Uses 80% margin (allows drops 20% too close)
   - **Failure**: Boxes can overlap or be too close

### Why Failures Are Rare:

- **Timing constraint**: Controller enforces minimum time between drops
- **Safety checks**: Even with faults, some conditions prevent drops
- **Physics**: Boxes move and fall, creating dynamic spacing
- **Removal**: Boxes fall off belt, clearing space

### Why Longer Trajectories Find More Failures:

- **Accumulation**: More boxes accumulate on belt over time
- **Movement**: Boxes move, creating varying spacing
- **Timing**: More drop attempts = more chances for fault conditions
- **Complexity**: More boxes = more potential collision pairs

---

## 14. SUMMARY TABLE

| Variable | Type | Purpose | Updates When |
|----------|------|---------|--------------|
| `positions` | `NDArray[float32]` | Box x-positions | Each step (belt movement) |
| `falling_heights` | `NDArray[float32]` | Box heights above belt | Each step (gravity) |
| `step_count` | `int` | Simulation step counter | Each step (+1) |
| `drop_package` | `bool` | Whether to drop box | Controller decision |
| `_steps_since_last_drop` | `int` | Steps since last drop | Each step (+1), reset on drop |
| `conveyor_belt_velocity` | `float` | Belt speed | Constant (config) |
| `dt` | `float` | Time step size | Constant (config) |
| `belt_length` | `float` | Belt length | Constant (config) |
| `box_width` | `float` | Box width | Constant (config) |
| `min_spacing` | `float` | Min spacing requirement | Constant (config, added dynamically) |
| `gravity` | `float` | Gravity acceleration | Constant (config) |
| `drop_start_height` | `float` | Initial drop height | Constant (config) |
| `mode` | `str` | Drop rate mode | Selected by commander |

---

## 15. FAILURE FINDER ARCHITECTURE

The failure finder system consists of multiple interconnected components that work together to discover failure trajectories. The architecture follows a hierarchical design:

```
RandomShootingFailureFinder (concrete implementation)
    ↓ extends
CommanderFailureFinder (abstract base class)
    ↓ implements
FailureFinder (interface)
```

### Components:
1. **FailureFinder**: Interface defining the `run()` method
2. **CommanderFailureFinder**: Base class managing trajectory sampling loop
3. **RandomShootingFailureFinder**: Concrete implementation using random sampling
4. **Commander**: Selects commands to guide controller
5. **InitialStateCommander**: Selects initial states for trajectories
6. **Trajectory Extension**: Core loop that generates and checks trajectories

---

## 16. COMMANDER FAILURE FINDER VARIABLES (`CommanderFailureFinder`)

### `_max_num_trajectories: int = 1000`
- **What it is**: Maximum number of different trajectories to try
- **How it works**:
  - Controls the outer loop: `for traj_idx in range(self._max_num_trajectories)`
  - Each iteration = one complete trajectory from start to finish
  - Early exit if failure found before reaching limit
- **Purpose**: Limits search space, prevents infinite loops
- **Impact on failure finding**:
  - **Higher** = more exploration = more likely to find failures
  - **Lower** = faster but may miss failures
- **Example**: 
  - `max_num_trajectories=50`: Try up to 50 different trajectories
  - If failure found on trajectory 15, stop (don't try remaining 35)

### `_max_trajectory_length: int = 100`
- **What it is**: Maximum number of actions in one trajectory
- **How it works**:
  - Controls termination: `return len(traj.actions) >= self._max_trajectory_length`
  - Each trajectory runs until: failure found OR length limit reached
  - Shorter = each trajectory explores less deeply
- **Purpose**: Prevents infinitely long trajectories
- **Impact on failure finding**:
  - **Higher** = deeper exploration per trajectory = more likely to find failures within one trajectory
  - **Lower** = shallower exploration = need more trajectories to find failures
- **Example**:
  - `max_trajectory_length=100`: Each trajectory can have up to 100 actions
  - If failure occurs at step 50, trajectory stops (doesn't continue to 100)

### `_seed: int = 0`
- **What it is**: Initial random seed for the failure finder
- **How it works**:
  - Used to initialize `_rng`
  - Determines all random choices in failure finding
  - Same seed = same sequence of trajectories (reproducibility)
- **Purpose**: Reproducibility and debugging
- **Impact**: 
  - Fixed seed = deterministic behavior (good for testing)
  - Different seeds = different exploration patterns

### `_rng: np.random.Generator`
- **What it is**: Random number generator (RNG) for sampling
- **How it works**:
  - Created from `_seed`: `self._rng = np.random.default_rng(seed)`
  - Used to generate sub-seeds for commanders and controllers
  - Maintains state across trajectory samples
- **Purpose**: Centralized randomness control
- **Flow**:
  ```
  Failure Finder (_rng, seed=123)
      ↓ samples seed
  Initial State Commander (seed=45892)
      ↓ samples initial state
  Trajectory starts from sampled state
      ↓ samples seed
  Command Commander (seed=78234)
      ↓ samples commands
  Trajectory executes with sampled commands
  ```

---

## 17. FAILURE FINDER EXECUTION FLOW (`run` method)

### Outer Loop: Trajectory Sampling
```python
for traj_idx in range(self._max_num_trajectories):
```
- **What it is**: Tries multiple different trajectories
- **Iteration variable**: `traj_idx` = trajectory index (0, 1, 2, ...)
- **Termination**: 
  - Exhausts all trajectories: `traj_idx == _max_num_trajectories - 1`
  - Finds failure: `if failure_found: return failure_traj`
- **Result**: Each iteration explores a completely different scenario

### Step 1: Get Initial State Space
```python
initial_space = env.get_initial_states()
```
- **What it is**: Set of all possible initial states
- **For conveyor belt**: `EnumSpace([ConveyorBeltState(positions=[], falling_heights=[], step_count=0)])`
- **Purpose**: Defines where trajectories can start
- **Note**: Single initial state (empty belt), but other environments may have multiple

### Step 2: Create Initial State Commander
```python
initializer = self.get_initial_state(initial_space, env, controller, failure_monitor)
```
- **What it is**: Object that selects specific initial state
- **For RandomShootingFailureFinder**: Returns `RandomInitialStateCommander`
- **Process**:
  1. Sample new seed from `_rng`: `seed = sample_seed_from_rng(self._rng)`
  2. Create initializer: `RandomInitialStateCommander(initial_space)`
  3. Seed it: `initializer.seed(seed)`
- **Purpose**: Choose starting point for trajectory

### Step 3: Sample Initial State
```python
initial_state = initializer.initialize()
```
- **What it is**: Actual initial state for this trajectory
- **How it works**: `return self.initial_space.sample()`
- **For conveyor belt**: Always returns empty belt (deterministic)
- **For other envs**: May return different starting configurations

### Step 4: Create Initial Trajectory
```python
init_traj: Trajectory = Trajectory([initial_state], [], [])
```
- **What it is**: Trajectory with only initial observation, no actions/commands yet
- **Structure**:
  - `observations`: `[initial_state]` (1 state)
  - `actions`: `[]` (0 actions)
  - `commands`: `[]` (0 commands)
- **Invariant**: `len(observations) == len(actions) + 1 == len(commands) + 1`

### Step 5: Create Commander
```python
commander = self.get_commander(env, controller, failure_monitor, traj_idx)
```
- **What it is**: Object that selects commands at each step
- **For RandomShootingFailureFinder**: Returns `RandomCommander`
- **Process**:
  1. Sample new seed from `_rng`: `seed = sample_seed_from_rng(self._rng)`
  2. Get command space from controller: `command_space = controller.get_command_space()`
  3. Create commander: `RandomCommander(command_space)`
  4. Seed it: `commander.seed(seed)`
- **Purpose**: Guide controller decisions throughout trajectory
- **Command space for conveyor belt**:
  ```python
  EnumSpace([
      ConveyorBeltCommand(mode="off"),
      ConveyorBeltCommand(mode="slow"),
      ConveyorBeltCommand(mode="mid"),
      ConveyorBeltCommand(mode="fast"),
  ])
  ```

### Step 6: Define Termination Function
```python
def _termination_fn(traj: Trajectory) -> bool:
    return len(traj.actions) >= self._max_trajectory_length
```
- **What it is**: Function that checks if trajectory should stop
- **Input**: Current trajectory
- **Output**: `True` = stop, `False` = continue
- **Logic**: Stop when trajectory reaches maximum length
- **Note**: Does NOT check for failures (that's handled separately)

### Step 7: Extend Trajectory Until Failure
```python
failure_traj, failure_found = extend_trajectory_until_failure(
    init_traj, env, commander, controller, failure_monitor,
    _termination_fn, self._rng
)
```
- **What it is**: Core loop that executes one complete trajectory
- **Inputs**:
  - `init_traj`: Starting trajectory (1 state, 0 actions)
  - `env`: Environment for physics simulation
  - `commander`: Command selector
  - `controller`: Action generator
  - `failure_monitor`: Failure detector
  - `_termination_fn`: When to stop
  - `self._rng`: RNG for state transitions
- **Outputs**:
  - `failure_traj`: Complete trajectory (with or without failure)
  - `failure_found`: `True` if failure detected, `False` otherwise
- **Details**: See Section 18 below

### Step 8: Check for Failure
```python
if failure_found:
    print(f"Found a failure after {traj_idx+1} trajectory samples")
    return failure_traj
```
- **What it is**: Early exit if failure found
- **Effect**: Stops trying more trajectories, returns failure immediately
- **Example**: If failure found on trajectory 5, don't try trajectories 6-1000

### Step 9: No Failure Found
```python
print("Failure finding failed.")
return None
```
- **What it is**: Exhausted all trajectories without finding failure
- **Result**: `None` indicates no failure found within limits

---

## 18. TRAJECTORY EXTENSION (`extend_trajectory_until_failure`)

This is the core algorithm that executes a single trajectory step-by-step, checking for failures at each step.

### Input Parameters

#### `trajectory: Trajectory`
- **What it is**: Partially complete trajectory to extend
- **Initial**: `Trajectory([initial_state], [], [])` (1 state, 0 actions)
- **Structure**: Always maintains invariant `len(states) == len(actions) + 1 == len(commands) + 1`

#### `env: ConstraintBasedEnvModel`
- **What it is**: Environment that defines physics/transitions
- **Role**: Given `(state, action)`, produces `next_state`

#### `commander: Commander`
- **What it is**: Selects commands at each step
- **Role**: Given current trajectory state, produces `command`

#### `controller: Controller`
- **What it is**: Converts commands to actions
- **Role**: Given `(state, command)`, produces `action`

#### `failure_monitor: FailureMonitor`
- **What it is**: Detects failures in states
- **Role**: Given `(command, action, state)`, returns `True` if failure

#### `termination_fn: Callable[[Trajectory], bool]`
- **What it is**: Function that decides when to stop trajectory
- **Role**: Returns `True` when trajectory should terminate (length limit)

#### `rng: np.random.Generator`
- **What it is**: RNG for state transitions
- **Role**: Seeds environment's state sampling

### Phase 1: Setup and Fast-Forward

#### Step 1: Copy Trajectory Data
```python
states = list(trajectory.observations)
actions = list(trajectory.actions)
commands = list(trajectory.commands)
```
- **Purpose**: Create mutable copies for extension
- **Initially**: `states=[initial_state]`, `actions=[]`, `commands=[]`

#### Step 2: Verify Invariant
```python
assert len(states) == len(commands) + 1 == len(actions) + 1
```
- **Purpose**: Ensure trajectory structure is valid
- **Invariant**: Always one more state than actions/commands
- **Reason**: State before first action, state after each action

#### Step 3: Reset All Components
```python
failure_monitor.reset(states[0])
controller.reset(states[0])
commander.reset(states[0])
```
- **Purpose**: Initialize all components from initial state
- **For conveyor belt**:
  - `failure_monitor.reset()`: No-op (memoryless)
  - `controller.reset()`: Sets `_steps_since_last_drop = 10**9`
  - `commander.reset()`: No-op (stateless random sampling)

#### Step 4: Fast-Forward Through Existing Trajectory
```python
for t in range(len(actions)):
    recovered_command = commander.get_command()
    assert recovered_command == commands[t]  # Verify determinism
    
    recovered_action = controller.step(states[t], commands[t])
    assert env.actions_are_equal(recovered_action, actions[t])  # Verify determinism
    
    failure_found = failure_monitor.step(commands[t], actions[t], states[t + 1])
    assert not failure_found  # Should not have failure in prefix
    
    commander.update(actions[t], states[t + 1])
```
- **Purpose**: Replay existing trajectory to restore internal states
- **Why necessary**: Controller/commander may have internal state that must be synchronized
- **Verification**: Asserts that replay produces same commands/actions (determinism)
- **Initially**: Loop doesn't execute (no existing actions)

### Phase 2: Extension Loop

#### Step 1: Get Current State
```python
state = states[-1]
```
- **Purpose**: Start extension from last state in trajectory
- **Initially**: `state = initial_state`

#### Step 2: Main Extension Loop
```python
while not termination_fn(trajectory):
```
- **Condition**: Continue until `len(trajectory.actions) >= max_trajectory_length`
- **OR**: Until failure found (break inside loop)

#### Step 3: Sample Command
```python
command = commander.get_command()
```
- **What it does**: Commander selects next command
- **For RandomCommander**: `return self.command_space.sample()`
- **For conveyor belt**: Randomly samples from `["off", "slow", "mid", "fast"]`
- **Example**: `ConveyorBeltCommand(mode="fast")`

#### Step 4: Generate Action
```python
action = controller.step(state, command)
```
- **What it does**: Controller converts command to action
- **Process** (for conveyor belt):
  1. Increment timer: `_steps_since_last_drop += 1`
  2. Get timing requirement: `steps_required = _mode_to_steps[command.mode]`
  3. Check if timing satisfied: `_steps_since_last_drop >= steps_required`
  4. If yes, check safety: `drop = _safe_to_drop(state)`
  5. If no, set `drop = False`
  6. Return: `ConveyorBeltAction(drop_package=drop)`
- **Example**: `ConveyorBeltAction(drop_package=True)`

#### Step 5: Simulate Environment
```python
next_states = env.get_next_states(state, action)
next_states.seed(sample_seed_from_rng(rng))
state = next_states.sample()
```
- **What it does**: Physics simulation produces next state
- **Process**:
  1. Get possible next states: `next_states = env.get_next_states(state, action)`
  2. Seed the space: `next_states.seed(...)` (for deterministic sampling)
  3. Sample one state: `state = next_states.sample()`
- **For conveyor belt**: Deterministic (single next state)
- **Result**: `state` now contains new state after applying action

#### Step 6: Update Commander
```python
commander.update(action, state)
```
- **What it does**: Inform commander of result (for stateful commanders)
- **For RandomCommander**: No-op (stateless)
- **For other commanders**: May update internal models, plans, etc.

#### Step 7: Extend Trajectory
```python
actions.append(action)
states.append(state)
commands.append(command)
```
- **What it does**: Add new step to trajectory
- **Order matters**: Command → Action → State (sequence of events)
- **Invariant maintained**: `len(states) == len(actions) + 1`

#### Step 8: Check for Failure
```python
if failure_monitor.step(command, action, state):
    return Trajectory(states, actions, commands), True
```
- **What it does**: Check if new state is a failure
- **For conveyor belt**: Calls `_check_failures(state)`
  - Computes gaps between boxes
  - Checks if any gap < `min_spacing` or gap < `box_width`
  - Returns `True` if violation detected
- **If failure**: Immediately return trajectory with `failure_found=True`
- **If no failure**: Continue loop

### Phase 3: Termination

#### Normal Termination (No Failure)
```python
return Trajectory(states, actions, commands), False
```
- **When**: Loop exited because `termination_fn` returned `True`
- **Meaning**: Trajectory reached maximum length without finding failure
- **Result**: Return complete trajectory with `failure_found=False`

---

## 19. RANDOM COMMANDER VARIABLES (`RandomCommander`)

### `command_space: Space[CommandType]`
- **What it is**: Set of all possible commands
- **For conveyor belt**: 
  ```python
  EnumSpace([
      ConveyorBeltCommand(mode="off"),
      ConveyorBeltCommand(mode="slow"),
      ConveyorBeltCommand(mode="mid"),
      ConveyorBeltCommand(mode="fast"),
  ])
  ```
- **How it works**: Provided by controller via `get_command_space()`
- **Purpose**: Defines what commands are available to sample
- **Seeding**: `command_space.seed(seed)` for deterministic sampling

### Methods

#### `reset(initial_state: ObsType) -> None`
- **What it does**: Reset commander for new trajectory
- **Implementation**: Pass (no internal state)
- **When called**: At start of each trajectory

#### `get_command() -> CommandType`
- **What it does**: Sample random command
- **Implementation**: `return self.command_space.sample()`
- **Probability**: Uniform over all commands (each has 25% chance for conveyor belt)
- **Determinism**: Depends on `command_space` seed

#### `update(action: ActType, next_state: ObsType) -> None`
- **What it does**: Receive feedback about action result
- **Implementation**: Pass (doesn't use feedback)
- **Purpose**: Hook for stateful commanders (not used here)

#### `seed(seed: int) -> None`
- **What it does**: Seed the command sampling
- **Implementation**: `self.command_space.seed(seed)`
- **Effect**: Determines sequence of commands sampled

---

## 20. RANDOM INITIAL STATE COMMANDER (`RandomInitialStateCommander`)

### `initial_space: Space[ObsType]`
- **What it is**: Set of all possible initial states
- **For conveyor belt**: 
  ```python
  EnumSpace([ConveyorBeltState(positions=[], falling_heights=[], step_count=0)])
  ```
- **How it works**: Provided by environment via `get_initial_states()`
- **Purpose**: Defines where trajectories can start
- **Note**: Single state for conveyor belt (deterministic), but could be multiple

### Methods

#### `initialize() -> ObsType`
- **What it does**: Sample initial state
- **Implementation**: `return self.initial_space.sample()`
- **For conveyor belt**: Always returns empty belt (single option)
- **For other envs**: May return different starting configurations

#### `seed(seed: int) -> None`
- **What it does**: Seed the initial state sampling
- **Implementation**: `self.initial_space.seed(seed)`
- **Effect**: Determines which initial state is sampled

---

## 21. TRAJECTORY STRUCTURE (`Trajectory`)

### `observations: list[ObsType]`
- **What it is**: Sequence of states visited
- **Length**: `n + 1` where `n` = number of actions
- **Index**: `observations[t]` = state at time step `t`
- **Example**: `[state_0, state_1, state_2, ...]`
- **Interpretation**: 
  - `observations[0]`: Initial state (before any action)
  - `observations[t]`: State after executing `actions[t-1]`

### `actions: list[ActType]`
- **What it is**: Sequence of actions executed
- **Length**: `n` (one less than observations)
- **Index**: `actions[t]` = action taken at time step `t`
- **Example**: `[action_0, action_1, action_2, ...]`
- **Interpretation**: `actions[t]` executed in `observations[t]`, resulting in `observations[t+1]`

### `commands: list[CommandType]`
- **What it is**: Sequence of commands that generated actions
- **Length**: `n` (same as actions)
- **Index**: `commands[t]` = command that produced `actions[t]`
- **Example**: `[command_0, command_1, command_2, ...]`
- **Interpretation**: `commands[t]` → controller → `actions[t]`

### Invariants
```python
len(observations) == len(actions) + 1
len(observations) == len(commands) + 1
len(actions) == len(commands)
```

### Example Trajectory
```
Time  | Observation                    | Command         | Action
------+--------------------------------+-----------------+------------------
  0   | positions=[], heights=[]       |                 |
      |                                | mode="fast"     | drop_package=False
  1   | positions=[], heights=[]       |                 |
      |                                | mode="fast"     | drop_package=False
  2   | positions=[], heights=[]       |                 |
      |                                | mode="fast"     | drop_package=True
  3   | positions=[0.0], heights=[1.0] |                 |
      |                                | mode="fast"     | drop_package=False
  4   | positions=[0.0], heights=[0.9] |                 |
      |                                | mode="mid"      | drop_package=False
  5   | positions=[0.01], heights=[0.0]|                 |
```

---

## 22. KEY VARIABLE INTERACTIONS IN FAILURE FINDING

### Seed Flow
```
FailureFinder._seed
    ↓ initializes
FailureFinder._rng (Generator)
    ↓ samples seed via sample_seed_from_rng()
InitialStateCommander seed
    ↓ seeds initial_space
Initial State sampled
    ↓ samples seed via sample_seed_from_rng()
Commander seed
    ↓ seeds command_space
Commands sampled randomly
    ↓ for each step in trajectory
Controller uses command + state
    ↓ produces action (with internal RNG if needed)
Environment uses action + state
    ↓ seeded via sample_seed_from_rng(FailureFinder._rng)
Next State sampled
```

### Trajectory Length Control
```
max_trajectory_length = 100
    ↓ defines
_termination_fn(traj): len(traj.actions) >= 100
    ↓ checked in
extend_trajectory_until_failure loop
    ↓ stops when
len(trajectory.actions) == 100
    OR
failure_found == True
```

### Trajectory Count Control
```
max_num_trajectories = 300
    ↓ defines
for traj_idx in range(300)
    ↓ tries up to
300 different trajectories
    ↓ but exits early if
failure_found == True
    ↓ which happens when
failure_monitor.step() returns True
```

### Failure Detection Flow
```
Current State
    ↓
Failure Monitor
    ↓ calls
_check_failures(state)
    ↓ computes
gaps = np.diff(sorted_positions)
    ↓ checks
gaps < min_spacing OR gaps < box_width
    ↓ returns
True (FAILURE) or False (SAFE)
    ↓ if True
Trajectory returned immediately
```

---

## 23. WHY FAILURES ARE FOUND (OR NOT FOUND)

### Factors Increasing Failure Probability

1. **Longer Trajectories** (`max_trajectory_length ↑`):
   - More time for boxes to accumulate
   - More drop attempts = more chances for faulty conditions
   - More state transitions = more opportunities for violations
   - **Example**: Length 100 vs 20 → 5x more drops → 5x more chances

2. **More Trajectory Samples** (`max_num_trajectories ↑`):
   - Explores more different scenarios
   - Different command sequences → different box patterns
   - Different random outcomes → varied spacing
   - **Example**: 300 trajectories vs 50 → 6x more exploration

3. **Controller Faults** (intentional in this system):
   - Fault 1: Allows drops when boxes almost landed (height 0.0-0.05)
   - Fault 2: Ignores box width in spacing check
   - Fault 3: Uses 80% margin instead of 100%
   - **Effect**: Creates windows where unsafe drops are permitted

4. **Aggressive Command Modes**:
   - `mode="fast"`: 20 steps between drops (fast accumulation)
   - `mode="mid"`: 120 steps between drops
   - `mode="slow"`: 200 steps between drops
   - **Random commander**: 25% chance of "fast" each step

5. **Scene Parameters**:
   - Smaller `min_spacing`: Easier to violate (0.1 is small)
   - Larger `box_width`: Easier to overlap (0.4 is large)
   - Faster `conveyor_belt_velocity`: Boxes move more between drops
   - Slower `gravity`: Boxes fall slower, longer in air

### Factors Decreasing Failure Probability

1. **Shorter Trajectories** (`max_trajectory_length ↓`):
   - Fewer boxes accumulate
   - Less time for fault conditions to align
   - **Example**: Length 15 might only get 1-2 boxes on belt

2. **Fewer Trajectory Samples** (`max_num_trajectories ↓`):
   - Less exploration of state space
   - May miss rare failure conditions
   - **Example**: 1 trajectory = single scenario, might be safe

3. **Safety Checks in Controller**:
   - Even with faults, some checks prevent drops
   - Timing constraint: minimum steps between drops
   - Falling height check: no drops if boxes falling (even with fault threshold)
   - Spacing check: no drops if too close (even with faulty margin)

4. **Box Removal**:
   - Boxes fall off belt after `belt_length / velocity` seconds
   - Clears space, reduces collision opportunities
   - **Example**: At velocity=2.0, length=3.0 → boxes leave after 1.5 seconds

5. **Conservative Commands**:
   - `mode="off"`: Never drops (0% failure chance)
   - `mode="slow"`: Rare drops (low accumulation)
   - **Random commander**: 25% chance of "off" each step

---

## 24. RELATIONSHIP BETWEEN VARIABLES

### Trajectory Length vs. Number of Trajectories

**Trade-off**: Depth vs. Breadth of exploration

- **Long trajectories** (`max_trajectory_length=100`):
  - Each trajectory explores deeply (many steps)
  - Finds failures that develop over time
  - But explores fewer different scenarios (less breadth)
  - **Best for**: Failures that require accumulation/setup

- **Short trajectories** (`max_trajectory_length=20`):
  - Each trajectory explores shallowly (few steps)
  - Needs more trajectories to find failures
  - But explores more different scenarios (more breadth)
  - **Best for**: Failures from specific initial conditions/command sequences

- **Example**:
  ```
  Config A: 50 trajectories × 100 steps = 5,000 total steps
  Config B: 300 trajectories × 15 steps = 4,500 total steps
  
  Config A: Deeper per trajectory (better for accumulation failures)
  Config B: More scenarios (better for rare condition failures)
  ```

### Seed vs. Determinism

- **Fixed seed** (`seed=42`):
  - Same trajectories every run
  - First trajectory always same commands/states
  - If trajectory 1 finds failure with length=100, always finds it
  - If trajectory 1 doesn't find failure with length=20, never finds it (needs more trajectories)
  - **Pro**: Reproducible, good for debugging
  - **Con**: Limited exploration

- **Random seed** (`seed=random.randint(...)`):
  - Different trajectories each run
  - Explores different parts of state space
  - May find failure on trajectory 1 sometimes, trajectory 5 other times
  - **Pro**: Broader exploration over multiple runs
  - **Con**: Non-reproducible, hard to debug specific failures

---

## 25. COMPLETE EXECUTION EXAMPLE

### Setup
```python
scene_spec = ConveyorBeltSceneSpec(
    box_width=0.3,
    conveyor_belt_velocity=2.0,
    dt=0.01,
    belt_length=3.0,
)
object.__setattr__(scene_spec, "min_spacing", 0.1)

env = ConveyorBeltEnv(scene_spec=scene_spec)
controller = ConveyorBeltController(seed=42, scene_spec=scene_spec)
failure_monitor = ConveyorBeltFailureMonitor(scene_spec)
failure_finder = RandomShootingFailureFinder(
    seed=42, max_num_trajectories=50, max_trajectory_length=20
)
```

### Execution Trace

#### Trajectory 0:

**Initialization**:
- `traj_idx = 0`
- Sample seed from RNG: `seed_0 = 123456` (example)
- Create initial state commander with `seed_0`
- Sample initial state: `state_0 = ConveyorBeltState(positions=[], heights=[], step=0)`
- Create commander with new seed: `seed_1 = 789012`
- Commander seeded with `seed_1`

**Step 0**:
- `commander.get_command()` → `ConveyorBeltCommand(mode="mid")` (random sample)
- `controller.step(state_0, command)`:
  - `_steps_since_last_drop = 10^9 + 1`
  - Required: `120` steps
  - `10^9 >= 120`? Yes
  - `_safe_to_drop(state_0)`: No falling boxes, no positions → `True`
  - `drop = True`
  - `_steps_since_last_drop = 0` (reset)
  - Return: `ConveyorBeltAction(drop_package=True)`
- `env.get_next_states(state_0, action)`:
  - Falling heights: (none to update)
  - Positions: (none to move)
  - New box: `positions=[0.0]`, `heights=[1.0]`
  - Remove off-belt: (none)
  - Return: `ConveyorBeltState(positions=[0.0], heights=[1.0], step=1)`
- `failure_monitor.step(command, action, state_1)`:
  - Only 1 box → no failure
  - Return: `False`
- Continue...

**Step 1**:
- `commander.get_command()` → `ConveyorBeltCommand(mode="fast")`
- `controller.step(state_1, command)`:
  - `_steps_since_last_drop = 1`
  - Required: `20` steps
  - `1 >= 20`? No
  - `drop = False`
  - Return: `ConveyorBeltAction(drop_package=False)`
- `env.get_next_states(state_1, action)`:
  - Falling heights: `[1.0 - 9.81*0.01] = [0.9019]`
  - Positions: `[0.0]` (still falling, doesn't move)
  - No new box
  - Return: `ConveyorBeltState(positions=[0.0], heights=[0.9019], step=2)`
- `failure_monitor.step(...)`: 1 box → no failure
- Continue...

**Steps 2-19**: (box falls, lands, moves, maybe another box drops, etc.)

**Step 20**: Trajectory reaches `max_trajectory_length=20`
- `termination_fn(trajectory)` returns `True`
- Exit loop, return `(trajectory, False)`

**Failure finder**:
- `failure_found = False`
- Continue to trajectory 1...

#### Trajectory 1, 2, ... 49:
Similar process, each with different random commands/outcomes

#### Trajectory 15:
(Example where failure found)

**Step 8**:
- State has 3 boxes: `positions=[0.5, 0.62, 1.5]`, `heights=[0.0, 0.03, 0.0]`
- Commander selects: `mode="fast"`
- Controller:
  - `_steps_since_last_drop = 20` (just reached threshold)
  - Required: `20` steps → OK
  - `_safe_to_drop(state)`:
    - Falling heights: `[0.0, 0.03, 0.0]`
    - Check: `0.03 > 0.05`? **NO** (FAULT: should check > 0.0)
    - No heights > 0.05, continue
    - Nearest position: `0.5`
    - Check: `0.5 < 0.1 * 0.8 = 0.08`? No
    - Return: `True` (INCORRECTLY allows drop)
  - Action: `drop_package=True`
- Environment:
  - New box: `positions=[0.5, 0.62, 1.5, 0.0]`, `heights=[0.0, 0.0, 0.0, 1.0]`
  - (Box 2 lands during this step, so height→0.0)
- Failure monitor:
  - Sort positions: `[0.0, 0.5, 0.62, 1.5]`
  - Gaps: `[0.5, 0.12, 0.88]`
  - Check: `0.12 < 0.3 (box_width)`? **YES**
  - Check: `0.12 < 0.1 (min_spacing)`? **YES**
  - **FAILURE DETECTED**
  - Return: `True`
- **Failure finder**:
  - `failure_found = True`
  - Print: "Found a failure after 16 trajectory samples"
  - Return `failure_traj`

**Result**: Failure found, no more trajectories tried

---

## 26. SUMMARY: ALL VARIABLES IN FAILURE FINDER

| Component | Variable | Type | Role | Affects Failure Finding |
|-----------|----------|------|------|------------------------|
| **CommanderFailureFinder** | `_max_num_trajectories` | `int` | Limit on trajectory attempts | More = more exploration = higher chance |
| | `_max_trajectory_length` | `int` | Limit on steps per trajectory | More = deeper per traj = higher chance |
| | `_seed` | `int` | Initial RNG seed | Fixed = deterministic, Random = varied |
| | `_rng` | `Generator` | Random number generator | Source of all randomness |
| **RandomCommander** | `command_space` | `Space` | Available commands | Defines command distribution |
| **RandomInitialState** | `initial_space` | `Space` | Available initial states | Defines starting points |
| **Trajectory** | `observations` | `list[State]` | States visited | Grows each step, checked by monitor |
| | `actions` | `list[Action]` | Actions taken | Grows each step, one per command |
| | `commands` | `list[Command]` | Commands selected | Grows each step, guides controller |
| **Extension Loop** | `traj_idx` | `int` | Current trajectory number | Increments until limit or failure |
| | `state` | `State` | Current state in trajectory | Updated each step by environment |
| | `command` | `Command` | Current command | Sampled each step by commander |
| | `action` | `Action` | Current action | Generated each step by controller |
| | `failure_found` | `bool` | Whether failure detected | True = stop immediately and return |

---

This comprehensive analysis now covers every variable in the entire conveyor belt failure finding system, from scene configuration through environment physics, controller decisions, failure monitoring, and the complete failure finder architecture.

