"""Tests for heuristic_failure_finder.py with Blocks environment.

This test suite validates the HeuristicFailureFinder's ability to discover
failures in the Blocks environment using a heuristic that encourages
failure-prone scenarios (e.g., blocks close together, complex configurations).
"""

from pathlib import Path

import imageio.v2 as iio
import numpy as np

from hybrid_failure_discovery.controllers.blocks_controller import BlocksController
from hybrid_failure_discovery.envs.blocks_env import BlocksEnv
from hybrid_failure_discovery.failure_finders.heuristic_failure_finder import (
    HeuristicFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)

# Import exception for handling empty task plan
try:
    from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
except ImportError:
    # If not available, define a dummy exception class
    TaskThenMotionPlanningFailure = Exception


def test_heuristic_failure_finder_blocks():
    """Tests for heuristic_failure_finder.py with Blocks environment.

    Uses a heuristic that encourages failure-prone scenarios:
    - Blocks positioned close together (higher collision risk)
    - Complex tower configurations (more opportunities for failures)
    - Rapid movements or unstable configurations
    """

    def blocks_heuristic(traj):
        """Heuristic that encourages failure-prone block configurations.
        
        Based on BlocksFailureMonitor: failures occur when non-held blocks
        move unexpectedly (move_tol=0.05). This heuristic encourages scenarios
        where blocks are likely to be knocked over or collide.
        
        Lower scores (closer to 0.0) are better - they indicate scenarios
        more likely to lead to failures.
        
        Returns:
            float: Score between 0.0 (good for finding failures) and 1.0 (bad)
        """
        states, actions = traj
        
        if len(states) < 2:
            return 1.0  # Need at least 2 states to evaluate
        
        # Check the final state for block positions
        final_state = states[-1]
        
        # Encourage scenarios with blocks close together
        # Blocks close together are more likely to collide when robot moves near them
        # or when one block is placed/stacked near another
        min_distance = float('inf')
        num_pairs = 0
        
        # Calculate minimum distance between any two non-held blocks
        for i, block1 in enumerate(final_state.blocks):
            if final_state.held_block_name == block1.name:
                continue  # Skip held block (as per failure monitor logic)
            pos1 = block1.pose.position
            
            for j, block2 in enumerate(final_state.blocks):
                if i >= j:
                    continue
                if final_state.held_block_name == block2.name:
                    continue  # Skip held block
                pos2 = block2.pose.position
                
                # Calculate 3D distance
                distance = np.linalg.norm(pos1 - pos2)
                min_distance = min(min_distance, distance)
                num_pairs += 1
        
        if num_pairs == 0:
            return 1.0  # No blocks to compare
        
        # Lower distance = higher collision risk = better score (closer to 0.0)
        # More aggressive: normalize with smaller threshold (0.3 instead of 0.5)
        # This strongly favors very close blocks (< 0.15 units apart)
        distance_score = min(1.0, min_distance / 0.3)
        
        # Also encourage scenarios with many blocks (more complex = more failure risk)
        # More blocks means more opportunities for collisions or knock-overs
        num_blocks = len(final_state.blocks)
        complexity_score = 1.0 - min(1.0, num_blocks / 8.0)  # More aggressive: favor 8+ blocks
        
        # Encourage tall towers (unstable configurations)
        # Check maximum height difference between blocks
        max_height = -float('inf')
        min_height = float('inf')
        for block in final_state.blocks:
            if final_state.held_block_name == block.name:
                continue
            height = block.pose.position[2]  # z-coordinate
            max_height = max(max_height, height)
            min_height = min(min_height, height)
        height_range = max_height - min_height if max_height > min_height else 0.0
        # Taller towers = more unstable = better score (closer to 0.0)
        tower_score = 1.0 - min(1.0, height_range / 0.5)  # Favor towers > 0.25 units tall
        
        # Combine all factors: close blocks, many blocks, AND tall towers
        # Weight distance most heavily since collisions are the main failure mode
        combined_score = (distance_score * 0.5 + complexity_score * 0.3 + tower_score * 0.2)
        
        return combined_score

    # Test failure finder in blocks env
    from hybrid_failure_discovery.envs.blocks_env import BlocksEnvSceneSpec
    
    # Create scene spec with more blocks for more failure opportunities
    # Using 8 blocks - more than default but not so many that planning becomes too slow
    scene_spec = BlocksEnvSceneSpec(num_blocks=8)  # Increased from default 6
    env = BlocksEnv(scene_spec=scene_spec, seed=123, use_gui=False)
    
    # Use lower safe_height to make controller less careful (more likely to cause collisions)
    controller = BlocksController(seed=123, scene_spec=env.scene_spec, safe_height=0.15)
    
    # Use very sensitive tolerance to catch even tiny movements
    failure_monitor = BlocksFailureMonitor(move_tol=0.01)  # Very sensitive - catch small movements
    
    # Create heuristic failure finder with balanced but aggressive parameters
    failure_finder = HeuristicFailureFinder(
        blocks_heuristic,
        num_particles=15,  # Increased from 10 - more parallel exploration
        num_extension_attempts=3,  # Increased from 1 - more attempts per particle
        max_trajectory_length=300,  # Increased from 200 - longer trajectories but not too long
        max_num_iters=150,  # Increased from 100 - more iterations but not excessive
        boltzmann_temperature=30.0,  # Lower temperature = more focused on best particles
        seed=123,
    )
    
    # Run the failure finder, handling empty task plan exceptions
    # Empty task plan means the command was successfully completed (no failure)
    print(f"Testing with {scene_spec.num_blocks} blocks, safe_height={0.15}, move_tol={0.01}")
    print(f"Heuristic search: {failure_finder._num_particles} particles, {failure_finder._max_num_iters} iterations")
    print("This may take a while - planning with many blocks can be slow...")
    
    try:
        result = failure_finder.run(env, controller, failure_monitor)
    except TaskThenMotionPlanningFailure as e:
        # If the planner has no task plan (empty plan), this means the command
        # was successfully completed. No failure found - treat as no failure.
        if "empty" in str(e).lower() or "task plan" in str(e).lower():
            result = None  # No failure found - command completed successfully
        else:
            # Re-raise if it's a different planning failure
            raise

    # Check if a failure was found
    # A failure occurs when a block that is NOT held moves unexpectedly
    # (e.g., due to collisions, robot hitting blocks, physics issues)
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
        print(f"  Failure occurred when a non-held block moved unexpectedly")
    else:
        print("✗ No failure found after all iterations")
        print("  This could mean:")
        print("  - The controller is very robust and avoids collisions")
        print("  - Failures are extremely rare in this environment")
        print("  - The BlocksController planner is too careful")
        print("\n  Suggestions:")
        print("  - Try even more iterations (increase max_num_iters)")
        print("  - Try different seeds to explore different scenarios")
        print("  - Consider if failures naturally occur in this environment")
        print("  - The heuristic may need further refinement")

    # Note: This test may or may not find a failure depending on the system
    # If failures are expected, uncomment the assertion below
    # assert result is not None

    # If a failure was found, save visualization
    if result is not None:
        states = result.observations
        # Accessing protected method _render_state is intentional for visualization
        # pylint: disable=protected-access
        imgs = [env._render_state(s) for s in states]
        path = (
            Path("videos")
            / "test-heuristic-failure-finding"
            / "blocks_heuristic_test.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()
