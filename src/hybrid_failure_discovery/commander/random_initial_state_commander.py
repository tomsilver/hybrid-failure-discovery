"""A commander that randomly samples from a command space."""

from gymnasium.core import ObsType
from gymnasium.spaces import Space

from hybrid_failure_discovery.commander.initial_state_commander import (
    InitialStateCommander,
)


class RandomInitialStateCommander(InitialStateCommander[ObsType]):
    """A commander that randomly samples from a command space."""

    def __init__(self, initial_space: Space[ObsType]):
        self.initial_space = initial_space

    def initialize(self) -> ObsType:
        return self.initial_space.sample()

    def seed(self, seed: int) -> None:
        """Seed the sampling."""
        self.initial_space.seed(seed)
