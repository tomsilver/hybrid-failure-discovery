"""A commander that randomly samples from a command space."""

from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.structs import CommandType


class RandomCommander(Commander[ObsType, ActType, CommandType]):
    """A commander that randomly samples from a command space."""

    def __init__(self, command_space: Space[CommandType]):
        self.command_space = command_space

    def reset(self, initial_state: ObsType) -> None:
        pass

    def get_command(self) -> CommandType:
        return self.command_space.sample()

    def update(self, action: ActType, next_state: ObsType) -> None:
        pass

    def seed(self, seed: int) -> None:
        """Seed the sampling."""
        self.command_space.seed(seed)
