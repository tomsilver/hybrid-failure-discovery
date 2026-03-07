"""Shared pytest configuration and fixtures."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

from gym_failure_discovery.utils import RecordBufferedVideo


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI flags."""
    parser.addoption(
        "--make-videos",
        action="store_true",
        default=False,
        help="Save videos for tests marked with @pytest.mark.make_videos.",
    )
    parser.addoption(
        "--run-llms",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.run_llms (skipped by default).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "make_videos: save a video when --make-videos is passed.",
    )
    config.addinivalue_line(
        "markers",
        "run_llms: test requires a real LLM call (skipped unless --run-llms).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip run_llms tests unless --run-llms is passed."""
    if config.getoption("--run-llms"):
        return
    skip = pytest.mark.skip(reason="needs --run-llms to run")
    for item in items:
        if "run_llms" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def maybe_record(
    request: pytest.FixtureRequest,
) -> Any:
    """Wrap an env with RecordVideo if --make-videos is passed.

    Usage in tests::

        @pytest.mark.make_videos
        def test_something(maybe_record):
            env = maybe_record(MyEnv())
            ...
    """
    make_videos = request.config.getoption("--make-videos")

    def _wrap(env: gym.Env) -> gym.Env:  # type: ignore[type-arg]
        if not make_videos:
            return env
        env.render_mode = "rgb_array"
        videos_dir = Path(__file__).parent.parent / "videos"
        videos_dir.mkdir(exist_ok=True)
        test_name = request.node.name
        return RecordBufferedVideo(
            env,
            video_folder=str(videos_dir),
            name_prefix=test_name,
            episode_trigger=lambda _: True,
        )

    return _wrap
