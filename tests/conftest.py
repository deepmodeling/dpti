import os
import sys

from _pytest.config import Config  # type: ignore

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def pytest_configure(config: Config) -> None:
    """Print paths information before any test collection starts."""
    print("\n" + "=" * 50)
    print(f"Running tests from: {os.path.abspath(__file__)}")
    print(f"Project root: {project_root}")
    print("=" * 50 + "\n")
