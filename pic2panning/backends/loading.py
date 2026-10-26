"""Import optional frameworks without loading them for ordinary jobs."""

from importlib import import_module
from types import ModuleType

from pic2panning.errors import DependencyError


def load_dependency(module: str) -> ModuleType:
    """Load a framework and explain how to install missing dependencies."""
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        message = (
            f"The model backend needs {error.name!r}. "
            "Run `uv sync --extra models` in Photo2Panning on this Mac."
        )
        raise DependencyError(message) from error
