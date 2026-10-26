"""Share command dispatch between the runnable examples."""

import sys
from collections.abc import Sequence

from pic2panning.cli import main as cli_main


def run_demo(
    command: str,
    defaults: tuple[str, ...],
    argv: Sequence[str] | None = None,
) -> int:
    """Run the public CLI with example defaults and explicit user options.

    Args:
        command: Public command to run.
        defaults: Options placed before the user arguments.
        argv: Arguments to parse, or the current process arguments.

    Returns:
        The command exit status.

    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    return cli_main([command, *defaults, *arguments])
