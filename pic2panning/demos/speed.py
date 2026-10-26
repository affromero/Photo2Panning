"""Change video playback speed while retaining its sound."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Render video at twice its speed, with an optional speed override."""
    return run_demo("render", ("--video", "--speed", "2"), argv)


if __name__ == "__main__":
    raise SystemExit(main())
