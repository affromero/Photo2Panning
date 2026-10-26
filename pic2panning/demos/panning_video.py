"""Reframe a video with a moving crop."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Run repeated video panning with the public rendering options."""
    return run_demo("render", ("--video", "--motion", "panning-lr"), argv)


if __name__ == "__main__":
    raise SystemExit(main())
