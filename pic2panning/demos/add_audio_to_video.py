"""Add audio with the render command's --audio and --audio-start options."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Render video with a soundtrack supplied through --audio."""
    return run_demo("render", ("--video",), argv)


if __name__ == "__main__":
    raise SystemExit(main())
