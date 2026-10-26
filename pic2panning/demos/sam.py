"""Create a text-guided mask through the public mask command."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Run text-guided masks through the public segmentation command."""
    return run_demo("mask", (), argv)


if __name__ == "__main__":
    raise SystemExit(main())
