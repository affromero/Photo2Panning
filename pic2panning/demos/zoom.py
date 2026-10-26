"""Zoom a photo with explicit input and output paths."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Run photo zooming with the public rendering options."""
    return run_demo("render", ("--motion", "zoom-in"), argv)


if __name__ == "__main__":
    raise SystemExit(main())
