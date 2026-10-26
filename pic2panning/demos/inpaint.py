"""Fill a masked region with --source, --mask, --prompt, and FLUX 1 Fill."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Run masked filling through the public generation command."""
    return run_demo("generate", ("--model", "flux1-fill"), argv)


if __name__ == "__main__":
    raise SystemExit(main())
