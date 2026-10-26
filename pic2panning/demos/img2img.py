"""Edit a reference photo with --source, --prompt, and quantized FLUX 3."""

from collections.abc import Sequence

from pic2panning.demos.common import run_demo


def main(argv: Sequence[str] | None = None) -> int:
    """Run reference-image editing through the public generation command."""
    return run_demo("generate", ("--model", "flux3"), argv)


if __name__ == "__main__":
    raise SystemExit(main())
