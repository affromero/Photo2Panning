"""Build the public CLI without loading media or model resources."""

import argparse
from collections.abc import Sequence
from typing import Never

from pic2panning.schemas.editing import Axis
from pic2panning.schemas.media import Motion
from pic2panning.schemas.models import Device, GenerationModel, Quantization


class Arguments(argparse.Namespace):
    """Declare the shape of parsed command-line options."""

    command: str
    sources: list[str]
    source: str | None
    output: str
    motion: Motion | None
    duration: float | None
    size: list[int]
    fps: int
    reverse: bool
    focus_center: bool
    audio: str | None
    audio_start: float
    loop_audio: bool
    cover: str | None
    speed: float
    pan_speed: float | None
    zoom: float
    video: bool
    overwrite: bool
    resize: list[int] | None
    crop: list[int] | None
    rotate: float | None
    brightness: float
    contrast: float
    saturation: float
    flip: Axis | None
    prompt: str
    model: GenerationModel
    weights: str | None
    quantization: int
    seed: int
    steps: int | None
    mask: str | None
    strength: float | None
    guidance: float | None
    threshold: float
    device: Device
    separate: bool
    job: str
    media: str


class ArgumentParser(argparse.ArgumentParser):
    """Let the outer CLI encode errors consistently as JSON."""

    def error(self, message: str) -> Never:
        """Raise a user-facing parse error without writing partial output."""
        raise ValueError(message)


def add_output(parser: argparse.ArgumentParser) -> None:
    """Add the common explicit destination and overwrite flag."""
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")


def add_render(parser: argparse.ArgumentParser) -> None:
    """Add timeline and audio options."""
    parser.add_argument("sources", nargs="+")
    add_output(parser)
    parser.add_argument("--motion", type=Motion, choices=list(Motion))
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[1080, 1920],
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--focus-center", action="store_true")
    parser.add_argument("--audio")
    parser.add_argument("--audio-start", type=float, default=0)
    parser.add_argument("--loop-audio", action="store_true")
    parser.add_argument("--cover")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--pan-speed", type=float)
    parser.add_argument("--zoom", type=float, default=4 / 3)
    parser.add_argument(
        "--video", action="store_true", help="Treat all inputs as videos"
    )


def add_edit(parser: argparse.ArgumentParser) -> None:
    """Add deterministic image transforms in their documented CLI order."""
    parser.add_argument("source")
    add_output(parser)
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM")
    )
    parser.add_argument(
        "--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument("--rotate", type=float)
    parser.add_argument("--flip", type=Axis, choices=list(Axis))
    parser.add_argument("--brightness", type=float, default=1)
    parser.add_argument("--contrast", type=float, default=1)
    parser.add_argument("--saturation", type=float, default=1)


def add_generate(parser: argparse.ArgumentParser) -> None:
    """Add quantized local model parameters."""
    add_output(parser)
    parser.add_argument("--source")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--model",
        type=GenerationModel,
        choices=list(GenerationModel),
        default=GenerationModel.FLUX3,
    )
    parser.add_argument("--weights")
    parser.add_argument(
        "--quantization",
        type=int,
        choices=[int(q) for q in Quantization],
        default=4,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--mask")
    parser.add_argument("--strength", type=float)
    parser.add_argument("--guidance", type=float)


def add_mask(parser: argparse.ArgumentParser) -> None:
    """Add text-guided mask controls."""
    parser.add_argument("source")
    add_output(parser)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--device", type=Device, choices=list(Device), default=Device.CPU
    )
    parser.add_argument("--separate", action="store_true")


def parse_arguments(argv: Sequence[str] | None) -> Arguments:
    """Parse one subcommand with a stable typed namespace."""
    parser = ArgumentParser(
        prog="pic2panning",
        description="Photo motion and local image editing on Apple Silicon",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_render(
        subparsers.add_parser(
            "render", help="Animate photos or process videos"
        )
    )
    add_edit(
        subparsers.add_parser(
            "edit", help="Crop, resize, rotate, or adjust an image"
        )
    )
    add_generate(
        subparsers.add_parser(
            "generate", help="Generate or edit with quantized MLX models"
        )
    )
    add_mask(
        subparsers.add_parser("mask", help="Find and segment objects by text")
    )
    for command in ("run", "validate"):
        subparsers.add_parser(
            command, help=f"{command.capitalize()} a JSON job"
        ).add_argument("job")
    subparsers.add_parser(
        "schema", help="Write the current JSON Schema to stdout"
    )
    subparsers.add_parser(
        "inspect", help="Probe a local media file"
    ).add_argument("media")
    return parser.parse_args(argv, namespace=Arguments())
