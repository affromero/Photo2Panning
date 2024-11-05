"""Demo for panning left to right."""

from typing import Literal, cast

import tyro

from pic2panning.main import main_images as main
from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import VALID_IMAGE_PROCESS, AudioOpts, Opts

logger = get_logger()


def demo_panning(
    movement: Literal["lr", "rl"] = "lr",
    /,
    add_reverse: bool = False,
    focus_center: bool = False,
    images: list[str] = ["DSC09552.JPG"],
    song: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_file: str = "assets/demo_panning.mp4",
) -> None:
    """Demo function for panning left to right."""
    opts = Opts(
        data=images,
        output_file=output_file,
        time=[4],
        ratio="16:9",
        audio=[AudioOpts(audio_file=song)],
        output_size=(1080, 1920),
        fps=[240],
        process=[cast(VALID_IMAGE_PROCESS, f"panning-{movement}")],
        add_reverse=add_reverse,
        focus_center=focus_center,
    )
    main(opts)


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_panning)
