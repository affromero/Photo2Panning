"""Demo for zoom-in."""

from typing import Literal, cast

import tyro

from pic2panning.main import main_images as main
from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import VALID_IMAGE_PROCESS, AudioOpts, Opts

logger = get_logger()


def demo_zoom(
    movement: Literal["in", "out"] = "in",
    /,
    add_reverse: bool = False,
    focus_center: bool = False,
    images: list[str] = [
        "https://images.pexels.com/photos/2113566/pexels-photo-2113566.jpeg"
    ],
    song: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
) -> None:
    """Demo function for zoom-in."""
    opts = Opts(
        data=images,
        output_file="assets/demo_zoom.mp4",
        time=[2],
        ratio="16:9",
        audio=[AudioOpts(audio_file=song)],
        output_size=None,
        fps=[240],
        process=[cast(VALID_IMAGE_PROCESS, f"zoom-{movement}")],
        add_reverse=add_reverse,
        focus_center=focus_center,
    )
    main(opts)


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_zoom)
