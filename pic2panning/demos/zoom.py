"""Demo for zoom-in."""

from typing import Literal, cast

import tyro

from pic2panning.main import main
from pic2panning.utils.options import VALID_MOVEMENT, AudioOpts, Opts


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
        images=images,
        output_file="assets/demo_zoom.mp4",
        time=[2],
        ratio="16:9",
        audio=[AudioOpts(song)],
        output_size=(6240 // 2, 4160 // 2),
        # output_size=None,
        fps=[30],
        movement=[cast(VALID_MOVEMENT, f"zoom-{movement}")],
        add_reverse=add_reverse,
        focus_center=focus_center,
    )
    main(opts)


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_zoom)
