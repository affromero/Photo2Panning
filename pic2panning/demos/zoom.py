"""Demo for zoom-in."""

from typing import Literal, cast

import tyro

from pic2panning import (  # type: ignore[attr-defined]
    VALID_MOVEMENT,
    AudioOpts,
    Opts,
    main,
)


def demo_zoom(
    movement: Literal["in", "out"] = "in",
    /,
    add_reverse: bool = False,
    focus_center: bool = False,
) -> None:
    """Demo function for zoom-in."""
    opts = Opts(
        images=[
            "https://images.pexels.com/photos/2113566/pexels-photo-2113566.jpeg"
        ],
        output_file="assets/demo_zoom.mp4",
        time=[5],
        ratio="16:9",
        audio=[AudioOpts("https://www.youtube.com/watch?v=dQw4w9WgXcQ")],
        # output_size=(1080, 1920),
        output_size=None,
        fps=[30],
        movement=[cast(VALID_MOVEMENT, f"zoom-{movement}")],
        add_reverse=add_reverse,
        focus_center=focus_center,
    )
    main(opts)


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_zoom)
