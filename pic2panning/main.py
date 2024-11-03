"""Main module for pic2panning."""

from typing import cast

import tyro
from PIL import Image

from pic2panning import (  # type: ignore[attr-defined]
    VALID_PANNING,
    VALID_ZOOM,
    Opts,
    create_panning_video,
    create_zoom_video,
)


def main(opts: Opts) -> None:
    """Create a video from the given images."""
    frame_list: list[list[Image.Image]] = []
    for idx, img in enumerate(opts.images):
        if opts.movement[idx].split("-")[0] == "panning":
            current_video = create_panning_video(
                img,
                opts.time[idx],
                opts.ratio,
                opts.fps[idx],
                opts.add_reverse,
                cast(VALID_PANNING, opts.movement[idx].split("-")[1]),
                output_size=opts.output_size,
            )
        elif opts.movement[idx].split("-")[0] == "zoom":
            current_video = create_zoom_video(
                img,
                opts.time[idx],
                opts.fps[idx],
                opts.add_reverse,
                cast(VALID_ZOOM, opts.movement[idx].split("-")[1]),
                opts.output_size,
            )
        frame_list.append(current_video)
    video = opts.make_video(frame_list)
    video.write_videofile(
        opts.output_file,
        audio=opts.audio is not None,
        audio_codec="aac",
        codec="libx264",
        fps=30 if not opts.focus_center else None,
    )


def cli() -> None:
    """Command line interface for pic2vid."""
    opts = tyro.cli(Opts)
    main(opts)


if __name__ == "__main__":
    """Run the CLI."""
    cli()
