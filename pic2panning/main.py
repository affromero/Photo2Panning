"""Main module for pic2panning."""

from typing import cast

import tyro
from PIL import Image

from pic2panning.utils import (
    VALID_PANNING,
    VALID_ZOOM,
    AudioOpts,
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
            )
        elif opts.movement[idx].split("-")[0] == "zoom":
            current_video = create_zoom_video(
                img,
                opts.time[idx],
                opts.fps[idx],
                opts.add_reverse,
                cast(VALID_ZOOM, opts.movement[idx].split("-")[1]),
            )
        frame_list.append(current_video)
    if opts.output_format == "gif":
        opts.make_gif(frame_list)
    else:
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


def demo_panning_lr() -> None:
    """Demo function for panning left to right."""
    opts = Opts(
        images=["IMG_0581.jpeg", "IMG_0485.jpeg"],
        output_file="output.mp4",
        time=[5],
        ratio="16:9",
        audio=AudioOpts("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        output_format="video",
        output_size=(1080, 1920),
        mail_to=None,
        fps=[240],
        movement=["panning-lr"],
        add_reverse=False,
        focus_center=False,
    )
    main(opts)


def demo_zoom_in() -> None:
    """Demo function for zoom-in."""
    opts = Opts(
        images=["IMG_0581.jpeg"],
        output_file="output.mp4",
        time=[5],
        ratio="16:9",
        audio=AudioOpts("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        output_format="video",
        # output_size=(1080, 1920),
        output_size=None,
        mail_to=None,
        fps=[30],
        movement=["zoom-out"],
        add_reverse=False,
        focus_center=False,
    )
    main(opts)


if __name__ == "__main__":
    """Run the CLI."""
    # cli()
    # demo_panning_lr()
    demo_zoom_in()
