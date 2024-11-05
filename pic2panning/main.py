"""Main module for pic2panning."""

import tempfile
from pathlib import Path
from typing import cast

import moviepy.editor as mpe
import tyro
from PIL import Image

from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import (
    VALID_IMAGE_PROCESS,
    VALID_PANNING,
    VALID_ZOOM,
    Opts,
)
from pic2panning.utils.process import (
    add_audio_to_video,
    add_image_to_video,
    create_panning_video,
    create_zoom_video,
    speed_up_video,
)

logger = get_logger()


def main_images(opts: Opts) -> None:
    """Create a video from the given images."""
    frame_list: list[list[Image.Image]] = []
    for idx, img in enumerate(opts.data):
        if opts.process[idx].split("-")[0] == "panning":
            current_video = create_panning_video(
                img,
                opts.time[idx],
                opts.ratio,
                opts.fps[idx],
                opts.add_reverse,
                cast(VALID_PANNING, opts.process[idx].split("-")[1]),
                output_size=opts.output_size,
            )
        elif opts.process[idx].split("-")[0] == "zoom":
            current_video = create_zoom_video(
                img,
                opts.time[idx],
                opts.fps[idx],
                opts.add_reverse,
                cast(VALID_ZOOM, opts.process[idx].split("-")[1]),
                opts.output_size,
            )
        else:
            raise ValueError(f"Invalid process: {opts.process[idx]}")
        frame_list.append(current_video)
    video = opts.make_video(frame_list)
    video.write_videofile(
        opts.output_file,
        audio=opts.audio is not None,
        audio_codec="aac",
        codec="libx264",
        fps=30 if not opts.focus_center else None,
    )


def main_videos(opts: Opts) -> None:
    """Create a video from the given images."""
    all_videos = []
    for idx, video in enumerate(opts.data):
        if opts.add_image is not None:
            temp_file_video = tempfile.NamedTemporaryFile(
                delete=True, suffix=".mp4"
            ).name
            temp_file_image = tempfile.NamedTemporaryFile(
                delete=True, suffix=".png"
            ).name
            cover_image = opts.add_image.create_text_image()
            cover_image.save(temp_file_image)
        if opts.process[idx] == "add-audio-to-video":
            if opts.audio is None:
                raise ValueError(
                    "Audio file is required for add-audio-to-video process."
                )
            if opts.add_image is not None:
                video = add_image_to_video(
                    video,
                    temp_file_image,
                    opts.add_image.time,
                    temp_file_video,
                    insert_at=opts.add_image.insert_at,
                )
            video = add_audio_to_video(
                video, opts.audio[idx], opts.output_file
            )
        elif opts.process[idx].split("-")[0] == "speed":
            speed = (
                opts.speed
                if opts.process[idx].split("-")[1] == "up"
                else 1 / opts.speed
            )
            video = speed_up_video(
                video, speed, opts.output_file, opts.output_size
            )
            if opts.add_image is not None:
                video_out = str(
                    Path(video).parent
                    / Path(
                        str(Path(video).stem) + "_cover" + Path(video).suffix
                    )
                )
                video = add_image_to_video(
                    video,
                    temp_file_image,
                    opts.add_image.time,
                    video_out,
                    insert_at=opts.add_image.insert_at,
                )
        else:
            raise ValueError(f"Invalid process: {opts.process[idx]}")

        all_videos.append(video)

    if len(all_videos) > 1:
        first_size = mpe.VideoFileClip(all_videos[0]).size
        logger.info("Resizing all videos to the size of the first video.")
        all_videos = [
            mpe.VideoFileClip(video).resize(first_size) for video in all_videos
        ]
        video_cat = mpe.concatenate_videoclips(all_videos, method="chain")
        video_cat.write_videofile(
            opts.output_file,
            codec="libx264",
            audio_codec="aac",
            fps=30 if not opts.focus_center else None,
        )


def cli() -> None:
    """Command line interface for pic2vid."""
    opts = tyro.cli(Opts)
    if isinstance(VALID_IMAGE_PROCESS, opts.process[0]):
        main_images(opts)
    else:
        main_videos(opts)


if __name__ == "__main__":
    """Run the CLI."""
    cli()
