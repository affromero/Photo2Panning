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
    create_panning_video_from_image,
    create_panning_video_from_video,
    create_zoom_video,
    read_video,
    save_video,
    speed_up_video,
)

logger = get_logger()


def get_temp_video() -> str:
    """Get a temporary video file."""
    return tempfile.NamedTemporaryFile(delete=True, suffix=".mp4").name


def get_temp_image() -> str:
    """Get a temporary image file."""
    return tempfile.NamedTemporaryFile(delete=True, suffix=".png").name


def main_images(opts: Opts) -> None:
    """Create a video from the given images."""
    frame_list: list[list[Image.Image]] = []
    for idx, img in enumerate(opts.data):
        if opts.process[idx].split("-")[0] == "panning":
            current_video = create_panning_video_from_image(
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
    save_video(
        video,
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
        output_file_temp = get_temp_video()
        if opts.add_image is not None:
            temp_file_image = get_temp_image()
            cover_image = opts.add_image.create_text_image()
            cover_image.save(temp_file_image)
            if opts.process[idx] == "speed":
                video = add_image_to_video(
                    video,
                    temp_file_image,
                    opts.add_image.time,
                    get_temp_video(),
                    insert_at=opts.add_image.insert_at,
                )
        if opts.process[idx] == "add-audio-to-video":
            if opts.audio is None:
                raise ValueError(
                    "Audio file is required for add-audio-to-video process."
                )
            video = add_audio_to_video(
                video, opts.audio[idx], output_file_temp
            )
        elif opts.process[idx] == "speed":
            speed = opts.speed
            video = speed_up_video(
                video, speed, output_file_temp, opts.output_size
            )
        elif opts.process[idx].split("-")[0] == "panning":
            list_images = create_panning_video_from_video(
                video,
                int(opts.speed),
                opts.ratio,
                output_size=opts.output_size,
            )
            _video = opts.make_video([list_images])
            video_size = read_video(video).size
            ffmpeg_params = ["-aspect", f"{video_size[1]}:{video_size[0]}"]
            save_video(
                _video,
                output_file_temp,
                audio=opts.audio is not None,
                audio_codec="aac",
                codec="libx264",
                fps=30 if not opts.focus_center else None,
                ffmpeg_params=ffmpeg_params,
            )
            # save again to fix aspect ratio
            # save_video(read_video(output_file_temp), output_file_temp)
            video = output_file_temp
        else:
            raise ValueError(f"Invalid process: {opts.process[idx]}")

        if opts.process[idx] != "speed" and opts.add_image is not None:
            video = add_image_to_video(
                video,
                temp_file_image,
                opts.add_image.time,
                get_temp_video(),
                insert_at=opts.add_image.insert_at,
            )

        all_videos.append(video)

    if len(all_videos) > 1:
        _all_videos = [read_video(i) for i in all_videos]
        video_cat = mpe.concatenate_videoclips(_all_videos, method="compose")
        # video_cat.write_videofile(
        #     opts.output_file,
        #     codec="libx264",
        #     audio_codec="aac",
        #     fps=30 if not opts.focus_center else None,
        # )
        save_video(
            video_cat,
            opts.output_file,
            codec="libx264",
            audio_codec="aac",
            fps=30 if not opts.focus_center else None,
        )
    else:
        logger.info(
            f"Renaming the only video to the output file to {opts.output_file}"
        )
        Path(all_videos[0]).rename(opts.output_file)


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
