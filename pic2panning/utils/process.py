"""Utility functions for creating videos and GIFs from images."""

import tempfile
from io import BytesIO
from typing import Literal

import moviepy.editor as mpe
import numpy as np
import requests
from PIL import Image, ImageFilter

from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import VALID_PANNING, VALID_ZOOM, AudioOpts

logger = get_logger()


def parse_aspect_ratio(ratio: str) -> tuple[int, int]:
    """Parse the aspect ratio string into a width and height."""
    width, height = map(int, ratio.split(":"))
    return width, height


def read_image(image_path: str) -> Image.Image:
    """Read an image from a file or from a URL."""
    if "http" in image_path:
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    logger.info(f"Reading image from {image_path} with size {img.size}")
    return img.convert("RGB")


def read_video(video_path: str) -> mpe.VideoFileClip:
    """Read a video from a file."""
    if "http" in video_path:
        response = requests.get(video_path)
        video_np = np.frombuffer(response.content, np.uint8)
        # save the video to a file
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4")
        with open(temp_file.name, "wb") as f:
            f.write(video_np)
        video = mpe.VideoFileClip(temp_file.name)
    else:
        video = mpe.VideoFileClip(video_path)
    logger.info(f"Reading video from {video_path} with size {video.size}")
    return video


def create_panning_video(
    image_path: str,
    duration: int,
    aspect_ratio: str,
    fps: int = 30,
    add_reverse: bool = False,
    movement: VALID_PANNING = "lr",
    output_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Create a panning video effect of an image from left to.

    right with a given aspect ratio.

    Args:
        image_path (str): Path of the input image.
        duration (int): Length of the video in seconds.
        aspect_ratio (str): Aspect ratio of the video in format 'height:width',
             e.g., '16:9'.
        fps (int): Frames per second of the video (default is 30).
        add_reverse (bool): Add reverse video (default is False).
        movement (str): Movement direction of the panning effect
            (default is 'lr').
        output_size (tuple[int, int]): Output video size
            in format (width, height).

    """
    current_video = []
    with read_image(image_path) as img:
        if output_size is not None:
            if movement in ["lr", "rl"]:
                # resize image keeping ratio to the output_size height
                img = img.resize(
                    (
                        int(img.width * output_size[1] / img.height),
                        output_size[1],
                    ),
                    Image.Resampling.LANCZOS,
                )
            else:
                # resize image keeping ratio to the output_size width
                img = img.resize(
                    (
                        output_size[0],
                        int(img.height * output_size[0] / img.width),
                    ),
                    Image.Resampling.LANCZOS,
                )
        # apply gaussian
        img = img.filter(ImageFilter.GaussianBlur(0.2))
        img_width, img_height = img.size

        # Parse aspect ratio
        aspect_width, aspect_height = parse_aspect_ratio(aspect_ratio)
        # Determine frame height and width
        float_ratio = aspect_width / aspect_height
        if float_ratio > 1:
            float_ratio = 1 / float_ratio

        if movement in ["lr", "rl"]:
            frame_height = img_height
            frame_width = int(frame_height * float_ratio)
        else:
            frame_width = img_width
            frame_height = int(frame_width * float_ratio)

        # Calculate total frames needed
        total_frames = int(duration * fps)

        # Calculate step for panning effect
        if movement in ["lr", "rl"]:
            left_most = 0
            last_left_most = img_width - frame_width
            all_left = np.linspace(left_most, last_left_most, total_frames)
            for left_int in all_left:
                left = left_int
                right = left + frame_width
                frame = img.crop((left, 0, right, img_height))
                current_video.append(frame)
            if movement == "rl":
                current_video = current_video[::-1]
        else:
            top_most = 0
            last_top_most = img_height - frame_height
            all_top = np.linspace(top_most, last_top_most, total_frames)
            for top_int in all_top:
                top = top_int
                bottom = top + frame_height
                frame = img.crop((0, top, img_width, bottom))
                current_video.append(frame)
            if movement == "du":
                current_video = current_video[::-1]
        if add_reverse:
            current_video.extend(current_video[::-1][1:])
    return current_video


def create_zoom_video(
    image_path: str,
    duration: int,
    fps: int = 30,
    add_reverse: bool = False,
    movement: VALID_ZOOM = "in",
    output_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Create a zoom-in video effect of an image from out to in.

    Args:
        image_path (str): Path of the input image.
        duration (int): Length of the video in seconds.
        fps (int): Frames per second of the video (default is 30).
        add_reverse (bool): Add reverse video (default is False).
        movement (str): Movement direction of the zoom effect (default is in).
        output_size (tuple[int, int]): Output video size in
            format (width, height).

    """
    current_video: list[Image.Image] = []
    with read_image(image_path) as img:
        if output_size is not None:
            if movement in ["lr", "rl"]:
                # resize image keeping ratio to the output_size height
                img = img.resize(
                    (
                        int(img.width * output_size[1] / img.height),
                        output_size[1],
                    ),
                    Image.Resampling.LANCZOS,
                )
            else:
                # resize image keeping ratio to the output_size width
                img = img.resize(
                    (
                        output_size[0],
                        int(img.height * output_size[0] / img.width),
                    ),
                    Image.Resampling.LANCZOS,
                )

        img = img.filter(ImageFilter.GaussianBlur(0.2))
        img_width, img_height = img.size

        # Calculate total frames needed
        total_frames = int(duration * fps)

        # how many steps until the size of the image is 25% of the original
        half_size_steps = np.linspace(1.0, 0.75, total_frames)
        for step in half_size_steps:
            left = int(img_width * (1 - step) / 2)
            top = int(img_height * (1 - step) / 2)
            right = img_width - left
            bottom = img_height - top
            frame = img.crop((left, top, right, bottom))
            current_video.append(frame)

        # resize all frames to the smallest frame size
        frame_sizes = [frame.size for frame in current_video]
        min_width = min([size[0] for size in frame_sizes])
        min_height = min([size[1] for size in frame_sizes])
        current_video = [
            frame.resize((min_width, min_height)) for frame in current_video
        ]
        if movement == "out":
            current_video = current_video[::-1]
        if add_reverse:
            current_video.extend(current_video[::-1][1:])
    return current_video


def add_audio_to_video(video: str, audio: AudioOpts, output_file: str) -> str:
    """Add audio to a video.

    Args:
        video (str): Path of the video.
        audio (str): Path of the youtube audio.
        output_file (str): Path of the output video.

    Returns:
        str: Path of the video with audio.

    """
    video_mpe = mpe.VideoFileClip(video)
    video_output = audio.set_audio(video_mpe)
    video_output.write_videofile(
        output_file, codec="libx264", audio_codec="aac"
    )
    return output_file


def speed_up_video(
    video: str,
    speed: float,
    output_file: str,
    output_size: tuple[int, int] | None = None,
) -> str:
    """Speed up a video.

    Args:
        video (str): Path of the video.
        speed (float): Speed factor.
        output_file (str): Path of the output video.
        output_size (tuple[int, int]): Output video size in
            format (width, height).

    Returns:
        str: Path of the sped up video.

    """
    video_mpe = read_video(video)
    if output_size:
        video_mpe = video_mpe.resize(
            width=output_size[0], height=output_size[1]
        )
    video_output = video_mpe.fx(mpe.vfx.speedx, speed)
    video_output.write_videofile(
        output_file, codec="libx264", audio_codec="aac"
    )
    return output_file


def add_image_to_video(
    video: str,
    image: str,
    time: int,
    output_file: str,
    insert_at: Literal["start", "end"],
) -> str:
    """Add an image to a video.

    Args:
        video (str): Path of the video.
        image (str): Path of the image.
        time (int): Time in seconds to add the image.
        output_file (str): Path of the output video.
        insert_at (str): Insert the image at the start or end of the video.

    Returns:
        str: Path of the video with the image added.

    """
    video_mpe = read_video(video)
    image_mpe = mpe.ImageClip(image)
    image_mpe = image_mpe.set_duration(time)
    if insert_at == "start":
        video_output = mpe.concatenate_videoclips([image_mpe, video_mpe])
    else:
        video_output = mpe.concatenate_videoclips([video_mpe, image_mpe])
    video_output.write_videofile(
        output_file, codec="libx264", audio_codec="aac"
    )
    return output_file
