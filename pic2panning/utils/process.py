"""Utility functions for creating videos and GIFs from images."""

from io import BytesIO

import numpy as np
import requests
from PIL import Image

from pic2panning.utils.options import VALID_PANNING, VALID_ZOOM


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
    return img


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
        img_width, img_height = img.size

        # Calculate total frames needed
        total_frames = int(duration * fps)

        # how many steps until the size of the image is half of the org size
        half_size_steps = np.linspace(1.0, 0.5, total_frames)
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
