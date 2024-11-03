"""Utility functions for creating videos and GIFs from images."""

import os
from dataclasses import field
from pathlib import Path
from typing import Literal, TypeAlias, cast

import moviepy.editor as mpe
import numpy as np
from PIL import Image
from pydantic.dataclasses import dataclass
from pytubefix import YouTube
from pytubefix.cli import on_progress

VALID_MOVEMENT: TypeAlias = Literal[
    "panning-lr",
    "panning-rl",
    "panning-ud",
    "panning-du",
    "zoom-in",
    "zoom-out",
]
VALID_PANNING: TypeAlias = Literal["lr", "rl", "ud", "du"]
VALID_ZOOM: TypeAlias = Literal["in", "out"]


@dataclass
class AudioOpts:
    """Options for audio in the video."""

    audio_file: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    """ YouTube link for audio. """

    start_at: int = 0
    """ Start audio at a specific time [seconds]. """

    def __post_init__(self) -> None:
        """Download audio from YouTube link."""
        if "http" in self.audio_file:
            self.audio_file = AudioOpts.download_audio(self.audio_file)

        if self.start_at:
            self.start_at = int(self.audio_file.split("t=")[1].split("s")[0])
        else:
            self.start_at = 0

    @staticmethod
    def download_audio(link: str) -> str:
        """Download audio from a YouTube URL and save as mp3."""
        # Create Youtube Object.
        yt = YouTube(link, on_progress_callback=on_progress)
        audio_dir = os.path.join(os.getcwd(), "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        audio_file = f"{audio_dir}/{yt.title}.mp3"
        if os.path.exists(audio_file):
            pass
        else:
            audio = yt.streams.filter(only_audio=True).first()
            if not audio:
                raise ValueError("No audio stream found.")
            print("Downloading Audio...")
            audio.download(filename=audio_file)
        return audio_file

    def set_audio(self, video: mpe.VideoFileClip) -> mpe.VideoFileClip:
        """Attach audio from YouTube link to a video."""
        audio_background = (
            mpe.AudioFileClip(self.audio_file)
            .subclip(self.start_at)
            .set_duration(video.duration)
        )
        video = video.set_audio(audio_background)
        # os.remove(self.audio_file)
        return video


@dataclass
class Opts:
    """Options for creating a video or GIF from images."""

    images: list[str]
    """ List of images to convert. """

    output_file: str = "output.mp4"
    """ Output file name. """

    time: list[int] = field(default_factory=lambda: [9])
    "" "Length of the video (approx.) [seconds]" ""

    ratio: str = "16:9"
    """ Output video size ratio. For example, '16:9', '4:3'. """

    audio: AudioOpts | list[AudioOpts] | None = None
    """ Audio options. """

    output_format: Literal["video", "gif"] = "video"
    """ Output type. """

    output_size: tuple[int, int] | None = (1080, 1920)
    """ Output video size. Format (width, height) """

    mail_to: str | None = None
    """ Email to send the Video/GIF. """

    fps: list[int] = field(default_factory=lambda: [30])
    """ Frames per second. """

    movement: list[VALID_MOVEMENT] = field(
        default_factory=lambda: ["panning-lr"]
    )
    """ Movement type. """

    add_reverse: bool = False
    """ Add reverse video. """

    focus_center: bool = False
    """ Focus on the center of the image slowing down the movement and audio.
    Cool for zoom-in and fade-in.
    It requires fps > 100, and best with time < 5. """

    def __post_init__(self) -> None:
        """Check if the files exist and if the number.

        of images and time are the same.
        """
        for img in self.images:
            if not os.path.exists(img):
                raise FileNotFoundError(f"File {img} not found.")

        # list of images and list of times must have the same length
        if len(self.time) == 1:
            self.time = self.time * len(self.images)
        if len(self.fps) == 1:
            self.fps = self.fps * len(self.images)
        if len(self.movement) == 1:
            self.movement = self.movement * len(self.images)
        if (
            len(self.images)
            != len(self.time)
            != len(self.fps)
            != len(self.movement)
        ):
            raise ValueError(
                "Number of images, time, fps, and movement must be the same."
            )
        if isinstance(self.audio, list) and len(self.audio) != len(
            self.images
        ):
            raise ValueError(
                "Number of audio options and images must be the same in "
                "case of multiple audio options."
            )

    def make_gif(self, img_list: list[list[Image.Image]]) -> None:
        """Create a GIF from a list of images."""
        frame_list_extended = img_list[0]
        for i in range(1, len(img_list)):
            frame_list_extended.extend(img_list[i])
        duration = 1 / self.fps[0]
        frame_list_extended[0].save(
            self.output_file,
            save_all=True,
            append_images=img_list[1:],
            duration=duration,
            loop=0,
        )

    def make_video(
        self, img_list: list[list[Image.Image]]
    ) -> mpe.VideoFileClip:
        """Create a video from a list of image frames.

        Args:
            img_list: List of frames. List of frames for each image.

        Returns:
            Video file clip.

        """
        videos = []
        for idx, img in enumerate(img_list):
            if self.output_size is not None:
                img_np_list = [
                    np.asarray(i.resize(self.output_size)) for i in img
                ]
            else:
                img_np_list = [np.asarray(i) for i in img]
            video = mpe.ImageSequenceClip(img_np_list, fps=self.fps[idx])
            if isinstance(self.audio, list):
                video = self.audio[idx].set_audio(video)
            videos.append(video)
        if len(videos) > 1:
            video_cat = cast(
                mpe.VideoFileClip,
                mpe.concatenate_videoclips(videos, method="chain"),
            )
        else:
            video_cat = videos[0]
        if isinstance(self.audio, AudioOpts):
            video_cat = self.audio.set_audio(video_cat)
        return video_cat


def parse_aspect_ratio(ratio: str) -> tuple[int, int]:
    """Parse the aspect ratio string into a width and height."""
    width, height = map(int, ratio.split(":"))
    return width, height


def create_panning_video(
    image_path: str,
    duration: int,
    aspect_ratio: str,
    fps: int = 30,
    add_reverse: bool = False,
    movement: VALID_PANNING = "lr",
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

    """
    current_video = []
    with Image.open(image_path) as img:
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
) -> list[Image.Image]:
    """Create a zoom-in video effect of an image from out to in.

    Args:
        image_path (str): Path of the input image.
        duration (int): Length of the video in seconds.
        fps (int): Frames per second of the video (default is 30).
        add_reverse (bool): Add reverse video (default is False).
        movement (str): Movement direction of the zoom effect (default is in).

    """
    current_video: list[Image.Image] = []
    with Image.open(image_path) as img:
        # resize image for modulo 64
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
