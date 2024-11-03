"""Options for creating a video or GIF from images."""

import os
from dataclasses import field
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

import moviepy.editor as mpe
import numpy as np
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pytubefix import YouTube
from pytubefix.cli import on_progress
from tyro.conf import arg

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
    def list_constructor(files: list[str]) -> list["AudioOpts"]:
        """Create a list of AudioOpts objects from a list of audio files."""
        return [AudioOpts(audio_file=i) for i in files]

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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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

    audio: (
        Annotated[list[AudioOpts], arg(constructor=AudioOpts.list_constructor)]
        | None
    ) = None
    """ Audio options. """

    output_size: tuple[int, int] | None = (1080, 1920)
    """ Output video size. Format (width, height) """

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
            if "http" not in img and not os.path.exists(img):
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

        if (
            self.audio is not None
            and len(self.audio) > 1
            and len(self.audio) != len(self.images)
        ):
            raise ValueError(
                "Number of audio options and images must be the same in "
                "case of multiple audio options."
            )

        # focus_center requires fps > 100
        if self.focus_center and any(i <= 100 for i in self.fps):
            print("Focus center requires fps > 100. Setting fps to 100.")
            self.fps = [i if i > 100 else 100 for i in self.fps]

        # focus_center requires time < 4
        if self.focus_center and any(i >= 5 for i in self.time):
            print("Focus center requires time < 5. Setting time to 3.")
            self.time = [i if i < 4 else 3 for i in self.time]

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
        audio = (
            self.audio[0]
            if isinstance(self.audio, list) and len(self.audio) == 1
            else self.audio
        )
        for idx, img in enumerate(img_list):
            if self.output_size is not None:
                img_np_list = [
                    np.asarray(i.resize(self.output_size)) for i in img
                ]
            else:
                img_np_list = [np.asarray(i) for i in img]
            video = mpe.ImageSequenceClip(img_np_list, fps=self.fps[idx])
            if isinstance(audio, list):
                video = audio[idx].set_audio(video)
            videos.append(video)
        if len(videos) > 1:
            video_cat = cast(
                mpe.VideoFileClip,
                mpe.concatenate_videoclips(videos, method="chain"),
            )
        else:
            video_cat = videos[0]
        if isinstance(audio, AudioOpts):
            video_cat = audio.set_audio(video_cat)
        return video_cat
