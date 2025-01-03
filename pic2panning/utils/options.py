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

from pic2panning.utils.create_text import create_text_image
from pic2panning.utils.logger import get_logger

logger = get_logger()

VALID_IMAGE_PROCESS: TypeAlias = Literal[
    "panning-lr",
    "panning-rl",
    "panning-ud",
    "panning-du",
    "zoom-in",
    "zoom-out",
]
VALID_VIDEO_PROCESS: TypeAlias = Literal[
    "add-audio-to-video", "speed", "panning-lr", "panning-rl"
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

        self.start_at = (
            int(self.audio_file.split("t=")[1].split("s")[0])
            if "t=" in self.audio_file and self.start_at == 0
            else self.start_at
        )

    @staticmethod
    def list_constructor(files: list[str]) -> list["AudioOpts"]:
        """Create a list of AudioOpts objects from a list of audio files."""
        return [AudioOpts(audio_file=i) for i in files]

    @staticmethod
    def download_audio(link: str) -> str:
        """Download audio from a YouTube URL and save as mp3."""
        # Create Youtube Object.
        yt = YouTube(link, on_progress_callback=on_progress)
        audio_dir = "audio"
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        _title = (
            yt.title.replace(" ", "_")
            .replace("|", "_")
            .replace("/", "_")
            .replace(")", "_")
            .replace("(", "_")
        )
        audio_file = f"{audio_dir}/{_title}.mp3"
        if os.path.exists(audio_file):
            pass
        else:
            audio = yt.streams.filter(only_audio=True).first()
            if not audio:
                raise ValueError("No audio stream found.")
            logger.info("Downloading Audio...")
            audio.download(
                output_path=audio_dir, filename=Path(audio_file).name
            )
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file {audio_file} not found.")
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
class CoverImageOpts:
    """Options for adding an image to the video."""

    text: str
    """ Text to add to the image. """

    size: tuple[int, int]
    """ Size of the image. """

    font_color: str = "white"
    """ Font color of the text. """

    font_path: str = ""
    """ Path to the font file. """

    justification: Literal["left", "center", "right"] = "center"
    """ Justification for the text. """

    time: int = 3
    """ Time to display the image [seconds]. """

    insert_at: Literal["start", "end"] = "start"
    """ Insert the image at the start or end of the video. """

    def __post_init__(self) -> None:
        """Check if the font file exists."""
        if self.font_path and not os.path.exists(self.font_path):
            raise FileNotFoundError(f"Font file {self.font_path} not found.")

    def create_text_image(self) -> Image.Image:
        """Create a black image with wrapped text, adjusting the font size.

        and justification.

        Args:
            image_size (Tuple[int, int]): Width and height of the image.

        Returns:
            Image.Image: A PIL Image object with the text rendered on it.

        """
        image = create_text_image(
            self.text,
            self.justification,
            self.size,
            font_path=self.font_path,
        )
        return image


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Opts:
    """Options for creating a video or GIF from images."""

    data: list[str]
    """ List of images/videos to convert. """

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

    output_size: tuple[int, int] = (1080, 1920)
    """ Output video size. Format (width, height) """

    fps: list[int] = field(default_factory=lambda: [30])
    """ Frames per second. """

    process: list[VALID_IMAGE_PROCESS | VALID_VIDEO_PROCESS] = field(
        default_factory=lambda: ["panning-lr"]
    )
    """ process type. """

    add_reverse: bool = False
    """ Add reverse video. """

    focus_center: bool = False
    """ Focus on the center of the image slowing down the movement and audio.
    Cool for zoom-in and fade-in.
    It requires fps > 100, and best with time < 5. """

    speed: float = 1.0
    """ Speed of the video. """

    add_image: CoverImageOpts | None = None
    """ Add an image to the video. """

    def __post_init__(self) -> None:
        """Check if the files exist and if the number.

        of images and time are the same.
        """
        for img in self.data:
            if "http" not in img and not os.path.exists(img):
                raise FileNotFoundError(f"File {img} not found.")

        # list of images and list of times must have the same length
        if len(self.time) == 1:
            self.time = self.time * len(self.data)
        if len(self.fps) == 1:
            self.fps = self.fps * len(self.data)
        if len(self.process) == 1:
            self.process = self.process * len(self.data)
        if (
            len(self.data)
            != len(self.time)
            != len(self.fps)
            != len(self.process)
        ):
            raise ValueError(
                "Number of data, time, fps, and process must be the same."
            )

        if (
            self.audio is not None
            and len(self.audio) > 1
            and len(self.audio) != len(self.data)
        ):
            raise ValueError(
                "Number of audio options and data must be the same in "
                "case of multiple audio options."
            )

        # focus_center requires fps > 100
        if self.focus_center and any(i <= 100 for i in self.fps):
            logger.info("Focus center requires fps > 100. Setting fps to 100.")
            self.fps = [i if i > 100 else 100 for i in self.fps]

        # focus_center requires time < 4
        if self.focus_center and any(i >= 5 for i in self.time):
            logger.info("Focus center requires time < 5. Setting time to 3.")
            self.time = [i if i < 4 else 3 for i in self.time]

        # for some reason in video processing, output_file cant have the same name
        # as any of the input files
        if self.output_file in self.data:
            raise ValueError(
                "Output file name cannot be the same as any of the input files.\n"
                f"Output file: {self.output_file}\n"
                f"Input files: {self.data}"
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
