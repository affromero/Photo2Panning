"""Replace a finished video's soundtrack while copying its video stream."""

import sys
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from typing import Annotated

import tyro
from klogr.path import get_suffix
from pydantic import field_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PydanticBaseModel,
)
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.media import AudioTrack
from pic2panning.tools.io import atomic_output, load_audio, require_file
from pic2panning.tools.media import normalize_audio, probe


class MusicOptions(PydanticBaseModel):
    """Attach a local track or audio URL to an existing video without resizing."""

    video: Annotated[NonEmptyString, tyro.conf.Positional]
    """Local source video; any existing soundtrack is replaced."""

    audio: Annotated[NonEmptyString, tyro.conf.Positional]
    """Local audio file or HTTP audio URL."""

    output: NonEmptyString
    """Destination MP4 file."""

    start: NonNegativeFloat = 0
    """Audio offset in seconds; the skipped prefix stays excluded when looping."""

    loop: bool = False
    """Repeat the selected audio; otherwise pad a short track with silence."""

    overwrite: bool = False
    """Replace an existing output while protecting both source files."""

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str) -> str:
        """Reject unsupported output containers before accessing inputs."""
        if get_suffix(value).lower() != ".mp4":
            raise ValueError("Soundtrack output must end in .mp4")
        return value


def add_music(options: MusicOptions) -> JobResult:
    """Copy the encoded video and replace its soundtrack atomically."""
    requested = AudioTrack(
        source=options.audio, start=options.start, loop=options.loop
    )
    video = require_file(options.video)
    with (
        atomic_output(
            options.output,
            (options.video, options.audio),
            overwrite=options.overwrite,
        ) as output,
        TemporaryDirectory(prefix="pic2panning-music-") as directory,
    ):
        source = probe(video)
        track = requested.model_copy(
            update={"source": load_audio(requested.source, directory)}
        )
        normalize_audio(video, output, source.duration, track=track)
        result = probe(output)
    return JobResult(
        outputs=(options.output,),
        width=result.width,
        height=result.height,
        frames=result.frames,
        duration=result.duration,
        details="Video stream copied; soundtrack replaced",
    )


def main() -> None:
    """Parse typed options and write a JSON result after successful processing."""
    options = tyro.cli(MusicOptions)
    with redirect_stdout(sys.stderr):
        result = add_music(options)
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
