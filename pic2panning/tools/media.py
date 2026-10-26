"""Stream pixels and normalize media using bounded FFmpeg subprocesses."""

import math
import shutil
import subprocess
import tempfile
from collections.abc import Generator, Iterable, Sequence
from contextlib import suppress
from fractions import Fraction
from typing import BinaryIO

from klogr.path import path_join, path_open
from PIL import Image

from pic2panning.errors import DependencyError, MediaError
from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.media import AudioTrack

COMMAND_TIMEOUT = 300


class Rotation(PydanticBaseModel):
    """The display orientation reported by FFprobe."""

    rotation: int = 0


class StreamInfo(PydanticBaseModel):
    """The selected FFprobe stream fields."""

    codec_type: str
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str = "0/0"
    nb_frames: str | None = None
    duration: str | None = None
    side_data_list: tuple[Rotation, ...] = ()


class FormatInfo(PydanticBaseModel):
    """The selected FFprobe container fields."""

    duration: str | None = None


class ProbeInfo(PydanticBaseModel):
    """A narrowly selected FFprobe JSON response."""

    streams: tuple[StreamInfo, ...]
    format: FormatInfo
    programs: tuple[object, ...] = ()
    stream_groups: tuple[object, ...] = ()


class MediaInfo(PydanticBaseModel):
    """Decoded display dimensions and the media timeline."""

    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool
    frames: int | None = None


def executable(name: str) -> str:
    """Find an external binary or explain the missing dependency."""
    command = shutil.which(name)
    if command is None:
        message = f"{name} is required. Install FFmpeg and put it on PATH."
        raise DependencyError(message)
    return command


def command_output(command: Sequence[str]) -> str:
    """Run a bounded external command and retain its complete diagnostics."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.CalledProcessError as error:
        message = f"Media command failed: {error.stderr.strip()}"
        raise MediaError(message) from error
    except subprocess.TimeoutExpired as error:
        message = f"Media command exceeded {COMMAND_TIMEOUT} seconds."
        raise MediaError(message) from error
    return result.stdout


def ffmpeg(arguments: Sequence[str]) -> None:
    """Run FFmpeg with consistent error reporting."""
    command_output(
        [
            executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            *arguments,
        ]
    )


def probe(source: str) -> MediaInfo:
    """Inspect a local video without decoding its frames."""
    output = command_output(
        [
            executable("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,width,height,avg_frame_rate,nb_frames,duration:"
                "stream_side_data=rotation:format=duration"
            ),
            "-of",
            "json",
            source,
        ]
    )
    info = ProbeInfo.model_validate_json(output)
    video = next(
        (stream for stream in info.streams if stream.codec_type == "video"),
        None,
    )
    if video is None or video.width is None or video.height is None:
        message = f"No video stream found in {source}."
        raise MediaError(message)
    width, height = video.width, video.height
    if any(item.rotation % 180 != 0 for item in video.side_data_list):
        width, height = height, width
    duration_value = (
        video.duration
        if video.duration not in {None, "N/A"}
        else info.format.duration
    )
    duration = (
        float(duration_value) if duration_value not in {None, "N/A"} else 0.0
    )
    try:
        fps = float(Fraction(video.avg_frame_rate))
    except ZeroDivisionError:
        fps = 0.0
    if not math.isfinite(duration) or duration <= 0:
        message = f"Cannot determine a positive video duration for {source}."
        raise MediaError(message)
    return MediaInfo(
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        has_audio=any(stream.codec_type == "audio" for stream in info.streams),
        frames=(
            int(video.nb_frames)
            if video.nb_frames and video.nb_frames.isdigit()
            else None
        ),
    )


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Reap a subprocess, including when a frame iterator is interrupted."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def stderr_message(stderr: BinaryIO) -> str:
    """Read a subprocess diagnostic file after its process exits."""
    stderr.seek(0)
    return stderr.read().decode("utf-8", errors="replace").strip()


def decode_frames(
    source: str, info: MediaInfo, fps: int, speed: float = 1
) -> Generator[Image.Image]:
    """Yield raw RGB frames resampled to the requested output timeline."""
    command = [
        executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"setpts=(PTS-STARTPTS)/{speed:.12g},fps={fps}:round=up",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=stderr
        )
        try:
            if process.stdout is None:
                message = "FFmpeg did not expose its decoded frame stream."
                raise MediaError(message)
            byte_count = info.width * info.height * 3
            while data := process.stdout.read(byte_count):
                if len(data) != byte_count:
                    message = "FFmpeg returned a truncated RGB frame."
                    raise MediaError(message)
                yield Image.frombytes("RGB", (info.width, info.height), data)
            process.wait(timeout=COMMAND_TIMEOUT)
            if process.returncode != 0:
                message = f"Video decoding failed: {stderr_message(stderr)}"
                raise MediaError(message)
        finally:
            stop_process(process)
            if process.stdout is not None:
                process.stdout.close()


def encode_frames(
    frames: Iterable[Image.Image], output: str, size: Size, fps: int
) -> int:
    """Write frames incrementally and return the exact encoded count."""
    pixel_format = (
        "yuv444p" if size.width % 2 or size.height % 2 else "yuv420p"
    )
    command = [
        executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size.width}x{size.height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        pixel_format,
        "-movflags",
        "+faststart",
        output,
    ]
    count = 0
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=stderr
        )
        try:
            if process.stdin is None:
                message = "FFmpeg did not expose its encoding input stream."
                raise MediaError(message)
            for frame in frames:
                with frame:
                    if frame.size != size.as_tuple or frame.mode != "RGB":
                        message = (
                            "Encoded frames must have the requested RGB size."
                        )
                        raise MediaError(message)
                    process.stdin.write(frame.tobytes())
                count += 1
            process.stdin.close()
            process.wait(timeout=COMMAND_TIMEOUT)
            if process.returncode != 0 or count == 0:
                message = f"Video encoding failed: {stderr_message(stderr)}"
                raise MediaError(message)
        except BrokenPipeError as error:
            process.wait(timeout=COMMAND_TIMEOUT)
            message = f"Video encoding failed: {stderr_message(stderr)}"
            raise MediaError(message) from error
        finally:
            stop_process(process)
            if process.stdin is not None and not process.stdin.closed:
                with suppress(BrokenPipeError):
                    process.stdin.close()
    return count


def tempo_filter(speed: float) -> str:
    """Express any positive audio speed using FFmpeg's supported factors."""
    factors: list[float] = []
    remainder = speed
    while remainder > 2:
        factors.append(2)
        remainder /= 2
    while remainder < 0.5:
        factors.append(0.5)
        remainder *= 2
    factors.append(remainder)
    return ",".join(f"atempo={factor:.12g}" for factor in factors)


def normalize_audio(
    video: str,
    output: str,
    duration: float,
    track: AudioTrack | None = None,
    speed: float = 1,
) -> None:
    """Attach stereo audio, padding short tracks to the exact video length."""
    arguments = ["-nostdin", "-y", "-i", video]
    if track is None:
        arguments.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"])
    else:
        if track.loop:
            arguments.extend(["-stream_loop", "-1"])
        if track.start:
            arguments.extend(["-ss", f"{track.start:.12g}"])
        arguments.extend(["-i", track.source])
    arguments.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-af",
            (
                "aformat=sample_fmts=fltp,"
                f"{tempo_filter(speed)},apad,atrim=duration={duration:.12g},"
                "asetpts=PTS-STARTPTS"
            ),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{duration:.12g}",
            output,
        ]
    )
    ffmpeg(arguments)


def concatenate(
    paths: Sequence[str], manifest: str, output: str, duration: float
) -> None:
    """Concatenate compatible encoded clips and normalize audio timestamps."""
    with path_open(manifest, "w") as stream:
        for path in paths:
            escaped = path.replace("'", "'\\''")
            stream.write(f"file '{escaped}'\n")
    ffmpeg(
        [
            "-nostdin",
            "-y",
            "-copyts",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            manifest,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-af",
            "aresample=async=1:first_pts=0",
            "-t",
            f"{duration:.12g}",
            "-movflags",
            "+faststart",
            output,
        ]
    )


def remove_audio(source: str, output: str) -> None:
    """Remove normalization silence while copying the encoded video exactly."""
    ffmpeg(
        [
            "-nostdin",
            "-y",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            output,
        ]
    )


def to_gif(source: str, output: str, fps: int) -> None:
    """Encode an animated GIF with a palette derived from the full timeline."""
    with tempfile.TemporaryDirectory(
        prefix="pic2panning-palette-"
    ) as directory:
        palette = path_join(directory, "palette.png")
        ffmpeg(
            [
                "-nostdin",
                "-y",
                "-i",
                source,
                "-vf",
                f"fps={fps},palettegen",
                "-frames:v",
                "1",
                "-update",
                "1",
                palette,
            ]
        )
        ffmpeg(
            [
                "-nostdin",
                "-y",
                "-i",
                source,
                "-i",
                palette,
                "-filter_complex",
                f"[0:v]fps={fps}[v];[v][1:v]paletteuse",
                "-an",
                "-loop",
                "0",
                output,
            ]
        )
