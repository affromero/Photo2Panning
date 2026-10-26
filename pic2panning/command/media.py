"""Typed photo, video, and deterministic image-editing commands."""

from typing import Annotated

import tyro

from pic2panning.command.base import OutputCommand
from pic2panning.schemas.base import Size
from pic2panning.schemas.editing import (
    Adjust,
    Axis,
    Crop,
    EditOperation,
    EditRequest,
    Flip,
    Resize,
    Rotate,
)
from pic2panning.schemas.media import (
    AudioTrack,
    Clip,
    Cover,
    Encoding,
    EncodingPreset,
    ImageClip,
    Motion,
    RenderRequest,
    VideoClip,
)


class RenderCommand(OutputCommand):
    """Animate photos or reframe videos with explicit timing and sound."""

    sources: Annotated[tuple[str, ...], tyro.conf.Positional]
    """Image paths or URLs, or video inputs when --video is selected."""

    motion: Motion | None = None
    """Photo pan or zoom; videos keep a stationary crop when omitted."""

    duration: float | None = None
    """Forward duration of each photo in seconds; default 5."""

    size: tuple[int, int] = (1080, 1920)
    """Output dimensions in width, height order."""

    fps: int = 30
    """Output frames per second; GIF frame rates must divide 100."""

    crf: int = 18
    """H.264 constant rate factor from 0 to 51; lower retains more detail."""

    preset: EncodingPreset = EncodingPreset.VERYFAST
    """H.264 compression effort; slower presets improve compression."""

    reverse: bool = False
    """Append the return motion for each photo."""

    focus_center: bool = False
    """Slow photo movement around the center of its path."""

    shutter_angle: float = 0
    """Photo motion exposure in degrees, from 0 to 360; try 180 for fast pans."""

    audio: str | None = None
    """Replacement soundtrack across the complete timeline."""

    audio_start: float = 0
    """Start the replacement soundtrack at this offset in seconds."""

    loop_audio: bool = False
    """Repeat the selected soundtrack after its start offset."""

    cover: str | None = None
    """Text for a three-second title card before the timeline."""

    speed: float = 1
    """Playback multiplier for video inputs, retaining audio pitch."""

    pan_speed: float | None = None
    """Video pan distance per output frame; default 5 pixels."""

    zoom: float = 4 / 3
    """Closest magnification for zoom motions."""

    video: bool = False
    """Treat all positional sources as videos."""

    def to_job(self) -> RenderRequest:
        """Build the validated timeline before reading sources or encoding."""
        clips: tuple[Clip, ...]
        if self.video:
            if self.duration is not None:
                raise ValueError("--duration requires image inputs")
            if self.reverse or self.focus_center:
                raise ValueError(
                    "--reverse and --focus-center require image inputs"
                )
            if self.shutter_angle:
                raise ValueError("--shutter-angle requires image inputs")
            clips = tuple(
                VideoClip(
                    source=source,
                    speed=self.speed,
                    motion=self.motion,
                    pan_speed=self.pan_speed
                    if self.pan_speed is not None
                    else 5,
                    zoom=self.zoom,
                )
                for source in self.sources
            )
        else:
            if self.speed != 1:
                raise ValueError("--speed requires --video")
            if self.pan_speed is not None:
                raise ValueError("--pan-speed requires --video")
            clips = tuple(
                ImageClip(
                    source=source,
                    duration=self.duration if self.duration is not None else 5,
                    motion=self.motion or Motion.PAN_LR,
                    reverse=self.reverse,
                    focus_center=self.focus_center,
                    shutter_angle=self.shutter_angle,
                    zoom=self.zoom,
                )
                for source in self.sources
            )
        if self.audio is None and (self.audio_start or self.loop_audio):
            raise ValueError("Audio offset and loop options require --audio")
        return RenderRequest(
            clips=clips,
            output=self.output,
            size=Size(width=self.size[0], height=self.size[1]),
            fps=self.fps,
            encoding=Encoding(crf=self.crf, preset=self.preset),
            audio=AudioTrack(
                source=self.audio, start=self.audio_start, loop=self.loop_audio
            )
            if self.audio is not None
            else None,
            cover=Cover(text=self.cover) if self.cover is not None else None,
            overwrite=self.overwrite,
        )


class EditCommand(OutputCommand):
    """Crop, resize, rotate, flip, and adjust a picture in that order."""

    source: Annotated[str, tyro.conf.Positional]
    """Input image path or HTTP URL."""

    crop: tuple[int, int, int, int] | None = None
    """Crop rectangle: left, top, right, bottom in source pixels."""

    resize: tuple[int, int] | None = None
    """Resize to an exact width and height."""

    rotate: float | None = None
    """Counterclockwise rotation in degrees, expanding the canvas."""

    flip: Axis | None = None
    """Reflect the image horizontally or vertically."""

    brightness: float = 1
    """Brightness multiplier; 1 keeps the current brightness."""

    contrast: float = 1
    """Contrast multiplier, applied after brightness."""

    saturation: float = 1
    """Color multiplier; 0 produces grayscale."""

    def to_job(self) -> EditRequest:
        """Build ordered image operations using the canonical request models."""
        operations: list[EditOperation] = []
        if self.crop is not None:
            left, top, right, bottom = self.crop
            operations.append(
                Crop(left=left, top=top, right=right, bottom=bottom)
            )
        if self.resize is not None:
            operations.append(
                Resize(size=Size(width=self.resize[0], height=self.resize[1]))
            )
        if self.rotate is not None:
            operations.append(Rotate(degrees=self.rotate))
        if self.flip is not None:
            operations.append(Flip(axis=self.flip))
        if (self.brightness, self.contrast, self.saturation) != (1, 1, 1):
            operations.append(
                Adjust(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                )
            )
        return EditRequest(
            source=self.source,
            output=self.output,
            operations=tuple(operations),
            overwrite=self.overwrite,
        )
