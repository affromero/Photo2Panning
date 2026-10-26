"""Image and video timeline requests."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from klogr.path import get_suffix
from PIL import ImageColor
from pydantic import Field, field_validator, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PositiveFloat,
    PydanticBaseModel,
    Size,
)


class Motion(StrEnum):
    """Movement of the crop window across the source."""

    PAN_LR = "panning-lr"
    PAN_RL = "panning-rl"
    PAN_UD = "panning-ud"
    PAN_DU = "panning-du"
    ZOOM_IN = "zoom-in"
    ZOOM_OUT = "zoom-out"


class CoverPosition(StrEnum):
    """Placement of a title card in the output timeline."""

    START = "start"
    END = "end"


class Alignment(StrEnum):
    """Horizontal text alignment."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AudioTrack(PydanticBaseModel):
    """Audio starts at an offset; short tracks pad with silence or loop."""

    source: NonEmptyString
    """Local audio path, HTTP media URL, or YouTube URL with its extra."""

    start: NonNegativeFloat = 0
    """Offset in seconds from the beginning of the audio source."""

    loop: bool = False
    """Repeat the selected audio; otherwise pad short tracks with silence."""


class ImageClip(PydanticBaseModel):
    """One photo animated for a duration rounded up to whole frames."""

    kind: Literal["image"] = "image"
    """Select a photo clip in a render timeline."""

    source: NonEmptyString
    """Local image path or HTTP media URL; EXIF orientation is applied."""

    duration: PositiveFloat = 5
    """Forward motion in seconds, rounded up to complete output frames."""

    motion: Motion = Motion.PAN_LR
    """Direction of the crop window or centered zoom across the image."""

    reverse: bool = False
    """Append the return path without duplicating the turning-point frame."""

    focus_center: bool = False
    """Slow movement around the center while retaining both endpoints."""

    zoom: Annotated[float, Field(gt=1, allow_inf_nan=False)] = 4 / 3
    """Magnification at the closest zoom endpoint; used by zoom motions."""

    audio: AudioTrack | None = None
    """Replacement sound for this clip; excludes global replacement audio."""


class VideoClip(PydanticBaseModel):
    """A video with optional speed change, moving crop, and replacement audio."""

    kind: Literal["video"] = "video"
    """Select a source video clip in a render timeline."""

    source: NonEmptyString
    """Local video path or HTTP media URL."""

    speed: PositiveFloat = 1
    """Playback multiplier; 2 halves duration and retains audio pitch."""

    motion: Motion | None = None
    """Bouncing pan or one zoom; null keeps a stationary centered crop."""

    pan_speed: PositiveFloat = 5
    """Pan distance in output pixels per frame, bouncing at source edges."""

    audio: AudioTrack | None = None
    """Replacement sound for this clip; excludes global replacement audio."""

    keep_audio: bool = True
    """Retain source sound when this clip has no replacement soundtrack."""

    zoom: Annotated[float, Field(gt=1, allow_inf_nan=False)] = 4 / 3
    """Magnification at the closest zoom endpoint; panning keeps its crop size."""


type Clip = Annotated[ImageClip | VideoClip, Field(discriminator="kind")]


class Cover(PydanticBaseModel):
    """A title card whose duration is independent of video speed."""

    text: NonEmptyString
    """Text to wrap and fit within a title card."""

    duration: PositiveFloat = 3
    """Title-card seconds, independent of source video playback speed."""

    position: CoverPosition = CoverPosition.START
    """Place the title card before or after the full clip timeline."""

    color: NonEmptyString = "white"
    """Pillow text color name or hexadecimal color."""

    background: NonEmptyString = "black"
    """Pillow background color name or hexadecimal color."""

    font: NonEmptyString | None = None
    """Local font file, or null to use Pillow's bundled default font."""

    align: Alignment = Alignment.CENTER
    """Horizontal alignment of wrapped title text."""

    @field_validator("color", "background")
    @classmethod
    def validate_color(cls, value: str) -> str:
        """Reject invalid colors before a timeline is rendered."""
        ImageColor.getrgb(value)
        return value


class RenderRequest(PydanticBaseModel):
    """Render a timeline to MP4 or GIF without stretching source images."""

    kind: Literal["render"] = "render"
    """Select timeline rendering in a JSON job."""

    clips: Annotated[tuple[Clip, ...], Field(min_length=1)]
    """Nonempty sequence of image or video clips, in playback order."""

    output: NonEmptyString
    """Local destination ending in .mp4 or .gif."""

    size: Size = Size(width=1080, height=1920)
    """Output width and height in pixels; MP4 requires even dimensions."""

    fps: Annotated[int, Field(ge=1, le=240, strict=True)] = 30
    """Shared output frame rate; frame counts determine encoded duration."""

    audio: AudioTrack | None = None
    """Replacement audio spanning the whole timeline, including the cover."""

    cover: Cover | None = None
    """Optional title card with a duration independent of clip speed."""

    overwrite: bool = False
    """Allow replacing an existing output; input aliases stay protected."""

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        """Reject incompatible output dimensions and ambiguous audio."""
        suffix = get_suffix(self.output).lower()
        if suffix not in {".mp4", ".gif"}:
            raise ValueError("Output must end in .mp4 or .gif")
        if suffix == ".mp4" and (self.size.width % 2 or self.size.height % 2):
            raise ValueError("MP4 width and height must be even")
        if self.audio and any(clip.audio for clip in self.clips):
            raise ValueError("Choose global audio or per-clip audio")
        if suffix == ".gif" and (
            self.audio or any(c.audio for c in self.clips)
        ):
            raise ValueError("GIF cannot contain audio")
        return self
