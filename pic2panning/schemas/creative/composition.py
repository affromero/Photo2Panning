"""World-canvas composition, animated layers, and exact-frame previews."""

import math
from enum import StrEnum
from itertools import pairwise
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
from pic2panning.schemas.creative.audio import MixTrack
from pic2panning.schemas.creative.depth import DepthOcclusion, DepthStyle
from pic2panning.schemas.media import Alignment, Encoding


class Easing(StrEnum):
    """Interpolation of the segment following a keyframe."""

    LINEAR = "linear"
    EASE_IN = "ease-in"
    EASE_OUT = "ease-out"
    EASE_IN_OUT = "ease-in-out"
    HOLD = "hold"


class CameraKeyframe(PydanticBaseModel):
    """Camera center in world pixels at a composition time in seconds."""

    at: NonNegativeFloat = 0
    x: float
    y: float
    zoom: PositiveFloat = 1
    ease: Easing = Easing.LINEAR


class LayerKeyframe(PydanticBaseModel):
    """Layer center and transform at a time relative to its start."""

    at: NonNegativeFloat = 0
    x: float
    y: float
    scale: PositiveFloat = 1
    rotation: float = 0
    opacity: Annotated[float, Field(ge=0, le=1)] = 1
    ease: Easing = Easing.LINEAR


class SourceTime(PydanticBaseModel):
    """Map a layer-local output time to an absolute video source time."""

    at: NonNegativeFloat
    source_time: NonNegativeFloat


class Cinemagraph(PydanticBaseModel):
    """White mask pixels select motion over an explicitly chosen still."""

    mask: NonEmptyString
    reference: NonEmptyString | None = None
    reference_time: NonNegativeFloat | None = None
    feather: NonNegativeFloat = 0

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        """Require exactly one clean plate image or frozen video timestamp."""
        if (self.reference is None) == (self.reference_time is None):
            raise ValueError(
                "Cinemagraph requires reference or reference_time, exclusively"
            )
        return self


class LayerBase(PydanticBaseModel):
    """A bounded time interval and an animated, centered layer rectangle."""

    id: NonEmptyString
    start: NonNegativeFloat = 0
    duration: PositiveFloat
    size: Size
    keyframes: Annotated[tuple[LayerKeyframe, ...], Field(min_length=1)]
    space: Literal["world", "viewport"] = "world"
    """World pixels, or output pixels authored at the composition's size.

    Reframing scales viewport centers with each output axis and graphic size
    uniformly with output width.
    """
    mask: NonEmptyString | None = None
    depth: DepthStyle | None = None
    occlusion: DepthOcclusion | None = None

    @model_validator(mode="after")
    def validate_keyframes(self) -> Self:
        """Reject ambiguous animation order or keyframes beyond the layer."""
        times = tuple(frame.at for frame in self.keyframes)
        if times[0] != 0 or any(
            right <= left for left, right in pairwise(times)
        ):
            raise ValueError("Layer keyframes must start at 0 and increase")
        if times[-1] > self.duration:
            raise ValueError("Layer keyframes exceed its duration")
        return self


class ImageLayer(LayerBase):
    """An oriented image placed on the world canvas or output viewport."""

    kind: Literal["image"] = "image"
    source: NonEmptyString
    fit: Literal["contain", "cover"] = "contain"


class VideoLayer(LayerBase):
    """A silent video layer; soundtrack selection is always explicit."""

    kind: Literal["video"] = "video"
    source: NonEmptyString
    fit: Literal["contain", "cover"] = "contain"
    source_start: NonNegativeFloat = 0
    speed: PositiveFloat = 1
    source_times: tuple[SourceTime, ...] = ()
    cinemagraph: Cinemagraph | None = None

    @model_validator(mode="after")
    def validate_source_times(self) -> Self:
        """Require monotone source motion, permitting explicit frozen holds."""
        if self.depth is not None:
            raise ValueError("Video layers cannot declare image depth effects")
        if not self.source_times:
            return self
        if self.source_start or self.speed != 1:
            raise ValueError("Choose source_times or source_start/speed")
        if self.source_times[0].at != 0:
            raise ValueError("Video source_times must start at 0")
        if self.source_times[-1].at > self.duration:
            raise ValueError("Video source_times exceed layer duration")
        for left, right in pairwise(self.source_times):
            if right.at <= left.at or right.source_time < left.source_time:
                raise ValueError(
                    "Video source_times must increase monotonically"
                )
        return self


class TextLayer(LayerBase):
    """Wrapped transparent text with animated placement and opacity."""

    kind: Literal["text"] = "text"
    text: NonEmptyString
    font: NonEmptyString | None = None
    font_size: Annotated[int, Field(gt=0, strict=True)] = 64
    color: NonEmptyString = "white"
    align: Alignment = Alignment.CENTER
    stroke_width: Annotated[int, Field(ge=0, strict=True)] = 0
    stroke_color: NonEmptyString = "black"
    shadow_color: NonEmptyString = "#00000080"
    shadow_offset: tuple[int, int] = (0, 0)
    spacing: Annotated[int, Field(ge=0, strict=True)] = 4

    @field_validator("color", "stroke_color", "shadow_color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        """Validate color names before a render begins."""
        ImageColor.getcolor(value, "RGBA")
        return value

    @model_validator(mode="after")
    def validate_depth(self) -> Self:
        """Depth effects require an image source and calibrated depth asset."""
        if self.depth is not None:
            raise ValueError("Text layers cannot declare depth effects")
        return self


type Layer = Annotated[
    ImageLayer | VideoLayer | TextLayer, Field(discriminator="kind")
]


class ComposeRequest(PydanticBaseModel):
    """Render animated layers through one camera with explicit audio mixing."""

    kind: Literal["compose"] = "compose"
    canvas: Size = Size(width=1920, height=1080)
    duration: PositiveFloat
    camera: Annotated[tuple[CameraKeyframe, ...], Field(min_length=1)]
    layers: Annotated[tuple[Layer, ...], Field(min_length=1)]
    audio: tuple[MixTrack, ...] = ()
    beats: tuple[NonNegativeFloat, ...] = ()
    background: NonEmptyString = "black"
    output: NonEmptyString
    size: Size = Size(width=1080, height=1920)
    fps: Annotated[int, Field(ge=1, le=240, strict=True)] = 30
    shutter_angle: Annotated[float, Field(ge=0, le=360)] = 0
    """Motion exposure in degrees; zero keeps instantaneous frame samples."""

    shutter_samples: Annotated[int, Field(ge=1, le=16, strict=True)] = 8
    """Evenly spaced exposure samples, integrated in linear light."""
    encoding: Encoding = Encoding()
    overwrite: bool = False

    @field_validator("background")
    @classmethod
    def validate_background(cls, value: str) -> str:
        """Require a valid opaque background color."""
        color = ImageColor.getrgb(value)
        if len(color) == 4 and color[3] != 255:
            raise ValueError("Composition background must be opaque")
        return value

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        """Validate all timing and cross-layer references before decoding."""
        if get_suffix(self.output).lower() != ".mp4":
            raise ValueError("Composition output must end in .mp4")
        if self.size.width % 2 or self.size.height % 2:
            raise ValueError("MP4 width and height must be even")
        times = tuple(frame.at for frame in self.camera)
        if (
            times[0] != 0
            or times[-1] > self.duration
            or any(right <= left for left, right in pairwise(times))
        ):
            raise ValueError(
                "Camera times must start at 0 and increase within duration"
            )
        return self

    @model_validator(mode="after")
    def validate_layers(self) -> Self:
        """Keep layers within the timeline and validate occlusion references."""
        ids = {layer.id for layer in self.layers}
        if len(ids) != len(self.layers):
            raise ValueError("Composition layer IDs must be unique")
        for layer in self.layers:
            if layer.start + layer.duration > self.duration + 1e-9:
                raise ValueError("Layer extends beyond composition duration")
            if layer.occlusion is not None:
                reference = next(
                    (
                        item
                        for item in self.layers
                        if item.id == layer.occlusion.layer_id
                    ),
                    None,
                )
                if (
                    reference is None
                    or reference.id == layer.id
                    or reference.depth is None
                ):
                    raise ValueError(
                        "Occlusion requires another layer with depth"
                    )
        return self

    @model_validator(mode="after")
    def validate_audio(self) -> Self:
        """Keep explicit sound and beat markers within the composition."""
        if any(
            track.start + track.duration > self.duration + 1e-9
            for track in self.audio
        ):
            raise ValueError("Audio track extends beyond composition duration")
        if any(track.ducking for track in self.audio) and not any(
            track.role == "voice" for track in self.audio
        ):
            raise ValueError("Ducking requires a voice track")
        if tuple(sorted(set(self.beats))) != self.beats or any(
            beat > self.duration for beat in self.beats
        ):
            raise ValueError(
                "Beat markers must be unique, ordered, and within duration"
            )
        return self


class PreviewRequest(PydanticBaseModel):
    """Save the same canonical RGB frame used by composition exports."""

    kind: Literal["preview"] = "preview"
    composition: ComposeRequest
    output: NonEmptyString
    frame: Annotated[int, Field(ge=0, strict=True)] | None = None
    at: NonNegativeFloat | None = None
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_preview(self) -> Self:
        """Select one frame index or time and a lossless PNG destination."""
        if get_suffix(self.output).lower() != ".png":
            raise ValueError("Preview output must end in .png")
        if self.frame is not None and self.at is not None:
            raise ValueError("Choose preview frame or at")
        if self.at is not None and self.at >= self.composition.duration:
            raise ValueError("Preview time is outside composition duration")
        if self.frame is not None and self.frame >= math.ceil(
            self.composition.duration * self.composition.fps
        ):
            raise ValueError("Preview frame is outside composition duration")
        return self
