"""Ordered image operations that can be serialized as JSON."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PydanticBaseModel,
    Size,
    validate_image_output,
)


class Axis(StrEnum):
    """Axis of an image reflection."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class Resize(PydanticBaseModel):
    """Resize to exact dimensions with Lanczos resampling."""

    op: Literal["resize"] = "resize"
    """Select exact image resizing in the ordered operation list."""

    size: Size
    """Target width and height in pixels; both dimensions are applied."""


class Crop(PydanticBaseModel):
    """Pixel crop using an exclusive right and bottom edge."""

    op: Literal["crop"] = "crop"
    """Select a rectangular crop in the ordered operation list."""

    left: Annotated[int, Field(ge=0, strict=True)]
    """Inclusive left edge in current-image pixels."""

    top: Annotated[int, Field(ge=0, strict=True)]
    """Inclusive top edge in current-image pixels."""

    right: Annotated[int, Field(gt=0, strict=True)]
    """Exclusive right edge; must stay inside the current image."""

    bottom: Annotated[int, Field(gt=0, strict=True)]
    """Exclusive bottom edge; must stay inside the current image."""

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Require a crop with positive area."""
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("Crop right/bottom must exceed left/top")
        return self


class Rotate(PydanticBaseModel):
    """Rotate counterclockwise, expanding the canvas to avoid clipping."""

    op: Literal["rotate"] = "rotate"
    """Select image rotation in the ordered operation list."""

    degrees: float
    """Counterclockwise angle; the canvas expands to retain the image."""


class Flip(PydanticBaseModel):
    """Reflect an image horizontally or vertically."""

    op: Literal["flip"] = "flip"
    """Select image reflection in the ordered operation list."""

    axis: Axis = Axis.HORIZONTAL
    """Horizontal mirrors left and right; vertical exchanges top and bottom."""


class Adjust(PydanticBaseModel):
    """Apply brightness, contrast, then saturation; 1 leaves each unchanged."""

    op: Literal["adjust"] = "adjust"
    """Select brightness, contrast, and saturation adjustment."""

    brightness: NonNegativeFloat = 1
    """Brightness multiplier; 1 keeps it unchanged and 0 produces black."""

    contrast: NonNegativeFloat = 1
    """Contrast multiplier, applied after brightness; 1 keeps it unchanged."""

    saturation: NonNegativeFloat = 1
    """Color multiplier, applied last; 0 is grayscale and 1 is unchanged."""


type EditOperation = Annotated[
    Resize | Crop | Rotate | Flip | Adjust, Field(discriminator="op")
]


class EditRequest(PydanticBaseModel):
    """Apply operations in their declared order and save a new image."""

    kind: Literal["edit"] = "edit"
    """Select deterministic image editing in a JSON job."""

    source: NonEmptyString
    """Local image path or HTTP media URL; EXIF orientation is applied."""

    output: NonEmptyString
    """Local PNG, JPEG, or WebP destination; the suffix selects its format."""

    operations: Annotated[tuple[EditOperation, ...], Field(min_length=1)]
    """Nonempty edit sequence applied in exactly the declared order."""

    overwrite: bool = False
    """Allow replacing an existing output; input aliases stay protected."""

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str) -> str:
        """Require a supported output format before reading the input."""
        return validate_image_output(value)
