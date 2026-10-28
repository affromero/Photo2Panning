"""Validated local repairs with source-pixel selections and context."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    PydanticBaseModel,
    validate_image_output,
)
from pic2panning.schemas.editing import Crop
from pic2panning.schemas.models import Quantization


class RepairRequest(PydanticBaseModel):
    """Repair a selected region through FLUX 3 without changing other pixels."""

    kind: Literal["repair"] = "repair"
    """Select context-crop FLUX 3 repair in a JSON job."""

    source: NonEmptyString
    """Image path or HTTP URL, using its EXIF-oriented pixel coordinates."""

    output: NonEmptyString
    """Lossless PNG destination at the original source dimensions."""

    region: Crop
    """Exact source-pixel selection, with exclusive right and bottom edges."""

    prompt: NonEmptyString
    """Describe the repaired content and the surrounding content to retain."""

    context: Annotated[int, Field(ge=0, le=4096, strict=True)] = 64
    """Source pixels to include around the region, clipped to image bounds."""

    max_edge: Annotated[
        int, Field(ge=16, le=4096, multiple_of=16, strict=True)
    ] = 768
    """Longest model image edge; aspect is preserved with 16-pixel padding."""

    mask: NonEmptyString | None = None
    """Optional full-source mask: black retains pixels, white selects edits."""

    feather: Annotated[int, Field(ge=0, le=128, strict=True)] = 0
    """Soften selection edges inward without editing black mask pixels."""

    seed: Annotated[int, Field(ge=0, strict=True)] = 0
    """FLUX 3 seed, repeated with the same image and model settings."""

    steps: Annotated[int, Field(gt=0, strict=True)] | None = None
    """Denoising steps; null selects the canonical FLUX 3 default."""

    quantization: Quantization = Quantization.Q4
    """FLUX 3 weight precision in bits."""

    weights: NonEmptyString | None = None
    """MFLUX checkpoint path or repository; null selects community weights."""

    overwrite: bool = False
    """Replace an existing destination while protecting every input file."""

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str) -> str:
        """Require PNG before opening inputs or loading a model."""
        return validate_image_output(value, mask_only=True)
