"""Shared validation policy for every request and result."""

from typing import Annotated

from klogr.path import get_suffix
from pydantic import BaseModel, ConfigDict, Field

type PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type NonEmptyString = Annotated[str, Field(min_length=1)]


def validate_image_output(output: str, *, mask_only: bool = False) -> str:
    """Require a supported image suffix before processing starts."""
    suffix = get_suffix(output).lower()
    allowed = {".png"} if mask_only else {".png", ".jpg", ".jpeg", ".webp"}
    if suffix not in allowed:
        message = (
            "Mask output must end in .png"
            if mask_only
            else "Image output must end in .png, .jpg, .jpeg, or .webp"
        )
        raise ValueError(message)
    return output


class PydanticBaseModel(BaseModel):
    """Reject unknown fields and non-finite numbers; keep requests immutable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        use_attribute_docstrings=True,
        str_strip_whitespace=True,
    )


class Size(PydanticBaseModel):
    """Pixel dimensions, always ordered width then height."""

    width: Annotated[int, Field(gt=0, strict=True)]
    """Horizontal size in pixels; width always precedes height."""

    height: Annotated[int, Field(gt=0, strict=True)]
    """Vertical size in pixels."""

    @property
    def as_tuple(self) -> tuple[int, int]:
        """Return dimensions in Pillow and FFmpeg order."""
        return self.width, self.height
