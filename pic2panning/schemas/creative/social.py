"""Social aspect variants, still selections, and optional safe guides."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PydanticBaseModel,
    Size,
)
from pic2panning.schemas.creative.composition import ComposeRequest


class SafeInsets(PydanticBaseModel):
    """Fractional preview-only margins measured from each viewport edge."""

    top: Annotated[float, Field(ge=0, lt=1)] = 0.1
    right: Annotated[float, Field(ge=0, lt=1)] = 0.08
    bottom: Annotated[float, Field(ge=0, lt=1)] = 0.15
    left: Annotated[float, Field(ge=0, lt=1)] = 0.08

    @model_validator(mode="after")
    def validate_rectangle(self) -> Self:
        """Leave a positive safe rectangle."""
        if self.top + self.bottom >= 1 or self.left + self.right >= 1:
            raise ValueError("Safe insets must leave a visible rectangle")
        return self


class SocialVariant(PydanticBaseModel):
    """One named viewport exported from the original world canvas."""

    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]
    size: Size

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        """Require H.264-compatible even dimensions."""
        if self.size.width % 2 or self.size.height % 2:
            raise ValueError("Social video dimensions must be even")
        return self


class SocialPackRequest(PydanticBaseModel):
    """Export aspect variants and matching cover/carousel stills."""

    kind: Literal["social-pack"] = "social-pack"
    composition: ComposeRequest
    output_directory: NonEmptyString
    variants: Annotated[tuple[SocialVariant, ...], Field(min_length=1)] = (
        SocialVariant(name="portrait", size=Size(width=1080, height=1920)),
        SocialVariant(name="feed", size=Size(width=1080, height=1350)),
        SocialVariant(name="square", size=Size(width=1080, height=1080)),
    )
    cover_at: NonNegativeFloat | None = 0
    carousel_at: tuple[NonNegativeFloat, ...] = ()
    safe_insets: SafeInsets = SafeInsets()
    preview_guides: bool = False
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_exports(self) -> Self:
        """Reject duplicate names and still selections outside the timeline."""
        names = tuple(variant.name.casefold() for variant in self.variants)
        if len(set(names)) != len(names):
            raise ValueError("Social variant names must be unique")
        moments = self.carousel_at + (
            (self.cover_at,) if self.cover_at is not None else ()
        )
        if any(moment >= self.composition.duration for moment in moments):
            raise ValueError("Social still times must be within composition")
        return self
