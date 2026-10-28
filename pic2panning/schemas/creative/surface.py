"""Depth-calibrated text placement on a selected planar image surface."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PositiveFloat,
    PydanticBaseModel,
)
from pic2panning.schemas.camera import Vector3

type SurfaceRegion = tuple[
    NonNegativeFloat, NonNegativeFloat, PositiveFloat, PositiveFloat
]


class LetterDrop(PydanticBaseModel):
    """Stagger glyphs through a bounded 3D fall before sentence transitions."""

    duration: PositiveFloat = 2
    stagger: Annotated[float, Field(ge=0, lt=1)] = 0.6
    """Fraction of duration over which glyph entrances begin, in reading order."""

    height: Annotated[float, Field(gt=0, le=20)] = 6
    """Initial displacement in font heights above the resting text plane."""

    rotation: Vector3 = (65, -25, -20)
    """Initial local X/Y/Z angles in degrees, settling to the declared pose."""

    bounce: Annotated[float, Field(ge=0, le=0.5)] = 0.15
    """Maximum rebound as a fraction of the initial fall height."""


class SurfaceAttachment(PydanticBaseModel):
    """Fit a plane inside a depth-enabled host image and place text on it.

    The region uses oriented source pixels with exclusive right and bottom
    edges. Text keyframe centers also use oriented host pixels; their rays
    intersect the fitted plane. Keyframe scale changes physical text width,
    and keyframe rotation adds clockwise roll within that plane.
    """

    layer_id: NonEmptyString
    region: SurfaceRegion
    tilt: Annotated[float, Field(ge=-80, le=80)] = 0
    """Right-hand rotation in degrees about the fitted surface's right axis."""

    yaw: Annotated[float, Field(ge=-80, le=80)] = 0
    """Right-hand rotation in degrees about the tilted surface's down axis."""

    roll: float = 0
    """Clockwise in-plane rotation, added to the text keyframe's rotation."""

    offset_metres: tuple[float, float] = (0, 0)
    """Additional right/down displacement along the fitted surface."""

    lift_metres: NonNegativeFloat = 0
    """Displacement toward the camera along the fitted surface normal."""

    depth_tolerance_metres: NonNegativeFloat = 0.02
    """Depth separation tolerated before source geometry hides the text."""

    feather_metres: NonNegativeFloat = 0.01
    """Smooth occlusion over this camera-Z depth interval."""

    max_relative_error: Annotated[float, Field(gt=0, le=0.25)] = 0.02
    """Largest accepted plane-fit RMS divided by median sampled depth."""

    @model_validator(mode="after")
    def validate_region(self) -> Self:
        """Require a source patch with positive area before loading media."""
        left, top, right, bottom = self.region
        if right <= left or bottom <= top:
            raise ValueError(
                "Surface region right/bottom must exceed left/top"
            )
        return self


class SurfacePlane(PydanticBaseModel):
    """Inspectable fit in the cached depth camera's metric coordinates."""

    center: Vector3
    normal: Vector3
    right: Vector3
    down: Vector3
    width_metres: PositiveFloat
    height_metres: PositiveFloat
    relative_error: NonNegativeFloat
    samples: Annotated[int, Field(ge=16, strict=True)]
    inlier_fraction: Annotated[float, Field(gt=0, le=1)]
