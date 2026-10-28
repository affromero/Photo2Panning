"""Validated, framework-independent messages for the native solid renderer."""

from typing import Annotated, Literal

from pydantic import Field

from pic2panning.schemas.base import (
    NonNegativeFloat,
    PositiveFloat,
    PydanticBaseModel,
    Size,
)
from pic2panning.schemas.camera import Matrix3, Matrix4, Vector3
from pic2panning.schemas.creative.surface import SurfacePlane
from pic2panning.tools.typography.contours import Contours

type Triangle = tuple[int, int, int]
type SurfaceQuad = tuple[Vector3, Vector3, Vector3, Vector3]


class SceneLight(PydanticBaseModel):
    """One metric world-space emitter with linear RGB radiance color."""

    position: Vector3
    energy: PositiveFloat
    size: PositiveFloat
    color: Vector3


class SolidScene(PydanticBaseModel):
    """Static calibrated geometry and explicitly selected render settings."""

    version: Literal[1] = 1
    size: Size
    intrinsics: Matrix3
    plane: SurfacePlane
    vertices: tuple[Vector3, ...]
    triangles: tuple[Triangle, ...]
    lights: tuple[SceneLight, ...]
    color: Vector3
    font_size: PositiveFloat
    thickness: PositiveFloat
    bevel: NonNegativeFloat
    roughness: Annotated[float, Field(ge=0, le=1)]
    metallic: Annotated[float, Field(ge=0, le=1)]
    ambient: NonNegativeFloat
    samples: Annotated[int, Field(gt=0, strict=True)]


class SolidGlyph(PydanticBaseModel):
    """Closed Cartesian pixel contours and their metric local-to-world map."""

    contours: Contours
    transform: Matrix4


class SolidFrame(PydanticBaseModel):
    """A bounded current set of complete solids and a receiving surface patch."""

    glyphs: tuple[SolidGlyph, ...]
    catcher: SurfaceQuad


class RenderReply(PydanticBaseModel):
    """A native worker acknowledgement or explicit diagnostic."""

    status: Literal["ready", "rendered", "error"]
    error: str = ""
