"""Metric depth artifacts and deterministic depth effects."""

import math
from typing import Annotated, Final, Literal, Self
from urllib.parse import urlparse

from klogr.path import get_suffix
from PIL import ImageColor
from pydantic import Field, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PositiveFloat,
    PydanticBaseModel,
)
from pic2panning.schemas.camera import Matrix3, Sha256, Vector3

MOGE_MODEL: Final = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION: Final = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
MOGE_SOURCE_REVISION: Final = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_SHA256: Final = (
    "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
)

type PixelCount = Annotated[int, Field(gt=0, strict=True)]
type UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
type BlurRadius = Annotated[float, Field(ge=0, le=64, allow_inf_nan=False)]
type ParallaxPixels = Annotated[
    float, Field(ge=-32, le=32, allow_inf_nan=False)
]


class GeometryOptions(PydanticBaseModel):
    """Shared source normalization and pinned local geometry options."""

    source: NonEmptyString
    """Image path or HTTP URL, EXIF-oriented and converted to RGB."""

    num_tokens: Annotated[int, Field(ge=1200, le=3600, strict=True)] = 1800
    """MoGe-2 inference token budget."""

    max_edge: Annotated[int, Field(ge=256, le=4096, strict=True)] = 1920
    """Maximum inference-grid edge, preserving the source field of view."""

    device: Literal["mps", "cpu"] = "mps"
    """Explicit inference device, with no device fallback."""

    overwrite: bool = False
    """Replace existing artifacts while protecting source inputs."""


class DepthRequest(GeometryOptions):
    """Extract reusable metric depth without reconstructing a Gaussian scene."""

    kind: Literal["depth"] = "depth"
    output: NonEmptyString
    """Float32 metric TIFF; sibling previews and metadata share its stem."""

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        """Validate all destinations before model or input access."""
        if urlparse(self.output).scheme in {"http", "https"}:
            raise ValueError("Depth output must be a local TIFF file")
        if get_suffix(self.output).lower() not in {".tif", ".tiff"}:
            raise ValueError("Depth output must end in .tif or .tiff")
        return self

    @property
    def stem(self) -> str:
        """Return the output path without its TIFF suffix."""
        return self.output[: -len(get_suffix(self.output))]

    @property
    def metadata(self) -> str:
        """JSON manifest published after every depth artifact."""
        return f"{self.stem}.depth.json"

    @property
    def preview(self) -> str:
        """Sixteen-bit near-white depth visualization."""
        return f"{self.stem}.preview.png"

    @property
    def normals(self) -> str:
        """RGB encoded camera-facing normals derived from metric depth."""
        return f"{self.stem}.normals.png"

    @property
    def mask(self) -> str:
        """White semantic foreground, independent of finite sky depth."""
        return f"{self.stem}.mask.png"

    @property
    def artifacts(self) -> tuple[str, ...]:
        """Enumerate every published destination before inference starts."""
        return (
            self.output,
            self.preview,
            self.normals,
            self.mask,
            self.metadata,
        )


class DepthResult(PydanticBaseModel):
    """Published metric depth, auxiliary artifacts, and source binding."""

    output: NonEmptyString
    metadata: NonEmptyString
    preview: NonEmptyString
    normals: NonEmptyString
    mask: NonEmptyString
    source_sha256: Sha256
    width: PixelCount
    height: PixelCount
    grid_width: PixelCount
    grid_height: PixelCount
    predictor: Literal["moge-2-vitl-normal"] = "moge-2-vitl-normal"

    @property
    def artifacts(self) -> tuple[str, ...]:
        """Enumerate outputs in their publication order."""
        return (
            self.output,
            self.preview,
            self.normals,
            self.mask,
            self.metadata,
        )


class DepthArtifact(PydanticBaseModel):
    """One exact output file, resolved relative to the manifest directory."""

    path: NonEmptyString
    sha256: Sha256


class DepthMetadata(PydanticBaseModel):
    """Bind metric fields to the original file and oriented pixel grid."""

    format_version: Literal[1] = 1
    source_sha256: Sha256
    width: PixelCount
    height: PixelCount
    grid_width: PixelCount
    grid_height: PixelCount
    intrinsics: Matrix3
    depth: DepthArtifact
    preview: DepthArtifact
    normals: DepthArtifact
    mask: DepthArtifact
    preview_near_metres: PositiveFloat
    preview_far_metres: PositiveFloat
    model: Literal["Ruicheng/moge-2-vitl-normal"] = MOGE_MODEL
    model_revision: NonEmptyString = MOGE_REVISION
    source_revision: NonEmptyString = MOGE_SOURCE_REVISION
    checkpoint_sha256: Sha256 = MOGE_SHA256
    device: Literal["mps", "cpu"]
    num_tokens: Annotated[int, Field(ge=1200, le=3600, strict=True)]
    max_edge: Annotated[int, Field(ge=256, le=4096, strict=True)]
    units: Literal["metres"] = "metres"
    depth_axis: Literal["camera-z"] = "camera-z"
    orientation: Literal["EXIF-transposed RGB"] = "EXIF-transposed RGB"
    coordinates: Literal["OpenCV x-right y-down z-forward"] = (
        "OpenCV x-right y-down z-forward"
    )
    pixel_centres: Literal["image edges plus 0.5"] = "image edges plus 0.5"
    mask_semantics: Literal[
        "white semantic foreground; sky depth retained"
    ] = "white semantic foreground; sky depth retained"
    preview_encoding: Literal["uint16 linear z, near=65535 far=0"] = (
        "uint16 linear z, near=65535 far=0"
    )
    normal_encoding: Literal[
        "depth-derived camera-facing unit XYZ mapped [-1,1] to RGB [0,255]"
    ] = "depth-derived camera-facing unit XYZ mapped [-1,1] to RGB [0,255]"
    license: Literal["MIT"] = "MIT"

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid calibration or inconsistent bounded image grids."""
        matrix = self.intrinsics
        standard = (
            (matrix[0][1], 0),
            (matrix[1][0], 0),
            (matrix[2][0], 0),
            (matrix[2][1], 0),
            (matrix[2][2], 1),
        )
        if (
            matrix[0][0] <= 0
            or matrix[1][1] <= 0
            or any(
                not math.isclose(value, expected, abs_tol=1e-7)
                for value, expected in standard
            )
        ):
            raise ValueError(
                "Depth intrinsics require positive unskewed focal lengths"
            )
        if self.grid_width > self.width or self.grid_height > self.height:
            raise ValueError(
                "Depth grid cannot exceed oriented source dimensions"
            )
        if self.preview_far_metres < self.preview_near_metres:
            raise ValueError("Depth preview bounds must be ordered")
        if (
            self.model_revision != MOGE_REVISION
            or self.source_revision != MOGE_SOURCE_REVISION
            or self.checkpoint_sha256 != MOGE_SHA256
        ):
            raise ValueError(
                "Depth provenance differs from the pinned predictor"
            )
        return self


class DepthStyle(PydanticBaseModel):
    """Deterministic effects using a bound cached metric depth asset."""

    metadata: NonEmptyString
    focus_metres: PositiveFloat = 3
    focus_end_metres: PositiveFloat | None = None
    blur_radius: BlurRadius = 0
    fog_density: NonNegativeFloat = 0
    fog_color: NonEmptyString = "white"
    light_direction: Vector3 = (0, 0, -1)
    ambient: UnitFloat = 1
    relight_strength: UnitFloat = 0
    """Blend depth-derived lighting over semantic foreground, preserving sky."""
    parallax_x: ParallaxPixels = 0
    """Maximum horizontal warp in pixels, scaled by bounded inverse depth."""
    parallax_y: ParallaxPixels = 0
    """Maximum vertical warp in pixels, scaled by bounded inverse depth."""
    exposed_pixels: Literal["edge-repeat"] = "edge-repeat"
    """Explicitly repeat sampled image edges in the small 2.5D warp."""

    @model_validator(mode="after")
    def validate_style(self) -> Self:
        """Reject invalid light vectors and colors before frame processing."""
        if not math.sqrt(sum(value * value for value in self.light_direction)):
            raise ValueError("Depth light direction cannot be zero")
        try:
            ImageColor.getrgb(self.fog_color)
        except ValueError as error:
            raise ValueError(
                "Depth fog color must be a Pillow color"
            ) from error
        return self


class DepthOcclusion(PydanticBaseModel):
    """Hide pixels behind nearer depth in the named source image layer."""

    layer_id: NonEmptyString
    plane_metres: PositiveFloat
    feather_metres: NonNegativeFloat = 0
