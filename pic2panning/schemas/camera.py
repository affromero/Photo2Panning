"""Validated source, calibration, and timing for Gaussian camera motion."""

import math
from enum import StrEnum
from typing import Annotated, Literal, Self
from urllib.parse import urlparse

from klogr.path import get_suffix, path_read_text
from pydantic import Field, JsonValue, TypeAdapter, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PositiveFloat,
    PydanticBaseModel,
    Size,
)
from pic2panning.schemas.media import Encoding

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]
type Vector4 = tuple[float, float, float, float]
type Matrix4 = tuple[Vector4, Vector4, Vector4, Vector4]
type Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def validate_rotation(rotation: Matrix3) -> None:
    """Require an orthonormal proper rotation in OpenCV coordinates."""
    for index, row in enumerate(rotation):
        length = sum(value * value for value in row)
        if not math.isclose(length, 1, abs_tol=1e-5):
            raise ValueError("Camera rotation rows must have unit length")
        for other in rotation[index + 1 :]:
            dot = sum(a * b for a, b in zip(row, other, strict=True))
            if not math.isclose(dot, 0, abs_tol=1e-5):
                raise ValueError("Camera rotation rows must be orthogonal")
    first, second, third = rotation
    determinant = (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )
    if not math.isclose(determinant, 1, abs_tol=1e-5):
        raise ValueError("Camera rotation must have determinant 1")


class SceneColorSpace(StrEnum):
    """The encoding of degree-zero Gaussian colors in the imported scene."""

    SRGB = "sRGB"
    LINEAR = "linearRGB"


class SceneCalibration(PydanticBaseModel):
    """Bind calibrated Gaussian geometry to the exact original image bytes."""

    width: Annotated[int, Field(gt=0, strict=True)]
    """Oriented source image width in pixels."""

    height: Annotated[int, Field(gt=0, strict=True)]
    """Oriented source image height in pixels."""

    intrinsics: Matrix3
    """Pixel camera matrix with positive focal lengths and zero skew."""

    world_to_camera: Matrix4
    """Proper OpenCV world-to-camera transform, with homogeneous last row."""

    sh_degree: Literal[0]
    """Only degree-zero spherical harmonic colors are supported."""

    sh_color_space: SceneColorSpace
    """Scene colors use sRGB or linear RGB before compositing."""

    z_percentiles_metres: dict[str, PositiveFloat]
    """Positive estimated depth percentiles, including foreground key '10'."""

    source_sha256: Sha256
    """Lowercase SHA256 of the original image file, before normalization."""

    scene_sha256: Sha256 | None = None
    """Optional lowercase SHA256 of the imported PLY scene."""

    @model_validator(mode="after")
    def validate_calibration(self) -> Self:
        """Reject geometry that cannot represent the declared source camera."""
        matrix = self.intrinsics
        if matrix[0][0] <= 0 or matrix[1][1] <= 0:
            raise ValueError("Camera focal lengths must be positive")
        standard = (
            (matrix[0][1], 0),
            (matrix[1][0], 0),
            (matrix[2][0], 0),
            (matrix[2][1], 0),
            (matrix[2][2], 1),
        )
        if any(
            not math.isclose(value, expected, abs_tol=1e-7)
            for value, expected in standard
        ):
            raise ValueError(
                "Camera intrinsics require zero skew and row 0,0,1"
            )
        world = self.world_to_camera
        if any(
            not math.isclose(value, expected, abs_tol=1e-7)
            for value, expected in zip(world[3], (0, 0, 0, 1), strict=True)
        ):
            raise ValueError("Camera extrinsics require last row 0,0,0,1")
        validate_rotation((world[0][:3], world[1][:3], world[2][:3]))
        if "10" not in self.z_percentiles_metres:
            raise ValueError("Scene depth percentiles require foreground '10'")
        return self

    @classmethod
    def from_sidecar(cls, path: str) -> SceneCalibration:
        """Read required calibration fields while allowing extra provenance."""
        document = TypeAdapter(dict[str, JsonValue]).validate_json(
            path_read_text(path)
        )
        return cls.model_validate(
            {
                name: document[name]
                for name in cls.model_fields
                if name in document
            }
        )


class CameraPose(PydanticBaseModel):
    """Independent pixel intrinsics and a proper OpenCV camera pose."""

    width: Annotated[int, Field(gt=0, strict=True)]
    height: Annotated[int, Field(gt=0, strict=True)]
    fx: PositiveFloat
    fy: PositiveFloat
    cx: float
    cy: float
    rotation: Matrix3
    translation: Vector3
    znear: PositiveFloat = 0.001
    zfar: PositiveFloat = 1_000_000

    @model_validator(mode="after")
    def validate_pose(self) -> Self:
        """Keep rotations proper and the far plane beyond the near plane."""
        validate_rotation(self.rotation)
        if self.zfar <= self.znear:
            raise ValueError("Camera far plane must exceed its near plane")
        return self


class CameraRequest(PydanticBaseModel):
    """Pan, pause, move through an imported 3D scene, return, and resume."""

    kind: Literal["camera"] = "camera"
    """Select Gaussian scene camera rendering in a JSON job."""

    source: NonEmptyString
    """Original image path or HTTP URL matching the calibration source hash."""

    scene: NonEmptyString
    """Local degree-zero Gaussian PLY scene, in calibrated world coordinates."""

    metadata: NonEmptyString
    """Local JSON calibration sidecar binding the scene to the source image."""

    output: NonEmptyString
    """Local silent MP4 destination, published after complete verification."""

    pan_start: Annotated[float, Field(ge=0, le=1)] = 0
    """Initial horizontal crop position, with 0 at the left and 1 at the right."""

    pause_at: Annotated[float, Field(ge=0, le=1)] = 0.5
    """Horizontal crop position at the pause and camera excursion endpoints."""

    pan_end: Annotated[float, Field(ge=0, le=1)] = 1
    """Final horizontal crop position after the pan resumes."""

    before: PositiveFloat = 4
    """Seconds spent panning from pan_start to pause_at."""

    duration: PositiveFloat = 5
    """Seconds for the complete camera excursion and return."""

    after: PositiveFloat = 3.2
    """Seconds spent panning from pause_at to pan_end."""

    hold: NonNegativeFloat = 0.25
    """Still seconds before and after the excursion; zero omits both holds."""

    position: Vector3 | None = None
    """Far offset in estimated metres: right, down, forward; negative Y rises.

    Null uses (0.02, -0.005, 0.01) times foreground depth percentile '10'.
    """

    size: Size = Size(width=1080, height=1920)
    """Even output dimensions, ordered width then height."""

    fps: Annotated[int, Field(ge=1, le=240, strict=True)] = 60
    """Output frames per second; phase durations round up to whole frames."""

    shutter_angle: Annotated[float, Field(ge=0, le=360)] = 180
    """Pan exposure in degrees; zero gives instantaneous photo samples."""

    encoding: Encoding = Encoding()
    """H.264 quality and compression effort for the complete timeline."""

    overwrite: bool = False
    """Replace an existing output while preserving source and scene inputs."""

    @model_validator(mode="after")
    def validate_camera_options(self) -> Self:
        """Reject invalid geometry and unrepresentable phase timing early."""
        if get_suffix(self.output).lower() != ".mp4":
            raise ValueError("Camera output must end in .mp4")
        if self.size.width % 2 or self.size.height % 2:
            raise ValueError("Camera output dimensions must be even")
        if any(
            urlparse(path).scheme in {"http", "https"}
            for path in (self.scene, self.metadata, self.output)
        ):
            raise ValueError(
                "Camera scene, metadata, and output must be local files"
            )
        if not (
            self.pan_start < self.pause_at < self.pan_end
            or self.pan_start > self.pause_at > self.pan_end
        ):
            raise ValueError("Pan positions must advance through pause_at")
        for name, seconds, minimum in (
            ("before", self.before, 2),
            ("duration", self.duration, 3),
            ("after", self.after, 2),
            ("hold", self.hold, 0),
        ):
            frames = seconds * self.fps
            if not math.isfinite(frames) or math.ceil(frames) < minimum:
                raise ValueError(
                    f"Camera {name} requires at least {minimum} frames"
                )
        return self
