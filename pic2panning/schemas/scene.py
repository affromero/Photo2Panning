"""Validated reconstruction requests and calibrated scene results."""

from typing import Annotated, Literal, Self
from urllib.parse import urlparse

from klogr.path import get_suffix
from pydantic import Field, model_validator

from pic2panning.schemas.base import NonEmptyString, PydanticBaseModel
from pic2panning.schemas.camera import Sha256

MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
MOGE_SOURCE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_SHA256 = (
    "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
)


def validate_scene_paths(output: str, metadata: str) -> None:
    """Require explicit PLY and JSON destinations before inference starts."""
    if any(
        urlparse(path).scheme in {"http", "https"}
        for path in (output, metadata)
    ):
        raise ValueError("Scene output and metadata must be local files")
    if get_suffix(output).lower() != ".ply":
        raise ValueError("Scene output must end in .ply")
    if get_suffix(metadata).lower() != ".json":
        raise ValueError("Scene metadata must end in .json")


class SceneRequest(PydanticBaseModel):
    """Reconstruct calibrated Gaussian geometry from one source image."""

    kind: Literal["scene"] = "scene"
    """Select local scene reconstruction in a JSON job."""

    source: NonEmptyString
    """Original image path or HTTP URL, oriented and converted to RGB."""

    output: NonEmptyString
    """Local PLY destination containing the reconstructed Gaussian scene."""

    metadata: NonEmptyString
    """Local JSON destination containing calibration and source identity."""

    num_tokens: Annotated[int, Field(ge=1200, le=3600, strict=True)] = 1800
    """MoGe-2 inference token budget; larger values increase detail and cost."""

    max_edge: Annotated[int, Field(ge=256, le=4096, strict=True)] = 1920
    """Maximum geometry image edge; the original photo and pan stay unchanged."""

    device: Literal["mps", "cpu"] = "mps"
    """Explicit inference device; failures never select another device."""

    overwrite: bool = False
    """Replace existing artifacts while preserving all source inputs."""

    @model_validator(mode="after")
    def validate_destinations(self) -> Self:
        """Reject unsupported scene artifact formats before accessing inputs."""
        validate_scene_paths(self.output, self.metadata)
        return self


class SceneResult(PydanticBaseModel):
    """Published scene artifacts, point count, and source identity."""

    output: NonEmptyString
    """Published Gaussian PLY scene path."""

    metadata: NonEmptyString
    """Published calibration JSON path, ready for CameraRequest.metadata."""

    points: Annotated[int, Field(gt=0, strict=True)]
    """Number of Gaussian primitives written to the scene."""

    source_sha256: Sha256
    """Lowercase SHA256 of the source image file before normalization."""

    predictor: Literal["moge-2-vitl-normal"] = "moge-2-vitl-normal"
    """The local geometry predictor used to construct the scene."""

    @model_validator(mode="after")
    def validate_artifact_paths(self) -> Self:
        """Keep result paths consistent with the declared scene formats."""
        validate_scene_paths(self.output, self.metadata)
        return self
