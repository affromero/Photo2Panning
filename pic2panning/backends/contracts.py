"""Typed boundaries for image generation and object segmentation."""

from typing import Protocol, runtime_checkable

from PIL import Image

from pic2panning.schemas.camera import (
    CameraPose,
    SceneCalibration,
    Vector3,
)
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.schemas.scene import SceneRequest


class ImageBackend(Protocol):
    """Generate an image without coupling callers to model frameworks."""

    def generate(
        self,
        request: GenerateRequest,
        image: Image.Image | None,
        mask: Image.Image | None,
    ) -> Image.Image:
        """Return one RGB image at the requested dimensions."""
        ...


@runtime_checkable
class ReferenceImageBackend(ImageBackend, Protocol):
    """Generate from multiple normalized, ordered conditioning images."""

    def generate_references(
        self,
        request: GenerateRequest,
        references: tuple[Image.Image, ...],
        mask: Image.Image | None,
    ) -> Image.Image:
        """Return RGB pixels using every reference in its declared order."""
        ...


class MaskBackend(Protocol):
    """Find separate object masks with white selected pixels."""

    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Return one full-resolution mask per detected object."""
        ...


class CameraScene(Protocol):
    """An explicitly loaded 3D scene with bounded rendering resources."""

    def render(self, camera: CameraPose) -> Image.Image:
        """Return independent RGB pixels at the requested camera pose."""
        ...

    def pose_from_offset(
        self, base: CameraPose, offset: Vector3, focus_distance: float
    ) -> CameraPose:
        """Move relative to the original camera and retain its optical aim."""
        ...

    def close(self) -> None:
        """Release resources owned by the loaded scene."""
        ...


class CameraBackend(Protocol):
    """Import a calibrated scene without coupling tools to GPU frameworks."""

    def load(self, scene: str, calibration: SceneCalibration) -> CameraScene:
        """Open the provided Gaussian scene or surface its loading failure."""
        ...


class SceneBackend(Protocol):
    """Predict reusable, calibrated geometry from an oriented source photo."""

    def predict(
        self,
        request: SceneRequest,
        image: Image.Image,
        output: str,
        source_sha256: str,
    ) -> tuple[SceneCalibration, int]:
        """Write a Gaussian PLY and return its calibration and point count."""
        ...
