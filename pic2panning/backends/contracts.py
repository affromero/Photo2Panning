"""Typed boundaries for image generation and object segmentation."""

from typing import Protocol

from PIL import Image

from pic2panning.schemas.models import GenerateRequest, MaskRequest


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


class MaskBackend(Protocol):
    """Find separate object masks with white selected pixels."""

    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Return one full-resolution mask per detected object."""
        ...
