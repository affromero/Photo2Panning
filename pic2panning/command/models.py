"""Typed optional-model commands without importing model frameworks."""

from typing import Annotated

import tyro

from pic2panning.command.base import OutputCommand
from pic2panning.schemas.base import Size
from pic2panning.schemas.models import (
    Device,
    GenerateRequest,
    GenerationModel,
    MaskRequest,
    Quantization,
)
from pic2panning.schemas.video import GenerateVideoRequest


class GenerateCommand(OutputCommand):
    """Generate or edit an image with an explicitly selected local MLX model."""

    prompt: str
    """Complete caption of the desired image, including content to retain."""

    source: str | None = None
    """Reference image path or URL; omit for text-to-image generation."""

    model: GenerationModel = GenerationModel.FLUX3
    """Model family for generation, reference editing, or masked fill."""

    weights: str | None = None
    """MFLUX checkpoint path or repository, replacing the default weights."""

    quantization: Quantization = Quantization.Q4
    """Weight precision in bits: 4, 6, or 8."""

    seed: int = 0
    """Seed for repeatable inference with the same model and settings."""

    steps: int | None = None
    """Denoising steps; omit to use the selected model's default."""

    size: tuple[int, int] = (512, 512)
    """Output width and height, both multiples of 16 pixels."""

    mask: str | None = None
    """Selection mask: white edits, black retains the source pixels."""

    strength: float | None = None
    """Reference-image strength for FLUX 1 schnell or dev."""

    guidance: float | None = None
    """Guidance for FLUX 1 dev or fill; distilled models reject it."""

    def to_job(self) -> GenerateRequest:
        """Validate model-specific options before loading weights."""
        return GenerateRequest(
            source=self.source,
            output=self.output,
            prompt=self.prompt,
            model=self.model,
            weights=self.weights,
            quantization=self.quantization,
            seed=self.seed,
            steps=self.steps,
            size=Size(width=self.size[0], height=self.size[1]),
            strength=self.strength,
            guidance=self.guidance,
            mask=self.mask,
            overwrite=self.overwrite,
        )


class MaskCommand(OutputCommand):
    """Find text-described objects and produce lossless segmentation masks."""

    source: Annotated[str, tyro.conf.Positional]
    """Input image path or HTTP URL."""

    prompt: str
    """Object labels to detect; separate multiple labels with periods."""

    threshold: float = 0.3
    """Minimum detection confidence, greater than 0 and at most 1."""

    device: Device = Device.CPU
    """Explicit CPU execution for the small local segmentation models."""

    separate: bool = False
    """Save one numbered mask per object instead of merging detections."""

    def to_job(self) -> MaskRequest:
        """Build the same validated mask request used by Python callers."""
        return MaskRequest(
            source=self.source,
            output=self.output,
            prompt=self.prompt,
            threshold=self.threshold,
            device=self.device,
            merge=not self.separate,
            overwrite=self.overwrite,
        )


class GenerateVideoCommand(OutputCommand):
    """Generate one local camera shot between first and last image guides."""

    first_frame: str
    """First guide image path or HTTP URL."""

    last_frame: str
    """Last guide image path or HTTP URL."""

    prompt: str
    """Describe a single continuous camera movement between the guides."""

    negative_prompt: str | None = None
    """Unwanted content or motion; omit to retain the model's default."""

    size: tuple[int, int] = (576, 1024)
    """Output width and height, both exact multiples of 64 pixels."""

    frames: int = 65
    """Generated frame count, at least 9 and equal to 8 times an integer plus 1."""

    fps: int = 24
    """Output frame rate, from 1 to 60 frames per second."""

    seed: int = 0
    """Seed for repeatable inference with the same model and settings."""

    steps: int = 20
    """First-stage denoising steps with classifier-free guidance."""

    refine_steps: int = 3
    """Second-stage refinement steps, from 1 to 3."""

    weights: str | None = None
    """Local LTX checkpoint directory; omit for the pinned q4 package."""

    text_weights: str | None = None
    """Local Gemma directory; omit for the pinned 4-bit text encoder."""

    low_memory: bool = True
    """Release model stages promptly; block streaming always stays enabled."""

    def to_job(self) -> GenerateVideoRequest:
        """Validate the complete shot before reading guides or loading models."""
        return GenerateVideoRequest(
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            output=self.output,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            size=Size(width=self.size[0], height=self.size[1]),
            frames=self.frames,
            fps=self.fps,
            seed=self.seed,
            steps=self.steps,
            refine_steps=self.refine_steps,
            weights=self.weights,
            text_weights=self.text_weights,
            low_memory=self.low_memory,
            overwrite=self.overwrite,
        )
