"""Local Apple Silicon model requests."""

from enum import IntEnum, StrEnum
from typing import Annotated, Literal, Self

from klogr.path import get_suffix
from pydantic import Field, field_validator, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    PositiveFloat,
    PydanticBaseModel,
    Size,
    validate_image_output,
)


class GenerationModel(StrEnum):
    """Explicit model families supported by the MLX backend."""

    FLUX3 = "flux3"
    FLUX1_SCHNELL = "flux1-schnell"
    FLUX1_DEV = "flux1-dev"
    FLUX1_FILL = "flux1-fill"


class Quantization(IntEnum):
    """Weight precision in bits."""

    Q4 = 4
    Q6 = 6
    Q8 = 8


class Device(StrEnum):
    """Explicit CPU execution for text-guided segmentation."""

    CPU = "cpu"


class Lora(PydanticBaseModel):
    """A local LoRA checkpoint and its contribution to the model."""

    path: NonEmptyString
    """Local LoRA checkpoint trained for the selected model architecture."""

    scale: float = 1
    """Adapter contribution; 0 disables its effect and 1 uses full strength."""


class GenerateRequest(PydanticBaseModel):
    """Generate or edit an image using quantized models through MLX."""

    kind: Literal["generate"] = "generate"
    """Select local MLX generation or prompted editing in a JSON job."""

    source: NonEmptyString | None = None
    """Reference image path or HTTP URL; null requests text-to-image."""

    output: NonEmptyString
    """PNG, JPEG, or WebP destination; masked edits require lossless PNG."""

    prompt: NonEmptyString
    """Complete caption of the desired image, including content to retain."""

    model: GenerationModel = GenerationModel.FLUX3
    """Explicit FLUX family; distilled FLUX 3 Klein is the default."""

    weights: NonEmptyString | None = None
    """MFLUX checkpoint path or repo; null selects community weights."""

    quantization: Quantization = Quantization.Q4
    """Weight precision in bits; 4-bit is the smallest supported precision."""

    seed: Annotated[int, Field(ge=0, strict=True)] = 0
    """Random seed to repeat inference with the same model and settings."""

    steps: Annotated[int, Field(gt=0, strict=True)] | None = None
    """Null uses 16 FLUX 3 steps, 4 schnell steps, or 28 dev/fill steps."""

    size: Size = Size(width=512, height=512)
    """Output pixels in multiples of 16; masked edits must match the source."""

    strength: Annotated[float, Field(gt=0, le=1)] | None = None
    """FLUX 1 schnell/dev image strength; null uses the backend default 0.7."""

    guidance: PositiveFloat | None = None
    """Null uses 3.5 for dev or 30 for fill; distilled models reject guidance."""

    mask: NonEmptyString | None = None
    """White edits, black retains pixels; size must match source and output."""

    loras: tuple[Lora, ...] = ()
    """Model-compatible LoRA checkpoints and their individual scales."""

    overwrite: bool = False
    """Allow replacing an existing output; input aliases stay protected."""

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str) -> str:
        """Require a supported output format before loading weights."""
        return validate_image_output(value)

    @model_validator(mode="after")
    def validate_model_options(self) -> Self:
        """Reject unsupported options before loading model weights."""
        if self.size.width % 16 or self.size.height % 16:
            raise ValueError("Model width and height must be multiples of 16")
        if self.mask and not self.source:
            raise ValueError("A mask requires a source image")
        if self.mask and get_suffix(self.output).lower() != ".png":
            raise ValueError("Masked image output must end in .png")
        if self.model == GenerationModel.FLUX1_FILL and (
            not self.source or not self.mask
        ):
            raise ValueError("FLUX1 Fill requires source and mask")
        if self.strength is not None and (
            not self.source
            or self.model
            in {GenerationModel.FLUX3, GenerationModel.FLUX1_FILL}
        ):
            raise ValueError("Strength requires FLUX1 schnell/dev img2img")
        if self.guidance is not None and self.model in {
            GenerationModel.FLUX3,
            GenerationModel.FLUX1_SCHNELL,
        }:
            raise ValueError("Guidance is unsupported by distilled models")
        return self

    @property
    def inference_steps(self) -> int:
        """Resolve the documented model-specific count without changing input."""
        if self.steps is not None:
            return self.steps
        return {
            GenerationModel.FLUX3: 16,
            GenerationModel.FLUX1_SCHNELL: 4,
            GenerationModel.FLUX1_DEV: 28,
            GenerationModel.FLUX1_FILL: 28,
        }[self.model]


class MaskRequest(PydanticBaseModel):
    """Detect text-described objects and produce full-resolution masks."""

    kind: Literal["mask"] = "mask"
    """Select text-guided object segmentation in a JSON job."""

    source: NonEmptyString
    """Image path or HTTP URL; masks use the EXIF-oriented dimensions."""

    output: NonEmptyString
    """Local PNG mask path; separate masks append _000, _001, and so on."""

    prompt: NonEmptyString
    """Object labels to detect; separate multiple labels with periods."""

    detector: NonEmptyString = "IDEA-Research/grounding-dino-tiny"
    """Grounding DINO checkpoint; the default uses the small Tiny variant."""

    segmenter: NonEmptyString = "facebook/sam2.1-hiera-tiny"
    """SAM 2 checkpoint; the default uses the small SAM 2.1 Tiny variant."""

    device: Device = Device.CPU
    """CPU execution for the small segmentation models."""

    threshold: Annotated[float, Field(gt=0, le=1)] = 0.3
    """Minimum object-detection confidence accepted for segmentation."""

    padding: Annotated[int, Field(ge=0, le=128, strict=True)] = 0
    """Expand each selection by this many pixels along both image axes."""

    merge: bool = True
    """Union masks into one output; false saves separate object masks."""

    overwrite: bool = False
    """Allow replacing existing masks; input aliases stay protected."""

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str) -> str:
        """Require lossless PNG masks before loading model weights."""
        return validate_image_output(value, mask_only=True)
