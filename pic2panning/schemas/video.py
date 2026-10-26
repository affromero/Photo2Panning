"""Joint endpoint conditioning for local generative video."""

from typing import Annotated, Literal, Self

from klogr.path import get_suffix
from pydantic import Field, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    PydanticBaseModel,
    Size,
)


class GenerateVideoRequest(PydanticBaseModel):
    """Generate one continuous shot guided by its first and last images."""

    kind: Literal["generate-video"] = "generate-video"
    """Select local MLX video generation in a JSON job."""

    first_frame: NonEmptyString
    """First image path or HTTP URL, oriented and center-cropped to fit."""

    last_frame: NonEmptyString
    """Last image path or HTTP URL, oriented and center-cropped to fit."""

    output: NonEmptyString
    """Local MP4 destination; generation produces a silent video."""

    prompt: NonEmptyString
    """Describe one continuous camera movement between the two images."""

    negative_prompt: NonEmptyString | None = None
    """Optional unwanted content or motion; null keeps the model's default."""

    size: Size = Size(width=576, height=1024)
    """Output width and height, both exact multiples of 64 pixels."""

    frames: Annotated[int, Field(ge=9, strict=True)] = 65
    """Generated frame count, at least 9 and exactly 8 times an integer plus 1."""

    fps: Annotated[int, Field(ge=1, le=60, strict=True)] = 24
    """Output frames per second; duration is frames divided by fps."""

    seed: Annotated[int, Field(ge=0, strict=True)] = 0
    """Seed for repeatable inference with the same models and settings."""

    steps: Annotated[int, Field(ge=1, strict=True)] = 20
    """First-stage denoising steps with classifier-free guidance."""

    refine_steps: Annotated[int, Field(ge=1, le=3, strict=True)] = 3
    """Second-stage refinement steps, from 1 to the model's full 3-step table."""

    weights: NonEmptyString | None = None
    """Local LTX checkpoint directory; null uses the pinned public q4 package."""

    text_weights: NonEmptyString | None = None
    """Local Gemma directory; null uses the pinned public 4-bit text encoder."""

    low_memory: bool = True
    """Release model stages promptly; transformer block streaming stays enabled."""

    overwrite: bool = False
    """Allow replacing an existing output; source aliases stay protected."""

    @model_validator(mode="after")
    def validate_video_options(self) -> Self:
        """Reject options the model would otherwise round or silently limit."""
        if get_suffix(self.output).lower() != ".mp4":
            raise ValueError("Generated video output must end in .mp4")
        if self.size.width % 64 or self.size.height % 64:
            raise ValueError("Video width and height must be multiples of 64")
        if self.frames % 8 != 1:
            raise ValueError(
                "Video frames must equal 8 times an integer plus 1"
            )
        return self
