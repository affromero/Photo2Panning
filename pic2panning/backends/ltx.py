"""Native MLX LTX video generation with jointly conditioned endpoints."""

import platform
import sys
from types import ModuleType
from typing import Protocol, cast

from klogr import get_logger
from klogr.path import path_expanduser, path_is_dir, path_resolve

from pic2panning.backends.loading import load_dependency
from pic2panning.errors import DependencyError
from pic2panning.schemas.video import GenerateVideoRequest

logger = get_logger()

VIDEO_REPOSITORY = "dgrauet/ltx-2.3-mlx-q4"
VIDEO_REVISION = "56a5866d638ecfe37c54d348e88938235185c2d4"
VIDEO_FILES = (
    "config.json",
    "embedded_config.json",
    "quantize_config.json",
    "split_model.json",
    "connector.safetensors",
    "transformer-dev.safetensors",
    "transformer-distilled.safetensors",
    "vae_encoder.safetensors",
    "vae_decoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
    "spatial_upscaler_x2_v1_1_config.json",
)
TEXT_REPOSITORY = "mlx-community/gemma-3-12b-it-4bit"
TEXT_REVISION = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
TEXT_FILES = ("*.json", "*.safetensors", "tokenizer.model")


class VideoBackend(Protocol):
    """Write a model-generated shot from two normalized endpoint images."""

    def generate(
        self,
        request: GenerateVideoRequest,
        first_frame: str,
        last_frame: str,
        output: str,
    ) -> None:
        """Write the requested silent MP4 or surface the inference failure."""
        ...


class SnapshotModule(Protocol):
    """The public Hugging Face download boundary used for pinned defaults."""

    def snapshot_download(
        self, repo_id: str, *, revision: str, allow_patterns: list[str]
    ) -> str:
        """Resolve selected files into the ordinary Hugging Face cache."""
        ...


class VideoPipeline(Protocol):
    """Public inference surface exposed by the pinned LTX pipeline."""

    generate_audio: bool

    def generate_and_save(self, **parameters: object) -> str:
        """Generate the jointly conditioned clip to the supplied destination."""
        ...


class PipelineFactory(Protocol):
    """Construct the selected external pipeline from local checkpoint paths."""

    def __call__(self, **parameters: object) -> VideoPipeline:
        """Load the two-stage MLX pipeline with explicit memory controls."""
        ...


class PipelineModule(Protocol):
    """The native LTX package's public two-stage export."""

    TI2VidTwoStagesPipeline: PipelineFactory


class GuideFactory(Protocol):
    """Construct image guides using the external package's canonical type."""

    def guide(
        self, path: str, frame_idx: int, strength: float, *, crf: int
    ) -> object:
        """Anchor a pixel-frame index without JPEG preprocessing loss."""
        ...


class ConditioningModule(Protocol):
    """The external package's canonical image-conditioning export."""

    ImageConditioningInput: GuideFactory


def video_dependency(module: str) -> ModuleType:
    """Explain the video extra when an optional framework is unavailable."""
    try:
        return load_dependency(module)
    except DependencyError as error:
        message = (
            f"Local video generation could not load {module!r}. "
            "Run `uv sync --extra video` in Photo2Panning on this Mac. "
            f"Dependency failure: {error.__cause__}"
        )
        raise DependencyError(message) from error


def checkpoint(
    local: str | None, repository: str, revision: str, files: tuple[str, ...]
) -> str:
    """Use an explicit local directory or fetch only pinned required files."""
    if local is not None:
        directory = path_resolve(path_expanduser(local))
        if not path_is_dir(directory):
            raise FileNotFoundError(
                f"Model checkpoint must be a local directory: {local}"
            )
        return directory
    hub = cast("SnapshotModule", video_dependency("huggingface_hub"))
    return hub.snapshot_download(
        repo_id=repository, revision=revision, allow_patterns=list(files)
    )


class LtxBackend:
    """Generate a single shot between endpoint guides on Apple Silicon."""

    def generate(
        self,
        request: GenerateVideoRequest,
        first_frame: str,
        last_frame: str,
        output: str,
    ) -> None:
        """Run the fixed two-stage model with explicit local memory settings."""
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise DependencyError(
                "The LTX backend requires an Apple Silicon Mac."
            )
        pipelines = cast(
            "PipelineModule", video_dependency("ltx_pipelines_mlx")
        )
        conditioning = cast(
            "ConditioningModule",
            video_dependency("ltx_pipelines_mlx.utils.types"),
        )
        weights = checkpoint(
            request.weights, VIDEO_REPOSITORY, VIDEO_REVISION, VIDEO_FILES
        )
        text_weights = checkpoint(
            request.text_weights, TEXT_REPOSITORY, TEXT_REVISION, TEXT_FILES
        )
        logger.info("Generating video locally with LTX endpoint guides on MLX")
        pipeline = pipelines.TI2VidTwoStagesPipeline(
            model_dir=weights,
            gemma_model_id=text_weights,
            low_memory=request.low_memory,
            low_ram_streaming=True,
        )
        pipeline.generate_audio = False
        guides = [
            conditioning.ImageConditioningInput.guide(
                first_frame, 0, 1.0, crf=0
            ),
            conditioning.ImageConditioningInput.guide(
                last_frame, request.frames - 1, 1.0, crf=0
            ),
        ]
        controls: dict[str, object] = {}
        if request.negative_prompt is not None:
            controls["negative_prompt"] = request.negative_prompt
        pipeline.generate_and_save(
            prompt=request.prompt,
            output_path=output,
            height=request.size.height,
            width=request.size.width,
            num_frames=request.frames,
            frame_rate=float(request.fps),
            seed=request.seed,
            stage1_steps=request.steps,
            stage2_steps=request.refine_steps,
            stg_scale=0.0,
            images=guides,
            refine_audio=True,
            **controls,
        )
