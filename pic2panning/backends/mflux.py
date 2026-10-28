"""Quantized FLUX generation and editing through Apple's MLX runtime."""

import platform
import sys
from tempfile import TemporaryDirectory
from typing import Protocol, cast

from klogr import get_logger
from klogr.path import path_join
from PIL import Image

from pic2panning.backends.loading import load_dependency
from pic2panning.errors import DependencyError
from pic2panning.schemas.models import GenerateRequest, GenerationModel

logger = get_logger()


class GeneratedImage(Protocol):
    """The image returned by MFLUX's generation methods."""

    image: Image.Image


class FluxModel(Protocol):
    """Framework boundary shared by MFLUX's model variants."""

    def generate_image(self, **parameters: object) -> GeneratedImage:
        """Generate with the selected model's documented parameters."""
        ...


class ModelFactory(Protocol):
    """Construct a selected MFLUX model without importing it eagerly."""

    def __call__(self, **parameters: object) -> FluxModel:
        """Load weights and return an image model."""
        ...


class Flux3Module(Protocol):
    """Public FLUX 3 variants exported by MFLUX."""

    Flux3Klein: ModelFactory
    Flux3KleinEdit: ModelFactory


class Flux1Module(Protocol):
    """A FLUX 1 implementation module."""

    Flux1: ModelFactory


class FluxFillModule(Protocol):
    """MFLUX's FLUX 1 Fill implementation module."""

    Flux1Fill: ModelFactory


class ModelConfigs(Protocol):
    """MFLUX model registry entry points."""

    def from_name(self, model_name: str) -> object:
        """Resolve a model name against MFLUX's registry."""
        ...


class ConfigModule(Protocol):
    """Public model configuration export."""

    ModelConfig: ModelConfigs


def model_factory(request: GenerateRequest) -> ModelFactory:
    """Select an implementation explicitly from the requested model."""
    if request.model == GenerationModel.FLUX3:
        module = cast(
            "Flux3Module", load_dependency("mflux.models.flux3.variants")
        )
        return (
            module.Flux3KleinEdit
            if request.source is not None or request.references
            else module.Flux3Klein
        )
    if request.model == GenerationModel.FLUX1_FILL:
        fill = cast(
            "FluxFillModule",
            load_dependency("mflux.models.flux.variants.fill.flux_fill"),
        )
        return fill.Flux1Fill
    flux = cast(
        "Flux1Module",
        load_dependency("mflux.models.flux.variants.txt2img.flux"),
    )
    return flux.Flux1


def canonical_default_checkpoint(request: GenerateRequest) -> str:
    """Resolve the selected checkpoint without importing a model package."""
    checkpoint_names = {
        GenerationModel.FLUX3: "flux-3-image-klein",
        GenerationModel.FLUX1_SCHNELL: "flux-1-schnell",
        GenerationModel.FLUX1_DEV: "flux-1-dev",
        GenerationModel.FLUX1_FILL: "flux-1-dev-fill",
    }
    return request.weights or (
        f"mflux-community/{checkpoint_names[request.model]}-mflux-"
        f"q{int(request.quantization)}"
    )


def model_parameters(request: GenerateRequest) -> dict[str, object]:
    """Build quantized weight options for a documented MFLUX variant."""
    names = {
        GenerationModel.FLUX3: "flux3-image-klein",
        GenerationModel.FLUX1_SCHNELL: "schnell",
        GenerationModel.FLUX1_DEV: "dev",
        GenerationModel.FLUX1_FILL: "dev-fill",
    }
    configs = cast(
        "ConfigModule",
        load_dependency("mflux.models.common.config.model_config"),
    )
    return {
        "quantize": int(request.quantization),
        "model_path": canonical_default_checkpoint(request),
        "model_config": configs.ModelConfig.from_name(names[request.model]),
        "lora_paths": [lora.path for lora in request.loras] or None,
        "lora_scales": [lora.scale for lora in request.loras] or None,
        "bake_lora": False,
    }


def inference_parameters(request: GenerateRequest) -> dict[str, object]:
    """Translate the model's explicit inference options and default guidance."""
    parameters: dict[str, object] = {
        "seed": request.seed,
        "prompt": request.prompt,
        "num_inference_steps": request.inference_steps,
        "width": request.size.width,
        "height": request.size.height,
    }
    if request.guidance is not None:
        parameters["guidance"] = request.guidance
    elif request.model == GenerationModel.FLUX1_FILL:
        parameters["guidance"] = 30.0
    elif request.model == GenerationModel.FLUX1_DEV:
        parameters["guidance"] = 3.5
    return parameters


class MfluxBackend:
    """Run FLUX with four-bit weights by default on Apple Silicon."""

    def generate(
        self,
        request: GenerateRequest,
        image: Image.Image | None,
        mask: Image.Image | None,
    ) -> Image.Image:
        """Run the chosen model, preserving source geometry and controls."""
        if request.references:
            raise ValueError(
                "Additional references require generate_references"
            )
        return self.generate_references(
            request, (image,) if image is not None else (), mask
        )

    def generate_references(
        self,
        request: GenerateRequest,
        references: tuple[Image.Image, ...],
        mask: Image.Image | None,
    ) -> Image.Image:
        """Run ordered conditioning through the selected MFLUX model."""
        if request.model != GenerationModel.FLUX3 and len(references) > 1:
            raise ValueError("Multiple conditioning images require FLUX 3")
        if sys.platform != "darwin" or platform.machine() != "arm64":
            message = "The MFLUX backend requires an Apple Silicon Mac."
            raise DependencyError(message)
        factory = model_factory(request)
        logger.info(
            f"Loading {request.model.value} with "
            f"{int(request.quantization)}-bit weights on MLX"
        )
        model = factory(**model_parameters(request))
        with TemporaryDirectory(prefix="pic2panning-model-") as directory:
            parameters = inference_parameters(request)
            if references:
                image_paths = []
                for index, image in enumerate(references):
                    image_path = path_join(directory, f"source-{index}.png")
                    image.save(image_path)
                    image_paths.append(image_path)
                if request.model == GenerationModel.FLUX3:
                    parameters["image_paths"] = image_paths
                else:
                    parameters["image_path"] = image_paths[0]
                    if request.model != GenerationModel.FLUX1_FILL:
                        parameters["image_strength"] = (
                            request.strength
                            if request.strength is not None
                            else 0.7
                        )
            if (
                request.model == GenerationModel.FLUX1_FILL
                and mask is not None
            ):
                mask_path = path_join(directory, "mask.png")
                mask.save(mask_path)
                parameters["masked_image_path"] = mask_path
            return model.generate_image(**parameters).image.convert("RGB")
