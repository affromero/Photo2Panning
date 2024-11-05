"""A module to handle the FLUX model for image generation."""

from typing import Any, Literal

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
)
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
from huggingface_hub import hf_hub_download
from optimum import quanto
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from transformers import T5EncoderModel

from pic2panning.utils.logger import get_logger

logger = get_logger()


@dataclass(config=ConfigDict(extra="forbid"))
class LoraOptions:
    """A class to handle LoRa options for the Flux model."""

    repo_name: str
    """Model name."""

    lora_scale: float
    """Lora scale."""

    ckpt_name: str | None = None
    """Adapter name."""

    def add_lora_to_pipe(self, pipe: ConfigMixin) -> None:
        """Add Lora weights to a given pipe object and fuse them together.

        This method takes a ConfigMixin object as an argument, adds Lora
            weights to it, and fuses the Lora weights with the pipe object.

        Arguments:
            self (LoraOptions): The instance of the LoraOptions class.
            pipe (ConfigMixin): An object of type ConfigMixin to which Lora
                weights will be added.

        Returns:
            None: This method does not return any value, it modifies the
                pipe object in-place.

        Example:
            >>> lora_options = LoraOptions()
            >>> pipe = ConfigMixin()
            >>> lora_options.add_lora_weights(pipe)

        Note:
            The Lora weights are fused with the pipe object, modifying it
                directly. Ensure to keep a copy of the original pipe object
                if needed.

        """
        if self.ckpt_name is not None:
            pipe.load_lora_weights(
                hf_hub_download(
                    repo_id=self.repo_name, filename=self.ckpt_name
                )
            )
        else:
            pipe.load_lora_weights(self.repo_name)
        pipe.fuse_lora(lora_scale=self.lora_scale)
        # pipe.unload_lora_weights()


@dataclass(config=ConfigDict(extra="forbid"))
class FluxModel:
    """A class to load and run the FLUX model."""

    def load_model(
        self,
        module: FluxImg2ImgPipeline | FluxInpaintPipeline,
        quantization: Literal["int4", "float8", "4bit"] | None = "float8",
        mode: Literal["dev", "schnell"] = "dev",
        dtype: torch.dtype = torch.bfloat16,
        with_lora: LoraOptions | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a model for the FluxModel class with various options.

        This function takes a module, quantization method, mode of operation,

        Arguments:
            self (FluxModel): The instance of the FluxModel class.
            module (FFluxImg2ImgPipeline | FluxInpaintPipeline):
                An instance of one of the Flux pipeline classes.
            quantization (str, optional):
                The quantization method to be used. Defaults to 'float8'.
            mode (str, optional):
                The mode of operation, either 'dev' or 'schnell'. Defaults
                to 'dev'.
            dtype (torch.dtype, optional):
                The data type to be used by torch. Defaults to
                torch.bfloat16.
            with_lora (LoraOptions | None, optional):
                An instance of LoraOptions if LoRa options are to be used,
                else None. Defaults to None.
            **kwargs:
                Additional keyword arguments.

        Returns:
            None: This function does not return any value.

        Example:
            >>> load_model(module, controlnet_model, quantization='float8',
                mode='dev', dtype=torch.bfloat16, with_lora=None)

        Note:
            This function modifies the FluxModel instance in-place by
                loading the specified model.

        """
        flux_repo = f"black-forest-labs/FLUX.1-{mode}"
        # flux_repo = "Freepik/flux.1-lite-8B-alpha"
        use_safetensors = kwargs.pop("use_safetensors", True)

        # download the model from the repo if it does not exist
        pipe = module.from_pretrained(
            flux_repo,
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        logger.log(
            f"Loading model from {flux_repo} with {quantization=} and {dtype=}"
        )
        if quantization is not None:
            transformer = FluxTransformer2DModel.from_single_file(
                hf_hub_download(
                    repo_id="Kijai/flux-fp8",
                    filename="flux1-dev-fp8.safetensors",
                ),
                torch_dtype=dtype,
                use_safetensors=use_safetensors,
            )
            quanto.quantize(transformer, weights=quanto.qfloat8)
            quanto.freeze(transformer)
            text_encoder_2 = T5EncoderModel.from_pretrained(
                flux_repo,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                use_safetensors=use_safetensors,
            )
            quanto.quantize(text_encoder_2, weights=quanto.qfloat8)
            quanto.freeze(text_encoder_2)

        if with_lora is not None:
            with_lora.add_lora_to_pipe(pipe)

        if quantization is not None:
            pipe.text_encoder_2 = text_encoder_2
            pipe.transformer = transformer
        self.pipe = pipe.to("cuda")

    def inference(self, **kwargs: Any) -> Image.Image:
        """Perform inference on the model.

        Arguments:
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The output of the model.

        Example:
            >>> inference(prompt="A cat", num_images=1)

        Note:
            This function performs inference on the model and returns the
                output.

        """
        # resize to 1024x1024
        original_size = kwargs["image"].size
        kwargs["image"] = kwargs["image"].resize((1024, 1024))
        if isinstance(self.pipe, FluxInpaintPipeline):
            # resize to 1024x1024
            kwargs["mask_image"] = kwargs["mask_image"].resize((1024, 1024))
        output = self.pipe(**kwargs).images[0]
        if "image" in kwargs:
            # resize to original size
            output = output.resize(original_size)
        return output
