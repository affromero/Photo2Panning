"""A demo script to run the Flux model on an image and save the output."""

from pathlib import Path

import tyro
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
from diffusers.utils.pil_utils import make_image_grid
from PIL import Image
from pydantic.dataclasses import dataclass

from pic2panning.utils.generative.flux import FluxModel
from pic2panning.utils.generative.sam import GroundingDino
from pic2panning.utils.logger import get_logger
from pic2panning.utils.process import read_image

logger = get_logger()


@dataclass
class Options:
    """Options for the image to image demo."""

    input: str = "DSC09552.JPG"
    """ A string representing the URL of the image to be processed. Default is 'https://images.pexels.com/photos/29188556/pexels-photo-29188556/free-photo-of-stunning-sunset-over-mulafossur-waterfall-faroe-islands.jpeg'."""

    query: str = "people."
    """ A string representing the query prompt for the grounding DINO model. Default is 'people'."""

    prompt: str = "A landscape of NYC. Beautiful sunset."
    """ A string representing the prompt for the Flux model. Default is 'A landscape'."""

    strength: float = 0.9
    """ A float representing the strength of the prompt for the Flux model. Default is 0.2."""

    output_file = "assets/demo_inpaint.png"
    """ A string representing the output file path. Default is 'assets/output_flux.png'."""


def get_sam(image_pil: Image.Image, query: str) -> Image.Image:
    """Get the SAM mask for the given image and query.

    This function takes an image and a query as input, runs a grounding DINO model on the image with the given query, and returns the SAM mask.

    Arguments:
        image_pil (Image.Image): An instance of the PIL Image class representing the input image.
        query (str): A string representing the query prompt for the grounding DINO model.

    Returns:
        Image.Image: An instance of the PIL Image class representing the SAM mask.

    Example:
        >>> get_sam(Image.open("image.jpg"), "cat")

    """
    opts = GroundingDino(
        dino_model_id="IDEA-Research/grounding-dino-base",
        sam_model_id="facebook/sam2.1-hiera-large",
        device="cuda",
    )
    mask_list = opts.run(
        image=image_pil,
        prompt=query,
        box_threshold=0.3,
        merge_masks=True,
        verbose=False,
    )
    image_out_pil = mask_list[0]
    return image_out_pil


def demo(opts: Options) -> None:
    """Process an image using a Flux model and save the output.

    This function takes an image URL as input, runs a Flux model on the image,

    Arguments:
        opts (Options): An instance of the Options class containing the input image URL, prompt, and strength.

    Returns:
        None: This function does not return any value, but saves the output as an image file.

    """
    flux_model = FluxModel()
    flux_model.load_model(FluxInpaintPipeline)
    landscape = read_image(opts.input)
    sam_mask = get_sam(landscape, opts.query).convert("RGB")
    output = flux_model.inference(
        image=landscape,
        mask_image=sam_mask,
        prompt=opts.input,
        strength=opts.strength,
        guidance_scale=3.5,
    )
    output_file = opts.output_file
    make_image_grid(
        [landscape] + [sam_mask] + [output],
        rows=1,
        cols=3,
    ).save(output_file)
    logger.success(
        f"Inference completed. Saving output to {Path(output_file).resolve()}"
    )


if __name__ == "__main__":
    """Run the demo."""
    opts = tyro.cli(Options)
    demo(opts)
