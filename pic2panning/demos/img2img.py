"""Image to image demo using Flux model."""

from pathlib import Path

import tyro
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from diffusers.utils.pil_utils import make_image_grid
from pydantic.dataclasses import dataclass

from pic2panning.utils.generative.flux import FluxModel
from pic2panning.utils.logger import get_logger
from pic2panning.utils.process import read_image

logger = get_logger()


@dataclass
class Options:
    """Options for the image to image demo."""

    input: str = "DSC09552.JPG"
    """ A string representing the URL of the image to be processed. Default is 'https://images.pexels.com/photos/29188556/pexels-photo-29188556/free-photo-of-stunning-sunset-over-mulafossur-waterfall-faroe-islands.jpeg'."""

    prompt: str = "A landscape"
    """ A string representing the prompt for the Flux model. Default is 'A landscape'."""

    strength: float = 0.3
    """ A float representing the strength of the prompt for the Flux model. Default is 0.2."""

    output_file: str = "assets/demo_flux.png"
    """ A string representing the output file path. Default is 'assets/output_flux.png'."""

    def __post_init__(self) -> None:
        """Create the output directory if it does not exist."""
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)


def demo(opts: Options) -> None:
    """Process an image using a Flux model and save the output.

    This function takes an image URL as input, runs a Flux model on the image,

    Arguments:
        opts (Options): An instance of the Options class containing the input image URL, prompt, and strength.

    Returns:
        None: This function does not return any value, but saves the output as an image file.

    """
    flux_model = FluxModel()
    flux_model.load_model(FluxImg2ImgPipeline)
    landscape = read_image(opts.input)
    output = flux_model.inference(
        image=landscape,
        prompt=opts.input,
        strength=opts.strength,
        guidance_scale=3.5,
    )
    output_file = opts.output_file
    make_image_grid(
        [landscape] + [output],
        rows=1,
        cols=2,
    ).save(output_file)
    logger.success(
        f"Inference completed. Saving output to {Path(output_file).resolve()}"
    )


if __name__ == "__main__":
    """Run the demo."""
    opts = tyro.cli(Options)
    demo(opts)
