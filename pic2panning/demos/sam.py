"""Demo for grounding DINO model with a given query."""

from pathlib import Path

import tyro
from diffusers.utils.pil_utils import make_image_grid

from pic2panning.utils.generative.sam import GroundingDino
from pic2panning.utils.logger import get_logger
from pic2panning.utils.process import read_image

logger = get_logger()


def demo(
    image: str = "DSC09552.JPG",
    query: str = "person.",
    output: str = "asset/demo_sam.jpg",
) -> None:
    """Process an image using a grounding DINO model with a given query and.

        save the debug output.

    This function takes an image URL and a query string as input, runs a
        grounding DINO model on the image with the given query, and saves
        the debug output as an image file.

    Arguments:
        image (str): A string representing the URL of the image to be
            processed.
            Default is
            'http://images.cocodataset.org/val2017/000000006471.jpg'.
        query (str): A string representing the query prompt for the
            grounding DINO model.
            Default is 'person'.
        output (str): A string representing the output file path.
            Default is 'assets/demo_sam.jpg'.

    Returns:
        None: This function does not return any value, but saves the output
            as an image file.

    Example:
        >>> run_dino_model("http://example.com/image.jpg", "cat")

    Note:
        The image file is saved in the current working directory.

    """
    image_pil = read_image(image)
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
    make_image_grid(
        [image_pil] + [image_out_pil],
        rows=1,
        cols=2,
    ).save(output)
    logger.success(f"Saved to {Path(output).resolve()}")


if __name__ == "__main__":
    tyro.cli(demo)
