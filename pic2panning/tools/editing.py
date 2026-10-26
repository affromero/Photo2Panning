"""Deterministic image editing shared by Python and JSON callers."""

from PIL import Image, ImageEnhance, ImageOps

from pic2panning.schemas.editing import (
    Adjust,
    Axis,
    Crop,
    EditOperation,
    EditRequest,
    Flip,
    Resize,
    Rotate,
)
from pic2panning.schemas.jobs import JobResult
from pic2panning.tools.io import atomic_output, read_image


def apply_operation(
    image: Image.Image, operation: EditOperation
) -> Image.Image:
    """Apply one validated edit, rejecting crops outside the current image."""
    match operation:
        case Resize(size=size):
            return image.resize(size.as_tuple, Image.Resampling.LANCZOS)
        case Crop(left=left, top=top, right=right, bottom=bottom):
            if right > image.width or bottom > image.height:
                raise ValueError(
                    "Crop bounds exceed the current image dimensions"
                )
            return image.crop((left, top, right, bottom))
        case Rotate(degrees=degrees):
            return image.rotate(
                degrees, resample=Image.Resampling.BICUBIC, expand=True
            )
        case Flip(axis=axis):
            return (
                ImageOps.mirror(image)
                if axis == Axis.HORIZONTAL
                else ImageOps.flip(image)
            )
        case Adjust(
            brightness=brightness, contrast=contrast, saturation=saturation
        ):
            adjusted = ImageEnhance.Brightness(image).enhance(brightness)
            adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)
            return ImageEnhance.Color(adjusted).enhance(saturation)


def edit(request: EditRequest) -> JobResult:
    """Apply ordered edits and publish the result without changing the input."""
    with atomic_output(
        request.output, [request.source], overwrite=request.overwrite
    ) as temporary:
        image = read_image(request.source)
        try:
            for operation in request.operations:
                previous = image
                image = apply_operation(previous, operation)
                previous.close()
            image.save(temporary)
            return JobResult(
                outputs=(request.output,),
                width=image.width,
                height=image.height,
            )
        finally:
            image.close()
