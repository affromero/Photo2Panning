"""Crop contextual model inputs and composite repairs into exact selections."""

import math
import tempfile
from contextlib import ExitStack

from klogr.path import path_join
from PIL import Image, ImageChops, ImageFilter, ImageOps

from pic2panning.backends.contracts import ImageBackend
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.repair import RepairRequest
from pic2panning.schemas.editing import Crop, Resize
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.models import GenerateRequest
from pic2panning.tools.editing import apply_operation
from pic2panning.tools.io import atomic_output, read_image
from pic2panning.tools.models import generate


def repair_inputs(request: RepairRequest) -> tuple[str, ...]:
    """List media inputs that must survive repair output publication."""
    return (request.source,) + ((request.mask,) if request.mask else ())


def selection_mask(request: RepairRequest, image: Image.Image) -> Image.Image:
    """Intersect a full-source mask with the region and feather inward."""
    region = request.region
    if region.right > image.width or region.bottom > image.height:
        raise ValueError("Repair region exceeds source image dimensions")
    with ExitStack() as images:
        rectangle = images.enter_context(Image.new("L", image.size))
        rectangle.paste(
            255, (region.left, region.top, region.right, region.bottom)
        )
        if request.mask is not None:
            mask = images.enter_context(read_image(request.mask))
            if mask.size != image.size:
                raise ValueError("Repair mask dimensions differ from source")
            gray = images.enter_context(mask.convert("L"))
            selection = images.enter_context(
                ImageChops.multiply(rectangle, gray)
            )
        else:
            selection = rectangle
        if selection.getbbox() is None:
            raise ValueError("Repair selection contains no selected pixels")
        if not request.feather:
            return selection.copy()
        padding = math.ceil(request.feather * 3)
        support = images.enter_context(selection.point([0] + [255] * 255))
        expanded = images.enter_context(
            ImageOps.expand(support, border=padding, fill=0)
        )
        blurred = images.enter_context(
            expanded.filter(ImageFilter.GaussianBlur(request.feather))
        )
        restored = images.enter_context(
            apply_operation(
                blurred,
                Crop(
                    left=padding,
                    top=padding,
                    right=padding + image.width,
                    bottom=padding + image.height,
                ),
            )
        )
        ramp = images.enter_context(
            restored.point([max(0, 2 * value - 255) for value in range(256)])
        )
        feathered = ImageChops.multiply(selection, ramp)
        if feathered.getbbox() is None:
            feathered.close()
            raise ValueError("Feather removes the entire repair selection")
        return feathered


def contextual_crop(request: RepairRequest, size: Size) -> Crop:
    """Expand the selected region while keeping context inside the source."""
    return Crop(
        left=max(0, request.region.left - request.context),
        top=max(0, request.region.top - request.context),
        right=min(size.width, request.region.right + request.context),
        bottom=min(size.height, request.region.bottom + request.context),
    )


def repair(
    request: RepairRequest, backend: ImageBackend | None = None
) -> JobResult:
    """Run canonical generation on context and publish only selected pixels."""
    with (
        ExitStack() as images,
        atomic_output(
            request.output,
            repair_inputs(request),
            overwrite=request.overwrite,
        ) as output,
        tempfile.TemporaryDirectory(prefix="pic2panning-repair-") as directory,
    ):
        image = images.enter_context(
            read_image(request.source, preserve_alpha=True)
        )
        selection = images.enter_context(selection_mask(request, image))
        crop = contextual_crop(
            request, Size(width=image.width, height=image.height)
        )
        context = images.enter_context(apply_operation(image, crop))
        scale = request.max_edge / max(context.size)
        content_size = Size(
            width=max(1, round(context.width * scale)),
            height=max(1, round(context.height * scale)),
        )
        rgb_context = images.enter_context(context.convert("RGB"))
        resized = images.enter_context(
            apply_operation(rgb_context, Resize(size=content_size))
        )
        model_size = Size(
            width=math.ceil(content_size.width / 16) * 16,
            height=math.ceil(content_size.height / 16) * 16,
        )
        model_image = images.enter_context(
            Image.new("RGB", model_size.as_tuple, "black")
        )
        model_image.paste(resized, (0, 0))
        source = path_join(directory, "context.png")
        generated_path = path_join(directory, "generated.png")
        model_image.save(source)
        generate(
            GenerateRequest(
                source=source,
                output=generated_path,
                prompt=request.prompt,
                size=model_size,
                seed=request.seed,
                steps=request.steps,
                quantization=request.quantization,
                weights=request.weights,
            ),
            backend,
        )
        generated = images.enter_context(read_image(generated_path))
        content = images.enter_context(
            apply_operation(
                generated,
                Crop(
                    left=0,
                    top=0,
                    right=content_size.width,
                    bottom=content_size.height,
                ),
            )
        )
        restored = images.enter_context(
            apply_operation(
                content,
                Resize(size=Size(width=context.width, height=context.height)),
            )
        )
        patch = images.enter_context(restored.convert(image.mode))
        patch_selection = images.enter_context(
            apply_operation(selection, crop)
        )
        result = images.enter_context(image.copy())
        result.paste(patch, (crop.left, crop.top), patch_selection)
        result.save(output)
        return JobResult(
            outputs=(request.output,),
            width=image.width,
            height=image.height,
            details="FLUX 3 context-crop repair",
        )
