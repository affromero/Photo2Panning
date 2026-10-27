"""Validated local model operations shared by the Python API and CLI."""

from contextlib import ExitStack

from klogr.path import get_suffix, path_dirname, path_join, path_stem
from PIL import Image, ImageChops, ImageFilter

from pic2panning.backends.contracts import ImageBackend, MaskBackend
from pic2panning.backends.mflux import MfluxBackend
from pic2panning.backends.segmentation import GroundedSamBackend
from pic2panning.errors import MediaError
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.tools.io import (
    atomic_output,
    read_image,
    require_file,
    validate_destination,
)


def generate(
    request: GenerateRequest, backend: ImageBackend | None = None
) -> JobResult:
    """Generate or edit an image with a selected local quantized model."""
    sources = tuple(
        source
        for source in (request.source, request.mask)
        if source is not None
    ) + tuple(lora.path for lora in request.loras)
    with (
        ExitStack() as images,
        atomic_output(
            request.output, sources, overwrite=request.overwrite
        ) as output,
    ):
        for lora in request.loras:
            require_file(lora.path)
        image = (
            images.enter_context(
                read_image(
                    request.source, preserve_alpha=request.mask is not None
                )
            )
            if request.source
            else None
        )
        mask_image = (
            images.enter_context(
                images.enter_context(read_image(request.mask)).convert("L")
            )
            if request.mask
            else None
        )
        if (
            mask_image is not None
            and image is not None
            and mask_image.size != image.size
        ):
            message = (
                "The mask and source image must have the same dimensions."
            )
            raise MediaError(message)
        if (
            mask_image is not None
            and image is not None
            and image.size != request.size.as_tuple
        ):
            message = (
                "Masked edits require output dimensions to match the source "
                "so unmasked pixels remain unchanged."
            )
            raise MediaError(message)
        selected = backend if backend is not None else MfluxBackend()
        reference = (
            images.enter_context(image.convert("RGB"))
            if image is not None and image.mode != "RGB"
            else image
        )
        generated = images.enter_context(
            images.enter_context(
                selected.generate(request, reference, mask_image)
            ).convert("RGB")
        )
        if generated.size != request.size.as_tuple:
            message = "The model returned an image with unexpected dimensions."
            raise MediaError(message)
        if mask_image is not None and image is not None:
            if image.mode != generated.mode:
                generated = images.enter_context(generated.convert(image.mode))
            generated = images.enter_context(
                Image.composite(generated, image, mask_image)
            )
        generated.save(output)
        return JobResult(
            outputs=(request.output,),
            width=generated.width,
            height=generated.height,
            details=f"{request.model.value}, {int(request.quantization)}-bit weights",
        )


def mask(
    request: MaskRequest, backend: MaskBackend | None = None
) -> JobResult:
    """Save separate or merged white-on-black masks for a text query."""
    if request.merge:
        validate_destination(
            request.output, (request.source,), overwrite=request.overwrite
        )
    with ExitStack() as images:
        image = images.enter_context(read_image(request.source))
        selected = backend if backend is not None else GroundedSamBackend()
        masks = tuple(
            images.enter_context(result)
            for result in selected.masks(request, image)
        )
        if not masks:
            message = f"No objects matched the mask prompt {request.prompt!r}."
            raise MediaError(message)
        if any(result.size != image.size for result in masks):
            message = "The model returned a mask with unexpected dimensions."
            raise MediaError(message)
        normalized = tuple(
            images.enter_context(result.convert("L")) for result in masks
        )
        if any(result.getbbox() is None for result in normalized):
            message = (
                "The segmentation model returned an empty mask for a detected "
                "object. Adjust the prompt or select a different model."
            )
            raise MediaError(message)
        if request.padding:
            normalized = tuple(
                images.enter_context(
                    selection.filter(
                        ImageFilter.MaxFilter(2 * request.padding + 1)
                    )
                )
                for selection in normalized
            )
        outputs: tuple[str, ...]
        if request.merge:
            merged = images.enter_context(Image.new("L", image.size))
            for selection in normalized:
                combined = images.enter_context(
                    ImageChops.lighter(merged, selection)
                )
                merged.close()
                merged = combined
            normalized = (merged,)
            outputs = (request.output,)
        else:
            suffix = get_suffix(request.output)
            outputs = tuple(
                path_join(
                    path_dirname(request.output),
                    f"{path_stem(request.output)}_{index:03d}{suffix}",
                )
                for index in range(len(normalized))
            )
        with ExitStack() as stack:
            temporary = [
                stack.enter_context(
                    atomic_output(
                        output, (request.source,), overwrite=request.overwrite
                    )
                )
                for output in outputs
            ]
            for output, selection in zip(temporary, normalized, strict=True):
                selection.save(output)
        return JobResult(
            outputs=outputs,
            width=image.width,
            height=image.height,
            details=f"{len(masks)} matching object(s)",
        )
