"""Stage social variants and stills before publishing completed artifacts."""

from contextlib import ExitStack

from klogr.path import path_join
from PIL import ImageDraw

from pic2panning.schemas.creative.composition import PreviewRequest
from pic2panning.schemas.creative.social import (
    SafeInsets,
    SocialPackRequest,
    SocialVariant,
)
from pic2panning.schemas.jobs import JobResult
from pic2panning.tools.creative.composition import (
    compose,
    composition_inputs,
    preview,
    reframe_composition,
)
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)
from pic2panning.tools.frames import frame_count
from pic2panning.tools.io import atomic_output, validate_destination


def variant_outputs(
    request: SocialPackRequest, variant: SocialVariant
) -> tuple[str, ...]:
    """Return stable names for a variant video and its selected stills."""
    stem = path_join(request.output_directory, variant.name)
    outputs = [f"{stem}.mp4"]
    if request.cover_at is not None:
        outputs.append(f"{stem}-cover.png")
    outputs.extend(
        f"{stem}-carousel-{index + 1:03d}.png"
        for index in range(len(request.carousel_at))
    )
    if request.preview_guides:
        outputs.append(f"{stem}-guides.png")
    return tuple(outputs)


def social_outputs(request: SocialPackRequest) -> tuple[str, ...]:
    """Enumerate every derived destination before touching source media."""
    return tuple(
        output
        for variant in request.variants
        for output in variant_outputs(request, variant)
    )


def draw_guides(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], insets: SafeInsets
) -> None:
    """Draw explicitly requested safe margins onto a separate guide preview."""
    width, height = size
    rectangle = (
        round(insets.left * width),
        round(insets.top * height),
        round((1 - insets.right) * width) - 1,
        round((1 - insets.bottom) * height) - 1,
    )
    draw.rectangle(rectangle, outline="#ffdd44", width=max(1, width // 250))


def stage_variant(
    request: SocialPackRequest,
    variant: SocialVariant,
    destinations: dict[str, str],
) -> None:
    """Render all artifacts for one aspect through the canonical frame path."""
    outputs = variant_outputs(request, variant)
    composition = reframe_composition(
        request.composition, variant.size
    ).model_copy(
        update={
            "output": destinations[outputs[0]],
            "overwrite": True,
        }
    )
    compose(composition)
    moments = (
        (request.cover_at,) if request.cover_at is not None else ()
    ) + request.carousel_at
    for output, moment in zip(outputs[1:], moments, strict=False):
        preview(
            PreviewRequest(
                composition=composition,
                output=destinations[output],
                at=moment,
                overwrite=True,
            )
        )
    if request.preview_guides:
        with (
            CompositionSession(composition, preview=True) as session,
            composition_frame(composition, 0, session) as image,
        ):
            draw_guides(ImageDraw.Draw(image), image.size, request.safe_insets)
            image.save(destinations[outputs[-1]], format="PNG")


def social_pack(request: SocialPackRequest) -> JobResult:
    """Validate every name and stage the whole pack before file publication."""
    outputs = social_outputs(request)
    inputs = composition_inputs(request.composition)
    for index, output in enumerate(outputs):
        others = outputs[:index] + outputs[index + 1 :]
        validate_destination(
            output, inputs + others, overwrite=request.overwrite
        )
    with ExitStack() as stack:
        destinations = {
            output: stack.enter_context(
                atomic_output(output, inputs, overwrite=request.overwrite)
            )
            for output in outputs
        }
        for variant in request.variants:
            stage_variant(request, variant, destinations)
    return JobResult(
        outputs=outputs,
        duration=frame_count(
            request.composition.duration, request.composition.fps
        )
        / request.composition.fps,
        details=f"Exported {len(request.variants)} social variants",
    )
