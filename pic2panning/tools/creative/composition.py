"""Canonical composition exports and exact-frame PNG previews."""

import math
import tempfile
from collections.abc import Iterator

from klogr.path import path_copy, path_dirname, path_join, path_read_text
from PIL import Image

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    ImageLayer,
    PreviewRequest,
    VideoLayer,
)
from pic2panning.schemas.creative.depth import DepthMetadata
from pic2panning.schemas.jobs import JobResult
from pic2panning.tools.creative.audio import mix_audio
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)
from pic2panning.tools.frames import frame_count
from pic2panning.tools.io import atomic_output, is_remote, require_file
from pic2panning.tools.media import encode_frames


def reframe_composition(request: ComposeRequest, size: Size) -> ComposeRequest:
    """Reframe viewport graphics from their authored output pixel coordinates.

    Centers follow the new viewport's width and height independently. Graphic
    and text scale follows width uniformly; world layers and camera stay in
    their original world coordinates.
    """
    width_ratio = size.width / request.size.width
    height_ratio = size.height / request.size.height
    layers = tuple(
        layer.model_copy(
            update={
                "keyframes": tuple(
                    frame.model_copy(
                        update={
                            "x": frame.x * width_ratio,
                            "y": frame.y * height_ratio,
                            "scale": frame.scale * width_ratio,
                        }
                    )
                    for frame in layer.keyframes
                )
            }
        )
        if layer.space == "viewport"
        else layer
        for layer in request.layers
    )
    return ComposeRequest.model_validate(
        request.model_copy(
            update={"size": size, "layers": layers}
        ).model_dump()
    )


def composition_inputs(request: ComposeRequest) -> tuple[str, ...]:
    """Enumerate source media, depth artifacts, fonts, and native executables."""
    sources: list[str] = [track.source for track in request.audio]
    for layer in request.layers:
        if isinstance(layer, ImageLayer | VideoLayer):
            sources.append(layer.source)
        else:
            if layer.font:
                sources.append(layer.font)
            if layer.solid and layer.solid.executable:
                sources.append(layer.solid.executable)
        if layer.mask:
            sources.append(layer.mask)
        if isinstance(layer, VideoLayer) and layer.cinemagraph:
            sources.append(layer.cinemagraph.mask)
            if layer.cinemagraph.reference:
                sources.append(layer.cinemagraph.reference)
        if layer.depth:
            manifest = require_file(layer.depth.metadata)
            sources.append(manifest)
            metadata = DepthMetadata.model_validate_json(
                path_read_text(manifest)
            )
            sources.extend(
                path_join(path_dirname(manifest), artifact.path)
                for artifact in (
                    metadata.depth,
                    metadata.mask,
                    metadata.normals,
                    metadata.preview,
                )
            )
    return tuple(dict.fromkeys(sources))


def composition_frames(
    request: ComposeRequest, session: CompositionSession
) -> Iterator[Image.Image]:
    """Yield owned RGB images without retaining rendered timeline frames."""
    for index in range(frame_count(request.duration, request.fps)):
        yield composition_frame(request, index, session)


def compose(request: ComposeRequest) -> JobResult:
    """Render a world canvas while protecting every source and output alias."""
    inputs = composition_inputs(request)
    for source in inputs:
        if not is_remote(source):
            require_file(source)
    with (
        atomic_output(
            request.output, inputs, overwrite=request.overwrite
        ) as destination,
        tempfile.TemporaryDirectory(prefix="pic2panning-export-") as directory,
        CompositionSession(request) as session,
    ):
        pixels = path_join(directory, "pixels.mp4")
        count = encode_frames(
            composition_frames(request, session),
            pixels,
            request.size,
            request.fps,
            request.encoding,
        )
        duration = count / request.fps
        if request.audio:
            mix_audio(pixels, destination, request.audio, duration)
        else:
            path_copy(pixels, destination)
    return JobResult(
        outputs=(request.output,),
        width=request.size.width,
        height=request.size.height,
        frames=count,
        duration=duration,
    )


def preview(request: PreviewRequest) -> JobResult:
    """Publish one canonical frame without encoding the preceding timeline."""
    composition = request.composition
    index = (
        request.frame
        if request.frame is not None
        else math.floor((request.at or 0) * composition.fps)
    )
    if not 0 <= index < frame_count(composition.duration, composition.fps):
        raise ValueError("Preview frame is outside composition duration")
    with (
        atomic_output(
            request.output,
            composition_inputs(composition),
            overwrite=request.overwrite,
        ) as destination,
        CompositionSession(composition, preview=True) as session,
        composition_frame(composition, index, session) as image,
    ):
        image.save(destination, format="PNG")
    return JobResult(
        outputs=(request.output,),
        width=composition.size.width,
        height=composition.size.height,
        frames=1,
        duration=1 / composition.fps,
    )
