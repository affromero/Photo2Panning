"""Validate and publish locally generated endpoint-conditioned video."""

import math
from tempfile import TemporaryDirectory

from klogr.path import path_join

from pic2panning.backends.ltx import LtxBackend, VideoBackend
from pic2panning.errors import MediaError
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.video import GenerateVideoRequest
from pic2panning.tools.frames import transform_frame
from pic2panning.tools.io import atomic_output, read_image
from pic2panning.tools.media import probe


def generate_video(
    request: GenerateVideoRequest, backend: VideoBackend | None = None
) -> JobResult:
    """Generate one shot atomically after normalizing both source images."""
    inputs = [request.first_frame, request.last_frame]
    inputs.extend(
        weights
        for weights in (request.weights, request.text_weights)
        if weights is not None
    )
    with (
        atomic_output(
            request.output, inputs, overwrite=request.overwrite
        ) as output,
        TemporaryDirectory(prefix="pic2panning-keyframes-") as directory,
    ):
        keyframes = []
        for name, source in (
            ("first.png", request.first_frame),
            ("last.png", request.last_frame),
        ):
            path = path_join(directory, name)
            with (
                read_image(source) as image,
                transform_frame(image, request.size, None, 0) as normalized,
            ):
                normalized.save(path)
            keyframes.append(path)
        selected = backend if backend is not None else LtxBackend()
        selected.generate(request, keyframes[0], keyframes[1], output)
        info = probe(output)
        if (info.width, info.height) != request.size.as_tuple:
            raise MediaError(
                "Generated video dimensions differ from the request"
            )
        if info.frames != request.frames:
            raise MediaError(
                "Generated video frame count differs from the request"
            )
        if not math.isclose(info.fps, request.fps, rel_tol=1e-6):
            raise MediaError(
                "Generated video frame rate differs from the request"
            )
        if not math.isclose(
            info.duration, request.frames / request.fps, abs_tol=0.001
        ):
            raise MediaError(
                "Generated video duration differs from the request"
            )
        if info.has_audio:
            raise MediaError("Generated video unexpectedly contains audio")
    return JobResult(
        outputs=(request.output,),
        width=info.width,
        height=info.height,
        frames=info.frames,
        duration=info.duration,
        details="Local LTX two-stage video with first and last image guides",
    )
