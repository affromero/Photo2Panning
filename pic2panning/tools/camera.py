"""Stream a calibrated pan, pause, 3D excursion, return, and resumed pan."""

import json
import math
from collections.abc import Generator, Iterator
from contextlib import closing
from tempfile import TemporaryDirectory

from klogr.path import path_stat, path_write_text
from PIL import Image

from pic2panning.backends.camera import NativeCameraBackend, NativeSceneBackend
from pic2panning.backends.contracts import (
    CameraBackend,
    CameraScene,
    SceneBackend,
)
from pic2panning.errors import MediaError
from pic2panning.schemas.base import PositiveFloat, PydanticBaseModel
from pic2panning.schemas.camera import (
    CameraPose,
    CameraRequest,
    SceneCalibration,
    Vector3,
)
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.media import ImageClip, Motion
from pic2panning.schemas.scene import (
    MOGE_MODEL,
    MOGE_REVISION,
    MOGE_SHA256,
    MOGE_SOURCE_REVISION,
    SceneRequest,
    SceneResult,
)
from pic2panning.tools.frames import (
    crop_dimensions,
    exposure_frame,
    transform_frame,
    weighted_linear_blend,
)
from pic2panning.tools.io import (
    atomic_output,
    file_sha256,
    materialize,
    read_image,
    require_file,
)
from pic2panning.tools.media import encode_frames, probe
from pic2panning.tools.render import repeated_frames


class ExcursionState(PydanticBaseModel):
    """The resolved original camera, displacement, and optical target."""

    base: CameraPose
    position: Vector3
    focus: PositiveFloat


def source_camera(
    calibration: SceneCalibration, request: CameraRequest, image: Image.Image
) -> CameraPose:
    """Express the exact paused crop as an off-axis calibrated camera."""
    width, height = crop_dimensions(image, request.size)
    left = (image.width - width) * request.pause_at
    top = (image.height - height) / 2
    scale = width / request.size.width
    matrix = calibration.intrinsics
    world = calibration.world_to_camera
    return CameraPose(
        width=request.size.width,
        height=request.size.height,
        fx=matrix[0][0] / scale,
        fy=matrix[1][1] / scale,
        cx=(matrix[0][2] - left) / scale,
        cy=(matrix[1][2] - top) / scale,
        rotation=(world[0][:3], world[1][:3], world[2][:3]),
        translation=(world[0][3], world[1][3], world[2][3]),
    )


def camera_offset(request: CameraRequest, focus: float) -> Vector3:
    """Resolve conservative defaults and reject a crossed target plane."""
    position = request.position or (focus * 0.02, -focus * 0.005, focus * 0.01)
    if position[2] >= focus - 0.001:
        raise ValueError("Camera position must stay before the focus plane")
    return position


def pan_frames(
    image: Image.Image,
    request: CameraRequest,
    positions: tuple[float, float],
    count: int,
) -> Iterator[Image.Image]:
    """Reach the pause with zero speed, using the canonical photo sampler."""
    start, end = positions
    clip = ImageClip(
        source=request.source,
        motion=Motion.PAN_LR,
        shutter_angle=request.shutter_angle,
    )
    for index in range(count):
        time = index / (count - 1)
        eased = time**3 * (10 + time * (-15 + time * 6))
        progress = start + (end - start) * eased
        if not request.shutter_angle or index in {0, count - 1}:
            yield transform_frame(image, request.size, Motion.PAN_LR, progress)
            continue
        velocity = 30 * time**2 * (1 - time) ** 2
        interval = (
            abs(end - start)
            * velocity
            * request.shutter_angle
            / (360 * (count - 1))
        )
        yield exposure_frame(image, clip, request.size, progress, interval)


def anchored_frame(
    image: Image.Image, anchor: Image.Image, weight: float
) -> Image.Image:
    """Blend a short low-motion handoff in linear light, quantizing once."""
    if image.mode != "RGB" or image.size != anchor.size:
        raise MediaError("Camera renderer must return RGB at the output size")
    return weighted_linear_blend(
        ((anchor.copy(), 1 - weight), (image.copy(), weight))
    )


def excursion_frames(
    scene: CameraScene,
    request: CameraRequest,
    state: ExcursionState,
    anchor: Image.Image,
) -> Iterator[Image.Image]:
    """Render the same scene along an eased camera path and exact return."""
    count = math.ceil(request.duration * request.fps)
    blend_count = min(10, max(1, (count - 1) // 4))
    for index in range(count):
        if index in {0, count - 1}:
            yield anchor.copy()
            continue
        amount = math.sin(math.pi * index / (count - 1)) ** 2
        position = state.position
        offset = (
            position[0] * amount,
            position[1] * amount,
            position[2] * amount,
        )
        pose = scene.pose_from_offset(state.base, offset, state.focus)
        edge = min(index, count - 1 - index)
        with scene.render(pose) as frame:
            if frame.mode != "RGB" or frame.size != anchor.size:
                raise MediaError(
                    "Camera renderer must return RGB at the output size"
                )
            if edge < blend_count:
                weight = 0.5 - 0.5 * math.cos(math.pi * edge / blend_count)
                yield anchored_frame(frame, anchor, weight)
            else:
                yield frame.copy()


def timeline_frames(
    image: Image.Image,
    request: CameraRequest,
    scene: CameraScene,
    calibration: SceneCalibration,
) -> Generator[Image.Image]:
    """Yield one bounded stream without retaining a completed video phase."""
    focus = calibration.z_percentiles_metres["10"]
    position = camera_offset(request, focus)
    base = source_camera(calibration, request, image)
    state = ExcursionState(base=base, position=position, focus=focus)
    with transform_frame(
        image, request.size, Motion.PAN_LR, request.pause_at
    ) as anchor:
        yield from pan_frames(
            image,
            request,
            (request.pan_start, request.pause_at),
            math.ceil(request.before * request.fps),
        )
        holds = math.ceil(request.hold * request.fps)
        yield from repeated_frames(anchor, holds)
        yield from excursion_frames(scene, request, state, anchor)
        yield from repeated_frames(anchor, holds)
        yield from pan_frames(
            image,
            request,
            (request.pause_at, request.pan_end),
            math.ceil(request.after * request.fps),
        )


def render_camera(
    request: CameraRequest, backend: CameraBackend | None = None
) -> JobResult:
    """Validate imported geometry, stream the timeline, and publish safely."""
    inputs = (request.source, request.scene, request.metadata)
    with (
        atomic_output(
            request.output, inputs, overwrite=request.overwrite
        ) as output,
        TemporaryDirectory(prefix="pic2panning-camera-") as directory,
    ):
        local = materialize(request.source, directory)
        scene_path = require_file(request.scene)
        metadata = require_file(request.metadata)
        calibration = SceneCalibration.from_sidecar(metadata)
        if file_sha256(local) != calibration.source_sha256:
            raise ValueError("Camera source differs from the calibrated image")
        if (
            calibration.scene_sha256 is not None
            and file_sha256(scene_path) != calibration.scene_sha256
        ):
            raise ValueError("Camera scene checksum differs from its metadata")
        camera_offset(request, calibration.z_percentiles_metres["10"])
        selected = backend if backend is not None else NativeCameraBackend()
        with read_image(local) as image:
            if image.size != (calibration.width, calibration.height):
                raise ValueError(
                    "Camera source dimensions differ from metadata"
                )
            with (
                closing(selected.load(scene_path, calibration)) as scene,
                closing(
                    timeline_frames(image, request, scene, calibration)
                ) as frames,
            ):
                count = encode_frames(
                    frames,
                    output,
                    request.size,
                    request.fps,
                    request.encoding,
                )
        info = probe(output)
        expected = sum(
            math.ceil(seconds * request.fps)
            for seconds in (
                request.before,
                request.duration,
                request.after,
                request.hold,
                request.hold,
            )
        )
        if (
            count != expected
            or info.frames != count
            or info.fps != request.fps
            or (info.width, info.height) != request.size.as_tuple
            or info.has_audio
            or not math.isclose(
                info.duration, count / request.fps, abs_tol=0.001
            )
        ):
            raise MediaError("Camera output failed media verification")
    return JobResult(
        outputs=(request.output,),
        width=info.width,
        height=info.height,
        frames=count,
        duration=info.duration,
        details="Calibrated 3D camera excursion with original photo handoffs",
    )


def reconstruct_scene(
    request: SceneRequest, backend: SceneBackend | None = None
) -> SceneResult:
    """Predict reusable photo geometry and publish its manifest last."""
    with (
        atomic_output(
            request.metadata,
            (request.source, request.output),
            overwrite=request.overwrite,
        ) as metadata,
        atomic_output(
            request.output,
            (request.source, request.metadata),
            overwrite=request.overwrite,
        ) as output,
        TemporaryDirectory(prefix="pic2panning-scene-") as directory,
    ):
        source = materialize(request.source, directory)
        source_hash = file_sha256(source)
        selected = backend if backend is not None else NativeSceneBackend()
        with read_image(source) as image:
            calibration, points = selected.predict(
                request, image, output, source_hash
            )
            if (calibration.width, calibration.height) != image.size:
                raise ValueError("Predicted calibration dimensions differ")
        if calibration.source_sha256 != source_hash:
            raise ValueError("Predicted calibration source checksum differs")
        if points <= 0:
            raise ValueError("Scene prediction must contain positive points")
        if not path_stat(output).st_size:
            raise ValueError("Scene prediction wrote an empty PLY")
        calibration = calibration.model_copy(
            update={"scene_sha256": file_sha256(output)}
        )
        result = SceneResult(
            output=request.output,
            metadata=request.metadata,
            points=points,
            source_sha256=source_hash,
        )
        document = calibration.model_dump(mode="json")
        document["reconstruction"] = {
            "model": MOGE_MODEL,
            "model_revision": MOGE_REVISION,
            "checkpoint_sha256": MOGE_SHA256,
            "source_revision": MOGE_SOURCE_REVISION,
            "license": "MIT",
            "device": request.device,
            "num_tokens": request.num_tokens,
            "max_edge": request.max_edge,
            "gaussians": points,
            "geometry": "Metric point field with analytic spherical scales",
            "pixel_sigma": 0.55,
            "mask_policy": "Retain finite raw predictions including sky",
            "coordinate_convention": "OpenCV x-right y-down z-forward",
            "pixel_convention": "Image edges; pixel centers at +0.5",
        }
        path_write_text(metadata, json.dumps(document, indent=2) + "\n")
    return result
