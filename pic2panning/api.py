"""Public functions shared by scripts, agents, demos, and the CLI."""

from pic2panning.backends.contracts import (
    CameraBackend,
    ImageBackend,
    MaskBackend,
    SceneBackend,
)
from pic2panning.backends.ltx import VideoBackend
from pic2panning.schemas.camera import CameraRequest
from pic2panning.schemas.editing import EditRequest
from pic2panning.schemas.jobs import Job, JobResult
from pic2panning.schemas.media import RenderRequest
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.schemas.scene import SceneRequest, SceneResult
from pic2panning.schemas.video import GenerateVideoRequest
from pic2panning.tools import camera as camera_rendering
from pic2panning.tools import editing, generation, models
from pic2panning.tools import render as rendering


def render(request: RenderRequest) -> JobResult:
    """Render a photo or video timeline to its declared output."""
    return rendering.render(request)


def render_camera(
    request: CameraRequest, backend: CameraBackend | None = None
) -> JobResult:
    """Render a calibrated imported scene between two portions of a photo pan."""
    return camera_rendering.render_camera(request, backend)


def reconstruct_scene(
    request: SceneRequest, backend: SceneBackend | None = None
) -> SceneResult:
    """Reconstruct reusable Gaussian geometry and its bound source calibration."""
    return camera_rendering.reconstruct_scene(request, backend)


def edit(request: EditRequest) -> JobResult:
    """Apply an ordered set of deterministic image edits."""
    return editing.edit(request)


def generate(
    request: GenerateRequest, backend: ImageBackend | None = None
) -> JobResult:
    """Generate or edit with a local model or an explicitly supplied backend."""
    return models.generate(request, backend)


def mask(
    request: MaskRequest, backend: MaskBackend | None = None
) -> JobResult:
    """Produce masks from a text query using a selected local backend."""
    return models.mask(request, backend)


def generate_video(
    request: GenerateVideoRequest, backend: VideoBackend | None = None
) -> JobResult:
    """Generate a local camera move guided by its first and last frames."""
    return generation.generate_video(request, backend)


def run_job(request: Job) -> JobResult | SceneResult:
    """Execute a validated request using its canonical library function."""
    result: JobResult | SceneResult
    match request:
        case RenderRequest():
            result = render(request)
        case CameraRequest():
            result = render_camera(request)
        case SceneRequest():
            result = reconstruct_scene(request)
        case EditRequest():
            result = edit(request)
        case GenerateRequest():
            result = generate(request)
        case MaskRequest():
            result = mask(request)
        case GenerateVideoRequest():
            result = generate_video(request)
    return result
