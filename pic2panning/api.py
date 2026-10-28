"""Public functions shared by scripts, agents, demos, and the CLI."""

from collections.abc import Callable

from pic2panning.backends.contracts import (
    CameraBackend,
    ImageBackend,
    MaskBackend,
    SceneBackend,
)
from pic2panning.backends.creative.depth import DepthBackend
from pic2panning.backends.ltx import VideoBackend
from pic2panning.schemas.camera import CameraRequest
from pic2panning.schemas.creative.audio import BeatRequest
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    PreviewRequest,
)
from pic2panning.schemas.creative.depth import DepthRequest, DepthResult
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.creative.project import ProjectRequest, ProjectResult
from pic2panning.schemas.creative.repair import RepairRequest
from pic2panning.schemas.creative.social import SocialPackRequest
from pic2panning.schemas.editing import EditRequest
from pic2panning.schemas.jobs import Job, JobResult
from pic2panning.schemas.media import RenderRequest
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.schemas.scene import SceneRequest, SceneResult
from pic2panning.schemas.video import GenerateVideoRequest
from pic2panning.tools import camera as camera_rendering
from pic2panning.tools import editing, generation, models
from pic2panning.tools import render as rendering
from pic2panning.tools.creative import audio, composition, depth, social
from pic2panning.tools.creative import repair as repairing
from pic2panning.tools.projects import runtime
from pic2panning.tools.projects.runtime import ProjectEvent

type OperationResult = JobResult | SceneResult | DepthResult
type MediaOperation = (
    RenderRequest
    | CameraRequest
    | SceneRequest
    | EditRequest
    | GenerateRequest
    | MaskRequest
    | GenerateVideoRequest
)
type CreativeOperation = (
    DepthRequest
    | ComposeRequest
    | PreviewRequest
    | BeatRequest
    | RepairRequest
    | SocialPackRequest
)


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


def extract_depth(
    request: DepthRequest, backend: DepthBackend | None = None
) -> DepthResult:
    """Extract source-bound metric depth and reusable visualizations."""
    return depth.extract_depth(request, backend)


def compose(request: ComposeRequest) -> JobResult:
    """Render a world canvas, animated layers, camera, and explicit audio."""
    return composition.compose(request)


def preview(request: PreviewRequest) -> JobResult:
    """Save the exact canonical frame used by the final composition."""
    return composition.preview(request)


def social_pack(request: SocialPackRequest) -> JobResult:
    """Export coordinated videos, cover images, and carousel stills."""
    return social.social_pack(request)


def detect_beats(request: BeatRequest) -> JobResult:
    """Analyze audio onset markers on the CPU."""
    return audio.detect_beats(request)


def repair(
    request: RepairRequest, backend: ImageBackend | None = None
) -> JobResult:
    """Edit an enlarged contextual region and preserve unselected pixels."""
    return repairing.repair(request, backend)


def run_media_operation(request: MediaOperation) -> OperationResult:
    """Dispatch photo, footage, segmentation, and reconstruction operations."""
    result: OperationResult
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


def run_creative_operation(request: CreativeOperation) -> OperationResult:
    """Dispatch depth, compositions, repair, audio analysis, and social packs."""
    result: OperationResult
    match request:
        case DepthRequest():
            result = extract_depth(request)
        case ComposeRequest():
            result = compose(request)
        case PreviewRequest():
            result = preview(request)
        case BeatRequest():
            result = detect_beats(request)
        case RepairRequest():
            result = repair(request)
        case SocialPackRequest():
            result = social_pack(request)
    return result


def run_operation(request: Operation) -> OperationResult:
    """Execute one operation without permitting nested project recursion."""
    if isinstance(
        request,
        RenderRequest
        | CameraRequest
        | SceneRequest
        | EditRequest
        | GenerateRequest
        | MaskRequest
        | GenerateVideoRequest,
    ):
        return run_media_operation(request)
    return run_creative_operation(request)


def run_project(
    request: ProjectRequest,
    progress: Callable[[ProjectEvent], None] | None = None,
) -> ProjectResult:
    """Execute a validated saved project with verified intermediate caching."""
    return runtime.run_project(request, run_operation, progress)


def run_job(request: Job) -> OperationResult | ProjectResult:
    """Execute the same validated job from the studio, CLI, or Python."""
    if isinstance(request, ProjectRequest):
        return run_project(request)
    return run_operation(request)
