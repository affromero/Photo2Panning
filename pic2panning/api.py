"""Public functions shared by scripts, agents, demos, and the CLI."""

from pic2panning.backends.contracts import ImageBackend, MaskBackend
from pic2panning.schemas.editing import EditRequest
from pic2panning.schemas.jobs import Job, JobResult
from pic2panning.schemas.media import RenderRequest
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.tools import editing, models
from pic2panning.tools import render as rendering


def render(request: RenderRequest) -> JobResult:
    """Render a photo or video timeline to its declared output."""
    return rendering.render(request)


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


def run_job(request: Job) -> JobResult:
    """Execute a validated request using its canonical library function."""
    match request:
        case RenderRequest():
            return render(request)
        case EditRequest():
            return edit(request)
        case GenerateRequest():
            return generate(request)
        case MaskRequest():
            return mask(request)
