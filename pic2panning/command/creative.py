"""Typed terminal entry points for projects, depth, and the local studio."""

from typing import Annotated, ClassVar, Literal

import tyro
from klogr.path import path_read_text
from pydantic import Field

from pic2panning.command.base import OutputCommand
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.audio import BeatRequest
from pic2panning.schemas.creative.depth import DepthRequest
from pic2panning.schemas.jobs import JOB_ADAPTER, Job


class DepthCommand(OutputCommand):
    """Extract metric TIFF, previews, normals, mask, and source-bound metadata."""

    source: Annotated[str, tyro.conf.Positional]
    num_tokens: int = 1800
    max_edge: int = 1920
    device: Literal["mps", "cpu"] = "mps"

    def to_job(self) -> DepthRequest:
        """Use the same validated depth request as Python and JSON."""
        return DepthRequest(
            source=self.source,
            output=self.output,
            num_tokens=self.num_tokens,
            max_edge=self.max_edge,
            device=self.device,
            overwrite=self.overwrite,
        )


class BeatCommand(OutputCommand):
    """Find music onset markers using bounded CPU audio analysis."""

    source: Annotated[str, tyro.conf.Positional]
    start: float = 0
    duration: float | None = None
    sensitivity: float = 1.5
    minimum_gap: float = 0.2

    def to_job(self) -> BeatRequest:
        """Validate analysis controls before decoding audio."""
        return BeatRequest(
            source=self.source,
            output=self.output,
            start=self.start,
            duration=self.duration,
            sensitivity=self.sensitivity,
            minimum_gap=self.minimum_gap,
            overwrite=self.overwrite,
        )


class JobFileCommand(PydanticBaseModel):
    """Execute a typed job file for an operation with nested controls."""

    job: Annotated[str, tyro.conf.Positional]
    operation_kind: ClassVar[str]

    def to_job(self) -> Job:
        """Reject a mismatched job before execution or input access."""
        request = JOB_ADAPTER.validate_json(path_read_text(self.job))
        if request.kind != self.operation_kind:
            raise ValueError(
                f"Expected {self.operation_kind!r} job, got {request.kind!r}"
            )
        return request


class ComposeCommand(JobFileCommand):
    """Render a saved layer composition with keyframes and explicit audio."""

    operation_kind: ClassVar[str] = "compose"


class PreviewCommand(JobFileCommand):
    """Save an exact composition frame without encoding a complete video."""

    operation_kind: ClassVar[str] = "preview"


class ProjectCommand(JobFileCommand):
    """Run a saved operation graph and reuse verified cached artifacts."""

    operation_kind: ClassVar[str] = "project"


class SocialPackCommand(JobFileCommand):
    """Export social aspect variants and matching cover/carousel images."""

    operation_kind: ClassVar[str] = "social-pack"


class RepairCommand(JobFileCommand):
    """Repair an enlarged contextual region while preserving other pixels."""

    operation_kind: ClassVar[str] = "repair"


class StudioCommand(PydanticBaseModel):
    """Open the local visual editor with saved projects and background jobs."""

    workspace: str = "~/Pictures/Photo2Panning"
    port: Annotated[int, Field(ge=0, le=65535, strict=True)] = 8765
    open_browser: bool = True
