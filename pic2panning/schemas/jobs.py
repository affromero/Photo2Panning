"""One discoverable JSON contract for all library operations."""

from typing import Annotated

from pydantic import Field, JsonValue, TypeAdapter

from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.creative.project import ProjectRequest

type Job = Annotated[
    Operation | ProjectRequest,
    Field(discriminator="kind"),
]

JOB_ADAPTER: TypeAdapter[Job] = TypeAdapter(Job)


class JobResult(PydanticBaseModel):
    """Paths and measured media properties returned by a completed job."""

    outputs: tuple[str, ...]
    """Paths published by the completed operation, including separate masks."""

    width: int | None = None
    """Output image or video width in pixels, when applicable."""

    height: int | None = None
    """Output image or video height in pixels, when applicable."""

    frames: int | None = None
    """Encoded frame count for a rendered timeline, when applicable."""

    duration: float | None = None
    """Encoded timeline seconds derived from output frames and frame rate."""

    details: str | None = None
    """Operation details, such as model precision or matching object count."""


def job_schema() -> dict[str, JsonValue]:
    """Build the current JSON Schema directly from validated request types."""
    return TypeAdapter(dict[str, JsonValue]).validate_python(
        JOB_ADAPTER.json_schema()
    )
