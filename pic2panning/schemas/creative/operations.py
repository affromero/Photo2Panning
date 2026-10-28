"""Executable operations shared by individual jobs and saved projects."""

from typing import Annotated

from pydantic import Field, TypeAdapter

from pic2panning.schemas.camera import CameraRequest
from pic2panning.schemas.creative.audio import BeatRequest
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    PreviewRequest,
)
from pic2panning.schemas.creative.depth import DepthRequest
from pic2panning.schemas.creative.repair import RepairRequest
from pic2panning.schemas.creative.social import SocialPackRequest
from pic2panning.schemas.editing import EditRequest
from pic2panning.schemas.media import RenderRequest
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.schemas.scene import SceneRequest
from pic2panning.schemas.video import GenerateVideoRequest

type Operation = Annotated[
    RenderRequest
    | CameraRequest
    | SceneRequest
    | EditRequest
    | GenerateRequest
    | MaskRequest
    | GenerateVideoRequest
    | DepthRequest
    | ComposeRequest
    | PreviewRequest
    | BeatRequest
    | RepairRequest
    | SocialPackRequest,
    Field(discriminator="kind"),
]

OPERATION_ADAPTER: TypeAdapter[Operation] = TypeAdapter(Operation)
