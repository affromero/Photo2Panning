"""JSON-producing command line for media tools and local model jobs."""

import json
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout

from klogr.path import path_read_text

from pic2panning.api import run_job
from pic2panning.command.camera import CameraCommand, SceneCommand
from pic2panning.command.creative import (
    BeatCommand,
    DepthCommand,
    JobFileCommand,
    StudioCommand,
)
from pic2panning.command.media import EditCommand, RenderCommand
from pic2panning.command.models import (
    GenerateCommand,
    GenerateVideoCommand,
    MaskCommand,
)
from pic2panning.command.parse import (
    InspectCommand,
    RunCommand,
    SchemaCommand,
    ValidateCommand,
    parse_command,
)
from pic2panning.schemas.jobs import JOB_ADAPTER, job_schema
from pic2panning.studio.server import serve_studio
from pic2panning.tools.media import probe


def run_utility(
    command: SchemaCommand | InspectCommand | StudioCommand,
) -> None:
    """Handle discovery and the editor without constructing a media job."""
    match command:
        case SchemaCommand():
            sys.stdout.write(json.dumps(job_schema(), indent=2) + "\n")
        case InspectCommand():
            sys.stdout.write(
                probe(command.media).model_dump_json(indent=2) + "\n"
            )
        case StudioCommand():
            serve_studio(
                command.workspace,
                command.port,
                open_browser=command.open_browser,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Write JSON to stdout on success and structured errors to stderr."""
    try:
        command = parse_command(argv)
        match command:
            case None:
                return 0
            case SchemaCommand() | InspectCommand() | StudioCommand():
                run_utility(command)
                return 0
            case ValidateCommand() | RunCommand():
                request = JOB_ADAPTER.validate_json(
                    path_read_text(command.job)
                )
                if isinstance(command, ValidateCommand):
                    sys.stdout.write(request.model_dump_json(indent=2) + "\n")
                    return 0
            case (
                RenderCommand()
                | CameraCommand()
                | SceneCommand()
                | EditCommand()
                | GenerateCommand()
                | MaskCommand()
                | GenerateVideoCommand()
                | DepthCommand()
                | BeatCommand()
                | JobFileCommand()
            ):
                request = command.to_job()
        with redirect_stdout(sys.stderr):
            result = run_job(request)
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    except (
        ValueError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        sys.stderr.write(
            json.dumps({"error": type(error).__name__, "message": str(error)})
            + "\n"
        )
        return 1
    return 0
