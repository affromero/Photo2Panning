"""Construct the terminal interface from typed Pydantic command models."""

import sys
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Annotated

import tyro

from pic2panning.command.media import EditCommand, RenderCommand
from pic2panning.command.models import (
    GenerateCommand,
    GenerateVideoCommand,
    MaskCommand,
)
from pic2panning.schemas.base import PydanticBaseModel


class RunCommand(PydanticBaseModel):
    """Execute a validated JSON job through the canonical library functions."""

    job: Annotated[str, tyro.conf.Positional]
    """JSON job file, with relative paths resolved from the working directory."""


class ValidateCommand(RunCommand):
    """Validate and print a JSON job without accessing its input resources."""


class SchemaCommand(PydanticBaseModel):
    """Write the authoritative runtime-generated job schema as JSON."""


class InspectCommand(PydanticBaseModel):
    """Inspect dimensions, timing, frame count, and audio in a local video."""

    media: Annotated[str, tyro.conf.Positional]
    """Local video file to inspect using FFprobe."""


type Command = (
    Annotated[RenderCommand, tyro.conf.subcommand(name="render")]
    | Annotated[EditCommand, tyro.conf.subcommand(name="edit")]
    | Annotated[GenerateCommand, tyro.conf.subcommand(name="generate")]
    | Annotated[MaskCommand, tyro.conf.subcommand(name="mask")]
    | Annotated[
        GenerateVideoCommand, tyro.conf.subcommand(name="generate-video")
    ]
    | Annotated[RunCommand, tyro.conf.subcommand(name="run")]
    | Annotated[ValidateCommand, tyro.conf.subcommand(name="validate")]
    | Annotated[SchemaCommand, tyro.conf.subcommand(name="schema")]
    | Annotated[InspectCommand, tyro.conf.subcommand(name="inspect")]
)


def parse_command(argv: Sequence[str] | None) -> Command | None:
    """Parse with Tyro, preserving help and translating failures for JSON CLI."""
    output = StringIO()
    diagnostics = StringIO()
    try:
        with redirect_stdout(output), redirect_stderr(diagnostics):
            command = tyro.cli(
                Command,
                args=argv,
                prog="pic2panning",
                description=(
                    "Photo motion and local image editing on Apple Silicon"
                ),
                config=(tyro.conf.EnumChoicesFromValues,),
            )
    except SystemExit as error:
        if error.code is None or error.code == 0:
            sys.stdout.write(output.getvalue())
            sys.stderr.write(diagnostics.getvalue())
            return None
        message = (
            diagnostics.getvalue().strip()
            or output.getvalue().strip()
            or "Invalid command-line arguments"
        )
        raise ValueError(message) from error
    sys.stdout.write(output.getvalue())
    sys.stderr.write(diagnostics.getvalue())
    return command
