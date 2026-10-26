"""JSON-producing command line for media tools and local model jobs."""

import json
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout

from klogr.path import path_read_text

from pic2panning.api import run_job
from pic2panning.command.parser import parse_arguments
from pic2panning.command.requests import to_job
from pic2panning.schemas.jobs import JOB_ADAPTER, job_schema
from pic2panning.tools.media import probe


def main(argv: Sequence[str] | None = None) -> int:
    """Write JSON to stdout on success and structured errors to stderr."""
    try:
        arguments = parse_arguments(argv)
        match arguments.command:
            case "schema":
                sys.stdout.write(json.dumps(job_schema(), indent=2) + "\n")
                return 0
            case "validate" | "run":
                request = JOB_ADAPTER.validate_json(
                    path_read_text(arguments.job)
                )
                if arguments.command == "validate":
                    sys.stdout.write(request.model_dump_json(indent=2) + "\n")
                    return 0
            case "inspect":
                sys.stdout.write(
                    probe(arguments.media).model_dump_json(indent=2) + "\n"
                )
                return 0
            case _:
                request = to_job(arguments)
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
