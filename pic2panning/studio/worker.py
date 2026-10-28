"""Interruptible subprocess boundary for canonical studio job execution."""

import signal
import subprocess
import sys
import traceback
from collections.abc import Sequence
from types import FrameType

from klogr.path import path_open, path_read_text

from pic2panning.api import run_job
from pic2panning.studio.schemas import JSON_OBJECT, STUDIO_JOB, WorkerResult
from pic2panning.tools.io import atomic_output


def interrupt(signal_number: int, frame: FrameType | None) -> None:
    """Raise through Python cleanup before the parent escalates termination."""
    location = frame.f_code.co_name if frame is not None else "worker"
    raise InterruptedError(
        f"Job cancelled by signal {signal_number} in {location}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one validated snapshot and atomically write its result."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        raise ValueError("Worker requires request and result filenames")
    source, destination = arguments
    signal.signal(signal.SIGTERM, interrupt)
    status = 0
    try:
        request = STUDIO_JOB.validate_json(path_read_text(source))
        result = run_job(request)
        response = WorkerResult(
            result=JSON_OBJECT.validate_python(result.model_dump(mode="json"))
        )
    except (
        ValueError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        sys.stderr.write(traceback.format_exc())
        response = WorkerResult(error=f"{type(error).__name__}: {error}")
        status = 1
    with (
        atomic_output(destination, (source,)) as temporary,
        path_open(temporary, "w") as handle,
    ):
        handle.write(response.model_dump_json() + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
