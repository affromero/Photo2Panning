"""Real subprocess work, failure reporting, and whole-process-group cancellation."""

import subprocess
import time
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from pic2panning.schemas.editing import Crop, EditRequest
from pic2panning.studio.schemas import ErrorResponse, JobRecord, JobSubmission
from pic2panning.tools.media import probe
from tests.studio import (
    running_studio,
    small_composition,
    small_edit,
    submit,
    write_source,
)


def process_groups() -> list[tuple[int, int, str]]:
    """Observe actual child process groups at the operating-system boundary."""
    result = subprocess.run(
        ["ps", "-ax", "-o", "pid=,pgid=,args="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return [
        (int(pid), int(group), command)
        for line in result.stdout.splitlines()
        for pid, group, command in (line.strip().split(maxsplit=2),)
    ]


def test_edit_job_publishes_real_resized_pixels_and_keeps_source(
    tmp_path: Path,
) -> None:
    """A canonical worker publishes its completed image and preserves input bytes."""
    source = write_source(tmp_path)
    original = source.read_bytes()
    with running_studio(tmp_path) as client:
        queued = submit(client, small_edit())
        result = client.wait(queued.id)
        assert result.status == "completed", result.model_dump_json(indent=2)
        assert result.outputs == ("exports/edited.png",)
        with Image.open(tmp_path / result.outputs[0]) as image:
            assert image.size == (8, 4)
            red = cast("tuple[int, int, int]", image.getpixel((0, 0)))
            blue = cast("tuple[int, int, int]", image.getpixel((7, 0)))
            assert red[0] > 200
            assert red[2] < 50
            assert blue[0] < 50
            assert blue[2] > 200
    assert source.read_bytes() == original


def test_compose_job_reports_encoded_frames_and_duration(
    tmp_path: Path,
) -> None:
    """A real FFmpeg subprocess exports the declared six-frame silent timeline."""
    write_source(tmp_path)
    with running_studio(tmp_path) as client:
        queued = submit(client, small_composition())
        result = client.wait(queued.id)
        assert result.status == "completed", result.model_dump_json(indent=2)
        assert result.outputs == ("exports/movie.mp4",)
        metadata = probe(str(tmp_path / result.outputs[0]))
        assert metadata.frames == 6
        assert metadata.duration == 1
        assert not metadata.has_audio


def test_worker_failure_retains_error_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    """Canonical source-bound errors surface with useful subprocess diagnostics."""
    write_source(tmp_path)
    invalid = EditRequest(
        source="source.png",
        output="exports/invalid.png",
        operations=(Crop(left=0, top=0, right=100, bottom=100),),
    )
    with running_studio(tmp_path) as client:
        response = client.post("/api/jobs", JobSubmission(job=invalid))
        assert response.status == 202
        result = client.wait(response.parsed(JobRecord).id)
        assert result.status == "failed"
        assert result.error is not None
        assert "Crop bounds exceed" in result.error
        assert "Traceback" in result.log
        assert not (tmp_path / "exports/invalid.png").exists()


@pytest.mark.parametrize("route", ["/api/jobs", "/api/validate"])
def test_job_submission_protects_source_hardlink_alias(
    tmp_path: Path, route: str
) -> None:
    """Even explicit overwrite cannot turn a queued output into a source alias."""
    source = write_source(tmp_path)
    alias = tmp_path / "alias.png"
    alias.hardlink_to(source)
    original = source.read_bytes()
    with running_studio(tmp_path) as client:
        request = small_edit("alias.png").model_copy(
            update={"overwrite": True}
        )
        response = client.post(route, JobSubmission(job=request))
        assert response.status == 400
        assert "aliases an input" in response.parsed(ErrorResponse).message
    assert source.read_bytes() == original


def test_cancelling_queued_and_running_jobs_stops_ffmpeg_descendants(
    tmp_path: Path,
) -> None:
    """Cancellation publishes no partial work and leaves no worker-group processes."""
    write_source(tmp_path)
    with running_studio(tmp_path) as client:
        active = submit(client, small_composition(duration=600, fps=60))
        deadline = time.monotonic() + 10
        worker_group: int | None = None
        while time.monotonic() < deadline:
            processes = process_groups()
            workers = [
                group
                for _, group, command in processes
                if "pic2panning.studio.worker" in command
                and active.id in command
                and str(tmp_path) in command
            ]
            if workers and any(
                group == workers[0] and "ffmpeg" in command
                for _, group, command in processes
            ):
                worker_group = workers[0]
                break
            snapshot = client.request(
                "GET", f"/api/job?id={active.id}"
            ).parsed(JobRecord)
            assert snapshot.status not in {"completed", "failed"}, (
                snapshot.model_dump_json(indent=2)
            )
            time.sleep(0.02)
        assert worker_group is not None, (
            "The real FFmpeg descendant did not start"
        )
        queued = submit(client, small_edit())
        response = client.request("POST", f"/api/cancel?id={queued.id}")
        assert response.status == 200
        assert response.parsed(JobRecord).status == "cancelled"
        response = client.request("POST", f"/api/cancel?id={active.id}")
        assert response.status == 200
        assert response.parsed(JobRecord).status == "cancelled"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
            group == worker_group for _, group, _ in process_groups()
        ):
            time.sleep(0.02)
        assert not any(
            group == worker_group for _, group, _ in process_groups()
        )
        assert not (tmp_path / "exports/movie.mp4").exists()
        assert not (tmp_path / "exports/edited.png").exists()


def test_worker_storage_failure_does_not_kill_following_jobs(
    tmp_path: Path,
) -> None:
    """A guarded run-directory failure is reported and the queue can recover."""
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    write_source(workspace)
    outside.mkdir()
    with running_studio(workspace) as client:
        private = workspace / ".studio"
        private.symlink_to(outside, target_is_directory=True)
        failed = client.wait(submit(client, small_edit()).id)
        assert failed.status == "failed"
        assert failed.error is not None
        assert "escapes" in failed.error
        assert not tuple(outside.iterdir())
        private.unlink()
        recovered = client.wait(
            submit(client, small_edit("exports/recovered.png")).id
        )
        assert recovered.status == "completed", recovered.model_dump_json(
            indent=2
        )
        assert (workspace / "exports/recovered.png").is_file()


@pytest.mark.parametrize("route", ["/api/jobs", "/api/validate"])
def test_existing_outputs_require_explicit_overwrite(
    tmp_path: Path, route: str
) -> None:
    """Preflight and submission leave an existing unrelated destination intact."""
    write_source(tmp_path)
    output = tmp_path / "existing.png"
    with Image.new("RGB", (8, 4), "lime") as image:
        image.save(output)
    original = output.read_bytes()
    with running_studio(tmp_path) as client:
        response = client.post(
            route, JobSubmission(job=small_edit("existing.png"))
        )
        assert response.status == 400
        assert response.parsed(ErrorResponse).error == "FileExistsError"
    assert output.read_bytes() == original
