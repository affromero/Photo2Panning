"""Fresh-process command execution for nested creative jobs."""

import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    PreviewRequest,
)
from pic2panning.schemas.creative.project import (
    ProjectDocument,
    ProjectNode,
    ProjectRequest,
    ProjectResult,
)
from pic2panning.schemas.creative.social import (
    SocialPackRequest,
    SocialVariant,
)
from pic2panning.schemas.jobs import Job, JobResult
from pic2panning.tools.media import probe


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Exercise terminal dispatch without sharing imported application state."""
    return subprocess.run(
        [sys.executable, "-m", "pic2panning", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("kind", ["compose", "preview", "social-pack"])
def test_creative_commands_publish_their_declared_media(
    kind: Literal["compose", "preview", "social-pack"],
    composition: ComposeRequest,
    tmp_path: Path,
) -> None:
    """CLI jobs produce the expected geometry and preserve the source photo."""
    source = tmp_path / "photo.png"
    original = source.read_bytes()
    request: Job
    if kind == "preview":
        request = PreviewRequest(
            composition=composition,
            output=str(tmp_path / "frame.png"),
            at=0.5,
        )
    elif kind == "social-pack":
        request = SocialPackRequest(
            composition=composition,
            output_directory=str(tmp_path / "social"),
            variants=(
                SocialVariant(name="square", size=Size(width=64, height=64)),
            ),
            cover_at=0.5,
        )
    else:
        request = composition
    job = tmp_path / "job.json"
    job.write_text(request.model_dump_json(), encoding="utf-8")
    process = command([kind, str(job)])
    assert process.returncode == 0, process.stdout + process.stderr
    result = JobResult.model_validate_json(process.stdout)
    assert all(Path(output).is_file() for output in result.outputs)
    assert source.read_bytes() == original
    if kind == "preview":
        with Image.open(result.outputs[0]) as frame:
            assert frame.size == (100, 100)
            assert frame.getpixel((0, 50)) == (255, 0, 0)
            assert frame.getpixel((99, 50)) == (0, 0, 255)
    else:
        video = probe(result.outputs[0])
        edge = 64 if kind == "social-pack" else 100
        assert (video.width, video.height) == (edge, edge)
        assert video.frames == 6


def test_project_command_reuses_verified_media_on_a_second_run(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A saved terminal project produces a report and reuses its finished movie."""
    request = ProjectRequest(
        project=ProjectDocument(
            name="CLI project",
            nodes=(ProjectNode(id="film", job=composition),),
        ),
        workspace=str(tmp_path),
        output="report.json",
        overwrite=True,
    )
    path = tmp_path / "project.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    first = command(["project", str(path)])
    assert first.returncode == 0, first.stdout + first.stderr
    original = Path(composition.output).read_bytes()
    second = command(["project", str(path)])
    assert second.returncode == 0, second.stdout + second.stderr
    result = ProjectResult.model_validate_json(second.stdout)
    assert result.nodes[0].cache_hit
    assert Path(composition.output).read_bytes() == original
    assert (
        ProjectResult.model_validate_json(
            (tmp_path / "report.json").read_text()
        )
        == result
    )


def test_creative_command_rejects_another_job_kind_before_writing(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A mismatched subcommand returns an actionable error without execution."""
    path = tmp_path / "job.json"
    path.write_text(composition.model_dump_json(), encoding="utf-8")
    result = command(["repair", str(path)])
    assert result.returncode == 1
    assert "Expected 'repair'" in result.stderr
    assert result.stdout == ""
    assert not Path(composition.output).exists()
