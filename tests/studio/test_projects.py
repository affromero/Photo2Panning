"""Cached graph execution and nested workspace confinement over actual HTTP."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from pic2panning.schemas.creative.project import (
    ProjectDocument,
    ProjectNode,
    ProjectRequest,
    ProjectResult,
)
from pic2panning.schemas.editing import Crop, EditRequest
from pic2panning.studio.schemas import (
    ErrorResponse,
    JobList,
    JobRecord,
    JobSubmission,
)
from tests.studio import running_studio, small_edit, write_source


def graph(workspace: Path) -> ProjectRequest:
    """Declare two real CPU edits linked through a named intermediate image."""
    return ProjectRequest(
        project=ProjectDocument(
            name="Cached HTTP graph",
            nodes=(
                ProjectNode(id="resize", job=small_edit()),
                ProjectNode(
                    id="crop",
                    job=EditRequest(
                        source="exports/edited.png",
                        output="exports/cropped.png",
                        operations=(Crop(left=1, top=0, right=7, bottom=4),),
                    ),
                    depends_on=("resize",),
                ),
                ProjectNode(
                    id="unrelated",
                    job=small_edit("exports/unrelated.png"),
                ),
            ),
        ),
        workspace=str(workspace),
        output="reports/graph.json",
        targets=("crop",),
        cache="reuse",
        overwrite=True,
    )


def test_project_http_reuses_cached_nodes_and_regenerates_changed_inputs(
    tmp_path: Path,
) -> None:
    """Real graph workers reuse intact files and invalidate downstream pixels."""
    source = write_source(tmp_path)
    original = source.read_bytes()
    unrelated = tmp_path / "exports/unrelated.png"
    unrelated.parent.mkdir()
    with Image.new("RGB", (3, 3), "yellow") as image:
        image.save(unrelated)
    unrelated_bytes = unrelated.read_bytes()
    request = graph(tmp_path)
    with running_studio(tmp_path) as client:
        results: list[ProjectResult] = []
        for _ in range(2):
            response = client.post("/api/jobs", JobSubmission(job=request))
            assert response.status == 202, response.body.decode()
            completed = client.wait(response.parsed(JobRecord).id)
            assert completed.status == "completed", completed.model_dump_json(
                indent=2
            )
            assert completed.result is not None
            assert "exports/unrelated.png" not in completed.outputs
            results.append(ProjectResult.model_validate(completed.result))
        assert not any(node.cache_hit for node in results[0].nodes)
        assert all(node.cache_hit for node in results[1].nodes)
        assert source.read_bytes() == original
        with Image.open(tmp_path / "exports/cropped.png") as cropped:
            assert cropped.size == (6, 4)
            red = cast("tuple[int, int, int]", cropped.getpixel((0, 0)))
            blue = cast("tuple[int, int, int]", cropped.getpixel((5, 0)))
            assert red[0] > 180
            assert blue[2] > 180
        with Image.new("RGB", (4, 2), "lime") as changed:
            changed.save(source)
        updated = source.read_bytes()
        response = client.post("/api/jobs", JobSubmission(job=request))
        assert response.status == 202, response.body.decode()
        completed = client.wait(response.parsed(JobRecord).id)
        assert completed.status == "completed", completed.model_dump_json(
            indent=2
        )
        assert completed.result is not None
        regenerated = ProjectResult.model_validate(completed.result)
        assert not any(node.cache_hit for node in regenerated.nodes)
        with Image.open(tmp_path / "exports/cropped.png") as cropped:
            assert cropped.getpixel((3, 2)) == (0, 255, 0)
        assert source.read_bytes() == updated
        assert unrelated.read_bytes() == unrelated_bytes
        report = ProjectResult.model_validate_json(
            (tmp_path / "reports/graph.json").read_text()
        )
        assert (
            report.nodes[-1].input_hashes != results[1].nodes[-1].input_hashes
        )


@pytest.mark.parametrize("escape", ["source", "output", "report", "workspace"])
@pytest.mark.parametrize("route", ["/api/jobs", "/api/validate"])
def test_nested_project_paths_cannot_select_another_workspace(
    tmp_path: Path, escape: str, route: str
) -> None:
    """Nested operations, reports, and workspace selectors stay confined."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    write_source(workspace)
    secret = write_source(outside)
    original = secret.read_bytes()
    operation = small_edit()
    if escape in {"source", "output"}:
        operation = operation.model_copy(
            update={
                escape: str(secret)
                if escape == "source"
                else str(outside / "generated.png")
            }
        )
    request = ProjectRequest(
        project=ProjectDocument(
            name="Confined",
            nodes=(ProjectNode(id="edit", job=operation),),
        ),
        workspace=str(outside) if escape == "workspace" else str(workspace),
        output=str(outside / "report.json")
        if escape == "report"
        else "reports/run.json",
    )
    with running_studio(workspace) as client:
        response = client.post(route, JobSubmission(job=request))
        assert response.status == 400
        assert "workspace" in response.parsed(ErrorResponse).message.lower()
        listed = client.request("GET", "/api/jobs").parsed(JobList)
        assert not listed.jobs
    assert secret.read_bytes() == original
    assert not (outside / "generated.png").exists()
    assert not (outside / "report.json").exists()
