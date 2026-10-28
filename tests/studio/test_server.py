"""Actual HTTP authentication, workspace protection, saves, and previews."""

import io
from pathlib import Path
from urllib.parse import urlencode

import pytest
from PIL import Image

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    LayerKeyframe,
    TextLayer,
)
from pic2panning.schemas.creative.project import ProjectDocument, ProjectNode
from pic2panning.studio.schemas import (
    Asset,
    Bootstrap,
    ErrorResponse,
    JobSubmission,
    PreviewResult,
    PreviewSubmission,
    ReframeSubmission,
    SavedProject,
    SaveProject,
    ValidationResult,
)
from tests.studio import (
    running_studio,
    small_composition,
    small_edit,
    write_source,
)


def test_bootstrap_lists_oriented_assets_and_session_token(
    tmp_path: Path,
) -> None:
    """The real editor discovers its source and receives a mutation token."""
    source = write_source(tmp_path)
    with running_studio(tmp_path) as client:
        response = client.request("GET", "/api/bootstrap")
        bootstrap = response.parsed(Bootstrap)
        assert response.status == 200
        assert len(bootstrap.token) >= 32
        asset = next(
            asset for asset in bootstrap.assets if asset.path == source.name
        )
        assert (asset.width, asset.height) == (4, 2)
        assert asset.kind == "image"
        assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "scenario",
    [
        "missing",
        "missing-token",
        "wrong-token",
        "cross-origin",
        "wrong-host",
    ],
)
def test_mutations_require_same_origin_and_session_token(
    tmp_path: Path, scenario: str
) -> None:
    """Missing credentials, hostile origins, and DNS rebinding cannot queue work."""
    write_source(tmp_path)
    payload = JobSubmission(job=small_edit()).model_dump_json().encode()
    with running_studio(tmp_path) as client:
        headers = {
            "missing": {},
            "missing-token": {"Origin": client.origin},
            "wrong-token": {
                "Origin": client.origin,
                "X-Studio-Token": "wrong",
            },
            "cross-origin": {
                "Origin": "http://external.test",
                "X-Studio-Token": client.token,
            },
            "wrong-host": {
                "Host": "external.test",
                "Origin": "http://external.test",
                "X-Studio-Token": client.token,
            },
        }[scenario]
        response = client.request("POST", "/api/jobs", payload, headers)
        assert response.status == 403
        assert response.parsed(ErrorResponse).error == "PermissionError"
        assert not (tmp_path / "exports").exists()


def test_cross_origin_reads_cannot_discover_token(tmp_path: Path) -> None:
    """A hostile page cannot read bootstrap or obtain an authorization token."""
    with running_studio(tmp_path) as client:
        response = client.request(
            "GET", "/api/bootstrap", headers={"Origin": "http://external.test"}
        )
        assert response.status == 403
        assert client.token.encode() not in response.body


def test_project_saves_increment_revision_and_reject_stale_edits(
    tmp_path: Path,
) -> None:
    """A stale editor receives a conflict without overwriting the newer project."""
    write_source(tmp_path)
    document = ProjectDocument(
        name="Original", nodes=(ProjectNode(id="resize", job=small_edit()),)
    )
    with running_studio(tmp_path) as client:
        response = client.post(
            "/api/project",
            SaveProject(
                path="projects/demo.json",
                project=document,
                expected_revision=0,
            ),
        )
        assert response.status == 200
        saved = response.parsed(SavedProject)
        assert saved.project.revision == 1
        changed = document.model_copy(update={"name": "Stale overwrite"})
        response = client.post(
            "/api/project",
            SaveProject(path=saved.path, project=changed, expected_revision=0),
        )
        assert response.status == 409
        loaded = client.request(
            "GET", "/api/project?" + urlencode({"path": saved.path})
        ).parsed(SavedProject)
        assert loaded.project.name == "Original"
        assert loaded.project.revision == 1
    assert (
        ProjectDocument.model_validate_json(
            (tmp_path / "projects/demo.json").read_text()
        ).name
        == "Original"
    )


def test_upload_roundtrips_bytes_and_supports_partial_reads(
    tmp_path: Path,
) -> None:
    """Uploaded media stays intact and file requests honor regular and suffix ranges."""
    with io.BytesIO() as buffer, Image.new("RGB", (6, 4), "lime") as image:
        image.save(buffer, format="PNG")
        original = buffer.getvalue()
    with running_studio(tmp_path) as client:
        response = client.request(
            "POST",
            "/api/upload?" + urlencode({"name": "family photo.png"}),
            original,
        )
        assert response.status == 201
        asset = response.parsed(Asset)
        assert (asset.width, asset.height) == (6, 4)
        route = "/api/file?" + urlencode({"path": asset.path})
        full = client.request("GET", route)
        assert full.status == 200
        assert full.body == original
        for requested, expected in (
            ("bytes=2-8", original[2:9]),
            ("bytes=-5", original[-5:]),
            ("bytes=4-", original[4:]),
        ):
            partial = client.request(
                "GET", route, headers={"Range": requested}
            )
            assert partial.status == 206
            assert partial.body == expected
            assert partial.headers["content-range"].endswith(
                f"/{len(original)}"
            )
        unsatisfied = client.request(
            "GET", route, headers={"Range": "bytes=99999-"}
        )
        assert unsatisfied.status == 416
        assert unsatisfied.body == b""
        head = client.request("HEAD", route, headers={"Range": "bytes=1-3"})
        assert head.status == 206
        assert head.headers["content-length"] == "3"
        assert head.body == b""


@pytest.mark.parametrize(
    "name", ["../escape.png", "subdir/escape.png", "..\\escape.png"]
)
def test_upload_rejects_traversal_names(tmp_path: Path, name: str) -> None:
    """Upload filenames cannot choose paths outside their assigned asset location."""
    with running_studio(tmp_path) as client:
        response = client.request(
            "POST",
            "/api/upload?" + urlencode({"name": name}),
            b"not published",
        )
        assert response.status == 400
        assert "filename" in response.parsed(ErrorResponse).message
    assert not (tmp_path / "assets").exists()


def test_file_and_operation_paths_cannot_follow_external_symlinks(
    tmp_path: Path,
) -> None:
    """Resolved file reads and output parents remain confined to the workspace."""
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    write_source(workspace)
    write_source(outside)
    with running_studio(workspace) as client:
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
        response = client.request(
            "GET", "/api/file?" + urlencode({"path": "escape/source.png"})
        )
        assert response.status == 400
        assert "escapes" in response.parsed(ErrorResponse).message
        response = client.post(
            "/api/jobs", JobSubmission(job=small_edit("escape/edited.png"))
        )
        assert response.status == 400
    assert not (outside / "edited.png").exists()


def test_upload_parent_symlink_cannot_write_outside_workspace(
    tmp_path: Path,
) -> None:
    """A redirected assets directory cannot receive a server-side upload."""
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    write_source(workspace)
    external = write_source(outside)
    original = external.read_bytes()
    with running_studio(workspace) as client:
        (workspace / "assets").symlink_to(outside, target_is_directory=True)
        response = client.request(
            "POST", "/api/upload?name=photo.png", original
        )
        assert response.status == 400
        assert "escapes" in response.parsed(ErrorResponse).message
    assert list(outside.iterdir()) == [external]


def test_asset_discovery_ignores_outside_symlink_without_breaking_bootstrap(
    tmp_path: Path,
) -> None:
    """One unrelated external symlink cannot hide the editor's valid local assets."""
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    write_source(workspace)
    external = write_source(outside)
    (workspace / "outside.png").symlink_to(external)
    with running_studio(workspace) as client:
        response = client.request("GET", "/api/bootstrap")
        assert response.status == 200
        assert tuple(
            asset.path for asset in response.parsed(Bootstrap).assets
        ) == ("source.png",)
        response = client.request("GET", "/api/file?path=outside.png")
        assert response.status == 400


def test_named_file_reads_reject_traversal_and_reserved_storage(
    tmp_path: Path,
) -> None:
    """External sources and private worker records never become download routes."""
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    write_source(workspace)
    write_source(outside)
    with running_studio(workspace) as client:
        for path in (
            "../outside/source.png",
            ".studio/runs/private.json",
            ".studio/previews/../runs/private.json",
        ):
            response = client.request(
                "GET", "/api/file?" + urlencode({"path": path})
            )
            assert response.status in {400, 403}
            assert (
                "escapes" in response.parsed(ErrorResponse).message
                or "reserved" in response.parsed(ErrorResponse).message
                or "preview" in response.parsed(ErrorResponse).message.lower()
            )


def test_preview_serves_canonical_pixels_and_rejects_invalid_frames(
    tmp_path: Path,
) -> None:
    """CPU previews return exact source pixels and never publish invalid selections."""
    write_source(tmp_path)
    with running_studio(tmp_path) as client:
        request = PreviewSubmission(composition=small_composition(), frame=3)
        response = client.post("/api/preview", request)
        assert response.status == 200
        preview = response.parsed(PreviewResult)
        image_response = client.request(
            "GET", "/api/file?" + urlencode({"path": preview.path})
        )
        with Image.open(io.BytesIO(image_response.body)) as image:
            assert image.size == (4, 2)
            assert image.getpixel((0, 0)) == (255, 0, 0)
            assert image.getpixel((3, 0)) == (0, 0, 255)
        response = client.post(
            "/api/preview", request.model_copy(update={"frame": 6})
        )
        assert response.status == 400
        assert "outside" in response.parsed(ErrorResponse).message


def test_validation_normalizes_requests_without_rendering(
    tmp_path: Path,
) -> None:
    """Preflight returns workspace-bound inputs and destinations without running a job."""
    write_source(tmp_path)
    with running_studio(tmp_path) as client:
        response = client.post(
            "/api/validate", JobSubmission(job=small_edit())
        )
        assert response.status == 200
        validated = response.parsed(ValidationResult)
        assert validated.outputs == (str(tmp_path / "exports/edited.png"),)
        assert not (tmp_path / "exports/edited.png").exists()


def test_reframed_and_bounded_previews_keep_viewport_headlines_visible(
    tmp_path: Path,
) -> None:
    """Aspect changes and reduced preview pixels retain text in its authored region."""
    write_source(tmp_path)
    base = small_composition()
    headline = TextLayer(
        id="headline",
        text="HELLO",
        font_size=72,
        size=Size(width=400, height=100),
        space="viewport",
        duration=1,
        keyframes=(LayerKeyframe(x=500, y=800),),
    )
    original = base.model_copy(
        update={
            "size": Size(width=1000, height=1000),
            "layers": (*base.layers, headline),
        }
    )
    with running_studio(tmp_path) as client:
        response = client.post(
            "/api/reframe",
            ReframeSubmission(
                composition=original, size=Size(width=200, height=200)
            ),
        )
        assert response.status == 200
        framed = response.parsed(ValidationResult).job
        assert isinstance(framed, ComposeRequest)
        for composition, expected_size in (
            (framed, (200, 200)),
            (original, (640, 640)),
        ):
            response = client.post(
                "/api/preview", PreviewSubmission(composition=composition)
            )
            assert response.status == 200
            result = response.parsed(PreviewResult)
            image_response = client.request(
                "GET", "/api/file?" + urlencode({"path": result.path})
            )
            with (
                Image.open(io.BytesIO(image_response.body)) as image,
                image.convert("L") as gray,
                gray.point(
                    [255 if value > 220 else 0 for value in range(256)]
                ) as white,
            ):
                assert image.size == expected_size
                bounds = white.getbbox()
                assert bounds is not None, "Viewport headline disappeared"
                assert abs((bounds[0] + bounds[2]) / 2 - image.width / 2) < 5
                assert image.height / 2 < bounds[1] < bounds[3] < image.height
