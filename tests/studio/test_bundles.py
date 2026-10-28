"""Real HTTP imports preserve and authenticate cached depth companions."""

from pathlib import Path
from urllib.parse import urlencode

import pytest

from pic2panning.schemas.creative.depth import DepthRequest, DepthResult
from pic2panning.studio.schemas import (
    Asset,
    DepthAttachment,
    ErrorResponse,
    VerifiedDepth,
)
from pic2panning.tools.creative.depth import extract_depth, load_depth
from pic2panning.tools.io import file_sha256
from tests.creative_depth.test_depth import PlaneBoundary
from tests.studio import StudioClient, running_studio, write_source

BATCH = "a1" * 16


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, DepthResult]:
    """Create real artifacts with only the external model boundary replaced."""
    directory = tmp_path / "outside" / "native"
    source = write_source(directory)
    result = extract_depth(
        DepthRequest(
            source=str(source),
            output=str(directory / "terrain.tiff"),
            device="cpu",
            num_tokens=1200,
            max_edge=256,
        ),
        PlaneBoundary(),
    )
    return source, result


def upload_file(client: StudioClient, path: Path, batch: str) -> Asset:
    """Send the unchanged bytes through the public grouped-upload route."""
    response = client.request(
        "POST",
        "/api/upload?" + urlencode({"name": path.name, "batch": batch}),
        path.read_bytes(),
    )
    assert response.status == 201, response.body.decode()
    return response.parsed(Asset)


@pytest.mark.parametrize("explicit_source", [False, True])
def test_imported_depth_bundle_preserves_paths_and_source_binding(
    tmp_path: Path,
    bundle: tuple[Path, DepthResult],
    *,
    explicit_source: bool,
) -> None:
    """Uploaded siblings load with exact metric data and their original photo."""
    source, result = bundle
    workspace = tmp_path / "workspace"
    with running_studio(workspace) as client:
        upload_file(client, source, "b2" * 16)
        imported = upload_file(client, source, BATCH)
        artifacts = {
            path: upload_file(client, Path(path), BATCH)
            for path in result.artifacts
        }
        metadata = artifacts[result.metadata]
        response = client.post(
            "/api/depth",
            DepthAttachment(
                metadata=metadata.path,
                source=imported.path if explicit_source else None,
            ),
        )
        assert response.status == 200, response.body.decode()
        verified = response.parsed(VerifiedDepth)
        assert verified.source == imported.path
        assert verified.metadata == metadata.path
        assert verified.focus_metres == pytest.approx(2)
        with load_depth(
            str(workspace / verified.metadata),
            str(workspace / verified.source),
        ) as asset:
            assert asset.depth.mode == "F"
            assert asset.depth.size == (4, 2)
            assert asset.depth.getpixel((3, 0)) == 2.75
            assert asset.metadata.source_sha256 == file_sha256(str(source))
        assert Path(metadata.path).parent == Path(imported.path).parent
        for path, artifact in artifacts.items():
            assert Path(artifact.path).name == Path(path).name
            assert (workspace / artifact.path).read_bytes() == Path(
                path
            ).read_bytes()


def test_imported_depth_rejects_changed_artifact_bytes(
    tmp_path: Path, bundle: tuple[Path, DepthResult]
) -> None:
    """A plausible TIFF with changed bytes cannot satisfy its saved manifest."""
    source, result = bundle
    with running_studio(tmp_path / "workspace") as client:
        imported = upload_file(client, source, BATCH)
        for path in result.artifacts:
            payload = Path(path).read_bytes()
            if path == result.output:
                payload += b"changed metric artifact"
            response = client.request(
                "POST",
                "/api/upload?"
                + urlencode({"name": Path(path).name, "batch": BATCH}),
                payload,
            )
            assert response.status == 201, response.body.decode()
        response = client.post(
            "/api/depth",
            DepthAttachment(
                metadata=f"assets/{BATCH}/{Path(result.metadata).name}",
                source=imported.path,
            ),
        )
        assert response.status == 400
        assert "checksum" in response.parsed(ErrorResponse).message.lower()


@pytest.mark.parametrize(
    ("name", "batch"),
    [
        ("../escape.png", BATCH),
        ("source.png", "../escape"),
        ("source.png", "z" * 32),
    ],
)
def test_grouped_upload_rejects_path_and_batch_traversal(
    tmp_path: Path, bundle: tuple[Path, DepthResult], name: str, batch: str
) -> None:
    """Invalid upload identifiers fail before writing any imported asset."""
    source = bundle[0]
    workspace = tmp_path / "workspace"
    with running_studio(workspace) as client:
        response = client.request(
            "POST",
            "/api/upload?" + urlencode({"name": name, "batch": batch}),
            source.read_bytes(),
        )
        assert response.status == 400
        assert not (workspace / "assets").exists()
        assert source.is_file()


def test_grouped_upload_never_replaces_an_existing_companion(
    tmp_path: Path, bundle: tuple[Path, DepthResult]
) -> None:
    """Duplicate filenames in one batch preserve the first uploaded bytes."""
    source = bundle[0]
    workspace = tmp_path / "workspace"
    with running_studio(workspace) as client:
        imported = upload_file(client, source, BATCH)
        response = client.request(
            "POST",
            "/api/upload?" + urlencode({"name": source.name, "batch": BATCH}),
            source.read_bytes() + b"replacement",
        )
        assert response.status == 400
        assert response.parsed(ErrorResponse).error == "FileExistsError"
        assert (workspace / imported.path).read_bytes() == source.read_bytes()
