"""Saved projects execute real operations and reuse authenticated outputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from pic2panning.api import run_operation, run_project
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
    PreviewRequest,
    TextLayer,
)
from pic2panning.schemas.creative.depth import (
    DepthArtifact,
    DepthMetadata,
    DepthRequest,
    DepthStyle,
)
from pic2panning.schemas.creative.project import (
    ProjectDocument,
    ProjectNode,
    ProjectRequest,
    ProjectResult,
)
from pic2panning.schemas.creative.surface import SolidText, SurfaceAttachment
from pic2panning.schemas.editing import Crop, EditRequest, Flip
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.models import GenerateRequest
from pic2panning.tools.creative.depth import load_depth
from pic2panning.tools.io import file_sha256
from pic2panning.tools.projects.cache import (
    CachedArtifact,
    CacheEntry,
    entry_path,
    fingerprint,
    input_hashes,
    read_entry,
    save_entry,
    workspace_lock,
)
from pic2panning.tools.projects.registry import file_inputs, resolve
from pic2panning.tools.projects.runtime import preflight, publish


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """Write source pixels whose horizontal order survives lossless edits."""
    source = tmp_path / "photo.png"
    with Image.new("RGB", (8, 4)) as image:
        image.putdata(
            [(column * 20, 40, 90) for _ in range(4) for column in range(8)]
        )
        image.save(source)
    return source


def edit_node(
    identifier: str,
    source: str = "photo.png",
    output: str = "edited.png",
    dependencies: tuple[str, ...] = (),
) -> ProjectNode:
    """Describe a real deterministic edit using project-relative paths."""
    return ProjectNode(
        id=identifier,
        job=EditRequest(source=source, output=output, operations=(Flip(),)),
        depends_on=dependencies,
    )


def project_request(
    workspace: Path,
    nodes: tuple[ProjectNode, ...],
    report: str = "report.json",
    *,
    overwrite: bool = False,
) -> ProjectRequest:
    """Create an ordinary saved graph without replacing the operation runner."""
    return ProjectRequest(
        project=ProjectDocument(name="Behavioral project", nodes=nodes),
        workspace=str(workspace),
        output=report,
        overwrite=overwrite,
    )


def test_project_runs_dependencies_and_reports_declared_paths(
    tmp_path: Path, photo: Path
) -> None:
    """Topological execution produces expected pixels and a durable report."""
    first = edit_node("first", output="first.png")
    second = ProjectNode(
        id="second",
        depends_on=("first",),
        job=EditRequest(
            source="first.png",
            output="second.png",
            operations=(Crop(left=0, top=0, right=4, bottom=4),),
        ),
    )
    result = run_project(project_request(tmp_path, (second, first)))
    assert tuple(node.id for node in result.nodes) == ("first", "second")
    assert all(not node.cache_hit for node in result.nodes)
    assert result.nodes[0].input_hashes[str(photo)] == file_sha256(str(photo))
    with Image.open(tmp_path / "second.png") as image:
        assert image.size == (4, 4)
        np.testing.assert_array_equal(
            np.asarray(image)[0, :, 0], [140, 120, 100, 80]
        )
    persisted = ProjectResult.model_validate_json(
        (tmp_path / "report.json").read_text()
    )
    assert persisted == result
    assert result.nodes[1].result["outputs"] == [str(tmp_path / "second.png")]
    assert ".pic2panning/cache" not in json.dumps(result.nodes[1].result)


def test_project_reuses_same_bytes_at_a_new_destination(
    tmp_path: Path, photo: Path
) -> None:
    """Changing output names republishes authenticated cached pixels."""
    first = run_project(project_request(tmp_path, (edit_node("edit"),)))
    second = run_project(
        project_request(
            tmp_path,
            (edit_node("edit", output="renamed.png"),),
            "renamed-report.json",
        )
    )
    assert photo.exists()
    assert second.nodes[0].cache_hit
    assert second.nodes[0].fingerprint == first.nodes[0].fingerprint
    assert (tmp_path / "renamed.png").read_bytes() == (
        tmp_path / "edited.png"
    ).read_bytes()
    assert second.nodes[0].result["outputs"] == [str(tmp_path / "renamed.png")]


def test_changed_source_bytes_regenerate_the_project(
    tmp_path: Path, photo: Path
) -> None:
    """Input content changes invalidate a cache even at the same filename."""
    request = project_request(tmp_path, (edit_node("edit"),))
    first = run_project(request)
    with Image.new("RGB", (8, 4), "blue") as image:
        image.save(photo)
    second = run_project(request.model_copy(update={"overwrite": True}))
    assert not second.nodes[0].cache_hit
    assert second.nodes[0].fingerprint != first.nodes[0].fingerprint
    with Image.open(tmp_path / "edited.png") as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_output_format_changes_do_not_reuse_png_bytes(
    tmp_path: Path, photo: Path
) -> None:
    """Destination suffix remains part of the processing cache identity."""
    first = run_project(project_request(tmp_path, (edit_node("edit"),)))
    second = run_project(
        project_request(
            tmp_path,
            (edit_node("edit", output="edited.jpg"),),
            "jpeg-report.json",
        )
    )
    assert photo.exists()
    assert not second.nodes[0].cache_hit
    assert second.nodes[0].fingerprint != first.nodes[0].fingerprint
    with Image.open(tmp_path / "edited.jpg") as image:
        assert image.format == "JPEG"


def test_changed_cached_bytes_are_rejected_before_republication(
    tmp_path: Path, photo: Path
) -> None:
    """Corrupt cache contents surface an error and preserve requested outputs."""
    first = run_project(project_request(tmp_path, (edit_node("edit"),)))
    index = Path(entry_path(str(tmp_path), first.nodes[0].fingerprint))
    entry = CacheEntry.model_validate_json(index.read_text())
    Path(entry.artifacts[0].path).write_bytes(b"corrupted cache pixels")
    with pytest.raises(ValueError, match=r"[Cc]ached artifact|checksum"):
        run_project(
            project_request(
                tmp_path,
                (edit_node("edit", output="new.png"),),
                "new-report.json",
            )
        )
    assert photo.exists()
    assert not (tmp_path / "new.png").exists()
    assert not (tmp_path / "new-report.json").exists()
    refreshed = project_request(
        tmp_path,
        (edit_node("edit", output="new.png"),),
        "new-report.json",
    ).model_copy(update={"cache": "refresh"})
    result = run_project(refreshed)
    assert not result.nodes[0].cache_hit
    with Image.open(tmp_path / "new.png") as image:
        assert image.size == (8, 4)


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_project_protects_linked_source_aliases_before_execution(
    tmp_path: Path, photo: Path, alias: str
) -> None:
    """Preflight rejects output aliases before a real edit can replace data."""
    output = tmp_path / "alias.png"
    if alias == "hardlink":
        os.link(photo, output)
    else:
        output.symlink_to(photo)
    original = photo.read_bytes()
    with pytest.raises(ValueError, match=r"alias|symbolic|depend"):
        run_project(
            project_request(
                tmp_path,
                (edit_node("edit", output="alias.png"),),
                overwrite=True,
            )
        )
    assert photo.read_bytes() == original
    assert not (tmp_path / "report.json").exists()


def test_missing_producer_dependency_fails_before_any_node_runs(
    tmp_path: Path, photo: Path
) -> None:
    """Reading a declared output requires an explicit dependency edge."""
    nodes = (
        edit_node("producer", output="produced.png"),
        edit_node("consumer", source="produced.png", output="final.png"),
    )
    with pytest.raises(ValueError, match="depend"):
        run_project(project_request(tmp_path, nodes))
    assert photo.exists()
    assert not (tmp_path / "produced.png").exists()
    assert not (tmp_path / "final.png").exists()


def test_duplicate_producers_fail_before_any_node_runs(
    tmp_path: Path, photo: Path
) -> None:
    """Two operations cannot publish the same artifact filename."""
    with pytest.raises(ValueError, match=r"Duplicate.*output"):
        run_project(
            project_request(tmp_path, (edit_node("one"), edit_node("two")))
        )
    assert photo.exists()
    assert not (tmp_path / "edited.png").exists()


def test_report_cannot_replace_a_source_file(tmp_path: Path) -> None:
    """The JSON report receives the same alias protection as media outputs."""
    source = tmp_path / "source.json"
    with Image.new("RGB", (4, 4), "red") as image:
        image.save(source, format="PNG")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="alias"):
        run_project(
            project_request(
                tmp_path,
                (edit_node("edit", source="source.json"),),
                "source.json",
                overwrite=True,
            )
        )
    assert source.read_bytes() == original
    assert not (tmp_path / "edited.png").exists()


def test_declared_outputs_cannot_modify_private_cache_storage(
    tmp_path: Path, photo: Path
) -> None:
    """Project files never overwrite their internal immutable cache objects."""
    with pytest.raises(ValueError, match="cache"):
        run_project(
            project_request(
                tmp_path,
                (edit_node("edit", output=".pic2panning/cache/edited.png"),),
            )
        )
    assert photo.exists()


def test_failed_operation_removes_its_unpublished_cache_stage(
    tmp_path: Path, photo: Path
) -> None:
    """A runtime crop error leaves no cache index or orphan generation folder."""
    node = ProjectNode(
        id="invalid_crop",
        job=EditRequest(
            source=photo.name,
            output="invalid.png",
            operations=(Crop(left=0, top=0, right=100, bottom=100),),
        ),
    )
    with pytest.raises(ValueError, match=r"[Cc]rop|bounds"):
        run_project(project_request(tmp_path, (node,)))
    cache = tmp_path / ".pic2panning" / "cache"
    assert not list(cache.rglob("*.json"))
    assert not list((cache / "objects").glob("*"))
    assert not (tmp_path / "invalid.png").exists()


def test_cache_off_keeps_generated_artifacts_outside_persistent_storage(
    tmp_path: Path, photo: Path
) -> None:
    """Disabling cache still publishes real edits and their result report."""
    request = project_request(tmp_path, (edit_node("edit"),))
    result = run_project(request.model_copy(update={"cache": "off"}))
    assert photo.exists()
    assert not result.nodes[0].cache_hit
    assert (tmp_path / "edited.png").exists()
    assert not (tmp_path / ".pic2panning" / "cache").exists()


def test_reference_order_and_content_change_generation_identity(
    tmp_path: Path, photo: Path
) -> None:
    """Every ordered conditioning image contributes its actual file content."""
    other = tmp_path / "reference.png"
    with Image.new("RGB", (8, 4), "blue") as image:
        image.save(other)
    request = GenerateRequest(
        output="generated.png",
        prompt="Keep both reference subjects",
        references=(photo.name, other.name),
    )
    first = resolve(request, str(tmp_path))
    reversed_order = resolve(
        request.model_copy(update={"references": (other.name, photo.name)}),
        str(tmp_path),
    )
    key = fingerprint(first, input_hashes(first), "same-processing-engine")
    assert key != fingerprint(
        reversed_order, input_hashes(reversed_order), "same-processing-engine"
    )
    with Image.new("RGB", (8, 4), "red") as image:
        image.save(other)
    assert key != fingerprint(
        first, input_hashes(first), "same-processing-engine"
    )


def depth_metadata(directory: Path, photo: Path) -> Path:
    """Prepare authenticated analytical depth files without model inference."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "depth": directory / "cached.tiff",
        "preview": directory / "cached.preview.png",
        "normals": directory / "cached.normals.png",
        "mask": directory / "cached.mask.png",
    }
    for name, mode, color in (
        ("depth", "F", 3.0),
        ("preview", "I;16", 65535),
        ("normals", "RGB", (128, 128, 0)),
        ("mask", "L", 255),
    ):
        with Image.new(mode, (8, 4), color) as image:
            image.save(paths[name])
    artifacts = {
        name: DepthArtifact(path=path.name, sha256=file_sha256(str(path)))
        for name, path in paths.items()
    }
    document = DepthMetadata(
        source_sha256=file_sha256(str(photo)),
        width=8,
        height=4,
        grid_width=8,
        grid_height=4,
        intrinsics=((8, 0, 4), (0, 4, 2), (0, 0, 1)),
        depth=artifacts["depth"],
        preview=artifacts["preview"],
        normals=artifacts["normals"],
        mask=artifacts["mask"],
        preview_near_metres=3,
        preview_far_metres=3,
        device="cpu",
        num_tokens=1200,
        max_edge=256,
    )
    target = directory / "cached.depth.json"
    target.write_text(document.model_dump_json())
    return target


def composition(photo: Path, metadata: Path | None = None) -> ComposeRequest:
    """Describe a small image composition served by the real frame pipeline."""
    return ComposeRequest(
        canvas=Size(width=8, height=4),
        duration=1,
        camera=(CameraKeyframe(x=4, y=2),),
        layers=(
            ImageLayer(
                id="photo",
                source=str(photo),
                duration=1,
                size=Size(width=8, height=4),
                keyframes=(LayerKeyframe(x=4, y=2),),
                depth=DepthStyle(metadata=str(metadata)) if metadata else None,
            ),
        ),
        output="composition-export.mp4",
        size=Size(width=8, height=4),
        fps=2,
    )


def test_custom_native_executable_resolves_under_workspace_and_changes_cache(
    tmp_path: Path, photo: Path
) -> None:
    """An explicit Blender binary contributes its actual bytes to project identity."""
    metadata = depth_metadata(tmp_path / "depth", photo)
    executable = tmp_path / "bin" / "blender"
    executable.parent.mkdir()
    executable.write_bytes(b"native executable version one")
    base = composition(Path(photo.name), metadata.relative_to(tmp_path))
    text = TextLayer(
        id="solid",
        text="X",
        font_size=2,
        size=Size(width=8, height=4),
        duration=1,
        keyframes=(LayerKeyframe(x=4, y=2),),
        surface=SurfaceAttachment(layer_id="photo", region=(0, 0, 8, 4)),
        solid=SolidText(executable="bin/blender"),
    )
    node = ProjectNode(
        id="compose",
        job=base.model_copy(update={"layers": (*base.layers, text)}),
    )
    request = project_request(tmp_path, (node,))
    operations, protected = preflight(
        request, str(tmp_path), request.project.nodes
    )
    resolved = operations[node.id]
    assert resolved.kind == "compose"
    caption = resolved.layers[-1]
    assert isinstance(caption, TextLayer)
    assert caption.solid is not None
    assert caption.solid.executable == str(executable)
    assert str(executable) in file_inputs(resolved)
    assert str(executable) in protected
    before = input_hashes(resolved)
    assert before[str(executable)] == file_sha256(str(executable))
    identity = fingerprint(resolved, before, "same-processing-engine")
    executable.write_bytes(b"native executable version two")
    after = input_hashes(resolved)
    assert after[str(executable)] != before[str(executable)]
    assert fingerprint(resolved, after, "same-processing-engine") != identity


def test_transitive_depth_artifacts_require_their_producer_dependency(
    tmp_path: Path, photo: Path
) -> None:
    """Manifest references participate in the graph before any work starts."""
    metadata = depth_metadata(tmp_path / "depth", photo)
    producer = edit_node("normals", output="depth/cached.normals.png")
    preview = ProjectNode(
        id="preview",
        job=PreviewRequest(
            composition=composition(photo, metadata), output="preview.png"
        ),
    )
    request = project_request(tmp_path, (producer, preview), overwrite=True)
    with pytest.raises(ValueError, match="depend"):
        preflight(request, str(tmp_path), request.project.nodes)


def test_model_free_project_preview_uses_real_composition_pixels(
    tmp_path: Path, photo: Path
) -> None:
    """The project runner reaches the same public preview operation."""
    node = ProjectNode(
        id="preview",
        job=PreviewRequest(
            composition=composition(photo), output="preview.png"
        ),
    )
    result = run_project(project_request(tmp_path, (node,)))
    assert not result.nodes[0].cache_hit
    with (
        Image.open(tmp_path / "preview.png") as image,
        Image.open(photo) as source,
    ):
        np.testing.assert_array_equal(np.asarray(image), np.asarray(source))


def test_preview_cache_ignores_its_unused_composition_export_name(
    tmp_path: Path, photo: Path
) -> None:
    """A preview depends on composition content and its own output format."""
    first_composition = composition(photo)
    first = ProjectNode(
        id="preview",
        job=PreviewRequest(composition=first_composition, output="one.png"),
    )
    run_project(project_request(tmp_path, (first,)))
    second = ProjectNode(
        id="preview",
        job=PreviewRequest(
            composition=first_composition.model_copy(
                update={"output": "renamed-composition.mp4"}
            ),
            output="two.png",
        ),
    )
    result = run_project(project_request(tmp_path, (second,), "two.json"))
    assert result.nodes[0].cache_hit
    assert (tmp_path / "one.png").read_bytes() == (
        tmp_path / "two.png"
    ).read_bytes()


def test_depth_manifest_becomes_visible_after_all_rebound_artifacts(
    tmp_path: Path, photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readers never see a new depth manifest referencing unpublished files."""
    metadata = depth_metadata(tmp_path / "cache", photo)
    sources = tuple(
        tmp_path / "cache" / name
        for name in (
            "cached.tiff",
            "cached.preview.png",
            "cached.normals.png",
            "cached.mask.png",
            "cached.depth.json",
        )
    )
    entry = CacheEntry(
        fingerprint="depth-publication",
        artifacts=tuple(
            CachedArtifact(path=str(path), sha256=file_sha256(str(path)))
            for path in sources
        ),
        result={"output": str(sources[0]), "metadata": str(metadata)},
        models={},
    )
    request = DepthRequest(
        source=str(photo), output=str(tmp_path / "published.tiff")
    )
    link = os.link
    observed: list[bool] = []

    def observe_publication(source: str, destination: str) -> None:
        """Observe actual filesystem visibility at the atomic link boundary."""
        link(source, destination)
        if destination == request.metadata:
            with load_depth(request.metadata, str(photo)) as asset:
                observed.append(asset.depth.getextrema() == (3.0, 3.0))

    monkeypatch.setattr(os, "link", observe_publication)
    publish(request, entry, (str(photo),), overwrite=False)
    assert observed == [True]


@pytest.mark.parametrize("symbolic", [False, True])
def test_cache_indices_cannot_reference_files_outside_owned_objects(
    tmp_path: Path, photo: Path, *, symbolic: bool
) -> None:
    """An edited index cannot republish an arbitrary file as generated media."""
    artifact = photo
    if symbolic:
        artifact = (
            tmp_path
            / ".pic2panning"
            / "cache"
            / "objects"
            / "entry"
            / "linked.png"
        )
        artifact.parent.mkdir(parents=True)
        artifact.symlink_to(photo)
    key = "a" * 64
    save_entry(
        str(tmp_path),
        CacheEntry(
            fingerprint=key,
            artifacts=(
                CachedArtifact(
                    path=str(artifact), sha256=file_sha256(str(photo))
                ),
            ),
            result={"outputs": [str(artifact)]},
            models={},
        ),
    )
    with pytest.raises(ValueError, match=r"cache|[Cc]ached|symbolic|outside"):
        read_entry(str(tmp_path), key, {})


@pytest.mark.parametrize("component", ["cache", "objects"])
def test_private_cache_directories_cannot_redirect_project_writes(
    tmp_path: Path, photo: Path, component: str
) -> None:
    """An internal directory symlink cannot redirect stages into other data."""
    external = tmp_path / "other-data"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("Keep unrelated files")
    cache = tmp_path / ".pic2panning" / "cache"
    target = cache if component == "cache" else cache / "objects"
    target.parent.mkdir(parents=True)
    target.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match=r"cache|[Cc]ached|symbolic"):
        run_project(project_request(tmp_path, (edit_node("edit"),)))
    assert photo.exists()
    assert tuple(external.iterdir()) == (marker,)
    assert marker.read_text() == "Keep unrelated files"
    assert not (tmp_path / "edited.png").exists()


def test_recorded_model_revision_changes_invalidate_a_cache_entry(
    tmp_path: Path, photo: Path
) -> None:
    """Identical input pixels cannot reuse an output from another model."""
    result = run_project(project_request(tmp_path, (edit_node("edit"),)))
    key = result.nodes[0].fingerprint
    entry = CacheEntry.model_validate_json(
        Path(entry_path(str(tmp_path), key)).read_text()
    )
    save_entry(
        str(tmp_path),
        entry.model_copy(update={"models": {"checkpoint": "a" * 40}}),
    )
    assert read_entry(str(tmp_path), key, {"checkpoint": "a" * 40}) is not None
    assert read_entry(str(tmp_path), key, {"checkpoint": "b" * 40}) is None
    assert photo.exists()


def test_kernel_lock_releases_when_a_worker_is_terminated(
    tmp_path: Path,
) -> None:
    """Canceling one worker cannot leave the workspace permanently locked."""
    program = """
import signal
import sys
from pic2panning.tools.projects.cache import workspace_lock
with workspace_lock(sys.argv[1]):
    print('locked', flush=True)
    signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("Worker did not expose stdout")
        assert process.stdout.readline() == b"locked\n"
        with (
            pytest.raises(RuntimeError, match=r"running|locked"),
            workspace_lock(str(tmp_path)),
        ):
            pass
        process.terminate()
        process.wait(timeout=15)
        with workspace_lock(str(tmp_path)):
            assert process.returncode is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def test_cyclic_project_is_rejected_during_schema_validation() -> None:
    """Cycles never reach filesystem access or the canonical operation runner."""
    with pytest.raises(ValidationError, match="cycle"):
        ProjectDocument(
            name="Cyclic",
            nodes=(
                edit_node("one", dependencies=("two",)),
                edit_node("two", dependencies=("one",)),
            ),
        )


def test_project_and_standalone_edits_produce_same_pixels(
    tmp_path: Path, photo: Path
) -> None:
    """Standalone dispatch writes the same deterministic edit bytes."""
    operation = EditRequest(
        source=str(photo),
        output=str(tmp_path / "standalone.png"),
        operations=(Flip(),),
    )
    standalone = run_operation(operation)
    assert isinstance(standalone, JobResult)
    run_project(project_request(tmp_path, (edit_node("edit"),)))
    assert standalone.outputs == (str(tmp_path / "standalone.png"),)
    assert (tmp_path / "standalone.png").read_bytes() == (
        tmp_path / "edited.png"
    ).read_bytes()
