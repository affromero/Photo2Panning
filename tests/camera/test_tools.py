"""Check reconstruction publication and camera timeline safety."""

import json
import shutil
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from pic2panning.api import reconstruct_scene, render_camera
from pic2panning.schemas.base import Size
from pic2panning.schemas.camera import (
    CameraPose,
    CameraRequest,
    SceneCalibration,
)
from pic2panning.schemas.scene import SceneRequest
from pic2panning.tools.camera import timeline_frames
from pic2panning.tools.io import file_sha256
from tests.camera.test_cli import CameraBoundary, calibration_document


class PredictionBoundary:
    """Write a synthetic model output through the real publication pipeline."""

    failure: bool
    points: int | None

    def __init__(
        self, *, failure: bool = False, points: int | None = None
    ) -> None:
        """Optionally reproduce a model failing after a partial write."""
        self.failure = failure
        self.points = points

    def predict(
        self,
        request: SceneRequest,
        image: Image.Image,
        output: str,
        source_sha256: str,
    ) -> tuple[SceneCalibration, int]:
        """Represent only the model/filesystem boundary, preserving real I/O."""
        Path(output).write_bytes(b"ply\nsynthetic test geometry\n")
        if self.failure:
            raise RuntimeError("Model failed after writing partial geometry")
        if request.device != "cpu":
            raise ValueError("This test boundary expects explicit CPU work")
        document = calibration_document()
        document.update(
            {
                "width": image.width,
                "height": image.height,
                "source_sha256": source_sha256,
            }
        )
        count = image.width if self.points is None else self.points
        return SceneCalibration.model_validate(document), count


class FailedCameraBoundary(CameraBoundary):
    """Fail a GPU render after the panning phase has reached FFmpeg."""

    @staticmethod
    def render(camera: CameraPose) -> Image.Image:
        """Expose a native failure without producing a replacement frame."""
        raise RuntimeError(f"Rendering failed at width {camera.width}")


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """Create independent source pixels for real filesystem operations."""
    source = tmp_path / "source.png"
    row = b"".join(bytes((x * 4, 0, 255 - x * 4)) for x in range(64))
    with Image.frombytes("RGB", (64, 32), row * 32) as image:
        image.save(source)
    return source


def test_reconstruction_publishes_bound_geometry_and_provenance(
    tmp_path: Path, photo: Path
) -> None:
    """A saved scene is reusable and its manifest binds exact artifact bytes."""
    output = tmp_path / "scene.ply"
    metadata = tmp_path / "scene.json"
    result = reconstruct_scene(
        SceneRequest(
            source=str(photo),
            output=str(output),
            metadata=str(metadata),
            device="cpu",
        ),
        PredictionBoundary(),
    )
    calibration = SceneCalibration.from_sidecar(str(metadata))
    assert result.points == 64
    assert result.output == str(output)
    assert result.metadata == str(metadata)
    assert calibration.source_sha256 == result.source_sha256
    assert calibration.source_sha256 == file_sha256(str(photo))
    assert calibration.scene_sha256 == file_sha256(str(output))
    document = json.loads(metadata.read_text())
    assert document["reconstruction"]["license"] == "MIT"
    assert document["reconstruction"]["device"] == "cpu"
    assert document["reconstruction"]["max_edge"] == 1920


@pytest.mark.parametrize("existing", [False, True])
def test_failed_reconstruction_preserves_prior_artifacts(
    tmp_path: Path, photo: Path, *, existing: bool
) -> None:
    """Partial geometry never replaces an existing scene or calibration."""
    output = tmp_path / "scene.ply"
    metadata = tmp_path / "scene.json"
    if existing:
        output.write_bytes(b"previous scene")
        metadata.write_text("previous calibration")
    with pytest.raises(RuntimeError, match="partial geometry"):
        reconstruct_scene(
            SceneRequest(
                source=str(photo),
                output=str(output),
                metadata=str(metadata),
                device="cpu",
                overwrite=existing,
            ),
            PredictionBoundary(failure=True),
        )
    if existing:
        assert output.read_bytes() == b"previous scene"
        assert metadata.read_text() == "previous calibration"
    else:
        assert not output.exists()
        assert not metadata.exists()
    assert not list(tmp_path.glob(".pic2panning-*"))


def test_invalid_prediction_result_cannot_publish_artifacts(
    tmp_path: Path, photo: Path
) -> None:
    """Validate runtime backend results before publishing either artifact."""
    output = tmp_path / "scene.ply"
    metadata = tmp_path / "scene.json"
    with pytest.raises(ValidationError, match="points"):
        reconstruct_scene(
            SceneRequest(
                source=str(photo),
                output=str(output),
                metadata=str(metadata),
                device="cpu",
            ),
            PredictionBoundary(points=True),
        )
    assert not output.exists()
    assert not metadata.exists()
    assert not list(tmp_path.glob(".pic2panning-*"))


@pytest.fixture
def camera_job(tmp_path: Path, photo: Path) -> CameraRequest:
    """Bind valid test geometry to its exact source pixels."""
    scene = tmp_path / "scene.ply"
    metadata = tmp_path / "scene.json"
    scene.write_bytes(b"synthetic geometry")
    document = calibration_document()
    document["source_sha256"] = file_sha256(str(photo))
    document["scene_sha256"] = file_sha256(str(scene))
    metadata.write_text(json.dumps(document))
    return CameraRequest(
        source=str(photo),
        scene=str(scene),
        metadata=str(metadata),
        output=str(tmp_path / "camera.mp4"),
        size=Size(width=32, height=64),
        before=0.3,
        duration=0.8,
        after=0.3,
        fps=10,
        hold=0,
    )


@pytest.mark.parametrize("damaged", ["source", "scene"])
def test_mismatched_geometry_is_rejected_before_loading_a_model(
    camera_job: CameraRequest, damaged: str
) -> None:
    """Modified source or scene files cannot reuse stale calibration."""
    damaged_path = (
        camera_job.source if damaged == "source" else camera_job.scene
    )
    Path(damaged_path).write_bytes(b"changed bytes")
    boundary = CameraBoundary()
    with pytest.raises(ValueError, match="differs"):
        render_camera(camera_job, boundary)
    assert boundary.loaded_scene is None
    assert not Path(camera_job.output).exists()
    assert not list(Path(camera_job.output).parent.glob(".pic2panning-*"))


@pytest.mark.parametrize("descending", [False, True])
def test_camera_returns_to_identical_photo_pixels_before_resuming(
    camera_job: CameraRequest, *, descending: bool
) -> None:
    """Both pan directions preserve exact pause anchors without holds."""
    if descending:
        camera_job = camera_job.model_copy(
            update={"pan_start": 1, "pan_end": 0}
        )
    calibration = SceneCalibration.from_sidecar(camera_job.metadata)
    boundary = CameraBoundary()
    with (
        Image.open(camera_job.source) as image,
        closing(
            timeline_frames(image, camera_job, boundary, calibration)
        ) as frames,
    ):
        pixels = consume_pixels(frames)
    assert len(pixels) == 14
    assert pixels[0] != pixels[-1]
    assert (pixels[0][0] > pixels[-1][0]) is descending
    assert pixels[2] == pixels[3] == pixels[10] == pixels[11]
    assert pixels[6] != pixels[3]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="This check exercises real FFmpeg"
)
def test_failed_camera_closes_its_scene_and_preserves_an_existing_output(
    camera_job: CameraRequest,
) -> None:
    """A native failure cleans its partial video and retains the prior file."""
    output = Path(camera_job.output)
    output.write_bytes(b"previous video")
    boundary = FailedCameraBoundary()
    with pytest.raises(RuntimeError, match="Rendering failed"):
        render_camera(
            camera_job.model_copy(update={"overwrite": True}), boundary
        )
    assert boundary.closed
    assert output.read_bytes() == b"previous video"
    assert not list(output.parent.glob(".pic2panning-*"))


def consume_pixels(frames: Iterator[Image.Image]) -> list[bytes]:
    """Retain tiny test pixels while closing each consumed frame."""
    result: list[bytes] = []
    for frame in frames:
        with frame:
            result.append(frame.tobytes())
    return result
