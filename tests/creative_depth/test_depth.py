"""Depth artifacts retain metric data and bind it to the source photo."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from pic2panning.backends.creative.depth import DepthPrediction
from pic2panning.schemas.creative.depth import (
    MOGE_REVISION,
    MOGE_SHA256,
    DepthMetadata,
    DepthRequest,
)
from pic2panning.tools.creative.depth import extract_depth, load_depth
from pic2panning.tools.io import file_sha256


class PlaneBoundary:
    """Return synthetic model outputs while exercising real artifact I/O."""

    malformed: str | None

    def __init__(self, malformed: str | None = None) -> None:
        """Optionally reproduce an invalid external model prediction."""
        self.malformed = malformed

    def predict(
        self, request: DepthRequest, image: Image.Image
    ) -> DepthPrediction:
        """Predict a finite depth ramp and an independent foreground mask."""
        if request.device != "cpu":
            raise ValueError("Synthetic boundary requires explicit CPU work")
        if self.malformed == "failure":
            raise RuntimeError("Synthetic depth prediction failed")
        width, height = image.size
        depth = Image.new("F", image.size)
        depth.putdata(
            [
                1.25 + 0.5 * column
                for _ in range(height)
                for column in range(width)
            ]
        )
        mask = Image.new("L", image.size, "white")
        if self.malformed == "depth-size":
            depth.close()
            depth = Image.new("F", (width + 1, height), 2)
        if self.malformed == "mask-size":
            mask.close()
            mask = Image.new("L", (width + 1, height), "white")
        if self.malformed == "depth-mode":
            depth.close()
            depth = Image.new("L", image.size, 2)
        if self.malformed in {"nan", "zero", "negative"}:
            value = {"nan": float("nan"), "zero": 0.0, "negative": -1.0}[
                self.malformed
            ]
            depth.putpixel((0, 0), value)
        return DepthPrediction(
            depth=depth,
            mask=mask,
            intrinsics=(
                (float(width), 0, width / 2),
                (0, float(height), height / 2),
                (0, 0, 1),
            ),
        )


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """Create a real source whose bytes and dimensions can be verified."""
    source = tmp_path / "photo.png"
    with Image.new("RGB", (8, 4), (32, 64, 96)) as image:
        image.save(source)
    return source


@pytest.fixture
def depth_request(tmp_path: Path, photo: Path) -> DepthRequest:
    """Declare a deterministic artifact set with no local model execution."""
    return DepthRequest(
        source=str(photo), output=str(tmp_path / "photo.tiff"), device="cpu"
    )


def test_extraction_retains_metric_floats_and_near_white_preview(
    depth_request: DepthRequest,
) -> None:
    """TIFF values stay in metres while the PNG preview has 16-bit range."""
    result = extract_depth(depth_request, PlaneBoundary())
    assert (result.width, result.height) == (8, 4)
    assert result.source_sha256 == file_sha256(depth_request.source)
    document = DepthMetadata.model_validate_json(
        Path(result.metadata).read_text()
    )
    assert document.units == "metres"
    assert document.depth_axis == "camera-z"
    assert document.orientation == "EXIF-transposed RGB"
    assert document.model_revision == MOGE_REVISION
    assert document.checkpoint_sha256 == MOGE_SHA256
    assert document.preview_near_metres < document.preview_far_metres
    for artifact in (
        document.depth,
        document.preview,
        document.normals,
        document.mask,
    ):
        assert artifact.sha256 == file_sha256(
            str(Path(result.metadata).parent / artifact.path)
        )
    assert set(depth_request.artifacts) == {
        result.output,
        result.metadata,
        result.preview,
        result.normals,
        result.mask,
    }
    with Image.open(result.output) as depth:
        assert depth.mode == "F"
        assert depth.size == (8, 4)
        np.testing.assert_array_equal(
            np.asarray(depth)[0],
            np.array([1.25 + 0.5 * column for column in range(8)]),
        )
    with Image.open(result.preview) as preview:
        pixels = np.asarray(preview)
        assert pixels.dtype == np.uint16
        assert int(pixels[0, 0]) > int(pixels[0, -1])
        assert int(pixels[0, 0]) > 60000
        assert int(pixels[0, -1]) < 5000
    with Image.open(result.mask) as mask:
        assert mask.mode == "L"
        assert mask.getextrema() == (255, 255)
    with load_depth(result.metadata, depth_request.source) as asset:
        assert asset.depth.mode == "F"
        assert asset.mask.mode == "L"
        assert asset.normals.mode == "RGB"
        assert asset.depth.getpixel((7, 0)) == 4.75


def test_exif_orientation_is_applied_before_predicting_depth(
    tmp_path: Path,
) -> None:
    """A rotated JPEG and its cached geometry use the same oriented grid."""
    source = tmp_path / "rotated.jpg"
    with Image.new("RGB", (8, 4), "red") as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(source, exif=exif)
    request = DepthRequest(
        source=str(source), output=str(tmp_path / "rotated.tif"), device="cpu"
    )
    result = extract_depth(request, PlaneBoundary())
    assert (result.width, result.height) == (4, 8)
    with load_depth(result.metadata, str(source)) as asset:
        assert asset.depth.size == (4, 8)


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".json", ".npy"])
def test_depth_requires_a_metric_tiff_destination(suffix: str) -> None:
    """Unsupported artifacts are rejected before the source is accessed."""
    with pytest.raises(ValidationError, match=r"[Tt][Ii][Ff]"):
        DepthRequest(source="missing.png", output="result" + suffix)


@pytest.mark.parametrize("index", range(5))
def test_every_artifact_is_protected_against_overwriting_the_source(
    tmp_path: Path, index: int
) -> None:
    """All derived destinations participate in source alias protection."""
    output = str(tmp_path / "result.tiff")
    proposed = DepthRequest(source="placeholder.png", output=output)
    source = Path(proposed.artifacts[index])
    with Image.new("RGB", (4, 4), "red") as image:
        image.save(source, format="PNG")
    original = source.read_bytes()
    request = DepthRequest(
        source=str(source), output=output, device="cpu", overwrite=True
    )
    with pytest.raises(ValueError, match="alias"):
        extract_depth(request, PlaneBoundary())
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".pic2panning-*"))


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_linked_artifacts_cannot_replace_source_bytes(
    depth_request: DepthRequest, photo: Path, alias: str
) -> None:
    """Source protection includes inode aliases and symbolic destinations."""
    target = Path(depth_request.preview)
    if alias == "hardlink":
        os.link(photo, target)
    else:
        target.symlink_to(photo)
    original = photo.read_bytes()
    request = depth_request.model_copy(update={"overwrite": True})
    with pytest.raises(ValueError, match=r"alias|symbolic"):
        extract_depth(request, PlaneBoundary())
    assert photo.read_bytes() == original


def test_existing_artifact_requires_explicit_overwrite(
    depth_request: DepthRequest,
) -> None:
    """A preview is as protected as the metric depth destination."""
    preview = Path(depth_request.preview)
    preview.write_bytes(b"previous preview")
    with pytest.raises(FileExistsError):
        extract_depth(depth_request, PlaneBoundary())
    assert preview.read_bytes() == b"previous preview"
    assert not Path(depth_request.output).exists()


@pytest.mark.parametrize("existing", [False, True])
def test_prediction_failure_preserves_all_previous_artifacts(
    depth_request: DepthRequest, *, existing: bool
) -> None:
    """Failed model work never publishes a partial depth set."""
    request = depth_request.model_copy(update={"overwrite": existing})
    if existing:
        for name in request.artifacts:
            Path(name).write_bytes(b"previous " + Path(name).name.encode())
    with pytest.raises(RuntimeError, match="prediction failed"):
        extract_depth(request, PlaneBoundary("failure"))
    for name in request.artifacts:
        path = Path(name)
        if existing:
            assert path.read_bytes() == b"previous " + path.name.encode()
        else:
            assert not path.exists()
    assert not list(Path(request.output).parent.glob(".pic2panning-*"))


@pytest.mark.parametrize(
    "malformed",
    ["depth-size", "mask-size", "depth-mode", "nan", "zero", "negative"],
)
def test_malformed_geometry_cannot_publish_any_artifact(
    depth_request: DepthRequest, malformed: str
) -> None:
    """Runtime predictions require aligned positive finite metric depth."""
    with pytest.raises(ValueError, match=r"depth|Depth|mask|Mask|dimensions"):
        extract_depth(depth_request, PlaneBoundary(malformed))
    assert not any(Path(name).exists() for name in depth_request.artifacts)


@pytest.mark.parametrize("index", range(5))
def test_loading_rejects_changed_artifacts(
    depth_request: DepthRequest, index: int
) -> None:
    """Every cached artifact is authenticated before it is trusted."""
    result = extract_depth(depth_request, PlaneBoundary())
    artifact = Path(depth_request.artifacts[index])
    if artifact == Path(result.metadata):
        document = json.loads(artifact.read_text())
        document["source_sha256"] = "0" * 64
        artifact.write_text(json.dumps(document))
    else:
        artifact.write_bytes(artifact.read_bytes() + b"changed artifact")
    with (
        pytest.raises(ValueError, match=r"checksum|source"),
        load_depth(result.metadata, depth_request.source),
    ):
        pass


def test_loading_rejects_a_different_source_even_at_the_same_size(
    depth_request: DepthRequest, tmp_path: Path
) -> None:
    """Depth cannot be reused with a second photo merely sharing its grid."""
    result = extract_depth(depth_request, PlaneBoundary())
    other = tmp_path / "other.png"
    with Image.new("RGB", (8, 4), "blue") as image:
        image.save(other)
    with (
        pytest.raises(ValueError, match=r"source|checksum"),
        load_depth(result.metadata, str(other)),
    ):
        pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 9),
        ("height", 5),
        ("grid_width", 7),
        ("grid_height", 3),
        ("model_revision", "other-revision"),
        ("checkpoint_sha256", "f" * 64),
    ],
)
def test_loading_rejects_changed_geometry_and_predictor_identity(
    depth_request: DepthRequest, field: str, value: str | int
) -> None:
    """A manifest must retain source calibration and the pinned predictor."""
    result = extract_depth(depth_request, PlaneBoundary())
    metadata = Path(result.metadata)
    document = json.loads(metadata.read_text())
    document[field] = value
    metadata.write_text(json.dumps(document))
    with (
        pytest.raises(ValueError, match=r"dimensions|provenance"),
        load_depth(result.metadata, depth_request.source),
    ):
        pass


def test_cached_depth_requires_metric_tiff_even_with_a_matching_checksum(
    depth_request: DepthRequest,
) -> None:
    """A valid checksum cannot turn normalized preview pixels into metres."""
    result = extract_depth(depth_request, PlaneBoundary())
    with Image.new("L", (8, 4), 128) as image:
        image.save(result.output, format="PNG")
    metadata = Path(result.metadata)
    document = json.loads(metadata.read_text())
    document["depth"]["sha256"] = file_sha256(result.output)
    metadata.write_text(json.dumps(document))
    with (
        pytest.raises(ValueError, match=r"mode|format"),
        load_depth(result.metadata, depth_request.source),
    ):
        pass


def test_depth_processing_imports_without_model_frameworks() -> None:
    """Cache loading and effects do not require Torch, MLX, or model packages."""
    program = """
import sys
from importlib.abc import MetaPathFinder

class BlockModels(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'mlx3d', 'moge', 'mflux'}:
            raise ImportError('model framework import: ' + fullname)

sys.meta_path.insert(0, BlockModels())
from pic2panning.tools.creative.depth import load_depth
from pic2panning.tools.creative.effects import apply_depth_style, depth_mask
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
