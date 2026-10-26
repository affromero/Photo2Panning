"""Exercise the real native Gaussian adapter with tiny synthetic scenes."""

import math
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import Mock

import numpy as np
import pytest
from klogr.path import path_write_text
from pydantic import TypeAdapter

from pic2panning.backends.camera import NativeCameraBackend
from pic2panning.backends.contracts import CameraScene
from pic2panning.backends.loading import load_dependency
from pic2panning.errors import MediaError
from pic2panning.schemas.camera import CameraPose, SceneCalibration

pytestmark = pytest.mark.models


class ArrayFactory(Protocol):
    """The optional MLX constructor used for malformed native outputs."""

    def array(self, values: object) -> object:
        """Create a real external buffer at the GPU boundary."""
        ...


class MetalDevice(Protocol):
    """The external runtime reports whether native Metal is usable."""

    def is_available(self) -> bool:
        """Return whether the current runner has a usable Metal device."""
        ...


class MetalModule(Protocol):
    """The deferred MLX module exposes its device availability probe."""

    metal: MetalDevice


def write_scene(
    path: Path,
    changes: dict[str, float] | None = None,
    *,
    rows: int = 64,
    columns: int = 128,
) -> None:
    """Write a small dense Gaussian grid without invoking model weights."""
    values = {
        "x": 0.0,
        "y": 0.0,
        "z": 3.0,
        "f_dc_0": 0.0,
        "f_dc_1": 0.0,
        "f_dc_2": 0.0,
        "opacity": math.inf,
        "scale_0": math.log(0.025),
        "scale_1": math.log(0.025),
        "scale_2": math.log(0.025),
        "rot_0": 1.0,
        "rot_1": 0.0,
        "rot_2": 0.0,
        "rot_3": 0.0,
    }
    values.update(changes or {})
    header = ["ply", "format ascii 1.0", f"element vertex {rows * columns}"]
    header.extend(f"property float {name}" for name in values)
    vertices = []
    for row in range(rows):
        for column in range(columns):
            point = {
                **values,
                "x": values["x"] + (column - (columns - 1) / 2) * 0.01,
                "y": values["y"] + (row - (rows - 1) / 2) * 0.01,
            }
            vertices.append(" ".join(map(str, point.values())))
    payload = "\n".join([*header, "end_header", *vertices, ""])
    path_write_text(path, payload)


def calibration(color_space: str = "sRGB") -> SceneCalibration:
    """Bind the synthetic scene to its simple reference camera."""
    return SceneCalibration.model_validate(
        {
            "width": 16,
            "height": 16,
            "intrinsics": ((16, 0, 8), (0, 16, 8), (0, 0, 1)),
            "world_to_camera": (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ),
            "sh_degree": 0,
            "sh_color_space": color_space,
            "z_percentiles_metres": {"10": 3},
            "source_sha256": "a" * 64,
        }
    )


def base_pose() -> CameraPose:
    """Use independent output intrinsics at the reference viewpoint."""
    return CameraPose(
        width=16,
        height=16,
        fx=16,
        fy=16,
        cx=8,
        cy=8,
        rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        translation=(0, 0, 0),
    )


@pytest.fixture
def scene_path(tmp_path: Path) -> Path:
    """Skip native checks when the optional camera framework is absent."""
    pytest.importorskip("mlx.core")
    module = cast("MetalModule", load_dependency("mlx.core"))
    if not module.metal.is_available():
        pytest.skip("Native Gaussian checks require an available Metal GPU")
    pytest.importorskip("mlx3d.splatting")
    path = tmp_path / "gaussian.ply"
    write_scene(path)
    return path


@pytest.fixture
def renderer(scene_path: Path) -> Iterator[CameraScene]:
    """Own the native renderer until its independent outputs are inspected."""
    with closing(
        NativeCameraBackend().load(str(scene_path), calibration())
    ) as scene:
        yield scene


@pytest.mark.parametrize(
    ("color_space", "expected"), [("sRGB", 128), ("linearRGB", 187)]
)
def test_scene_colors_use_declared_transfer_and_allow_opaque_logits(
    scene_path: Path, color_space: str, expected: int
) -> None:
    """Render real Gaussians, retaining light through linear compositing."""
    scene = NativeCameraBackend().load(
        str(scene_path), calibration(color_space)
    )
    with closing(scene), scene.render(base_pose()) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 16)
        pixel = TypeAdapter(tuple[int, int, int]).validate_python(
            image.getpixel((8, 8))
        )
        assert pixel == pytest.approx((expected,) * 3, abs=2)
        corner = TypeAdapter(tuple[int, int, int]).validate_python(
            image.getpixel((0, 0))
        )
        assert sum(corner) < sum(pixel)


def test_offset_moves_in_rotated_original_camera_basis(
    renderer: CameraScene,
) -> None:
    """Maintain the optical target from a translated, rotated base camera."""
    base = base_pose().model_copy(
        update={
            "rotation": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
            "translation": (2, 3, 4),
        }
    )
    assert renderer.pose_from_offset(base, (0, 0, 0), 10) is base
    moved = renderer.pose_from_offset(base, (1, -0.5, 0.25), 10)
    rotation = np.asarray(moved.rotation)
    shift = np.asarray(moved.translation)
    expected_eye = np.array((-3.5, 1, -3.75))
    target = np.array((-3, 2, 6))
    np.testing.assert_allclose(rotation @ expected_eye + shift, 0, atol=1e-6)
    projected_target = rotation @ target + shift
    np.testing.assert_allclose(projected_target[:2], 0, atol=1e-6)
    assert projected_target[2] > 0
    assert (moved.fx, moved.fy, moved.cx, moved.cy) == (
        base.fx,
        base.fy,
        base.cx,
        base.cy,
    )


def test_one_gaussian_uses_the_native_renderer_without_small_buffer_failure(
    scene_path: Path,
) -> None:
    """Render a valid minimal PLY without changing backends or its colors."""
    write_scene(scene_path, {"x": 3 / 32, "y": 3 / 32}, rows=1, columns=1)
    with (
        closing(
            NativeCameraBackend().load(str(scene_path), calibration())
        ) as scene,
        scene.render(base_pose()) as image,
    ):
        center = TypeAdapter(tuple[int, int, int]).validate_python(
            image.getpixel((8, 8))
        )
        corner = TypeAdapter(tuple[int, int, int]).validate_python(
            image.getpixel((0, 0))
        )
        assert center == pytest.approx((128, 128, 128), abs=2)
        assert corner == (0, 0, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"x": math.nan},
        {"f_dc_0": math.inf},
        {"opacity": math.nan},
        {"scale_0": math.inf},
        {"rot_0": 0.0},
    ],
)
def test_invalid_scene_parameters_fail_before_native_rendering(
    scene_path: Path, changes: dict[str, float]
) -> None:
    """Reject unsafe activated buffers instead of invoking native kernels."""
    write_scene(scene_path, changes)
    with pytest.raises(MediaError, match=r"Gaussian|scales|quaternion"):
        NativeCameraBackend().load(str(scene_path), calibration())


@pytest.mark.parametrize("malformed", ["size", "nan", "missing"])
def test_invalid_native_frames_are_rejected(
    renderer: CameraScene,
    monkeypatch: pytest.MonkeyPatch,
    malformed: str,
) -> None:
    """Keep broken external image buffers away from video encoding."""
    factory = cast("ArrayFactory", load_dependency("mlx.core"))
    output = (
        {}
        if malformed == "missing"
        else {
            "image": factory.array(
                np.full(
                    (1, 1, 3) if malformed == "size" else (16, 16, 3),
                    math.nan if malformed == "nan" else 0,
                    dtype=np.float32,
                )
            )
        }
    )
    monkeypatch.setattr(
        "mlx3d.splatting.FastGaussianRenderer.render",
        Mock(return_value=output),
    )
    with pytest.raises(MediaError, match=r"size|finite"):
        renderer.render(base_pose())


def test_close_keeps_returned_pixels_independent_and_prevents_reuse(
    renderer: CameraScene,
) -> None:
    """Release GPU resources without invalidating an already returned image."""
    with renderer.render(base_pose()) as image:
        before = image.tobytes()
        renderer.close()
        renderer.close()
        assert image.tobytes() == before
        with pytest.raises(RuntimeError, match="closed"):
            renderer.render(base_pose())
        with pytest.raises(RuntimeError, match="closed"):
            renderer.pose_from_offset(base_pose(), (0.1, 0, 0), 3)
