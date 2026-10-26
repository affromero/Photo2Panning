"""Render degree-zero Gaussian scenes through native MLX Metal kernels."""

import math
from typing import Final, Protocol, cast

import mlx.core as mx
import numpy as np
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from pydantic import TypeAdapter

from pic2panning.backends.loading import load_dependency
from pic2panning.errors import MediaError
from pic2panning.schemas.camera import (
    CameraPose,
    Matrix3,
    SceneCalibration,
    SceneColorSpace,
    Vector3,
)

SH_C0 = math.sqrt(1 / (4 * math.pi))
COLOR_SHAPE: Final = "n 3"
IMAGE_SHAPE: Final = "h w 3"
MIN_NATIVE_POINTS = 8192


class GaussianScene(Protocol):
    """The public parameter surface of an imported Gaussian scene."""

    params: dict[str, mx.array]
    scales_act: mx.array
    opacities_act: mx.array
    active_sh_degree: int
    num_gaussians: int


class SceneFactory(Protocol):
    """The external library's standard Gaussian PLY loader."""

    def load_ply(self, path: str) -> GaussianScene:
        """Load log scales, logit opacities, WXYZ rotations, and SH data."""
        ...


class NativeRenderer(Protocol):
    """The native forward renderer exposes lazy MLX image buffers."""

    def render(self, camera: object) -> dict[str, object]:
        """Rasterize the retained scene at an explicit camera pose."""
        ...


class RendererFactory(Protocol):
    """Keep external constructor options at the library boundary."""

    def __call__(self, **parameters: object) -> NativeRenderer:
        """Build the forward-only Metal renderer."""
        ...


class SplatModule(Protocol):
    """The public native scene-loading and rendering exports."""

    GaussianModel: SceneFactory
    FastGaussianRenderer: RendererFactory


class CameraFactory(Protocol):
    """The external OpenCV camera constructor."""

    def __call__(self, **parameters: object) -> object:
        """Construct a camera with independent pose and intrinsics."""
        ...


class CameraModule(Protocol):
    """Public OpenCV camera construction and look-at transformation."""

    Camera: CameraFactory

    def look_at(
        self, eye: object, at: object, up: object
    ) -> tuple[object, object]:
        """Return world-to-camera rotation and translation."""
        ...


@jaxtyped(typechecker=beartype)
def srgb_to_linear(
    colors: Float[mx.array, COLOR_SHAPE],
) -> Float[mx.array, COLOR_SHAPE]:
    """Decode scene display colors before alpha compositing."""
    return mx.where(
        colors <= 0.04045,
        colors / 12.92,
        ((colors + 0.055) / 1.055) ** 2.4,
    )


@jaxtyped(typechecker=beartype)
def display_pixels(
    colors: Float[mx.array, IMAGE_SHAPE],
) -> UInt8[np.ndarray, IMAGE_SHAPE]:
    """Encode composited linear RGB to independent display pixels."""
    clipped = mx.clip(colors, 0, 1)
    display = mx.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * clipped ** (1 / 2.4) - 0.055,
    )
    pixels = mx.round(mx.clip(display, 0, 1) * 255).astype(mx.uint8)
    mx.eval(pixels)
    return np.asarray(pixels)


def validate_scene(scene: GaussianScene) -> None:
    """Reject malformed buffers before they reach the native kernels."""
    count = scene.num_gaussians
    if scene.active_sh_degree != 0 or count < 1:
        raise MediaError("Camera scenes require nonempty degree-zero colors")
    buffers = (
        (scene.params["means"], (count, 3)),
        (scene.params["quats"], (count, 4)),
        (scene.scales_act, (count, 3)),
        (scene.opacities_act, (count,)),
        (scene.params["sh_dc"], (count, 1, 3)),
    )
    for values, shape in buffers:
        if values.shape != shape or not bool(
            mx.all(mx.isfinite(values)).item()
        ):
            raise MediaError("The scene contains invalid Gaussian buffers")
    if not bool(mx.all(scene.scales_act > 0).item()):
        raise MediaError("Gaussian scales must be positive")
    if not bool(mx.all(mx.sum(scene.params["quats"] ** 2, axis=1) > 0).item()):
        raise MediaError("Gaussian rotations must have nonzero quaternions")


class SplatRenderer:
    """Own one loaded Gaussian scene and its reusable native renderer."""

    scene: GaussianScene | None
    native: NativeRenderer | None
    cameras: CameraModule

    def __init__(self, scene: str, calibration: SceneCalibration) -> None:
        """Load geometry and decode its declared color space once."""
        self.scene = None
        self.native = None
        module = cast("SplatModule", load_dependency("mlx3d.splatting"))
        self.cameras = cast("CameraModule", load_dependency("mlx3d.cameras"))
        try:
            loaded = module.GaussianModel.load_ply(scene)
            self.scene = loaded
            validate_scene(loaded)
            rgb = mx.clip(loaded.params["sh_dc"][:, 0, :] * SH_C0 + 0.5, 0, 1)
            colors = (
                srgb_to_linear(rgb)
                if calibration.sh_color_space == SceneColorSpace.SRGB
                else rgb
            )
            means = loaded.params["means"]
            quats = loaded.params["quats"]
            scales = loaded.scales_act
            opacities = loaded.opacities_act
            padding = max(0, MIN_NATIVE_POINTS - loaded.num_gaussians)
            if padding:
                # The upstream compositor requires device-addressed vectors.
                # MLX binds small buffers in constant memory. Zero-opacity
                # padding keeps each vector buffer at least 64 KiB without
                # changing the scene contribution or switching renderers.
                means = mx.concatenate([means, mx.zeros((padding, 3))])
                quats = mx.concatenate(
                    [
                        quats,
                        mx.tile(mx.array((1.0, 0.0, 0.0, 0.0)), (padding, 1)),
                    ]
                )
                scales = mx.concatenate([scales, mx.full((padding, 3), 1e-6)])
                opacities = mx.concatenate([opacities, mx.zeros((padding,))])
                colors = mx.concatenate([colors, mx.zeros((padding, 3))])
            self.native = module.FastGaussianRenderer(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                t_min=1e-4,
                color_refresh=0,
                antialias=False,
            )
        except BaseException:
            self.close()
            raise

    def render(self, pose: CameraPose) -> Image.Image:
        """Return independently owned RGB pixels at the requested pose."""
        if self.native is None:
            raise RuntimeError("The camera renderer is closed")
        camera = self.cameras.Camera(
            R=mx.array(pose.rotation, dtype=mx.float32),
            t=mx.array(pose.translation, dtype=mx.float32),
            fx=pose.fx,
            fy=pose.fy,
            cx=pose.cx,
            cy=pose.cy,
            width=pose.width,
            height=pose.height,
            znear=pose.znear,
            zfar=pose.zfar,
        )
        image = self.native.render(camera).get("image")
        if not isinstance(image, mx.array) or image.shape != (
            pose.height,
            pose.width,
            3,
        ):
            raise MediaError("The native renderer returned an unexpected size")
        if not bool(mx.all(mx.isfinite(image)).item()):
            raise MediaError("The native renderer returned non-finite pixels")
        pixels = display_pixels(image)
        return Image.frombytes(
            "RGB", (pose.width, pose.height), pixels.tobytes()
        )

    def pose_from_offset(
        self, base: CameraPose, offset: Vector3, focus_distance: float
    ) -> CameraPose:
        """Move in original-camera coordinates and aim at its fixed target."""
        if self.native is None:
            raise RuntimeError("The camera renderer is closed")
        if offset == (0.0, 0.0, 0.0):
            return base
        if not math.isfinite(focus_distance) or focus_distance <= base.znear:
            raise ValueError("Camera focus must lie beyond its near plane")
        if not all(math.isfinite(value) for value in offset):
            raise ValueError("Camera offsets must be finite")
        if offset[2] >= focus_distance - base.znear:
            raise ValueError("Camera offset must stay before the focus plane")
        inverse_rotation = mx.array(base.rotation, dtype=mx.float32).T
        eye = -(
            inverse_rotation @ mx.array(base.translation, dtype=mx.float32)
        )
        position = eye + inverse_rotation @ mx.array(offset, dtype=mx.float32)
        target = eye + inverse_rotation @ mx.array(
            (0, 0, focus_distance), dtype=mx.float32
        )
        up = inverse_rotation @ mx.array((0, -1, 0), dtype=mx.float32)
        rotation, translation = self.cameras.look_at(position, target, up)
        rotation_array = cast("mx.array", rotation)
        translation_array = cast("mx.array", translation)
        return CameraPose.model_validate(
            {
                **base.model_dump(),
                "rotation": TypeAdapter(Matrix3).validate_python(
                    rotation_array.tolist()
                ),
                "translation": TypeAdapter(Vector3).validate_python(
                    translation_array.tolist()
                ),
            }
        )

    def close(self) -> None:
        """Release owned arrays and unreferenced Metal cache allocations."""
        self.native = None
        self.scene = None
        mx.clear_cache()
