"""Pinned MIT MoGe2 geometry with analytic Gaussian pixel footprints."""

import math
from typing import Final, Protocol, cast, runtime_checkable

import mlx.core as mx
import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, Float32, jaxtyped
from PIL import Image

from pic2panning.backends.creative.moge import (
    pixel_intrinsics,
    predict_fields,
)
from pic2panning.backends.loading import load_dependency
from pic2panning.schemas.camera import (
    SceneCalibration,
    SceneColorSpace,
)
from pic2panning.schemas.scene import SceneRequest

POINTS_SHAPE: Final = "points 3"
DEPTH_SHAPE: Final = "height width"


@runtime_checkable
class GaussianScene(Protocol):
    """Canonical Gaussian parameters and standard PLY export."""

    params: dict[str, mx.array]

    def save_ply(self, path: str) -> None:
        """Write SH colors, log scales, quaternions, and opacity logits."""
        ...


class GaussianFactory(Protocol):
    """Canonical point initialization without an all-pairs distance pass."""

    @jaxtyped(typechecker=beartype)
    def from_points(
        self,
        points: Float[mx.array, POINTS_SHAPE],
        colors: Float[mx.array, POINTS_SHAPE],
        *,
        sh_degree: int,
        initial_opacity: float,
        scale_init_max_ref: int,
    ) -> GaussianScene:
        """Initialize source-colored degree-zero Gaussians."""
        ...


class GaussianModule(Protocol):
    """Keep MLX3D's package types at an explicit boundary."""

    GaussianModel: GaussianFactory


@jaxtyped(typechecker=beartype)
def foreground_depths(
    depth: Float32[np.ndarray, DEPTH_SHAPE],
    mask: Bool[np.ndarray, DEPTH_SHAPE],
) -> dict[str, float]:
    """Use semantic foreground for camera displacement, retaining sky splats."""
    selected = depth[mask]
    if not selected.size:
        raise ValueError("MoGe returned no valid foreground geometry")
    return {
        str(percentile): float(np.percentile(selected, percentile))
        for percentile in (1, 10, 50, 90, 99)
    }


class MoGeSceneBackend:
    """Predict real metric point geometry and export source-colored splats."""

    def predict(
        self,
        request: SceneRequest,
        image: Image.Image,
        output: str,
        source_sha256: str,
    ) -> tuple[SceneCalibration, int]:
        """Build one isotropic Gaussian per finite predicted source pixel."""
        with image.copy() as geometry_image:
            geometry_image.thumbnail(
                (request.max_edge, request.max_edge), Image.Resampling.LANCZOS
            )
            geometry_size = geometry_image.size
            fields = predict_fields(request, geometry_image)
            colors = (
                np.asarray(geometry_image).reshape(-1, 3).astype(np.float32)
                / 255
            )
        depth = fields["depth"]
        points = fields["points"]
        mask = fields["mask"] > 0.5
        intrinsics = pixel_intrinsics(fields["intrinsics"], image.size)
        calibration = SceneCalibration.model_validate(
            {
                "width": image.width,
                "height": image.height,
                "intrinsics": intrinsics,
                "world_to_camera": (
                    (1, 0, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                ),
                "sh_degree": 0,
                "sh_color_space": SceneColorSpace.SRGB,
                "z_percentiles_metres": foreground_depths(depth, mask),
                "source_sha256": source_sha256,
            }
        )
        geometry_intrinsics = pixel_intrinsics(
            fields["intrinsics"], geometry_size
        )
        focal = math.sqrt(
            geometry_intrinsics[0][0] * geometry_intrinsics[1][1]
        )
        scales = np.maximum(depth.reshape(-1) * (0.55 / focal), 1e-6)
        module = cast("GaussianModule", load_dependency("mlx3d.splatting"))
        scene = module.GaussianModel.from_points(
            mx.array(points.reshape(-1, 3)),
            mx.array(colors),
            sh_degree=0,
            initial_opacity=0.999,
            scale_init_max_ref=1,
        )
        try:
            scene.params["scales"] = mx.repeat(
                mx.log(mx.array(scales))[:, None], 3, axis=1
            )
            mx.eval(scene.params)
            scene.save_ply(output)
        finally:
            del scene
            mx.clear_cache()
        return calibration, geometry_size[0] * geometry_size[1]
