"""Pinned MIT MoGe2 geometry with analytic Gaussian pixel footprints."""

import gc
import math
import os
from typing import Final, Protocol, Self, cast, runtime_checkable

import mlx.core as mx
import numpy as np
import torch
from beartype import beartype
from huggingface_hub import hf_hub_download
from jaxtyping import Bool, Float, Float32, Shaped, jaxtyped
from PIL import Image

from pic2panning.backends.loading import load_dependency
from pic2panning.schemas.camera import (
    Matrix3,
    SceneCalibration,
    SceneColorSpace,
)
from pic2panning.schemas.scene import (
    MOGE_MODEL,
    MOGE_REVISION,
    MOGE_SHA256,
    SceneRequest,
)
from pic2panning.tools.io import file_sha256

IMAGE_SHAPE: Final = "3 height width"
FIELD_SHAPE: Final = "..."
POINTS_SHAPE: Final = "points 3"
DEPTH_SHAPE: Final = "height width"
MATRIX_SHAPE: Final = "3 3"


class Model(Protocol):
    """The official model interface at its untyped package boundary."""

    def to(self, device: str) -> Self:
        """Select exactly the requested device."""
        ...

    def eval(self) -> Self:
        """Disable training behavior."""
        ...

    def enable_pytorch_native_sdpa(self) -> None:
        """Use native PyTorch attention on the local device."""
        ...

    @jaxtyped(typechecker=beartype)
    def infer(
        self,
        image: Float[torch.Tensor, IMAGE_SHAPE],
        *,
        num_tokens: int,
        apply_mask: bool,
        use_fp16: bool,
    ) -> dict[str, Shaped[torch.Tensor, FIELD_SHAPE]]:
        """Predict metric points and calibration without discarding sky."""
        ...


class ModelFactory(Protocol):
    """Load the verified local checkpoint through the canonical loader."""

    def from_pretrained(self, checkpoint: str) -> Model:
        """Load weights and embedded architecture without backbone fetching."""
        ...


class ModelModule(Protocol):
    """Expose the upstream version-two factory."""

    MoGeModel: ModelFactory


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
def cpu_float(
    tensor: Shaped[torch.Tensor, FIELD_SHAPE],
) -> Float32[np.ndarray, FIELD_SHAPE]:
    """Detach model output into an independently owned float32 field."""
    return tensor.detach().to("cpu", torch.float32).numpy().copy()


@jaxtyped(typechecker=beartype)
def predict_fields(
    request: SceneRequest, image: Image.Image
) -> dict[str, Float32[np.ndarray, FIELD_SHAPE]]:
    """Release the PyTorch model before loading Gaussian Metal resources."""
    if request.device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise ValueError(
                "Disable PYTORCH_ENABLE_MPS_FALLBACK for explicit MPS work"
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device is unavailable")
    checkpoint = hf_hub_download(
        repo_id=MOGE_MODEL, filename="model.pt", revision=MOGE_REVISION
    )
    if file_sha256(checkpoint) != MOGE_SHA256:
        raise ValueError(
            "MoGe checkpoint checksum differs from pinned weights"
        )
    module = cast("ModelModule", load_dependency("moge.model.v2"))
    model = module.MoGeModel.from_pretrained(checkpoint)
    try:
        model = model.to(request.device).eval()
        model.enable_pytorch_native_sdpa()
        tensor = (
            torch.from_numpy(np.asarray(image).copy())
            .to(request.device, torch.float32)
            .permute(2, 0, 1)
            / 255
        )
        with torch.inference_mode():
            prediction = model.infer(
                tensor,
                num_tokens=request.num_tokens,
                apply_mask=False,
                use_fp16=False,
            )
        required = ("points", "depth", "mask", "intrinsics")
        if not set(required).issubset(prediction):
            raise ValueError("MoGe returned incomplete geometry fields")
        fields = {key: cpu_float(prediction[key]) for key in required}
        del prediction, tensor
        return fields
    finally:
        del model
        gc.collect()
        if request.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()


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


@jaxtyped(typechecker=beartype)
def pixel_intrinsics(
    normalized: Float32[np.ndarray, MATRIX_SHAPE],
    size: tuple[int, int],
) -> Matrix3:
    """Preserve edge-based pixel centers shared by MoGe, MLX3D, and Pillow."""
    matrix = normalized.copy()
    matrix[0] *= size[0]
    matrix[1] *= size[1]
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


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
        if (
            depth.shape != (geometry_size[1], geometry_size[0])
            or points.shape != (geometry_size[1], geometry_size[0], 3)
            or mask.shape != depth.shape
        ):
            raise ValueError("MoGe output dimensions differ from source")
        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(points).all(axis=-1)
            & (points[..., 2] > 0)
        )
        if not bool(valid.all()):
            raise ValueError(
                "MoGe raw geometry contains invalid source pixels"
            )
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
                "z_percentiles_metres": foreground_depths(depth, mask & valid),
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
