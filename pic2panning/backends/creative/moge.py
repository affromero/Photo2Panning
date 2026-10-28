"""Canonical pinned MoGe inference, independent of Gaussian MLX resources."""

import gc
import os
from typing import Final, Protocol, Self, cast

import numpy as np
import torch
from beartype import beartype
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Float32, Shaped, jaxtyped
from PIL import Image

from pic2panning.backends.creative.depth import DepthPrediction
from pic2panning.backends.loading import load_dependency
from pic2panning.schemas.camera import Matrix3
from pic2panning.schemas.creative.depth import (
    MOGE_MODEL,
    MOGE_REVISION,
    MOGE_SHA256,
    DepthRequest,
    GeometryOptions,
)
from pic2panning.tools.io import file_sha256

IMAGE_SHAPE: Final = "3 height width"
FIELD_SHAPE: Final = "..."
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
        """Load weights and architecture without backbone fetching."""
        ...


class ModelModule(Protocol):
    """Expose the upstream version-two factory."""

    MoGeModel: ModelFactory


@jaxtyped(typechecker=beartype)
def cpu_float(
    tensor: Shaped[torch.Tensor, FIELD_SHAPE],
) -> Float32[np.ndarray, FIELD_SHAPE]:
    """Detach model output into an independently owned float32 field."""
    return tensor.detach().to("cpu", torch.float32).numpy().copy()


@jaxtyped(typechecker=beartype)
def predict_fields(
    request: GeometryOptions, image: Image.Image
) -> dict[str, Float32[np.ndarray, FIELD_SHAPE]]:
    """Release the PyTorch model before callers allocate other resources."""
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
        validate_fields(fields, image.size)
        return fields
    finally:
        del model
        gc.collect()
        if request.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()


@jaxtyped(typechecker=beartype)
def validate_fields(
    fields: dict[str, Float32[np.ndarray, FIELD_SHAPE]],
    size: tuple[int, int],
) -> None:
    """Reject malformed geometry consistently for scene and depth outputs."""
    depth = fields["depth"]
    points = fields["points"]
    mask = fields["mask"]
    intrinsics = fields["intrinsics"]
    shape = (size[1], size[0])
    if (
        depth.shape != shape
        or points.shape != (*shape, 3)
        or mask.shape != shape
        or intrinsics.shape != (3, 3)
    ):
        raise ValueError("MoGe output dimensions differ from source")
    if not bool(
        (
            np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(points).all(axis=-1)
            & (points[..., 2] > 0)
            & np.isfinite(mask)
        ).all()
    ):
        raise ValueError("MoGe raw geometry contains invalid source pixels")
    if not bool((mask > 0.5).any()):
        raise ValueError("MoGe returned no valid foreground geometry")
    if (
        not bool(np.isfinite(intrinsics).all())
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or intrinsics[0, 1] != 0
        or intrinsics[1, 0] != 0
        or not bool(np.array_equal(intrinsics[2], (0, 0, 1)))
    ):
        raise ValueError("MoGe intrinsics require finite unskewed calibration")


@jaxtyped(typechecker=beartype)
def pixel_intrinsics(
    normalized: Float32[np.ndarray, MATRIX_SHAPE],
    size: tuple[int, int],
) -> Matrix3:
    """Preserve image-edge coordinates with pixel centers at plus 0.5."""
    matrix = normalized.copy()
    matrix[0] *= size[0]
    matrix[1] *= size[1]
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


class MoGeDepthBackend:
    """Return independent metric images without loading Gaussian libraries."""

    @staticmethod
    def predict(request: DepthRequest, image: Image.Image) -> DepthPrediction:
        """Retain finite sky depth and preserve semantic foreground separately."""
        with image.copy() as geometry_image:
            geometry_image.thumbnail(
                (request.max_edge, request.max_edge), Image.Resampling.LANCZOS
            )
            fields = predict_fields(request, geometry_image)
            intrinsics = pixel_intrinsics(
                fields["intrinsics"], geometry_image.size
            )
        return DepthPrediction(
            depth=Image.fromarray(fields["depth"]),
            mask=Image.fromarray(
                ((fields["mask"] > 0.5) * 255).astype(np.uint8)
            ),
            intrinsics=intrinsics,
        )
