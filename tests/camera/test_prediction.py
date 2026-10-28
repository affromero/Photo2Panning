"""Exercise MoGe prediction at external model and checkpoint boundaries."""

import hashlib
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, Self, cast

import numpy as np
import pytest
from PIL import Image
from pydantic import TypeAdapter

from pic2panning.backends.contracts import SceneBackend
from pic2panning.backends.loading import load_dependency
from pic2panning.schemas.camera import SceneColorSpace
from pic2panning.schemas.creative.depth import MOGE_MODEL, MOGE_REVISION
from pic2panning.schemas.scene import SceneRequest

pytestmark = pytest.mark.models


class TensorSurface(Protocol):
    """Read actual tensor inputs without importing an optional type eagerly."""

    shape: tuple[int, ...]

    def cpu(self) -> Self:
        """Make the model's observable input available as primitive values."""
        ...

    def tolist(self) -> object:
        """Return values for validation at the external tensor boundary."""
        ...


class TorchFactory(Protocol):
    """Construct real CPU tensors while keeping optional imports lazy."""

    float32: object

    def tensor(self, values: object, *, dtype: object) -> object:
        """Return actual framework tensor buffers from Python field values."""
        ...


class ModelBoundary:
    """A fake external predictor producing calibrated plane geometry."""

    torch: TorchFactory
    device: str | None
    native_attention: bool
    input_shape: tuple[int, int, int] | None
    input_pixels: list[list[list[float]]]
    options: dict[str, object]
    malformed: str | None

    def __init__(self, torch: TorchFactory) -> None:
        """Start with no model observations or malformed output selected."""
        self.torch = torch
        self.device = None
        self.native_attention = False
        self.input_shape = None
        self.input_pixels = []
        self.options = {}
        self.malformed = None

    def to(self, device: str) -> Self:
        """Record the exact externally selected inference device."""
        self.device = device
        return self

    def eval(self) -> Self:
        """Expose the upstream inference-mode interface."""
        return self

    def enable_pytorch_native_sdpa(self) -> None:
        """Record native attention rather than requiring CUDA extensions."""
        self.native_attention = True

    def infer(self, image: object, **options: object) -> dict[str, object]:
        """Return real CPU tensors with known depth, projection, and sky."""
        tensor = cast("TensorSurface", image)
        channels, height, width = TypeAdapter(
            tuple[int, int, int]
        ).validate_python(tensor.shape)
        self.input_shape = (channels, height, width)
        self.input_pixels = TypeAdapter(
            list[list[list[float]]]
        ).validate_python(tensor.cpu().tolist())
        self.options = options
        depth = [[2.0 for column in range(width)] for row in range(height)]
        points = [
            [
                [
                    2 * ((column + 0.5) / width - 0.5),
                    2 * ((row + 0.5) / height - 0.5),
                    2.0,
                ]
                for column in range(width)
            ]
            for row in range(height)
        ]
        mask = [
            [float(row > 0) for column in range(width)]
            for row in range(height)
        ]
        intrinsics = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
        shape_fields: dict[
            str, list[list[float]] | list[list[list[float]]]
        ] = {"depth-shape": depth, "points-shape": points, "mask-shape": mask}
        if self.malformed in shape_fields:
            shape_fields[self.malformed].pop()
        elif self.malformed == "depth-nan":
            depth[0][0] = math.nan
        elif self.malformed == "depth-zero":
            depth[0][0] = 0
        elif self.malformed == "point-nan":
            points[0][0][0] = math.nan
        elif self.malformed == "point-behind":
            points[0][0][2] = -1
        elif self.malformed == "mask-empty":
            mask = [[0.0 for column in range(width)] for row in range(height)]
        elif self.malformed == "intrinsics-skew":
            intrinsics[0][1] = 0.1
        fields = {
            name: self.torch.tensor(values, dtype=self.torch.float32)
            for name, values in (
                ("depth", depth),
                ("points", points),
                ("mask", mask),
                ("intrinsics", intrinsics),
            )
        }
        if self.malformed == "missing-depth":
            del fields["depth"]
        return fields


class ModelFactory:
    """Expose the real checkpoint filename received by the external loader."""

    model: ModelBoundary
    checkpoint: str | None

    def __init__(self, model: ModelBoundary) -> None:
        """Retain the selected provider without touching checkpoint files."""
        self.model = model
        self.checkpoint = None

    def from_pretrained(self, checkpoint: str) -> ModelBoundary:
        """Record external weight selection and return the model boundary."""
        self.checkpoint = checkpoint
        return self.model


class ModelModule(ModuleType):
    """Supply only the official version-two model factory export."""

    MoGeModel: ModelFactory

    def __init__(self, factory: ModelFactory) -> None:
        """Replace the external package without patching application helpers."""
        super().__init__("moge.model.v2")
        self.MoGeModel = factory


class DownloadBoundary:
    """Resolve a tiny local checkpoint at the Hugging Face network boundary."""

    checkpoint: str
    requested: tuple[str, str, str] | None

    def __init__(self, checkpoint: str) -> None:
        """Keep download arguments observable without performing a download."""
        self.checkpoint = checkpoint
        self.requested = None

    def __call__(self, repo_id: str, filename: str, revision: str) -> str:
        """Return the local bytes that the real adapter must authenticate."""
        self.requested = (repo_id, filename, revision)
        return self.checkpoint


class BackendModule(Protocol):
    """Construct the production adapter after optional imports are checked."""

    MoGeSceneBackend: type[SceneBackend]


class MetalDevice(Protocol):
    """The real Metal availability check required for native export."""

    def is_available(self) -> bool:
        """Report a usable GPU, rather than successful import alone."""
        ...


class MetalModule(Protocol):
    """Expose the native device probe without executing a GPU operation."""

    metal: MetalDevice


class GaussianData(Protocol):
    """Read canonical Gaussian fields after native PLY loading."""

    params: dict[str, object]
    num_gaussians: int


class GaussianReader(Protocol):
    """Inspect exported geometry with the existing native scene loader."""

    def load_ply(self, path: str) -> GaussianData:
        """Decode the actual scene file without a separate parser dependency."""
        ...


class GaussianModule(Protocol):
    """Expose MLX3D's public Gaussian model loader."""

    GaussianModel: GaussianReader


class PredictionHarness:
    """Typed observations around the unchanged production adapter."""

    backend: SceneBackend
    model: ModelBoundary
    factory: ModelFactory
    download: DownloadBoundary
    checkpoint: Path

    def __init__(
        self, module: ModuleType, torch: TorchFactory, checkpoint: Path
    ) -> None:
        """Create external fake providers and the real adapter instance."""
        self.backend = cast("BackendModule", module).MoGeSceneBackend()
        self.model = ModelBoundary(torch)
        self.factory = ModelFactory(self.model)
        self.download = DownloadBoundary(str(checkpoint))
        self.checkpoint = checkpoint


@pytest.fixture
def prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PredictionHarness:
    """Replace only model loading and checkpoint network boundaries."""
    torch = cast("TorchFactory", pytest.importorskip("torch"))
    pytest.importorskip("mlx.core")
    pytest.importorskip("jaxtyping")
    pytest.importorskip("beartype")
    pytest.importorskip("huggingface_hub")
    module = load_dependency("pic2panning.backends.moge")
    inference = load_dependency("pic2panning.backends.creative.moge")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"Synthetic authenticated checkpoint")
    harness = PredictionHarness(module, torch, checkpoint)
    monkeypatch.setattr(inference, "hf_hub_download", harness.download)
    monkeypatch.setattr(
        inference,
        "MOGE_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    monkeypatch.setitem(
        sys.modules, "moge.model.v2", ModelModule(harness.factory)
    )
    return harness


@pytest.fixture
def metal_export(prediction: PredictionHarness) -> PredictionHarness:
    """Run real export only when the optional native Metal device exists."""
    module = cast("MetalModule", load_dependency("mlx.core"))
    if not module.metal.is_available():
        pytest.skip("Gaussian export requires a usable Metal GPU")
    pytest.importorskip("mlx3d.splatting")
    return prediction


def request(tmp_path: Path, **options: object) -> SceneRequest:
    """Build a small explicit CPU job with ordinary artifact destinations."""
    return SceneRequest.model_validate(
        {
            "source": "source.png",
            "output": str(tmp_path / "scene.ply"),
            "metadata": str(tmp_path / "scene.json"),
            "device": "cpu",
            "num_tokens": 1200,
            "max_edge": 256,
            **options,
        }
    )


def test_prediction_exports_source_colors_and_pixel_center_calibration(
    tmp_path: Path, metal_export: PredictionHarness
) -> None:
    """Real tensor conversion and MLX export preserve a calibrated RGB plane."""
    job = request(tmp_path)
    with Image.new("RGB", (4, 2), "red") as source:
        source.putpixel((1, 0), (0, 255, 0))
        calibration, count = metal_export.backend.predict(
            job, source, job.output, "a" * 64
        )
        expected_colors = np.asarray(source).reshape(-1, 3) / 255
    module = cast("GaussianModule", load_dependency("mlx3d.splatting"))
    scene = module.GaussianModel.load_ply(job.output)
    points = np.asarray(scene.params["means"])
    assert count == 8
    assert scene.num_gaussians == 8
    np.testing.assert_allclose(
        np.asarray(calibration.intrinsics),
        [[4.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
    )
    assert calibration.sh_color_space is SceneColorSpace.SRGB
    assert calibration.source_sha256 == "a" * 64
    assert calibration.z_percentiles_metres["10"] == pytest.approx(2)
    np.testing.assert_allclose(
        4 * points[:, 0] / points[:, 2] + 2,
        np.tile(np.arange(4) + 0.5, 2),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        2 * points[:, 1] / points[:, 2] + 1,
        np.repeat(np.arange(2) + 0.5, 4),
        atol=1e-6,
    )
    colors = (
        np.asarray(scene.params["sh_dc"])[:, 0, :]
        * math.sqrt(1 / (4 * math.pi))
        + 0.5
    )
    np.testing.assert_allclose(colors, expected_colors, atol=1e-6)
    np.testing.assert_allclose(
        np.exp(np.asarray(scene.params["scales"])), 0.38890873, rtol=1e-5
    )
    assert bool(np.all(np.asarray(scene.params["opacities"]) > 6))
    assert metal_export.factory.checkpoint == str(metal_export.checkpoint)
    assert metal_export.download.requested == (
        MOGE_MODEL,
        "model.pt",
        MOGE_REVISION,
    )
    assert metal_export.model.device == "cpu"
    assert metal_export.model.native_attention
    assert metal_export.model.options == {
        "num_tokens": 1200,
        "apply_mask": False,
        "use_fp16": False,
    }
    assert metal_export.model.input_pixels[0][0][0] == 1
    assert metal_export.model.input_pixels[1][0][1] == 1


def test_geometry_bound_retains_original_camera_dimensions(
    tmp_path: Path, metal_export: PredictionHarness
) -> None:
    """Only the geometry grid shrinks; the source camera keeps its full field."""
    job = request(tmp_path)
    with Image.new("RGB", (640, 320), "green") as source:
        calibration, count = metal_export.backend.predict(
            job, source, job.output, "b" * 64
        )
        assert source.size == (640, 320)
    assert metal_export.model.input_shape == (3, 128, 256)
    assert count == 256 * 128
    assert (calibration.width, calibration.height) == (640, 320)
    np.testing.assert_allclose(
        np.asarray(calibration.intrinsics),
        [[640.0, 0.0, 320.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]],
    )
    module = cast("GaussianModule", load_dependency("mlx3d.splatting"))
    scene = module.GaussianModel.load_ply(job.output)
    points = np.asarray(scene.params["means"])
    assert scene.num_gaussians == count
    projected_x = 640 * points[:, 0] / points[:, 2] + 320
    assert float(projected_x.min()) == pytest.approx(1.25)
    assert float(projected_x.max()) == pytest.approx(638.75)


def test_bad_checkpoint_is_rejected_before_model_dispatch(
    tmp_path: Path, prediction: PredictionHarness
) -> None:
    """A downloaded file must match its pinned checksum before being loaded."""
    prediction.checkpoint.write_bytes(b"Changed checkpoint bytes")
    job = request(tmp_path)
    with (
        Image.new("RGB", (4, 2)) as image,
        pytest.raises(ValueError, match="checksum"),
    ):
        prediction.backend.predict(job, image, job.output, "a" * 64)
    assert prediction.factory.checkpoint is None
    assert prediction.model.device is None
    assert not Path(job.output).exists()


def test_unavailable_mps_fails_without_selecting_cpu_or_downloading(
    tmp_path: Path,
    prediction: PredictionHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device failure surfaces before checkpoint access or model loading."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    job = request(tmp_path, device="mps")
    with (
        Image.new("RGB", (4, 2)) as image,
        pytest.raises(RuntimeError, match="unavailable"),
    ):
        prediction.backend.predict(job, image, job.output, "a" * 64)
    assert prediction.download.requested is None
    assert prediction.model.device is None
    assert not Path(job.output).exists()


def test_mps_cpu_fallback_setting_is_rejected_explicitly(
    tmp_path: Path,
    prediction: PredictionHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An environment fallback cannot silently change the requested device."""
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    job = request(tmp_path, device="mps")
    with (
        Image.new("RGB", (4, 2)) as image,
        pytest.raises(ValueError, match="FALLBACK"),
    ):
        prediction.backend.predict(job, image, job.output, "a" * 64)
    assert prediction.download.requested is None
    assert prediction.model.device is None


@pytest.mark.parametrize(
    "malformed",
    [
        "depth-shape",
        "points-shape",
        "mask-shape",
        "depth-nan",
        "depth-zero",
        "point-nan",
        "point-behind",
        "mask-empty",
        "intrinsics-skew",
        "missing-depth",
    ],
)
def test_invalid_prediction_fields_cannot_publish_geometry(
    tmp_path: Path, prediction: PredictionHarness, malformed: str
) -> None:
    """Broken dimensions, finite depth, foreground, and calibration fail early."""
    prediction.model.malformed = malformed
    job = request(tmp_path)
    with (
        Image.new("RGB", (4, 2)) as image,
        pytest.raises(
            ValueError, match=r"dimensions|geometry|skew|[Mm]issing"
        ),
    ):
        prediction.backend.predict(job, image, job.output, "a" * 64)
    assert not Path(job.output).exists()
