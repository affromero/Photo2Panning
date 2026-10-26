"""Lazy native camera rendering without optional imports at discovery time."""

from typing import Protocol, cast

from PIL import Image

from pic2panning.backends.contracts import CameraScene, SceneBackend
from pic2panning.backends.loading import load_dependency
from pic2panning.errors import DependencyError
from pic2panning.schemas.camera import SceneCalibration
from pic2panning.schemas.scene import SceneRequest


class RendererFactory(Protocol):
    """Construct a native renderer at its explicit dependency boundary."""

    def __call__(
        self, scene: str, calibration: SceneCalibration
    ) -> CameraScene:
        """Load the selected scene on the supported local GPU."""
        ...


class RendererModule(Protocol):
    """The native backend export loaded only by camera execution."""

    SplatRenderer: RendererFactory


class NativeCameraBackend:
    """Load the Metal renderer when a validated camera job executes."""

    def load(self, scene: str, calibration: SceneCalibration) -> CameraScene:
        """Explain the camera extra and preserve the dependency error."""
        try:
            module = cast(
                "RendererModule",
                load_dependency("pic2panning.backends.splatting"),
            )
        except DependencyError as error:
            raise DependencyError(
                "Camera rendering requires an Apple Silicon Mac and "
                "the camera extra. Run `uv sync --extra camera`. "
                f"Dependency failure: {error.__cause__}"
            ) from error
        return module.SplatRenderer(scene, calibration)


class PredictorFactory(Protocol):
    """Create one predictor at the optional framework boundary."""

    def __call__(self) -> SceneBackend:
        """Construct a stateless photo reconstruction adapter."""
        ...


class PredictorModule(Protocol):
    """The optional MoGe adapter's public factory."""

    MoGeSceneBackend: PredictorFactory


class NativeSceneBackend:
    """Load the pinned, permissively licensed predictor on execution."""

    def predict(
        self,
        request: SceneRequest,
        image: Image.Image,
        output: str,
        source_sha256: str,
    ) -> tuple[SceneCalibration, int]:
        """Explain missing camera dependencies without device fallback."""
        try:
            module = cast(
                "PredictorModule",
                load_dependency("pic2panning.backends.moge"),
            )
        except DependencyError as error:
            raise DependencyError(
                "Photo reconstruction requires the camera extra. "
                "Run `uv sync --extra camera`. "
                f"Dependency failure: {error.__cause__}"
            ) from error
        return module.MoGeSceneBackend().predict(
            request, image, output, source_sha256
        )
