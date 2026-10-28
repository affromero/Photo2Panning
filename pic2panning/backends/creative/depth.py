"""Framework-free depth contracts and lazy native inference loading."""

from typing import Protocol, cast

from PIL import Image
from pydantic import ConfigDict

from pic2panning.backends.loading import load_dependency
from pic2panning.errors import DependencyError
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.camera import Matrix3
from pic2panning.schemas.creative.depth import DepthRequest


class DepthPrediction(PydanticBaseModel):
    """Independent metric image fields returned across the model boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    depth: Image.Image
    mask: Image.Image
    intrinsics: Matrix3

    def close(self) -> None:
        """Release both returned pixel buffers after artifact processing."""
        self.depth.close()
        self.mask.close()


class DepthBackend(Protocol):
    """Predict metric depth without exposing model framework arrays."""

    def predict(
        self, request: DepthRequest, image: Image.Image
    ) -> DepthPrediction:
        """Return depth and semantic foreground in a calibrated bounded grid."""
        ...


class DepthFactory(Protocol):
    """Create the inference adapter at the optional dependency boundary."""

    def __call__(self) -> DepthBackend:
        """Construct the stateless pinned geometry predictor."""
        ...


class DepthModule(Protocol):
    """Expose only the native depth factory to core processing."""

    MoGeDepthBackend: DepthFactory


class NativeDepthBackend:
    """Load Torch and the pinned MoGe model only when extraction executes."""

    @staticmethod
    def predict(request: DepthRequest, image: Image.Image) -> DepthPrediction:
        """Surface dependency and device failures without selecting fallback."""
        try:
            module = cast(
                "DepthModule",
                load_dependency("pic2panning.backends.creative.moge"),
            )
        except DependencyError as error:
            raise DependencyError(
                "Depth extraction requires the camera extra. "
                "Run `uv sync --extra camera`. "
                f"Dependency failure: {error.__cause__}"
            ) from error
        return module.MoGeDepthBackend().predict(request, image)
