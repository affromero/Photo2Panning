"""Typed terminal controls for an imported Gaussian camera timeline."""

from typing import Annotated, Literal

import tyro

from pic2panning.command.base import OutputCommand
from pic2panning.schemas.base import Size
from pic2panning.schemas.camera import CameraRequest, Vector3
from pic2panning.schemas.media import Encoding, EncodingPreset
from pic2panning.schemas.scene import SceneRequest


class SceneCommand(OutputCommand):
    """Reconstruct a source photo into a reusable calibrated Gaussian scene."""

    output: str
    """Local Gaussian scene destination ending in .ply."""

    source: Annotated[str, tyro.conf.Positional]
    """Original image path or HTTP URL."""

    metadata: str
    """JSON calibration destination beside the PLY scene."""

    num_tokens: int = 1800
    """Inference token budget, from 1200 to 3600; larger costs more memory."""

    max_edge: int = 1920
    """Maximum geometry image edge, from 256 to 4096; keeps the source intact."""

    device: Literal["mps", "cpu"] = "mps"
    """Explicit geometry inference device; no automatic fallback."""

    def to_job(self) -> SceneRequest:
        """Validate scene output paths and model controls before inference."""
        return SceneRequest(
            source=self.source,
            output=self.output,
            metadata=self.metadata,
            num_tokens=self.num_tokens,
            max_edge=self.max_edge,
            device=self.device,
            overwrite=self.overwrite,
        )


class CameraCommand(OutputCommand):
    """Pan a photo, pause for a real 3D excursion, return, and resume."""

    output: str
    """Local silent video destination ending in .mp4."""

    source: Annotated[str, tyro.conf.Positional]
    """Original image path or URL matching the scene calibration hash."""

    scene: str
    """Local degree-zero Gaussian PLY scene."""

    metadata: str
    """Local JSON source calibration and provenance sidecar."""

    pan_start: float = 0
    """Initial horizontal crop position, from 0 at left to 1 at right."""

    pause_at: float = 0.5
    """Crop position at the pause, excursion start, and exact return."""

    pan_end: float = 1
    """Final horizontal crop position after the pan resumes."""

    before: float = 4
    """Seconds for the initial pan, including both crop endpoints."""

    duration: float = 5
    """Seconds for the complete camera excursion and return."""

    after: float = 3.2
    """Seconds for the resumed pan."""

    hold: float = 0.25
    """Still seconds on each side of the excursion; 0 omits holds."""

    position: Vector3 | None = None
    """Far offset in estimated metres: right, down, forward; negative Y rises."""

    size: tuple[int, int] = (1080, 1920)
    """Even output width and height."""

    fps: int = 60
    """Output frames per second, from 1 to 240."""

    shutter_angle: float = 180
    """Pan exposure from 0 to 360 degrees; 0 keeps instantaneous samples."""

    crf: int = 18
    """H.264 constant rate factor from 0 to 51; lower retains more detail."""

    preset: EncodingPreset = EncodingPreset.VERYFAST
    """H.264 compression effort; slower presets improve compression."""

    def to_job(self) -> CameraRequest:
        """Validate every camera option before source or scene access."""
        return CameraRequest(
            source=self.source,
            scene=self.scene,
            metadata=self.metadata,
            output=self.output,
            pan_start=self.pan_start,
            pause_at=self.pause_at,
            pan_end=self.pan_end,
            before=self.before,
            duration=self.duration,
            after=self.after,
            hold=self.hold,
            position=self.position,
            size=Size(width=self.size[0], height=self.size[1]),
            fps=self.fps,
            shutter_angle=self.shutter_angle,
            encoding=Encoding(crf=self.crf, preset=self.preset),
            overwrite=self.overwrite,
        )
