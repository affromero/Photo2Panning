"""Bounded video sampling shared by composition previews and exports."""

import math
from collections.abc import Generator

from PIL import Image

from pic2panning.errors import MediaError
from pic2panning.schemas.creative.composition import VideoLayer
from pic2panning.tools.media import MediaInfo, decode_frames


def source_time(layer: VideoLayer, local_time: float) -> float:
    """Evaluate a monotone source-time curve, holding after its last point."""
    if not layer.source_times:
        return layer.source_start + local_time * layer.speed
    for left, right in zip(
        layer.source_times, layer.source_times[1:], strict=False
    ):
        if local_time < right.at:
            fraction = (local_time - left.at) / (right.at - left.at)
            return left.source_time + fraction * (
                right.source_time - left.source_time
            )
    return layer.source_times[-1].source_time


class VideoSampler:
    """Retain one CFR frame, sharing exact FFmpeg sampling between both paths."""

    source: str
    info: MediaInfo
    fps: int
    preview: bool
    decoded: Generator[Image.Image] | None
    current: Image.Image | None
    index: int

    def __init__(
        self,
        source: str,
        info: MediaInfo,
        *,
        preview: bool,
    ) -> None:
        """Use a source sampling grid shared by both rendering paths."""
        if not math.isfinite(info.fps) or info.fps <= 0:
            raise MediaError("Cannot determine a positive video frame rate")
        self.source = source
        self.info = info
        self.fps = max(1, math.ceil(info.fps))
        self.preview = preview
        self.decoded = None
        self.current = None
        self.index = -1

    def seek(self, seconds: float) -> Image.Image:
        """Select one frame with the canonical CFR decoder and bounded RAM."""
        sampler = VideoSampler(self.source, self.info, preview=False)
        try:
            return sampler.frame(seconds)
        finally:
            sampler.close()

    def frame(self, seconds: float) -> Image.Image:
        """Return an owned source frame without retaining a video in RAM."""
        if seconds < 0 or seconds > self.info.duration + 1e-9:
            raise MediaError("Video source time is outside its duration")
        maximum = max(0, math.ceil(self.info.duration * self.fps) - 1)
        wanted = min(maximum, math.floor(seconds * self.fps + 1e-8))
        if wanted < self.index:
            if not self.preview:
                raise MediaError("Export video sampling must move forward")
            self.close()
        if self.decoded is None:
            self.decoded = decode_frames(self.source, self.info, self.fps)
        while self.index < wanted:
            try:
                selected = next(self.decoded)
            except StopIteration:
                if self.current is None:
                    raise MediaError(
                        "Video contains no decoded frames"
                    ) from None
                break
            if self.current is not None:
                self.current.close()
            self.current = selected
            self.index += 1
        if self.current is None:
            raise MediaError("Video contains no decoded frames")
        return self.current.copy()

    def close(self) -> None:
        """Close decoded pixels and reap an interrupted FFmpeg subprocess."""
        if self.decoded is not None:
            self.decoded.close()
            self.decoded = None
        if self.current is not None:
            self.current.close()
            self.current = None
        self.index = -1
