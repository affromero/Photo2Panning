"""Compute image and video viewports without retaining rendered frames."""

import math
from collections.abc import Iterable, Iterator

import numpy as np
from PIL import Image

from pic2panning.schemas.base import Size
from pic2panning.schemas.media import ImageClip, Motion, VideoClip

SHUTTER_SAMPLES = 8
LINEAR_SRGB = np.array(
    [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in (level / 255 for level in range(256))
    ],
    dtype=np.float32,
)
LINEAR_SRGB.setflags(write=False)


def weighted_linear_blend(
    samples: Iterable[tuple[Image.Image, float]],
) -> Image.Image:
    """Average RGB samples in linear light, closing each consumed image."""
    linear = None
    size = None
    total = 0.0
    for image, weight in samples:
        with image:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    "Image weights must be finite and nonnegative"
                )
            if image.mode != "RGB":
                raise ValueError("Image blending requires RGB samples")
            if size is not None and size != image.size:
                raise ValueError("Image samples must share their dimensions")
            size = image.size
            sample = LINEAR_SRGB[np.asarray(image)] * weight
            if linear is None:
                linear = sample
            else:
                linear += sample
            total += weight
    if linear is None or total <= 0 or not math.isfinite(total):
        raise ValueError("Image blending requires a positive total weight")
    linear /= total
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055,
    )
    pixels = np.rint(np.clip(srgb * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(pixels)


def frame_count(duration: float, fps: int) -> int:
    """Round a requested duration up to a representable frame count."""
    return max(1, math.ceil(duration * fps))


def center_progress(progress: float) -> float:
    """Slow motion near the center while retaining exact endpoints."""
    return progress + 0.8 * math.sin(2 * math.pi * progress) / (2 * math.pi)


def crop_dimensions(image: Image.Image, size: Size) -> tuple[float, float]:
    """Return the largest source viewport matching the output aspect ratio."""
    scale = min(image.width / size.width, image.height / size.height)
    return (
        min(float(image.width), size.width * scale),
        min(float(image.height), size.height * scale),
    )


def transform_frame(
    image: Image.Image,
    size: Size,
    motion: Motion | None,
    progress: float,
    zoom: float = 4 / 3,
) -> Image.Image:
    """Crop and resize one frame while preserving the source proportions."""
    width, height = crop_dimensions(image, size)
    if motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT):
        zoom_progress = 1 - progress if motion == Motion.ZOOM_OUT else progress
        factor = 1 + (zoom - 1) * zoom_progress
        width /= factor
        height /= factor
    left = (image.width - width) / 2
    top = (image.height - height) / 2
    if motion == Motion.PAN_LR:
        left = (image.width - width) * progress
    elif motion == Motion.PAN_RL:
        left = (image.width - width) * (1 - progress)
    elif motion == Motion.PAN_UD:
        top = (image.height - height) * progress
    elif motion == Motion.PAN_DU:
        top = (image.height - height) * (1 - progress)
    return image.resize(
        size.as_tuple,
        resample=Image.Resampling.LANCZOS,
        box=(left, top, left + width, top + height),
    )


def image_frames(
    image: Image.Image, clip: ImageClip, size: Size, fps: int
) -> Iterator[Image.Image]:
    """Yield a deterministic animation, including an optional return journey."""
    count = frame_count(clip.duration, fps)
    total = count * 2 - 1 if clip.reverse else count
    for index in range(total):
        step = index if index < count else 2 * count - 2 - index
        progress = step / (count - 1) if count > 1 else 0.0
        if clip.shutter_angle and 0 < step < count - 1:
            yield exposure_frame(
                image,
                clip,
                size,
                progress,
                clip.shutter_angle / (360 * (count - 1)),
            )
            continue
        if clip.focus_center:
            progress = center_progress(progress)
        yield transform_frame(image, size, clip.motion, progress, clip.zoom)


def exposure_frame(
    image: Image.Image,
    clip: ImageClip,
    size: Size,
    progress: float,
    interval: float,
) -> Image.Image:
    """Integrate an RGB exposure in linear light without retaining subframes."""

    def samples() -> Iterator[tuple[Image.Image, float]]:
        for index in range(SHUTTER_SAMPLES):
            offset = ((index + 0.5) / SHUTTER_SAMPLES - 0.5) * interval
            sample_progress = min(1.0, max(0.0, progress + offset))
            if clip.focus_center:
                sample_progress = center_progress(sample_progress)
            yield (
                transform_frame(
                    image, size, clip.motion, sample_progress, clip.zoom
                ),
                1.0,
            )

    return weighted_linear_blend(samples())


def video_frame(
    image: Image.Image,
    clip: VideoClip,
    size: Size,
    index: int,
    count: int,
) -> Image.Image:
    """Apply a repeated bouncing pan or a single zoom to a video frame."""
    progress = min(index / (count - 1), 1) if count > 1 else 0.0
    if clip.motion in (
        Motion.PAN_LR,
        Motion.PAN_RL,
        Motion.PAN_UD,
        Motion.PAN_DU,
    ):
        width, height = crop_dimensions(image, size)
        horizontal = clip.motion in (Motion.PAN_LR, Motion.PAN_RL)
        distance = image.width - width if horizontal else image.height - height
        scale = width / size.width if horizontal else height / size.height
        if distance > 0:
            offset = (index * clip.pan_speed * scale) % (2 * distance)
            progress = min(offset, 2 * distance - offset) / distance
        else:
            progress = 0.0
    return transform_frame(image, size, clip.motion, progress, clip.zoom)
