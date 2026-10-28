"""Deterministic cached-depth effects using only Pillow and NumPy."""

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageColor, ImageFilter

from pic2panning.schemas.camera import Matrix3
from pic2panning.schemas.creative.depth import DepthStyle

if TYPE_CHECKING:
    from pic2panning.tools.creative.depth import DepthAsset


def rescale_depth_style(
    style: DepthStyle,
    source_size: tuple[int, int],
    working_size: tuple[int, int],
) -> DepthStyle:
    """Keep source-pixel blur and displacement constant on a smaller grid."""
    if any(value <= 0 for value in (*source_size, *working_size)):
        raise ValueError("Depth working dimensions must be positive")
    scale_x = working_size[0] / source_size[0]
    scale_y = working_size[1] / source_size[1]
    if scale_x > 1 or scale_y > 1:
        raise ValueError("Depth working grid cannot exceed source dimensions")
    if abs(
        working_size[0] * source_size[1] - working_size[1] * source_size[0]
    ) > max(source_size):
        raise ValueError(
            "Depth working grid must preserve source aspect ratio"
        )
    return style.model_copy(
        update={
            "blur_radius": style.blur_radius * math.sqrt(scale_x * scale_y),
            "parallax_x": style.parallax_x * scale_x,
            "parallax_y": style.parallax_y * scale_y,
        }
    )


def validate_depth(depth: Image.Image) -> None:
    """Require finite positive float metric pixels, without normalization."""
    if depth.mode != "F":
        raise ValueError("Metric depth must use Pillow mode F")
    values = np.asarray(depth)
    if not bool((np.isfinite(values) & (values > 0)).all()):
        raise ValueError("Metric depth must contain finite positive metres")


def normal_image(depth: Image.Image, intrinsics: Matrix3) -> Image.Image:
    """Encode camera-facing unit normals from calibrated metric geometry."""
    validate_depth(depth)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    if fx <= 0 or fy <= 0:
        raise ValueError("Depth normals require positive focal lengths")
    z = np.asarray(depth, dtype=np.float64)
    rows, columns = np.indices(z.shape, dtype=np.float64)
    points = np.stack(
        (
            (columns + 0.5 - intrinsics[0][2]) * z / fx,
            (rows + 0.5 - intrinsics[1][2]) * z / fy,
            z,
        ),
        axis=-1,
    )
    horizontal = (
        points[:, np.minimum(np.arange(depth.width) + 1, depth.width - 1)]
        - points[:, np.maximum(np.arange(depth.width) - 1, 0)]
    )
    vertical = (
        points[np.minimum(np.arange(depth.height) + 1, depth.height - 1)]
        - points[np.maximum(np.arange(depth.height) - 1, 0)]
    )
    if depth.width == 1:
        horizontal[..., 0] = z / fx
    if depth.height == 1:
        vertical[..., 1] = z / fy
    normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if not bool((np.isfinite(lengths) & (lengths > 0)).all()):
        raise ValueError("Depth geometry cannot produce finite unit normals")
    normals /= lengths
    normals *= np.where(
        np.sum(normals * points, axis=-1, keepdims=True) > 0, -1, 1
    )
    encoded = np.rint((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(encoded)


def depth_mask(
    depth: Image.Image, plane_metres: float, feather_metres: float = 0
) -> Image.Image:
    """Select nearer geometry in white, feathering around the metric plane."""
    validate_depth(depth)
    if not math.isfinite(plane_metres) or plane_metres <= 0:
        raise ValueError("Occlusion plane must be finite positive metres")
    if not math.isfinite(feather_metres) or feather_metres < 0:
        raise ValueError("Depth feather must be finite nonnegative metres")
    values = np.asarray(depth)
    if not feather_metres:
        return Image.fromarray(
            ((values < plane_metres) * 255).astype(np.uint8)
        )
    alpha = np.clip(
        (plane_metres + feather_metres / 2 - values) / feather_metres,
        0,
        1,
    )
    return Image.fromarray(np.rint(alpha * 255).astype(np.uint8))


def depth_blur(
    image: Image.Image,
    depth: Image.Image,
    focus_metres: float,
    blur_radius: float,
) -> Image.Image:
    """Composite blurred depth bands from far to near to retain sharp edges."""
    validate_depth(depth)
    if image.size != depth.size:
        raise ValueError("Depth blur image dimensions differ from depth")
    if not math.isfinite(focus_metres) or focus_metres <= 0:
        raise ValueError("Focus must be finite positive metres")
    if not math.isfinite(blur_radius) or not 0 <= blur_radius <= 64:
        raise ValueError("Depth blur radius must be between 0 and 64 pixels")
    if not blur_radius:
        return image.convert("RGB")
    values = np.asarray(depth)
    with image.convert("RGB") as rgb:
        colors = np.asarray(rgb, dtype=np.float32)
    centres = np.unique(np.quantile(values, np.linspace(0, 1, 16)))
    bands = np.searchsorted((centres[:-1] + centres[1:]) / 2, values)
    result = image.convert("RGBA")
    for index in reversed(range(len(centres))):
        selected = bands == index
        if not bool(selected.any()):
            continue
        distance = float(np.median(values[selected]))
        radius = blur_radius * min(abs(distance - focus_metres) / distance, 1)
        with Image.fromarray(selected.astype(np.uint8) * 255) as selection:
            bounds = selection.getbbox()
            if bounds is None:
                continue
            padding = math.ceil(radius * 4) + 4
            left = max(0, bounds[0] - padding)
            top = max(0, bounds[1] - padding)
            right = min(image.width, bounds[2] + padding)
            bottom = min(image.height, bounds[3] + padding)
            local_colors = colors[top:bottom, left:right]
            pixels = np.rint(
                local_colors * selected[top:bottom, left:right, None]
            ).astype(np.uint8)
            with (
                Image.fromarray(pixels) as premultiplied,
                selection.crop((left, top, right, bottom)) as local_mask,
                premultiplied.filter(
                    ImageFilter.GaussianBlur(radius)
                ) as blurred,
                local_mask.filter(ImageFilter.GaussianBlur(radius)) as opacity,
            ):
                alpha = np.asarray(opacity, dtype=np.float32) / 255
                straight = np.divide(
                    np.asarray(blurred, dtype=np.float32),
                    alpha[..., None],
                    out=np.zeros_like(local_colors),
                    where=alpha[..., None] > 0,
                )
                rgba = np.concatenate(
                    (straight.clip(0, 255), alpha[..., None] * 255), axis=-1
                )
                with Image.fromarray(np.rint(rgba).astype(np.uint8)) as layer:
                    result.alpha_composite(layer, (left, top))
    try:
        with (
            result.convert("RGB") as blurred_result,
            image.convert("RGB") as original,
            Image.fromarray(
                (np.isclose(values, focus_metres, rtol=1e-6) * 255).astype(
                    np.uint8
                )
            ) as focused,
        ):
            return Image.composite(original, blurred_result, focused)
    finally:
        result.close()


def fog(
    image: Image.Image, depth: Image.Image, density: float, color: str
) -> Image.Image:
    """Apply metric exponential atmospheric attenuation in linear light."""
    validate_depth(depth)
    if image.size != depth.size:
        raise ValueError("Fog image dimensions differ from depth")
    if not math.isfinite(density) or density < 0:
        raise ValueError("Fog density must be finite and nonnegative")
    if not density:
        return image.convert("RGB")
    target = np.asarray(ImageColor.getrgb(color)[:3], dtype=np.float32) / 255
    with image.convert("RGB") as rgb:
        pixels = np.asarray(rgb, dtype=np.float32) / 255
    source_linear = np.where(
        pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4
    )
    target_linear = np.where(
        target <= 0.04045, target / 12.92, ((target + 0.055) / 1.055) ** 2.4
    )
    transmission = np.exp(-density * np.asarray(depth))[..., None]
    linear = source_linear * transmission + target_linear * (1 - transmission)
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear ** (1 / 2.4) - 0.055,
    )
    return Image.fromarray(
        np.rint(encoded * 255).clip(0, 255).astype(np.uint8)
    )


def _relight(
    image: Image.Image, normals: Image.Image, style: DepthStyle
) -> Image.Image:
    """Mix camera-space Lambertian shading with the photographed colors."""
    vectors = np.asarray(normals, dtype=np.float32) / 127.5 - 1
    vectors /= np.maximum(
        np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9
    )
    light = np.asarray(style.light_direction, dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.maximum(np.sum(vectors * light, axis=-1), 0)
    illumination = style.ambient + (1 - style.ambient) * diffuse
    gain = 1 - style.relight_strength + style.relight_strength * illumination
    with image.convert("RGB") as rgb:
        pixels = np.asarray(rgb, dtype=np.float32) * gain[..., None]
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))


def _focus(style: DepthStyle, progress: float) -> float:
    """Interpolate a rack focus in metric distance at validated progress."""
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("Depth effect progress must be between zero and one")
    end = style.focus_end_metres or style.focus_metres
    return style.focus_metres + (end - style.focus_metres) * progress


def apply_depth_parallax(
    image: Image.Image,
    asset: DepthAsset,
    style: DepthStyle,
    progress: float,
) -> Image.Image:
    """Inverse-warp RGB, RGBA, or L pixels with bounded edge-repeat sampling."""
    focus = _focus(style, progress)
    if image.mode not in {"RGB", "RGBA", "L"}:
        raise ValueError("Depth parallax supports RGB, RGBA, and L images")
    validate_depth(asset.depth)
    if not (style.parallax_x or style.parallax_y) or not progress:
        return image.copy()
    with asset.depth.resize(image.size, Image.Resampling.BILINEAR) as depth:
        amount = np.clip(focus / np.asarray(depth) - 1, -1, 1) * progress
    rows, columns = np.indices((image.height, image.width), dtype=np.float32)
    x = np.clip(columns - style.parallax_x * amount, 0, image.width - 1)
    y = np.clip(rows - style.parallax_y * amount, 0, image.height - 1)
    left, top = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    right, bottom = (
        np.minimum(left + 1, image.width - 1),
        np.minimum(top + 1, image.height - 1),
    )
    fraction_x, fraction_y = x - left, y - top
    pixels = np.asarray(image, dtype=np.float32)
    if image.mode != "L":
        fraction_x, fraction_y = fraction_x[..., None], fraction_y[..., None]
    result = (
        pixels[top, left] * (1 - fraction_x) * (1 - fraction_y)
        + pixels[top, right] * fraction_x * (1 - fraction_y)
        + pixels[bottom, left] * (1 - fraction_x) * fraction_y
        + pixels[bottom, right] * fraction_x * fraction_y
    )
    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8))


def apply_depth_style(
    image: Image.Image,
    asset: DepthAsset,
    style: DepthStyle,
    progress: float,
) -> Image.Image:
    """Apply cached metric focus, fog, relighting, then shared parallax."""
    focus = _focus(style, progress)
    result = image.convert("RGB")
    try:
        with asset.depth.resize(
            image.size, Image.Resampling.BILINEAR
        ) as depth:
            if style.blur_radius:
                changed = depth_blur(result, depth, focus, style.blur_radius)
                result.close()
                result = changed
            if style.fog_density:
                changed = fog(
                    result, depth, style.fog_density, style.fog_color
                )
                result.close()
                result = changed
        if style.relight_strength:
            with (
                asset.normals.resize(
                    image.size, Image.Resampling.BILINEAR
                ) as normals,
                asset.mask.resize(
                    image.size, Image.Resampling.BILINEAR
                ) as foreground,
                _relight(result, normals, style) as relit,
            ):
                changed = Image.composite(relit, result, foreground)
            result.close()
            result = changed
        return apply_depth_parallax(result, asset, style, progress)
    finally:
        result.close()
