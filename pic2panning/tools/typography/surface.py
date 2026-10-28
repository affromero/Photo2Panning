"""Bounded robust surface fitting from an existing verified depth asset."""

import math

import numpy as np

from pic2panning.schemas.camera import Vector3
from pic2panning.schemas.creative.surface import (
    SurfaceAttachment,
    SurfacePlane,
)
from pic2panning.tools.creative.depth import DepthAsset


def dot(left: Vector3, right: Vector3) -> float:
    """Return the scalar product of two explicit camera-space vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def cross(left: Vector3, right: Vector3) -> Vector3:
    """Return a camera-space cross product without an array API boundary."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def unit(vector: Vector3) -> Vector3:
    """Normalize a finite, non-degenerate vector."""
    length = math.sqrt(dot(vector, vector))
    if not math.isfinite(length) or length < 1e-10:
        raise ValueError("Surface geometry contains a degenerate direction")
    return vector[0] / length, vector[1] / length, vector[2] / length


def source_ray(asset: DepthAsset, point: tuple[float, float]) -> Vector3:
    """Unproject an oriented source pixel center into a camera ray."""
    document = asset.metadata
    x = point[0] * document.grid_width / document.width
    y = point[1] * document.grid_height / document.height
    intrinsics = document.intrinsics
    return (
        (x - intrinsics[0][2]) / intrinsics[0][0],
        (y - intrinsics[1][2]) / intrinsics[1][1],
        1,
    )


def intersect_plane(
    asset: DepthAsset, plane: SurfacePlane, point: tuple[float, float]
) -> Vector3:
    """Intersect a calibrated source ray with the fitted metric plane."""
    ray = source_ray(asset, point)
    denominator = dot(plane.normal, ray)
    if abs(denominator) < 1e-7:
        raise ValueError("Surface anchor ray is parallel to the fitted plane")
    depth = dot(plane.normal, plane.center) / denominator
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError("Surface anchor intersects behind the camera")
    return ray[0] * depth, ray[1] * depth, depth


def patch_samples(
    asset: DepthAsset, attachment: SurfaceAttachment
) -> tuple[tuple[float, float, float], ...]:
    """Unproject at most 4096 foreground depth samples from the source patch."""
    document = asset.metadata
    left, top, right, bottom = attachment.region
    if right > document.width or bottom > document.height:
        raise ValueError("Surface region exceeds oriented source dimensions")
    width_ratio = document.grid_width / document.width
    height_ratio = document.grid_height / document.height
    x_start = max(0, math.ceil(left * width_ratio - 0.5))
    y_start = max(0, math.ceil(top * height_ratio - 0.5))
    x_stop = min(document.grid_width, math.ceil(right * width_ratio - 0.5))
    y_stop = min(document.grid_height, math.ceil(bottom * height_ratio - 0.5))
    if x_stop <= x_start or y_stop <= y_start:
        raise ValueError("Surface region contains no cached depth samples")
    x_indices = np.unique(
        np.rint(
            np.linspace(x_start, x_stop - 1, min(64, x_stop - x_start))
        ).astype(np.int64)
    )
    y_indices = np.unique(
        np.rint(
            np.linspace(y_start, y_stop - 1, min(64, y_stop - y_start))
        ).astype(np.int64)
    )
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    depths = np.asarray(asset.depth)[y_grid, x_grid]
    foreground = np.asarray(asset.mask)[y_grid, x_grid] > 0
    finite = np.isfinite(depths) & (depths > 0)
    accepted = foreground & finite
    if float(accepted.mean()) < 0.6 or int(accepted.sum()) < 16:
        raise ValueError("Surface region has insufficient foreground depth")
    x_values = x_grid[accepted].astype(np.float64) + 0.5
    y_values = y_grid[accepted].astype(np.float64) + 0.5
    z_values = depths[accepted].astype(np.float64)
    intrinsics = document.intrinsics
    x_values = (x_values - intrinsics[0][2]) / intrinsics[0][0] * z_values
    y_values = (y_values - intrinsics[1][2]) / intrinsics[1][1] * z_values
    return tuple(
        (float(x), float(y), float(z))
        for x, y, z in zip(x_values, y_values, z_values, strict=True)
    )


def fit_points(
    samples: tuple[tuple[float, float, float], ...],
    max_relative_error: float,
) -> SurfacePlane:
    """Fit a dominant plane and reject unsupported or nonplanar selections."""
    points = np.asarray(samples, dtype=np.float64)
    count = len(points)
    if count < 16:
        raise ValueError("Surface fit requires at least 16 depth samples")
    selected = np.arange(count)
    trim_count = max(16, math.ceil(count * 0.8))
    for _ in range(4):
        center = points[selected].mean(axis=0)
        _, singular, basis = np.linalg.svd(
            points[selected] - center, full_matrices=False
        )
        if singular[1] < 1e-7 or singular[1] / singular[0] < 1e-4:
            raise ValueError("Surface region cannot define a stable plane")
        normal = basis[-1]
        distances = np.abs((points - center) @ normal)
        selected = np.argsort(distances, kind="stable")[:trim_count]
    center = points[selected].mean(axis=0)
    _, singular, basis = np.linalg.svd(
        points[selected] - center, full_matrices=False
    )
    if singular[1] < 1e-7 or singular[1] / singular[0] < 1e-4:
        raise ValueError("Surface region cannot define a stable plane")
    normal = basis[-1]
    if float(normal @ center) > 0:
        normal = -normal
    distances = np.abs((points - center) @ normal)
    median_depth = float(np.median(points[:, 2]))
    relative_error = float(
        np.sqrt(np.mean(distances[selected] ** 2)) / median_depth
    )
    inliers = float((distances <= max_relative_error * median_depth).mean())
    if relative_error > max_relative_error or inliers < 0.6:
        raise ValueError(
            "Selected depth is not planar enough; choose a smaller surface region"
        )
    normal_tuple = float(normal[0]), float(normal[1]), float(normal[2])
    right = unit(
        (
            1 - normal_tuple[0] ** 2,
            -normal_tuple[0] * normal_tuple[1],
            -normal_tuple[0] * normal_tuple[2],
        )
    )
    down = unit(cross(right, normal_tuple))
    relative = points[selected] - center
    width = float(np.ptp(relative @ np.asarray(right)))
    height = float(np.ptp(relative @ np.asarray(down)))
    if width <= 1e-7 or height <= 1e-7:
        raise ValueError("Surface patch has no usable physical extent")
    return SurfacePlane(
        center=(float(center[0]), float(center[1]), float(center[2])),
        normal=normal_tuple,
        right=right,
        down=down,
        width_metres=width,
        height_metres=height,
        relative_error=relative_error,
        samples=count,
        inlier_fraction=inliers,
    )


def fit_surface(
    asset: DepthAsset, attachment: SurfaceAttachment
) -> SurfacePlane:
    """Fit one inspectable plane from a verified, bounded source selection."""
    plane = fit_points(
        patch_samples(asset, attachment), attachment.max_relative_error
    )
    left, top, right, bottom = attachment.region
    center = intersect_plane(
        asset, plane, ((left + right) / 2, (top + bottom) / 2)
    )
    # Preserve the selected patch's full physical width after robust trimming.
    corners = tuple(
        intersect_plane(asset, plane, point)
        for point in (
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        )
    )
    horizontal = tuple(dot(point, plane.right) for point in corners)
    vertical = tuple(dot(point, plane.down) for point in corners)
    return plane.model_copy(
        update={
            "center": center,
            "width_metres": max(horizontal) - min(horizontal),
            "height_metres": max(vertical) - min(vertical),
        }
    )
