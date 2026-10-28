"""Project text onto a fitted plane and hide it behind nearer source pixels."""

import math

import numpy as np
from PIL import Image

from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.camera import Vector3
from pic2panning.schemas.creative.composition import LayerKeyframe
from pic2panning.schemas.creative.surface import (
    SurfaceAttachment,
    SurfacePlane,
)
from pic2panning.tools.creative.depth import DepthAsset
from pic2panning.tools.typography.surface import (
    cross,
    dot,
    intersect_plane,
    unit,
)

type Perspective = tuple[
    float, float, float, float, float, float, float, float
]
type Quad = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


class SurfaceProjection(PydanticBaseModel):
    """Source geometry; an edge-on plane has no invertible texture warp."""

    corners: Quad
    inverse: Perspective | None
    center: Vector3
    normal: Vector3


def edge_on_plane(right: Vector3, down: Vector3, center: Vector3) -> bool:
    """Recognize valid planes whose projected area is geometrically zero."""
    normal = cross(right, down)
    normal_length = math.sqrt(dot(normal, normal))
    axis_product = math.sqrt(dot(right, right) * dot(down, down))
    if not axis_product or normal_length <= axis_product * 1e-12:
        raise ValueError("Text plane has degenerate axes")
    distance = math.sqrt(dot(center, center))
    return abs(dot(normal, center)) <= normal_length * distance * 1e-12


def rotate_axis(vector: Vector3, axis: Vector3, degrees: float) -> Vector3:
    """Apply a right-hand camera-space rotation about a unit direction."""
    wrapped = degrees % 360
    if wrapped % 90 == 0:
        quarter = int(wrapped / 90)
        cosine = (1.0, 0.0, -1.0, 0.0)[quarter]
        sine = (0.0, 1.0, 0.0, -1.0)[quarter]
    else:
        radians = math.radians(wrapped)
        cosine, sine = math.cos(radians), math.sin(radians)
    perpendicular = cross(axis, vector)
    parallel = dot(axis, vector) * (1 - cosine)
    return (
        vector[0] * cosine + perpendicular[0] * sine + axis[0] * parallel,
        vector[1] * cosine + perpendicular[1] * sine + axis[1] * parallel,
        vector[2] * cosine + perpendicular[2] * sine + axis[2] * parallel,
    )


def oriented_plane(
    plane: SurfacePlane, attachment: SurfaceAttachment, pose: LayerKeyframe
) -> tuple[Vector3, Vector3, Vector3]:
    """Apply local tilt, yaw, and clockwise text roll to the fitted axes."""
    return local_axes(
        (plane.right, plane.down, plane.normal),
        (
            attachment.tilt + pose.rotation_x,
            attachment.yaw + pose.rotation_y,
            attachment.roll + pose.rotation,
        ),
    )


def local_axes(
    basis: tuple[Vector3, Vector3, Vector3], angles: Vector3
) -> tuple[Vector3, Vector3, Vector3]:
    """Rotate glyph axes in a local frame, retaining clockwise in-plane roll."""
    right = basis[0]
    tilt, yaw, roll = angles
    down = rotate_axis(basis[1], right, tilt)
    normal = rotate_axis(basis[2], right, tilt)
    right = rotate_axis(right, down, yaw)
    normal = unit(rotate_axis(normal, down, yaw))
    radians = math.radians(roll)
    cosine, sine = math.cos(radians), math.sin(radians)
    rolled_right = (
        right[0] * cosine + down[0] * sine,
        right[1] * cosine + down[1] * sine,
        right[2] * cosine + down[2] * sine,
    )
    rolled_down = (
        down[0] * cosine - right[0] * sine,
        down[1] * cosine - right[1] * sine,
        down[2] * cosine - right[2] * sine,
    )
    return unit(rolled_right), unit(rolled_down), normal


def surface_projection(
    asset: DepthAsset,
    attachment: SurfaceAttachment,
    plane: SurfacePlane,
    pose: LayerKeyframe,
    texture: Size,
) -> SurfaceProjection:
    """Compute a calibrated perspective warp without using noisy anchor depth."""
    right, down, normal = oriented_plane(plane, attachment, pose)
    anchor = intersect_plane(asset, plane, (pose.x, pose.y))
    center = (
        anchor[0]
        + plane.right[0] * attachment.offset_metres[0]
        + plane.down[0] * attachment.offset_metres[1]
        + plane.normal[0] * attachment.lift_metres,
        anchor[1]
        + plane.right[1] * attachment.offset_metres[0]
        + plane.down[1] * attachment.offset_metres[1]
        + plane.normal[1] * attachment.lift_metres,
        anchor[2]
        + plane.right[2] * attachment.offset_metres[0]
        + plane.down[2] * attachment.offset_metres[1]
        + plane.normal[2] * attachment.lift_metres,
    )
    return project_plane(
        asset,
        plane.model_copy(
            update={
                "center": center,
                "right": right,
                "down": down,
                "normal": normal,
                "width_metres": plane.width_metres * pose.scale,
            }
        ),
        texture,
    )


def project_plane(
    asset: DepthAsset, plane: SurfacePlane, texture: Size
) -> SurfaceProjection:
    """Project one metric rectangle, retaining its own per-pixel depth plane."""
    center, right, down, normal = (
        plane.center,
        plane.right,
        plane.down,
        plane.normal,
    )
    width = plane.width_metres
    height = width * texture.height / texture.width
    corners_3d = tuple(
        (
            center[0] + right[0] * u * width + down[0] * v * height,
            center[1] + right[1] * u * width + down[1] * v * height,
            center[2] + right[2] * u * width + down[2] * v * height,
        )
        for u, v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
    )
    if any(point[2] <= 1e-6 for point in corners_3d):
        raise ValueError("Surface text projects behind the camera")
    document = asset.metadata
    intrinsics = np.asarray(document.intrinsics, dtype=np.float64).copy()
    intrinsics[0] *= document.width / document.grid_width
    intrinsics[1] *= document.height / document.grid_height
    basis = np.column_stack(
        (
            np.asarray(right) * width / texture.width,
            np.asarray(down) * height / texture.height,
            np.asarray(center)
            - np.asarray(right) * width / 2
            - np.asarray(down) * height / 2,
        )
    )
    homography = intrinsics @ basis
    if not bool(np.isfinite(homography).all()):
        raise ValueError("Surface text projection is nonfinite")
    projected = np.asarray(corners_3d) @ intrinsics.T
    projected = projected[:, :2] / projected[:, 2:3]
    corners: Quad = (
        (float(projected[0, 0]), float(projected[0, 1])),
        (float(projected[1, 0]), float(projected[1, 1])),
        (float(projected[2, 0]), float(projected[2, 1])),
        (float(projected[3, 0]), float(projected[3, 1])),
    )
    if edge_on_plane(right, down, center):
        return SurfaceProjection(
            corners=corners, inverse=None, center=center, normal=normal
        )
    if float(np.linalg.cond(homography)) > 1e12:
        raise ValueError("Surface text projection is degenerate")
    inverse = np.linalg.inv(homography)
    if abs(float(inverse[2, 2])) < 1e-12:
        raise ValueError("Surface text perspective crosses the camera horizon")
    inverse /= inverse[2, 2]
    return SurfaceProjection(
        corners=corners,
        inverse=(
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[0, 2]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(inverse[1, 2]),
            float(inverse[2, 0]),
            float(inverse[2, 1]),
        ),
        center=center,
        normal=normal,
    )


def surface_visibility(
    asset: DepthAsset,
    attachment: SurfaceAttachment,
    projection: SurfaceProjection,
    bounds: tuple[int, int, int, int],
) -> Image.Image:
    """Compare source camera-Z with the projected plane at each covered pixel."""
    left, top, right, bottom = bounds
    document = asset.metadata
    columns, rows = np.meshgrid(
        np.arange(left, right, dtype=np.float64) + 0.5,
        np.arange(top, bottom, dtype=np.float64) + 0.5,
    )
    grid_x = columns * document.grid_width / document.width
    grid_y = rows * document.grid_height / document.height
    intrinsics = document.intrinsics
    normal = projection.normal
    denominator = (
        normal[0] * (grid_x - intrinsics[0][2]) / intrinsics[0][0]
        + normal[1] * (grid_y - intrinsics[1][2]) / intrinsics[1][1]
        + normal[2]
    )
    projected_depth = np.divide(
        dot(normal, projection.center),
        denominator,
        out=np.full_like(denominator, np.inf),
        where=np.abs(denominator) > 1e-10,
    )
    width_scale = document.grid_width / document.width
    height_scale = document.grid_height / document.height
    with (
        asset.depth.transform(
            (right - left, bottom - top),
            Image.Transform.AFFINE,
            (
                width_scale,
                0,
                left * width_scale,
                0,
                height_scale,
                top * height_scale,
            ),
            resample=Image.Resampling.BILINEAR,
        ) as selected_depth,
        asset.mask.transform(
            selected_depth.size,
            Image.Transform.AFFINE,
            (
                width_scale,
                0,
                left * width_scale,
                0,
                height_scale,
                top * height_scale,
            ),
            resample=Image.Resampling.NEAREST,
        ) as foreground,
    ):
        source_depth = np.asarray(selected_depth, dtype=np.float64)
        gap = (
            source_depth - projected_depth + attachment.depth_tolerance_metres
        )
        visible = (
            np.clip(gap / attachment.feather_metres + 1, 0, 1)
            if attachment.feather_metres
            else (gap >= 0).astype(np.float64)
        )
        valid = np.isfinite(projected_depth) & (projected_depth > 0)
        visible = np.where(np.asarray(foreground) > 0, visible, 1) * valid
    return Image.fromarray(np.rint(visible * 255).astype(np.uint8))


def render_surface(
    bitmap: Image.Image,
    asset: DepthAsset,
    attachment: SurfaceAttachment,
    plane: SurfacePlane,
    pose: LayerKeyframe,
) -> Image.Image:
    """Project an RGBA texture into the host's source grid with depth occlusion."""
    projection = surface_projection(
        asset,
        attachment,
        plane,
        pose,
        Size(width=bitmap.width, height=bitmap.height),
    )
    result = Image.new("RGBA", (asset.metadata.width, asset.metadata.height))
    try:
        tile = surface_tile(bitmap, asset, attachment, projection)
        if tile is not None:
            pixels, position = tile
            with pixels:
                result.paste(pixels, position)
    except BaseException:
        result.close()
        raise
    else:
        return result


def local_perspective(
    inverse: Perspective, left: int, top: int
) -> Perspective:
    """Rebase one inverse homography to the clipped output tile's origin."""
    a, b, c, d, e, f, g, h = inverse
    denominator = 1 + g * left + h * top
    if abs(denominator) < 1e-12:
        raise ValueError("Text perspective crosses the image horizon")
    return (
        a / denominator,
        b / denominator,
        (a * left + b * top + c) / denominator,
        d / denominator,
        e / denominator,
        (d * left + e * top + f) / denominator,
        g / denominator,
        h / denominator,
    )


def projection_bounds(
    corners: Quad, output: Size
) -> tuple[int, int, int, int] | None:
    """Clip a finite, valid projected rectangle to the actual output frame."""
    xs, ys = zip(*corners, strict=True)
    left = max(0, math.floor(min(xs)))
    top = max(0, math.floor(min(ys)))
    right = min(output.width, math.ceil(max(xs)))
    bottom = min(output.height, math.ceil(max(ys)))
    return (
        (left, top, right, bottom) if right > left and bottom > top else None
    )


def surface_tile(
    bitmap: Image.Image,
    asset: DepthAsset,
    attachment: SurfaceAttachment,
    projection: SurfaceProjection,
) -> tuple[Image.Image, tuple[int, int]] | None:
    """Own only the covered glyph/texture ROI while retaining camera-Z occlusion."""
    if projection.inverse is None:
        return None
    bounds = projection_bounds(
        projection.corners,
        Size(width=asset.metadata.width, height=asset.metadata.height),
    )
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    inverse = local_perspective(projection.inverse, left, top)
    rgba = bitmap.convert("RGBA")
    try:
        warped = rgba.transform(
            (right - left, bottom - top),
            Image.Transform.PERSPECTIVE,
            inverse,
            resample=Image.Resampling.BICUBIC,
        )
    finally:
        rgba.close()
    try:
        with (
            warped.getchannel("A") as alpha,
            surface_visibility(
                asset, attachment, projection, (left, top, right, bottom)
            ) as visible,
        ):
            opacity = (
                np.asarray(alpha, dtype=np.float32)
                * np.asarray(visible, dtype=np.float32)
                / 255
            )
            with Image.fromarray(np.rint(opacity).astype(np.uint8)) as covered:
                warped.putalpha(covered)
    except BaseException:
        warped.close()
        raise
    else:
        return warped, (left, top)
