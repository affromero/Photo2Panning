"""Perspective rotation into the output frame without clipping glyph textures."""

import numpy as np
from PIL import Image

from pic2panning.schemas.base import PositiveFloat, PydanticBaseModel, Size
from pic2panning.schemas.camera import Vector3
from pic2panning.tools.typography.projection import (
    Perspective,
    Quad,
    edge_on_plane,
    local_perspective,
    projection_bounds,
    rotate_axis,
)
from pic2panning.tools.typography.surface import cross, unit

type Affine = tuple[float, float, float, float, float, float]


class TextOrientation(PydanticBaseModel):
    """Virtual-camera X/Y/Z angles in degrees and authored-pixel focal length."""

    rotation_x: float = 0
    rotation_y: float = 0
    rotation_z: float = 0
    perspective: PositiveFloat = 1200


class TextPlacement(PydanticBaseModel):
    """The compositor's exact inverse output-to-bitmap placement mapping."""

    output: Size
    authored: Size
    inverse: Affine


class TextPlane(PydanticBaseModel):
    """One camera-space rectangle measured in authored layer pixels."""

    center: Vector3
    right: Vector3
    down: Vector3
    width: PositiveFloat
    height: PositiveFloat
    perspective: PositiveFloat


class TextProjection(PydanticBaseModel):
    """Output geometry; an edge-on plane has no inverse texture sampling map."""

    corners: Quad
    inverse: Perspective | None


def rotation_axes(
    orientation: TextOrientation,
) -> tuple[Vector3, Vector3, Vector3]:
    """Rotate around camera X, then Y, then Z while retaining an orthonormal frame."""
    right: Vector3 = (1, 0, 0)
    down: Vector3 = (0, 1, 0)
    for axis, angle in (
        ((1.0, 0.0, 0.0), orientation.rotation_x),
        ((0.0, 1.0, 0.0), orientation.rotation_y),
        ((0.0, 0.0, 1.0), orientation.rotation_z),
    ):
        right = rotate_axis(right, axis, angle)
        down = rotate_axis(down, axis, angle)
    toward = cross(down, right)
    return unit(right), unit(down), unit(toward)


def text_plane_projection(
    geometry: TextPlane,
    placement: TextPlacement,
    reference: Size,
    texture: Size,
) -> TextProjection:
    """Compose physical perspective with the existing affine layer/camera map."""
    right, down, center = (
        np.asarray(geometry.right),
        np.asarray(geometry.down),
        np.asarray(geometry.center),
    )
    corners_3d = np.asarray(
        [
            center + right * u * geometry.width + down * v * geometry.height
            for u, v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
        ]
    )
    if bool((corners_3d[:, 2] <= 1e-6).any()):
        raise ValueError("3D text projects behind the virtual camera")
    intrinsics = np.asarray(
        (
            (
                geometry.perspective
                * reference.width
                / placement.authored.width,
                0,
                reference.width / 2,
            ),
            (
                0,
                geometry.perspective
                * reference.height
                / placement.authored.height,
                reference.height / 2,
            ),
            (0, 0, 1),
        ),
        dtype=np.float64,
    )
    basis = np.column_stack(
        (
            right * geometry.width / texture.width,
            down * geometry.height / texture.height,
            center - right * geometry.width / 2 - down * geometry.height / 2,
        )
    )
    a, b, c, d, e, f = placement.inverse
    inverse_affine = np.asarray(((a, b, c), (d, e, f), (0, 0, 1)))
    affine_determinant = float(np.linalg.det(inverse_affine))
    if not np.isfinite(affine_determinant) or abs(affine_determinant) < 1e-12:
        raise ValueError("3D text placement is degenerate")
    homography = np.linalg.inv(inverse_affine) @ intrinsics @ basis
    if not bool(np.isfinite(homography).all()):
        raise ValueError("3D text projection is nonfinite")
    corners = (
        np.asarray(
            (
                (0, 0, 1),
                (texture.width, 0, 1),
                (texture.width, texture.height, 1),
                (0, texture.height, 1),
            ),
            dtype=np.float64,
        )
        @ homography.T
    )
    corners = corners[:, :2] / corners[:, 2:3]
    quad: Quad = (
        (float(corners[0, 0]), float(corners[0, 1])),
        (float(corners[1, 0]), float(corners[1, 1])),
        (float(corners[2, 0]), float(corners[2, 1])),
        (float(corners[3, 0]), float(corners[3, 1])),
    )
    if edge_on_plane(geometry.right, geometry.down, geometry.center):
        return TextProjection(corners=quad, inverse=None)
    if float(np.linalg.cond(homography)) > 1e12:
        raise ValueError("3D text projection is degenerate")
    inverse = np.linalg.inv(homography)
    if abs(float(inverse[2, 2])) < 1e-12:
        raise ValueError("3D text projection crosses the camera horizon")
    inverse /= inverse[2, 2]
    return TextProjection(
        corners=quad,
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
    )


def text_tile(
    bitmap: Image.Image, projection: TextProjection, output: Size
) -> tuple[Image.Image, tuple[int, int]] | None:
    """Warp only a covered output ROI for one owned texture or glyph."""
    if projection.inverse is None:
        return None
    bounds = projection_bounds(projection.corners, output)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    inverse = local_perspective(projection.inverse, left, top)
    with bitmap.convert("RGBA") as rgba:
        warped = rgba.transform(
            (right - left, bottom - top),
            Image.Transform.PERSPECTIVE,
            inverse,
            resample=Image.Resampling.BICUBIC,
        )
    return warped, (left, top)


def project_text(
    bitmap: Image.Image,
    orientation: TextOrientation,
    placement: TextPlacement,
) -> Image.Image:
    """Rotate a bitmap in 3D, retaining corners outside its original texture box."""
    if not (
        orientation.rotation_x
        or orientation.rotation_y
        or orientation.rotation_z
    ):
        with bitmap.convert("RGBA") as rgba:
            return rgba.transform(
                placement.output.as_tuple,
                Image.Transform.AFFINE,
                placement.inverse,
                resample=Image.Resampling.BICUBIC,
            )
    right, down, _ = rotation_axes(orientation)
    reference = Size(width=bitmap.width, height=bitmap.height)
    geometry = TextPlane(
        center=(0, 0, orientation.perspective),
        right=right,
        down=down,
        width=placement.authored.width,
        height=placement.authored.height,
        perspective=orientation.perspective,
    )
    projection = text_plane_projection(
        geometry, placement, reference, reference
    )
    result = Image.new("RGBA", placement.output.as_tuple)
    try:
        tile = text_tile(bitmap, projection, placement.output)
        if tile is not None:
            pixels, position = tile
            with pixels:
                result.paste(pixels, position)
    except BaseException:
        result.close()
        raise
    else:
        return result
