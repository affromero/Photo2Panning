"""Build real letter volumes from canonical ink and calibrated source depth."""

from typing import cast

import numpy as np
from PIL import Image, ImageColor

from pic2panning.backends.solid.contracts import (
    SceneLight,
    SolidFrame,
    SolidGlyph,
    SolidScene,
    SurfaceQuad,
    Triangle,
)
from pic2panning.backends.solid.session import SolidSession
from pic2panning.schemas.base import PositiveFloat, PydanticBaseModel, Size
from pic2panning.schemas.camera import Matrix3, Matrix4, Vector3
from pic2panning.schemas.creative.composition import LayerKeyframe, TextLayer
from pic2panning.schemas.creative.surface import SolidLight, SurfacePlane
from pic2panning.tools.creative.depth import DepthAsset
from pic2panning.tools.typography.contours import Contours, mask_contours
from pic2panning.tools.typography.drop import (
    SurfaceContext,
    fall_pose,
    glyph_progress,
)
from pic2panning.tools.typography.morph import TextAnimator
from pic2panning.tools.typography.projection import (
    local_axes,
    oriented_plane,
    surface_projection,
)
from pic2panning.tools.typography.surface import dot
from pic2panning.tools.typography.text import GlyphInk


class GlyphGeometry(PydanticBaseModel):
    """One glyph's metric center, oriented basis, and authored pixel scale."""

    center: Vector3
    basis: tuple[Vector3, Vector3, Vector3]
    pixel_metres: PositiveFloat


class GlyphBody(PydanticBaseModel):
    """One closed glyph outline and its independently posed physical volume."""

    contours: Contours
    geometry: GlyphGeometry


def linear_color(color: str) -> Vector3:
    """Convert the declared sRGB color into physical shader input channels."""
    values = cast(
        "tuple[int, int, int, int]", ImageColor.getcolor(color, "RGBA")
    )
    channels = tuple(
        value / 255 / 12.92
        if value / 255 <= 0.04045
        else ((value / 255 + 0.055) / 1.055) ** 2.4
        for value in values[:3]
    )
    return channels[0], channels[1], channels[2]


def scene_light(light: SolidLight, plane: SurfacePlane) -> SceneLight:
    """Keep photographed lighting fixed in the fitted patch's metric frame."""
    position = tuple(
        plane.center[index]
        + plane.right[index] * light.position[0]
        + plane.down[index] * light.position[1]
        + plane.normal[index] * light.position[2]
        for index in range(3)
    )
    return SceneLight(
        position=(position[0], position[1], position[2]),
        energy=light.energy,
        size=light.size,
        color=linear_color(light.color),
    )


def foreground_mesh(
    asset: DepthAsset, plane: SurfacePlane, size: Size
) -> tuple[tuple[Vector3, ...], tuple[Triangle, ...]]:
    """Unproject bounded source holdouts above the fitted surface's noise band."""
    depth = np.asarray(asset.depth)
    foreground = np.asarray(asset.mask)
    matrix = asset.metadata.intrinsics
    stride = max(
        1, min(depth.shape[1] // size.width, depth.shape[0] // size.height)
    )
    rows = tuple(
        sorted({*range(0, depth.shape[0], stride), depth.shape[0] - 1})
    )
    columns = tuple(
        sorted({*range(0, depth.shape[1], stride), depth.shape[1] - 1})
    )
    vertices: list[Vector3] = []
    heights: list[float] = []
    for row in rows:
        for column in columns:
            z = float(depth[row, column])
            point = (
                (column + 0.5 - matrix[0][2]) / matrix[0][0] * z,
                (row + 0.5 - matrix[1][2]) / matrix[1][1] * z,
                z,
            )
            vertices.append(point)
            separation = sum(
                (value - center) * normal
                for value, center, normal in zip(
                    point, plane.center, plane.normal, strict=True
                )
            )
            heights.append(separation if foreground[row, column] else -1)
    noise = max(0.001, plane.relative_error * plane.center[2] * 8)
    faces: list[Triangle] = []
    width = len(columns)
    for row in range(len(rows) - 1):
        for column in range(width - 1):
            top = row * width + column
            for indices in (
                (top, top + 1, top + width),
                (top + 1, top + width + 1, top + width),
            ):
                if min(heights[index] for index in indices) <= noise:
                    continue
                depths = tuple(vertices[index][2] for index in indices)
                if max(depths) - min(depths) > max(depths) * 0.05:
                    continue
                faces.append(indices)
    return tuple(vertices), tuple(faces)


def solid_scene(
    layer: TextLayer, asset: DepthAsset, plane: SurfacePlane, size: Size
) -> SolidScene:
    """Create one model-free, typed scene with source-grid camera calibration."""
    if layer.solid is None:
        raise ValueError("Solid text needs explicit rendering options")
    options = layer.solid
    document = asset.metadata
    matrix = document.intrinsics
    x_scale, y_scale = (
        size.width / document.grid_width,
        size.height / document.grid_height,
    )
    intrinsics: Matrix3 = (
        (
            matrix[0][0] * x_scale,
            matrix[0][1] * x_scale,
            matrix[0][2] * x_scale,
        ),
        (
            matrix[1][0] * y_scale,
            matrix[1][1] * y_scale,
            matrix[1][2] * y_scale,
        ),
        matrix[2],
    )
    vertices, triangles = foreground_mesh(asset, plane, size)
    lights = (options.key,) + ((options.fill,) if options.fill else ())
    return SolidScene(
        size=size,
        intrinsics=intrinsics,
        plane=plane,
        vertices=vertices,
        triangles=triangles,
        lights=tuple(scene_light(light, plane) for light in lights),
        color=linear_color(layer.color),
        font_size=layer.font_size,
        thickness=options.thickness,
        bevel=options.bevel,
        roughness=options.roughness,
        metallic=options.metallic,
        ambient=options.ambient,
        samples=options.samples,
    )


def glyph_transform(
    center: Vector3,
    basis: tuple[Vector3, Vector3, Vector3],
    pixel_metres: float,
) -> Matrix4:
    """Map Cartesian glyph pixels, including actual extrusion, into world metres."""
    right, down, normal = basis
    return (
        (
            right[0] * pixel_metres,
            -down[0] * pixel_metres,
            normal[0] * pixel_metres,
            center[0],
        ),
        (
            right[1] * pixel_metres,
            -down[1] * pixel_metres,
            normal[1] * pixel_metres,
            center[1],
        ),
        (
            right[2] * pixel_metres,
            -down[2] * pixel_metres,
            normal[2] * pixel_metres,
            center[2],
        ),
        (0, 0, 0, 1),
    )


def glyph_body(ink: GlyphInk, geometry: GlyphGeometry) -> GlyphBody:
    """Trace ink once into Cartesian contours around its canonical tile center."""
    traced = mask_contours(ink.mask)
    contours: Contours = tuple(
        tuple(
            (x - ink.mask.width / 2, ink.mask.height / 2 - y) for x, y in loop
        )
        for loop in traced
    )
    return GlyphBody(contours=contours, geometry=geometry)


def glyph_floor(body: GlyphBody, plane: SurfacePlane, depth: float) -> float:
    """Find the actual lowest body point along the fitted table's normal."""
    center, basis, pixel_metres = (
        body.geometry.center,
        body.geometry.basis,
        body.geometry.pixel_metres,
    )
    right, down, normal = basis
    normal_right = dot(plane.normal, right)
    normal_up = -dot(plane.normal, down)
    normal_out = dot(plane.normal, normal)
    center_height = dot(plane.normal, center) - dot(plane.normal, plane.center)
    return min(
        center_height
        + pixel_metres * (normal_right * x + normal_up * y + normal_out * z)
        for loop in body.contours
        for x, y in loop
        for z in (0, depth)
    )


def move_body(
    body: GlyphBody, direction: Vector3, distance: float
) -> GlyphBody:
    """Translate a body without changing its complete outline or local axes."""
    center = body.geometry.center
    return body.model_copy(
        update={
            "geometry": body.geometry.model_copy(
                update={
                    "center": (
                        center[0] + direction[0] * distance,
                        center[1] + direction[1] * distance,
                        center[2] + direction[2] * distance,
                    )
                }
            )
        }
    )


def solid_glyph(layer: TextLayer, body: GlyphBody) -> SolidGlyph:
    """Require a visible-camera complete volume before exporting its transform."""
    if layer.solid is None:
        raise ValueError("Solid text needs explicit rendering options")
    geometry = body.geometry
    location, pixel_metres = geometry.center, geometry.pixel_metres
    right, down, normal = geometry.basis
    depth = layer.font_size * layer.solid.thickness
    nearest = min(
        location[2]
        + pixel_metres * (right[2] * x - down[2] * y + normal[2] * z)
        for loop in body.contours
        for x, y in loop
        for z in (0, depth)
    )
    if nearest <= 1e-6:
        raise ValueError("Solid glyph projects behind the source camera")
    return SolidGlyph(
        contours=body.contours,
        transform=glyph_transform(location, geometry.basis, pixel_metres),
    )


def solid_frame(
    layer: TextLayer,
    context: SurfaceContext,
    animator: TextAnimator,
    seconds: float,
    pose: LayerKeyframe,
) -> SolidFrame:
    """Use the canonical ink timeline and drop physics without a flat text phase."""
    if layer.surface is None or layer.solid is None:
        raise ValueError("Solid text requires a fitted source surface")
    asset, plane = context.asset, context.plane
    projection = surface_projection(
        asset, layer.surface, plane, pose, layer.size
    )
    basis = oriented_plane(plane, layer.surface, pose)
    pixel_metres = plane.width_metres * pose.scale / layer.size.width
    inks = animator.ink_frame(seconds)
    bodies: list[GlyphBody] = []
    try:
        for ink in inks:
            horizontal = (
                ink.x + ink.mask.width / 2 - layer.size.width / 2
            ) * pixel_metres
            vertical = (
                ink.y + ink.mask.height / 2 - layer.size.height / 2
            ) * pixel_metres
            center = tuple(
                projection.center[axis]
                + basis[0][axis] * horizontal
                + basis[1][axis] * vertical
                for axis in range(3)
            )
            bodies.append(
                glyph_body(
                    ink,
                    GlyphGeometry(
                        center=(center[0], center[1], center[2]),
                        basis=basis,
                        pixel_metres=pixel_metres,
                    ),
                )
            )
    finally:
        for ink in inks:
            ink.close()
    depth = layer.font_size * layer.solid.thickness
    rest_clearance = (
        max(
            0.0,
            -min(
                (
                    glyph_floor(body, plane, depth)
                    for body in bodies
                    if body.contours
                ),
                default=0.0,
            ),
        )
        + 0.0001
    )
    glyphs: list[SolidGlyph] = []
    for index, body in enumerate(bodies):
        if not body.contours:
            continue
        posed = move_body(body, plane.normal, rest_clearance)
        if (
            layer.letter_drop is not None
            and seconds < layer.letter_drop.duration
        ):
            phase = glyph_progress(
                layer.letter_drop, seconds, index, len(bodies)
            )
            if phase <= 0:
                continue
            height, angles = fall_pose(layer.letter_drop, phase)
            posed = posed.model_copy(
                update={
                    "geometry": posed.geometry.model_copy(
                        update={"basis": local_axes(basis, angles)}
                    )
                }
            )
            posed = move_body(
                posed,
                plane.normal,
                height
                * layer.letter_drop.height
                * layer.font_size
                * pixel_metres,
            )
            extra = max(0.0, 0.0001 - glyph_floor(posed, plane, depth))
            posed = move_body(posed, plane.normal, extra)
        glyphs.append(solid_glyph(layer, posed))
    width = plane.width_metres * pose.scale * 2
    height = max(plane.height_metres * 3, layer.size.height * pixel_metres * 3)
    separation = dot(plane.normal, projection.center) - dot(
        plane.normal, plane.center
    )
    anchor = tuple(
        projection.center[axis] - plane.normal[axis] * separation
        for axis in range(3)
    )
    corners = tuple(
        tuple(
            anchor[axis]
            + plane.right[axis] * x * width
            + plane.down[axis] * y * height
            for axis in range(3)
        )
        for x, y in ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5))
    )
    catcher: SurfaceQuad = (
        (corners[0][0], corners[0][1], corners[0][2]),
        (corners[1][0], corners[1][1], corners[1][2]),
        (corners[2][0], corners[2][1], corners[2][2]),
        (corners[3][0], corners[3][1], corners[3][2]),
    )
    return SolidFrame(glyphs=tuple(glyphs), catcher=catcher)


class SolidRenderer:
    """Borrow canonical ink/depth and own a persistent explicit Metal session."""

    layer: TextLayer
    context: SurfaceContext
    animator: TextAnimator
    session: SolidSession

    def __init__(
        self,
        layer: TextLayer,
        context: SurfaceContext,
        animator: TextAnimator,
        directory: str,
        *,
        render_size: Size | None = None,
    ) -> None:
        """Build cached source geometry once at the requested host density."""
        if layer.solid is None:
            raise ValueError("Solid text needs explicit rendering options")
        self.layer, self.context, self.animator = layer, context, animator
        asset, plane = context.asset, context.plane
        size = render_size or Size(
            width=asset.metadata.width, height=asset.metadata.height
        )
        self.session = SolidSession(
            solid_scene(layer, asset, plane, size),
            directory,
            layer.solid.executable,
        )

    def frame(self, seconds: float, pose: LayerKeyframe) -> Image.Image:
        """Render complete physical solids at every hold, drop, and morph time."""
        return self.session.frame(
            solid_frame(self.layer, self.context, self.animator, seconds, pose)
        )

    def close(self) -> None:
        """Release native resources while leaving borrowed ink and depth alive."""
        self.session.close()
