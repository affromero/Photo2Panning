"""Bounded, seekable individual glyph falls in a free or fitted 3D plane."""

import math

from PIL import Image, ImageChops

from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.camera import Vector3
from pic2panning.schemas.creative.composition import (
    LayerKeyframe,
    TextLayer,
)
from pic2panning.schemas.creative.surface import (
    LetterDrop,
    SurfaceAttachment,
    SurfacePlane,
)
from pic2panning.tools.creative.depth import DepthAsset
from pic2panning.tools.typography.projection import (
    local_axes,
    oriented_plane,
    project_plane,
    render_surface,
    surface_projection,
    surface_tile,
)
from pic2panning.tools.typography.rotation import (
    TextOrientation,
    TextPlacement,
    TextPlane,
    project_text,
    rotation_axes,
    text_plane_projection,
    text_tile,
)
from pic2panning.tools.typography.text import Glyph, TextLayout


class SurfaceContext(PydanticBaseModel):
    """Borrow a verified host asset and its once-fitted geometry."""

    asset: DepthAsset
    attachment: SurfaceAttachment
    plane: SurfacePlane


def glyph_progress(
    options: LetterDrop, seconds: float, index: int, count: int
) -> float:
    """Resolve one reading-order entrance independently of preview seek order."""
    start = (
        options.duration * options.stagger * index / (count - 1)
        if count > 1
        else 0
    )
    interval = options.duration * (1 - options.stagger)
    return min(1.0, max(0.0, (seconds - start) / interval))


def fall_pose(options: LetterDrop, progress: float) -> tuple[float, Vector3]:
    """Accelerate to impact, rebound above the plane, and settle every angle."""
    impact = 0.72
    if progress < impact:
        height = 1 - (progress / impact) ** 2
    else:
        rebound = (progress - impact) / (1 - impact)
        height = (
            options.bounce * math.sin(math.pi * rebound) ** 2 * (1 - rebound)
        )
    angular = (1 - progress) ** 2
    return height, (
        options.rotation[0] * angular,
        options.rotation[1] * angular,
        options.rotation[2] * angular,
    )


def masked_texture(
    bitmap: Image.Image, mask: Image.Image | None
) -> Image.Image:
    """Return an owned texture with the caller's declared source mask applied."""
    image = bitmap.convert("RGBA")
    if mask is None:
        return image
    try:
        with (
            image.getchannel("A") as alpha,
            ImageChops.multiply(alpha, mask) as selected,
        ):
            image.putalpha(selected)
    except BaseException:
        image.close()
        raise
    else:
        return image


class LetterDropRenderer:
    """Borrow one pinned layout and own one complete texture and prefix cache.

    The layout and optional mask must remain alive for this renderer's life.
    Returned images are caller-owned. Opacity and host transport are applied
    by the compositor, once, after these geometric operations.
    """

    layer: TextLayer
    layout: TextLayout
    options: LetterDrop
    mask: Image.Image | None
    complete: int
    settled: Image.Image
    final_texture: Image.Image

    def __init__(
        self,
        layer: TextLayer,
        layout: TextLayout,
        mask: Image.Image | None = None,
    ) -> None:
        """Prepare bounded storage without caching any rendered movie frames."""
        if layer.letter_drop is None:
            raise ValueError(
                "LetterDropRenderer requires a letter_drop option"
            )
        if mask is not None and (
            mask.mode != "L" or mask.size != layout.image.size
        ):
            raise ValueError("Letter drop mask must match its text texture")
        self.layer = layer
        self.layout = layout
        self.options = layer.letter_drop
        self.mask = mask
        self.complete = 0
        self.final_texture = masked_texture(layout.image, mask)
        try:
            self.settled = Image.new("RGBA", layout.image.size)
        except BaseException:
            self.final_texture.close()
            raise

    def close(self) -> None:
        """Release only the buffers owned here, preserving the borrowed layout."""
        self.final_texture.close()
        self.settled.close()

    def glyph_texture(self, glyph: Glyph) -> Image.Image:
        """Sample an optional whole-text mask in this padded glyph's coordinates."""
        if self.mask is None:
            return glyph.image.copy()
        with self.mask.transform(
            glyph.image.size,
            Image.Transform.AFFINE,
            (1, 0, glyph.x, 0, 1, glyph.y),
            resample=Image.Resampling.BILINEAR,
        ) as selected:
            return masked_texture(glyph.image, selected)

    def prepare(self, seconds: float) -> tuple[float, ...]:
        """Update a single settled prefix and reset it on backwards seeks."""
        glyphs = self.layout.glyphs
        phases = tuple(
            glyph_progress(self.options, seconds, index, len(glyphs))
            for index in range(len(glyphs))
        )
        complete = sum(phase >= 1 for phase in phases)
        if complete < self.complete:
            self.settled.paste((0, 0, 0, 0), (0, 0, *self.settled.size))
            self.complete = 0
        for glyph in glyphs[self.complete : complete]:
            left, top = math.floor(glyph.x), math.floor(glyph.y)
            size = (
                math.ceil(glyph.x + glyph.image.width) - left,
                math.ceil(glyph.y + glyph.image.height) - top,
            )
            with (
                self.glyph_texture(glyph) as bitmap,
                bitmap.transform(
                    size,
                    Image.Transform.AFFINE,
                    (1, 0, left - glyph.x, 0, 1, top - glyph.y),
                    resample=Image.Resampling.BICUBIC,
                ) as placed,
            ):
                self.settled.alpha_composite(placed, (left, top))
        self.complete = complete
        return phases

    def free_frame(
        self, seconds: float, pose: LayerKeyframe, placement: TextPlacement
    ) -> Image.Image:
        """Fall through a virtual 3D camera directly into the output frame."""
        orientation = TextOrientation(
            rotation_x=pose.rotation_x,
            rotation_y=pose.rotation_y,
            perspective=self.layer.perspective,
        )
        if seconds >= self.options.duration:
            return project_text(self.final_texture, orientation, placement)
        phases = self.prepare(seconds)
        result = project_text(self.settled, orientation, placement)
        basis = rotation_axes(orientation)
        reference = Size(
            width=self.layout.image.width, height=self.layout.image.height
        )
        try:
            for glyph, phase in zip(self.layout.glyphs, phases, strict=True):
                if not 0 < phase < 1:
                    continue
                with self.glyph_texture(glyph) as bitmap:
                    geometry = self.free_glyph(
                        glyph, phase, basis, placement.authored
                    )
                    projection = text_plane_projection(
                        geometry,
                        placement,
                        reference,
                        Size(width=bitmap.width, height=bitmap.height),
                    )
                    tile = text_tile(bitmap, projection, placement.output)
                    if tile is not None:
                        pixels, position = tile
                        with pixels:
                            result.alpha_composite(pixels, position)
        except BaseException:
            result.close()
            raise
        else:
            return result

    def free_glyph(
        self,
        glyph: Glyph,
        progress: float,
        basis: tuple[Vector3, Vector3, Vector3],
        authored: Size,
    ) -> TextPlane:
        """Place one falling glyph in authored-pixel virtual-camera units."""
        height, angles = fall_pose(self.options, progress)
        lift = self.options.height * self.layer.font_size * height
        horizontal = (
            (glyph.x + glyph.image.width / 2 - self.layout.image.width / 2)
            * authored.width
            / self.layout.image.width
        )
        vertical = (
            (glyph.y + glyph.image.height / 2 - self.layout.image.height / 2)
            * authored.height
            / self.layout.image.height
        )
        right, down, _ = local_axes(basis, angles)
        return TextPlane(
            center=(
                basis[0][0] * horizontal + basis[1][0] * vertical,
                basis[0][1] * horizontal + basis[1][1] * vertical - lift,
                self.layer.perspective
                + basis[0][2] * horizontal
                + basis[1][2] * vertical
                - lift * 0.18,
            ),
            right=right,
            down=down,
            width=glyph.image.width * authored.width / self.layout.image.width,
            height=glyph.image.height
            * authored.height
            / self.layout.image.height,
            perspective=self.layer.perspective,
        )

    def surface_frame(
        self, seconds: float, pose: LayerKeyframe, context: SurfaceContext
    ) -> Image.Image:
        """Fall above the fitted host, with each moving glyph's own metric depth."""
        if seconds >= self.options.duration:
            return render_surface(
                self.final_texture,
                context.asset,
                context.attachment,
                context.plane,
                pose,
            )
        phases = self.prepare(seconds)
        result = render_surface(
            self.settled,
            context.asset,
            context.attachment,
            context.plane,
            pose,
        )
        try:
            for glyph, phase in zip(self.layout.glyphs, phases, strict=True):
                if not 0 < phase < 1:
                    continue
                with self.glyph_texture(glyph) as bitmap:
                    plane = self.surface_glyph(glyph, phase, context, pose)
                    projection = project_plane(
                        context.asset,
                        plane,
                        Size(width=bitmap.width, height=bitmap.height),
                    )
                    tile = surface_tile(
                        bitmap,
                        context.asset,
                        context.attachment,
                        projection,
                    )
                    if tile is not None:
                        pixels, position = tile
                        with pixels:
                            result.alpha_composite(pixels, position)
        except BaseException:
            result.close()
            raise
        else:
            return result

    def surface_glyph(
        self,
        glyph: Glyph,
        progress: float,
        context: SurfaceContext,
        pose: LayerKeyframe,
    ) -> SurfacePlane:
        """Lift one projected glyph along the fitted table's physical normal."""
        texture = Size(
            width=self.layout.image.width, height=self.layout.image.height
        )
        base = surface_projection(
            context.asset, context.attachment, context.plane, pose, texture
        )
        basis = oriented_plane(context.plane, context.attachment, pose)
        height, angles = fall_pose(self.options, progress)
        per_pixel = (
            context.plane.width_metres * pose.scale / self.layout.image.width
        )
        horizontal = (
            glyph.x + glyph.image.width / 2 - self.layout.image.width / 2
        ) * per_pixel
        vertical = (
            glyph.y + glyph.image.height / 2 - self.layout.image.height / 2
        ) * per_pixel
        lift = self.options.height * self.layer.font_size * height * per_pixel
        right, down, normal = local_axes(basis, angles)
        return context.plane.model_copy(
            update={
                "center": (
                    base.center[0]
                    + basis[0][0] * horizontal
                    + basis[1][0] * vertical
                    + context.plane.normal[0] * lift,
                    base.center[1]
                    + basis[0][1] * horizontal
                    + basis[1][1] * vertical
                    + context.plane.normal[1] * lift,
                    base.center[2]
                    + basis[0][2] * horizontal
                    + basis[1][2] * vertical
                    + context.plane.normal[2] * lift,
                ),
                "right": right,
                "down": down,
                "normal": normal,
                "width_metres": glyph.image.width * per_pixel,
            }
        )
