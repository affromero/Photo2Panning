"""Calibrated table fitting, perspective geometry, and source-depth occlusion."""

import math
from typing import cast

import numpy as np
import pytest
from PIL import Image, ImageChops
from pydantic import ValidationError

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import LayerKeyframe
from pic2panning.schemas.creative.surface import SurfaceAttachment
from pic2panning.tools.creative.depth import DepthAsset
from pic2panning.tools.typography.projection import (
    render_surface,
    surface_projection,
)
from pic2panning.tools.typography.surface import (
    dot,
    fit_surface,
    intersect_plane,
)


def attachment() -> SurfaceAttachment:
    """Select a source patch with known metric dimensions."""
    return SurfaceAttachment(layer_id="table", region=(40, 40, 120, 80))


def test_surface_fit_recovers_metric_extent_and_source_anchor(
    table: DepthAsset,
) -> None:
    """A calibrated plane retains its orientation, dimensions, and anchor ray."""
    plane = fit_surface(table, attachment())
    assert plane.center == pytest.approx((0, 0, 5), abs=1e-7)
    assert plane.normal == pytest.approx((0, 0, -1), abs=1e-7)
    assert plane.right == pytest.approx((1, 0, 0), abs=1e-7)
    assert plane.down == pytest.approx((0, 1, 0), abs=1e-7)
    assert plane.width_metres == pytest.approx(2.5)
    assert plane.height_metres == pytest.approx(1.25)
    assert plane.relative_error < 1e-7
    assert plane.inlier_fraction == 1
    assert plane.samples >= 16
    assert intersect_plane(table, plane, (96, 68)) == pytest.approx(
        (0.5, 0.25, 5)
    )


def test_projection_preserves_text_aspect_instead_of_stretching_to_region(
    table: DepthAsset,
) -> None:
    """A four-to-one texture stays four-to-one on a two-to-one fitted patch."""
    options = attachment()
    plane = fit_surface(table, options)
    projected = surface_projection(
        table,
        options,
        plane,
        LayerKeyframe(x=80, y=60),
        Size(width=80, height=20),
    )
    for corner, expected in zip(
        projected.corners,
        ((40, 50), (120, 50), (120, 70), (40, 70)),
        strict=True,
    ):
        assert corner == pytest.approx(expected)
    with (
        Image.new("RGBA", (80, 20), "red") as text,
        render_surface(
            text, table, options, plane, LayerKeyframe(x=80, y=60)
        ) as image,
    ):
        assert image.getpixel((80, 60)) == (255, 0, 0, 255)
        assert image.getpixel((80, 45)) == (0, 0, 0, 0)
        assert image.getpixel((20, 60)) == (0, 0, 0, 0)
        assert image.size == (160, 120)


def test_surface_anchor_uses_fitted_plane_beneath_a_foreground_object(
    table: DepthAsset,
) -> None:
    """A closer cup hides the text without pulling its anchor off the table."""
    options = attachment()
    plane = fit_surface(table, options)
    table.depth.paste(3, (76, 50, 85, 70))
    assert intersect_plane(table, plane, (80, 60)) == pytest.approx((0, 0, 5))
    with (
        Image.new("RGBA", (80, 20), "white") as text,
        render_surface(
            text, table, options, plane, LayerKeyframe(x=80, y=60)
        ) as result,
    ):
        assert cast("tuple[int, ...]", result.getpixel((80, 60)))[3] == 0
        assert result.getpixel((60, 60)) == (255, 255, 255, 255)


def test_tilted_surface_uses_per_pixel_depth_for_occlusion(
    table: DepthAsset,
) -> None:
    """One sloped text plane is visible in front and hidden behind the table."""
    options = attachment().model_copy(
        update={"tilt": 30, "depth_tolerance_metres": 0, "feather_metres": 0}
    )
    plane = fit_surface(table, options)
    with (
        Image.new("RGBA", (80, 40), "white") as text,
        render_surface(
            text, table, options, plane, LayerKeyframe(x=80, y=60)
        ) as result,
    ):
        # Tilting about camera X places the lower half behind the flat table.
        assert cast("tuple[int, ...]", result.getpixel((80, 52)))[3] == 255
        assert cast("tuple[int, ...]", result.getpixel((80, 68)))[3] == 0


def test_fit_handles_a_sloping_table_and_rejects_nearby_object_outliers(
    table: DepthAsset,
) -> None:
    """Robust unprojection recovers a dominant real plane rather than a depth slope."""
    rows, columns = np.indices((120, 160), dtype=np.float64)
    rays_x, rays_y = (columns + 0.5 - 80) / 160, (rows + 0.5 - 60) / 160
    normal = (0.2, -0.6, -0.8)
    length = math.sqrt(sum(value * value for value in normal))
    normal = normal[0] / length, normal[1] / length, normal[2] / length
    depths = -4 / (normal[0] * rays_x + normal[1] * rays_y + normal[2])
    with Image.fromarray(depths.astype(np.float32)) as sloping:
        table.depth.paste(sloping)
    table.depth.paste(2, (75, 52, 84, 68))
    plane = fit_surface(table, attachment())
    assert abs(dot(plane.normal, (normal[0], normal[1], normal[2]))) > 0.9999
    assert plane.relative_error < 1e-6
    assert plane.inlier_fraction > 0.9


def test_clockwise_roll_and_local_offsets_change_projected_geometry(
    table: DepthAsset,
) -> None:
    """Roll rotates in the source image and metric offsets retain calibration."""
    options = attachment().model_copy(
        update={"offset_metres": (0.5, 0.25), "roll": 90}
    )
    projection = surface_projection(
        table,
        options,
        fit_surface(table, options),
        LayerKeyframe(x=80, y=60),
        Size(width=80, height=20),
    )
    assert projection.center == pytest.approx((0.5, 0.25, 5))
    assert projection.corners[1][1] > projection.corners[0][1]
    assert projection.corners[1][0] == pytest.approx(projection.corners[0][0])


def test_depth_tolerance_and_feather_preserve_alpha_without_modifying_rgb(
    table: DepthAsset,
) -> None:
    """Declared metric feathering softens only partially hidden text coverage."""
    options = attachment().model_copy(
        update={"depth_tolerance_metres": 0.1, "feather_metres": 0.2}
    )
    plane = fit_surface(table, options)
    table.depth.paste(4.8, (50, 50, 70, 70))
    with (
        Image.new("RGBA", (80, 20), (200, 60, 20, 128)) as text,
        render_surface(
            text, table, options, plane, LayerKeyframe(x=80, y=60)
        ) as result,
    ):
        pixel = cast("tuple[int, ...]", result.getpixel((60, 60)))
        # Pillow's premultiplied perspective resampling rounds at 8 bits.
        assert all(
            abs(actual - expected) <= 1
            for actual, expected in zip(pixel[:3], (200, 60, 20), strict=True)
        )
        assert 60 <= pixel[3] <= 68
        assert cast("tuple[int, ...]", result.getpixel((100, 60)))[3] == 128


@pytest.mark.parametrize("invalid", [(0, 0, 200, 100), (0, 0, 1, 1)])
def test_surface_fit_rejects_outside_or_undersampled_regions(
    table: DepthAsset, invalid: tuple[int, int, int, int]
) -> None:
    """Invalid selections fail before any distorted text can be rendered."""
    with pytest.raises(ValueError, match="Surface region"):
        fit_surface(
            table,
            SurfaceAttachment(layer_id="table", region=invalid),
        )


def test_surface_fit_rejects_missing_foreground_and_curved_depth(
    table: DepthAsset,
) -> None:
    """Sky-only selections and strongly curved geometry do not pretend to be planes."""
    table.mask.paste(0, (0, 0, 160, 120))
    with pytest.raises(ValueError, match="insufficient foreground"):
        fit_surface(table, attachment())
    table.mask.paste(255, (0, 0, 160, 120))
    rows, columns = np.indices((120, 160), dtype=np.float64)
    depths = 5 + ((columns - 80) / 12) ** 2 + ((rows - 60) / 12) ** 2
    with Image.fromarray(depths.astype(np.float32)) as curved:
        table.depth.paste(curved)
    with pytest.raises(ValueError, match="not planar enough"):
        fit_surface(table, attachment())


def test_invalid_surface_region_and_nonfinite_angles_fail_validation() -> None:
    """Malformed geometry never reaches the depth or image processing code."""
    with pytest.raises(ValidationError):
        SurfaceAttachment(layer_id="table", region=(50, 0, 20, 30))
    with pytest.raises(ValidationError):
        SurfaceAttachment(
            layer_id="table", region=(0, 0, 10, 10), roll=float("nan")
        )


def test_surface_projection_rejects_text_crossing_behind_the_camera(
    table: DepthAsset,
) -> None:
    """An impossible plane does not become a mirrored or clipped caption."""
    options = attachment().model_copy(update={"yaw": 75})
    with pytest.raises(ValueError, match="behind the camera"):
        surface_projection(
            table,
            options,
            fit_surface(table, options),
            LayerKeyframe(x=80, y=60, scale=8),
            Size(width=80, height=20),
        )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_attached_text_flips_through_an_edge_on_plane(
    table: DepthAsset, axis: str
) -> None:
    """Attachment and keyframe angles combine through 90 into an exact half-turn."""
    options = SurfaceAttachment(
        layer_id="table",
        region=(40, 40, 120, 80),
        tilt=30 if axis == "x" else 0,
        yaw=30 if axis == "y" else 0,
    )
    plane = fit_surface(table, options)
    with Image.new("RGBA", (80, 20), "red") as bitmap:
        bitmap.paste("blue", (0, 0, 30, 8))
        for frame in range(61):
            angle = frame * 2.5
            with (
                render_surface(
                    bitmap,
                    table,
                    options,
                    plane,
                    LayerKeyframe(
                        x=80,
                        y=60,
                        rotation_x=angle if axis == "x" else 0,
                        rotation_y=angle if axis == "y" else 0,
                    ),
                ) as actual,
                actual.getchannel("A") as alpha,
            ):
                if frame == 24:
                    assert alpha.getbbox() is None
                elif frame in {0, 12, 36, 48, 60}:
                    assert alpha.getbbox() is not None
                if frame == 60:
                    flip = (
                        Image.Transpose.FLIP_TOP_BOTTOM
                        if axis == "x"
                        else Image.Transpose.FLIP_LEFT_RIGHT
                    )
                    with (
                        bitmap.transpose(flip) as mirrored,
                        render_surface(
                            mirrored,
                            table,
                            attachment(),
                            plane,
                            LayerKeyframe(x=80, y=60),
                        ) as expected,
                        ImageChops.difference(actual, expected) as difference,
                    ):
                        assert difference.getbbox(alpha_only=False) is None
