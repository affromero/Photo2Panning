"""Closed glyph solids retain subpixel boundaries, holes, and thin strokes."""

import math
from typing import cast

import numpy as np
import pytest
from PIL import Image

from pic2panning.tools.typography.contours import (
    Contour,
    Contours,
    Point2,
    contour_area,
    mask_contours,
)


def contains(point: Point2, contour: Contour) -> bool:
    """Test the visible footprint of a returned geometric loop."""
    x, y = point
    inside = False
    for first, second in zip(contour, contour[1:] + contour[:1], strict=True):
        if (first[1] > y) != (second[1] > y):
            crossing = first[0] + (y - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if x < crossing:
                inside = not inside
    return inside


def ink_at(point: Point2, contours: Contours) -> bool:
    """Apply even-odd filling to exteriors, holes, and nested islands."""
    return sum(contains(point, loop) for loop in contours) % 2 == 1


def assert_pixel_footprint(mask: Image.Image, contours: Contours) -> None:
    """Reproject solid footprints onto every original pixel center."""
    for row in range(mask.height):
        for column in range(mask.width):
            expected = cast("int", mask.getpixel((column, row))) >= 128
            assert ink_at((column + 0.5, row + 0.5), contours) == expected


def assert_simple(contour: Contour) -> None:
    """No non-adjacent boundary edges may cross in a filled glyph body."""
    assert len(contour) >= 3
    assert contour[0] != contour[-1]
    assert len(set(contour)) == len(contour)
    edges = tuple(zip(contour, contour[1:] + contour[:1], strict=True))
    for index, (a, b) in enumerate(edges):
        for other, (c, d) in enumerate(edges[index + 2 :], index + 2):
            if index == 0 and other == len(edges) - 1:
                continue
            first = (
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            ) * ((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]))
            second = (
                (d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])
            ) * ((d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0]))
            assert not (first < 0 and second < 0)


def test_rectangle_retains_source_pixel_edges_and_closed_winding() -> None:
    """A geometric solid occupies the same rectangle as its source ink."""
    with Image.new("L", (10, 12)) as mask:
        mask.paste(255, (2, 3, 7, 8))
        contours = mask_contours(mask)
        assert len(contours) == 1
        loop = contours[0]
        assert min(x for x, _ in loop) == 2
        assert max(x for x, _ in loop) == 7
        assert min(y for _, y in loop) == 3
        assert max(y for _, y in loop) == 8
        assert contour_area(loop) > 0
        assert_simple(loop)
        assert_pixel_footprint(mask, contours)


def test_hole_is_retained_with_opposite_winding() -> None:
    """A hollow letter remains hollow after conversion to filled polygons."""
    with Image.new("L", (12, 12)) as mask:
        mask.paste(255, (1, 1, 11, 11))
        mask.paste(0, (4, 4, 8, 8))
        contours = mask_contours(mask)
        assert len(contours) == 2
        assert contour_area(contours[0]) > 0
        assert contour_area(contours[1]) < 0
        assert ink_at((2, 6), contours)
        assert not ink_at((6, 6), contours)
        assert_pixel_footprint(mask, contours)


def test_nested_island_remains_visible_inside_a_hole() -> None:
    """Nested components alternate fill without losing the innermost island."""
    with Image.new("L", (16, 16)) as mask:
        mask.paste(255, (1, 1, 15, 15))
        mask.paste(0, (4, 4, 12, 12))
        mask.paste(255, (6, 6, 10, 10))
        contours = mask_contours(mask)
        assert len(contours) == 3
        assert [math.copysign(1, contour_area(loop)) for loop in contours] == [
            1,
            -1,
            1,
        ]
        assert_pixel_footprint(mask, contours)


def test_border_ink_is_closed_without_clipping_the_original_image() -> None:
    """Zero padding retains edge-touching text as a complete solid."""
    with Image.new("L", (7, 5), 255) as mask:
        contours = mask_contours(mask)
        assert len(contours) == 1
        assert min(x for x, _ in contours[0]) == 0
        assert max(x for x, _ in contours[0]) == 7
        assert min(y for _, y in contours[0]) == 0
        assert max(y for _, y in contours[0]) == 5
        assert_pixel_footprint(mask, contours)


def test_antialiased_ink_boundary_retains_subpixel_position() -> None:
    """A fractional crossing is interpolated instead of rounded to pixels."""
    with Image.new("L", (4, 3)) as mask:
        mask.putdata([0, 64, 192, 255] * 3)
        contours = mask_contours(mask)
        left = min(x for x, _ in contours[0])
        assert left == pytest.approx(1.5 + (127.5 - 64) / 128)
        assert left != round(left)
        assert_pixel_footprint(mask, contours)


def test_single_pixel_strokes_survive_without_resampling() -> None:
    """A thin letter stem keeps its full extent and every covered center."""
    with Image.new("L", (9, 13)) as mask:
        mask.paste(255, (4, 1, 5, 12))
        contours = mask_contours(mask)
        assert len(contours) == 1
        assert (
            max(x for x, _ in contours[0]) - min(x for x, _ in contours[0])
            == 1
        )
        assert_pixel_footprint(mask, contours)


def test_faint_ink_keeps_nonzero_geometry_until_the_threshold() -> None:
    """A vanishing component stays a small closed loop before it disappears."""
    with Image.new("L", (3, 3)) as mask:
        mask.putpixel((1, 1), 128)
        contours = mask_contours(mask)
        assert len(contours) == 1
        assert 0 < contour_area(contours[0]) < 0.001
        assert_simple(contours[0])
        assert ink_at((1.5, 1.5), contours)
        mask.putpixel((1, 1), 127)
        assert mask_contours(mask) == ()


@pytest.mark.parametrize("case", range(16))
def test_all_cell_patterns_preserve_ink_without_crossed_boundaries(
    case: int,
) -> None:
    """Every marching-square pattern closes and reproduces its pixel centers."""
    with Image.new("L", (2, 2)) as mask:
        mask.putdata([255 if case & bit else 0 for bit in (1, 2, 8, 4)])
        contours = mask_contours(mask)
        for loop in contours:
            assert_simple(loop)
            assert contour_area(loop) > 0
        assert_pixel_footprint(mask, contours)


@pytest.mark.parametrize("values", [(255, 0, 0, 255), (0, 255, 255, 0)])
def test_exact_saddle_keeps_diagonal_ink_as_separate_solids(
    values: tuple[int, int, int, int],
) -> None:
    """An exact saddle tie does not create a crossing or a false ink bridge."""
    with Image.new("L", (2, 2)) as mask:
        mask.putdata(values)
        contours = mask_contours(mask)
        assert len(contours) == 2
        assert not ink_at((1, 1), contours)
        assert_pixel_footprint(mask, contours)


def test_asymptotic_decider_keeps_a_real_saddle_bridge() -> None:
    """Stronger diagonal ink connects when the bilinear field has a positive bridge."""
    with Image.new("L", (2, 2)) as mask:
        mask.putdata([255, 1, 1, 255])
        contours = mask_contours(mask)
        assert len(contours) == 1
        assert ink_at((1, 1), contours)
        assert_simple(contours[0])
        assert_pixel_footprint(mask, contours)


def test_invalid_modes_and_oversized_masks_fail_explicitly() -> None:
    """Inputs cannot silently change color interpretation or contour resolution."""
    with (
        Image.new("RGB", (2, 2)) as color,
        Image.new("L", (4097, 1)) as too_wide,
        Image.new("L", (2200, 2000)) as too_large,
    ):
        for mask, reason in (
            (color, "L-mode"),
            (too_wide, "dimension"),
            (too_large, "dimension"),
        ):
            with pytest.raises(ValueError, match=reason):
                mask_contours(mask)


def test_excessive_islands_fail_instead_of_dropping_components() -> None:
    """A complex mask fails without quietly discarding visible islands."""
    with Image.new("L", (54, 54)) as mask:
        for row in range(0, 54, 3):
            for column in range(0, 54, 3):
                mask.putpixel((column, row), 255)
        with pytest.raises(ValueError, match="contour count"):
            mask_contours(mask)


def test_excessive_segments_fail_without_reducing_detail() -> None:
    """Pathological pixel complexity cannot trigger unbounded geometry work."""
    rows, columns = np.indices((128, 1024))
    with (
        Image.fromarray(((rows + columns) % 2 * 255).astype(np.uint8)) as mask,
        pytest.raises(ValueError, match="segment limit"),
    ):
        mask_contours(mask)
