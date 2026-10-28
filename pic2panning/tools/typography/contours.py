"""Bounded subpixel ink contours for filled, extruded glyph geometry."""

from typing import Final

import numpy as np
from PIL import Image

type Point2 = tuple[float, float]
type Contour = tuple[Point2, ...]
type Contours = tuple[Contour, ...]
type GridEdge = tuple[int, int, int]
type CellValues = tuple[float, float, float, float]

ISO_LEVEL: Final = 127.5
MAX_MASK_EDGE: Final = 4096
MAX_MASK_PIXELS: Final = 4_194_304
MAX_SEGMENTS: Final = 65_536
MAX_CONTOURS: Final = 256


def contour_area(contour: Contour) -> float:
    """Return signed pixel area, retaining precision for microscopic loops."""
    if len(contour) < 3:
        return 0
    origin_x, origin_y = contour[0]
    return (
        sum(
            (a[0] - origin_x) * (b[1] - origin_y)
            - (b[0] - origin_x) * (a[1] - origin_y)
            for a, b in zip(contour, contour[1:] + contour[:1], strict=True)
        )
        / 2
    )


def cell_pairs(case: int, values: CellValues) -> tuple[tuple[int, int], ...]:
    """Resolve crossings with a bilinear asymptotic decider at saddle cells."""
    crossings = tuple(
        edge
        for edge in range(4)
        if bool(case & (1 << edge)) != bool(case & (1 << ((edge + 1) % 4)))
    )
    if len(crossings) == 2:
        return ((crossings[0], crossings[1]),)
    if len(crossings) != 4:
        raise ValueError("Contour cell has invalid edge crossings")
    a, b, c, d = (value - ISO_LEVEL for value in values)
    determinant = a * c - b * d
    # Exact ties retain disconnected diagonal ink, without crossing segments.
    if determinant > 0 or (determinant == 0 and case == 10):
        return ((0, 1), (2, 3))
    return ((0, 3), (1, 2))


def edge_point(
    edge: int, x: int, y: int, values: CellValues
) -> tuple[GridEdge, Point2]:
    """Locate one iso-crossing by its shared integer grid-edge identity."""
    a, b, c, d = values
    keys: tuple[GridEdge, ...] = (
        (0, x, y),
        (1, x + 1, y),
        (0, x, y + 1),
        (1, x, y),
    )
    origins = (
        (x - 0.5, y - 0.5),
        (x + 0.5, y - 0.5),
        (x - 0.5, y + 0.5),
        (x - 0.5, y - 0.5),
    )
    endpoints = ((a, b), (b, c), (d, c), (a, d))
    before, after = endpoints[edge]
    fraction = (ISO_LEVEL - before) / (after - before)
    left, top = origins[edge]
    return keys[edge], (
        left + (fraction if edge in {0, 2} else 0),
        top + (fraction if edge in {1, 3} else 0),
    )


def oriented_segment(
    pair: tuple[int, int], x: int, y: int, values: CellValues
) -> tuple[tuple[GridEdge, Point2], tuple[GridEdge, Point2]]:
    """Orient a segment so ink lies on its left in image XY coordinates."""
    first = edge_point(pair[0], x, y, values)
    second = edge_point(pair[1], x, y, values)
    dx = second[1][0] - first[1][0]
    dy = second[1][1] - first[1][1]
    u = (first[1][0] + second[1][0]) / 2 - x + 0.5
    v = (first[1][1] + second[1][1]) / 2 - y + 0.5
    a, b, c, d = values
    gradient_x = (b - a) * (1 - v) + (c - d) * v
    gradient_y = (d - a) * (1 - u) + (c - b) * u
    facing = -dy * gradient_x + dx * gradient_y
    if facing == 0:
        raise ValueError("Contour cell has an indeterminate ink side")
    return (first, second) if facing > 0 else (second, first)


def contour_graph(
    mask: Image.Image,
) -> tuple[dict[GridEdge, GridEdge], dict[GridEdge, Point2]]:
    """Build a directed boundary graph with zero padding around border ink."""
    samples = np.pad(np.asarray(mask, dtype=np.float64), 1)
    a, b = samples[:-1, :-1], samples[:-1, 1:]
    c, d = samples[1:, 1:], samples[1:, :-1]
    cases = (
        (a > ISO_LEVEL).astype(np.uint8)
        + 2 * (b > ISO_LEVEL).astype(np.uint8)
        + 4 * (c > ISO_LEVEL).astype(np.uint8)
        + 8 * (d > ISO_LEVEL).astype(np.uint8)
    )
    active = (cases != 0) & (cases != 15)
    if int(active.sum()) > MAX_SEGMENTS:
        raise ValueError("Glyph contour exceeds the segment limit")
    rows, columns = np.nonzero(active)
    outgoing: dict[GridEdge, GridEdge] = {}
    incoming: dict[GridEdge, GridEdge] = {}
    points: dict[GridEdge, Point2] = {}
    for row, column in zip(rows, columns, strict=True):
        x, y = int(column), int(row)
        values = (
            float(a[y, x]),
            float(b[y, x]),
            float(c[y, x]),
            float(d[y, x]),
        )
        for pair in cell_pairs(int(cases[y, x]), values):
            first, second = oriented_segment(pair, x, y, values)
            start, end = first[0], second[0]
            if start in outgoing or end in incoming:
                raise ValueError("Glyph contour has a branched boundary")
            outgoing[start], incoming[end] = end, start
            points[start], points[end] = first[1], second[1]
            if len(outgoing) > MAX_SEGMENTS:
                raise ValueError("Glyph contour exceeds the segment limit")
    if outgoing.keys() != incoming.keys():
        raise ValueError("Glyph contour has an open boundary")
    return outgoing, points


def clean_contour(points: Contour) -> Contour:
    """Remove only exactly collinear points, with zero approximation error."""
    kept: list[Point2] = []
    for index, current in enumerate(points):
        previous, following = (
            points[index - 1],
            points[(index + 1) % len(points)],
        )
        before = current[0] - previous[0], current[1] - previous[1]
        after = following[0] - current[0], following[1] - current[1]
        if (
            before[0] * after[1] - before[1] * after[0] != 0
            or before[0] * after[0] + before[1] * after[1] <= 0
        ):
            kept.append(current)
    result = tuple(kept)
    if len(result) < 3 or contour_area(result) == 0:
        return ()
    start = min(range(len(result)), key=result.__getitem__)
    return result[start:] + result[:start]


def mask_contours(mask: Image.Image) -> Contours:
    """Extract closed subpixel loops, preserving holes and single-pixel ink.

    Input is an L-mode coverage mask. The 127.5 iso-level is interpolated
    between pixel centers, without resampling. Coordinates retain the source
    image grid, where pixel (0, 0) has center (0.5, 0.5). Loops are implicitly
    cyclic, with no duplicate endpoint. Exteriors run clockwise in the image
    (positive signed area); holes run counterclockwise (negative area).

    Masks accept at most 4096 pixels per edge and 4,194,304 total pixels.
    Complexity is bounded to 65,536 segments and 256 loops per glyph. Invalid
    modes, excessive complexity, and broken boundaries raise ValueError.
    """
    if mask.mode != "L":
        raise ValueError("Glyph contour requires an L-mode coverage mask")
    if (
        max(mask.size) > MAX_MASK_EDGE
        or mask.width * mask.height > MAX_MASK_PIXELS
    ):
        raise ValueError("Glyph contour mask exceeds the dimension limit")
    outgoing, points = contour_graph(mask)
    contours: list[Contour] = []
    while outgoing:
        start = min(outgoing)
        current = start
        boundary: list[Point2] = []
        while True:
            boundary.append(points[current])
            following = outgoing.pop(current, None)
            if following is None:
                raise ValueError("Glyph contour has an open boundary")
            if following == start:
                break
            current = following
        cleaned = clean_contour(tuple(boundary))
        if cleaned:
            contours.append(cleaned)
        if len(contours) > MAX_CONTOURS:
            raise ValueError("Glyph exceeds the contour count limit")
    return tuple(
        sorted(
            contours,
            key=lambda loop: (
                min(point[0] for point in loop),
                min(point[1] for point in loop),
                -abs(contour_area(loop)),
            ),
        )
    )
