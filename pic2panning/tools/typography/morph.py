"""Glyph correspondence and signed-distance sentence morphs on the CPU."""

import math
from collections import OrderedDict
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import zip_longest

import numpy as np
from PIL import Image, ImageColor, ImageFilter

from pic2panning.schemas.creative.composition import TextLayer
from pic2panning.schemas.creative.typography import TextCue
from pic2panning.tools.typography.text import Glyph, GlyphInk, TextLayout

FIELD_EDGE = 128


def smooth(progress: float) -> float:
    """Use a clamped quintic curve with zero endpoint speed and acceleration."""
    value = min(1.0, max(0.0, progress))
    return value * value * value * (value * (value * 6 - 15) + 10)


def distance_line(values: Sequence[float]) -> list[float]:
    """Compute exact squared Euclidean distance along one row in linear time."""
    count = len(values)
    vertices = [0] * count
    boundaries = [math.inf] * (count + 1)
    boundaries[0] = -math.inf
    last = 0
    for coordinate in range(1, count):
        previous = vertices[last]
        crossing = (
            values[coordinate]
            + coordinate * coordinate
            - values[previous]
            - previous * previous
        ) / (2 * (coordinate - previous))
        while crossing <= boundaries[last]:
            last -= 1
            previous = vertices[last]
            crossing = (
                values[coordinate]
                + coordinate * coordinate
                - values[previous]
                - previous * previous
            ) / (2 * (coordinate - previous))
        last += 1
        vertices[last] = coordinate
        boundaries[last] = crossing
        boundaries[last + 1] = math.inf
    result: list[float] = []
    last = 0
    for coordinate in range(count):
        while boundaries[last + 1] < coordinate:
            last += 1
        previous = vertices[last]
        result.append((coordinate - previous) ** 2 + values[previous])
    return result


def squared_distances(
    pixels: Sequence[Sequence[bool]], *, inside: bool
) -> list[list[float]]:
    """Apply separable one-dimensional transforms to a bounded binary mask."""
    height, width = len(pixels), len(pixels[0])
    far = float(width * width + height * height + 1)
    rows = [
        distance_line([0.0 if value is inside else far for value in row])
        for row in pixels
    ]
    result = [row.copy() for row in rows]
    for column in range(width):
        distances = distance_line([row[column] for row in rows])
        for row, value in enumerate(distances):
            result[row][column] = value
    return result


def distance_field(mask: Image.Image) -> Image.Image:
    """Prepare a bounded signed-distance field while retaining raster geometry."""
    size = mask.size
    factor = min(1.0, FIELD_EDGE / max(size))
    working = (
        max(1, round(mask.width * factor)),
        max(1, round(mask.height * factor)),
    )
    with mask.resize(working, Image.Resampling.LANCZOS) as reduced:
        pixels = (np.asarray(reduced) >= 128).tolist()
    exterior = np.asarray(squared_distances(pixels, inside=True))
    interior = np.asarray(squared_distances(pixels, inside=False))
    field = (np.sqrt(interior) - np.sqrt(exterior)) / factor
    with Image.fromarray(field.astype(np.float32)) as image:
        return image.resize(size, Image.Resampling.BILINEAR)


def morph_mask(
    fields: tuple[Image.Image, Image.Image],
    masks: tuple[Image.Image, Image.Image],
    progress: float,
) -> Image.Image:
    """Interpolate actual ink boundaries, preserving endpoint antialiasing."""
    before, after = (np.asarray(image) for image in fields)
    first, second = (
        np.asarray(image, dtype=np.float32) / 255 for image in masks
    )
    reconstructed_before = np.clip(before / 2 + 0.5, 0, 1)
    reconstructed_after = np.clip(after / 2 + 0.5, 0, 1)
    field = (1 - progress) * before + progress * after
    target_area = float((1 - progress) * first.sum() + progress * second.sum())
    if target_area <= 0:
        return Image.new("L", masks[0].size)
    threshold = float(
        np.quantile(field, 1 - min(1.0, target_area / field.size))
    )
    maximum = min(masks[0].size) * 0.15
    adjustment = min(maximum, max(-maximum, -threshold))
    adjustment *= 4 * progress * (1 - progress)
    interpolated = np.clip((field + adjustment) / 2 + 0.5, 0, 1)
    alpha = (
        interpolated
        + (1 - progress) * (first - reconstructed_before)
        + progress * (second - reconstructed_after)
    )
    return Image.fromarray(
        np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    )


def colored_mask(mask: Image.Image, color: str) -> Image.Image:
    """Convert ink coverage to one declared RGBA color without losing alpha."""
    rgba = ImageColor.getcolor(color, "RGBA")
    if not isinstance(rgba, tuple) or len(rgba) != 4:
        raise ValueError("Text colors require four RGBA channels")
    image = Image.new("RGBA", mask.size, rgba)
    with mask.point(
        [round(value * rgba[3] / 255) for value in range(256)]
    ) as alpha:
        image.putalpha(alpha)
    return image


def styled_mask(mask: Image.Image, layer: TextLayer) -> Image.Image:
    """Apply the layer's existing color, outline, and offset-shadow controls."""
    image = Image.new("RGBA", mask.size)
    if layer.shadow_offset != (0, 0):
        with (
            Image.new("L", mask.size) as moved,
        ):
            moved.paste(mask, layer.shadow_offset)
            with colored_mask(moved, layer.shadow_color) as shadow:
                image.alpha_composite(shadow)
    if layer.stroke_width:
        with (
            mask.filter(
                ImageFilter.MaxFilter(layer.stroke_width * 2 + 1)
            ) as outlined,
            colored_mask(outlined, layer.stroke_color) as stroke,
        ):
            image.alpha_composite(stroke)
    with colored_mask(mask, layer.color) as fill:
        image.alpha_composite(fill)
    return image


def paste_glyph(
    destination: Image.Image,
    sprite: Image.Image,
    position: tuple[float, float],
) -> None:
    """Place a small owned sprite with subpixel translation and bounded memory."""
    x, y = position
    left, top = math.floor(x), math.floor(y)
    with sprite.transform(
        (sprite.width + 1, sprite.height + 1),
        Image.Transform.AFFINE,
        (1, 0, left - x, 0, 1, top - y),
        Image.Resampling.BICUBIC,
    ) as shifted:
        destination.alpha_composite(shifted, (left, top))


class GlyphPair:
    """Own the two endpoint ink masks and their optional boundary fields."""

    masks: tuple[Image.Image, Image.Image]
    fields: tuple[Image.Image, Image.Image] | None
    origins: tuple[tuple[float, float], tuple[float, float]]
    matched: bool
    entering: bool
    leaving: bool
    sprite: Image.Image | None

    def __init__(self, before: Glyph | None, after: Glyph | None) -> None:
        """Prepare one correspondence, including insertion and deletion cases."""
        if before is None and after is None:
            raise ValueError(
                "A glyph correspondence requires visible endpoints"
            )
        endpoints = [glyph for glyph in (before, after) if glyph is not None]
        width = max(glyph.mask.width for glyph in endpoints)
        height = max(glyph.mask.height for glyph in endpoints)
        prepared: list[Image.Image] = []
        positions: list[tuple[float, float]] = []
        self.fields = None
        self.sprite = None
        self.matched = (
            before is not None
            and after is not None
            and before.text == after.text
        )
        self.entering, self.leaving = before is None, after is None
        try:
            for glyph in (before, after):
                mask = Image.new("L", (width, height))
                prepared.append(mask)
                if glyph is not None:
                    offset = (
                        (width - glyph.mask.width) // 2,
                        (height - glyph.mask.height) // 2,
                    )
                    mask.paste(glyph.mask, offset)
                    positions.append(
                        (glyph.x - offset[0], glyph.y - offset[1])
                    )
            self.masks = prepared[0], prepared[1]
            self.origins = (
                (positions[0], positions[-1])
                if len(positions) == 2
                else (positions[0], positions[0])
            )
            if self.matched and before is not None:
                self.sprite = before.image.copy()
            if not (self.matched or self.entering or self.leaving):
                first = distance_field(self.masks[0])
                try:
                    second = distance_field(self.masks[1])
                except BaseException:
                    first.close()
                    raise
                self.fields = first, second
        except BaseException:
            for mask in prepared:
                mask.close()
            raise

    def close(self) -> None:
        """Release fields and masks before another sentence pair is prepared."""
        for image in (
            self.masks
            + (self.fields or ())
            + ((self.sprite,) if self.sprite is not None else ())
        ):
            image.close()

    def coverage(self, progress: float, *, fade: bool = True) -> Image.Image:
        """Return owned deformed ink, without dissolving whole sentences."""
        if self.fields is not None:
            return morph_mask(self.fields, self.masks, progress)
        mask = self.masks[1 if self.entering else 0].copy()
        if not (self.entering or self.leaving):
            return mask
        visible = progress if self.entering else 1 - progress
        cut = round(mask.height * (1 - visible))
        if self.entering:
            mask.paste(0, (0, 0, mask.width, cut))
        else:
            mask.paste(0, (0, mask.height - cut, mask.width, mask.height))
        if not fade:
            return mask
        with mask.point(
            [round(value * visible) for value in range(256)]
        ) as faded:
            mask.paste(faded)
        return mask

    def position(
        self, progress: float, cue: TextCue, layer: TextLayer
    ) -> tuple[float, float]:
        """Share the same curved sentence-space path across 2D and solid type."""
        first, second = self.origins
        x = first[0] + (second[0] - first[0]) * progress
        y = first[1] + (second[1] - first[1]) * progress
        movement = min(
            1.0,
            math.hypot(second[0] - first[0], second[1] - first[1])
            / layer.font_size,
        )
        wave = math.sin(progress * math.pi)
        y += wave * cue.travel * layer.font_size * movement * 0.3
        if self.entering:
            y += (1 - progress) * cue.travel * layer.font_size
        elif self.leaving:
            y -= progress * cue.travel * layer.font_size
        return x, y

    def ink(self, progress: float, cue: TextCue, layer: TextLayer) -> GlyphInk:
        """Sample closed-solid ink without bitmap styling or opacity fades."""
        x, y = self.position(progress, cue, layer)
        wave = math.sin(progress * math.pi)
        with self.coverage(progress, fade=False) as coverage:
            if self.matched:
                return GlyphInk(coverage.copy(), (x, y))
            width = max(1, round(coverage.width * (1 + wave * 0.12)))
            height = max(1, round(coverage.height * (1 - wave * 0.08)))
            x -= (width - coverage.width) / 2
            y -= (height - coverage.height) / 2
            return GlyphInk(
                coverage.resize((width, height), Image.Resampling.LANCZOS),
                (x, y),
            )

    def draw(
        self,
        destination: Image.Image,
        progress: float,
        cue: TextCue,
        layer: TextLayer,
    ) -> None:
        """Move and reshape one correspondence along a smooth curved path."""
        x, y = self.position(progress, cue, layer)
        wave = math.sin(progress * math.pi)
        if self.sprite is not None:
            paste_glyph(destination, self.sprite, (x, y))
            return
        with self.coverage(progress) as coverage:
            if not self.matched:
                width = max(1, round(coverage.width * (1 + wave * 0.12)))
                height = max(1, round(coverage.height * (1 - wave * 0.08)))
                x -= (width - coverage.width) / 2
                y -= (height - coverage.height) / 2
                image = coverage.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            else:
                image = coverage.copy()
            with image:
                blurred = (
                    image.filter(ImageFilter.GaussianBlur(cue.blur * wave))
                    if not self.matched
                    else image.copy()
                )
                with blurred, styled_mask(blurred, layer) as sprite:
                    paste_glyph(destination, sprite, (x, y))


def correspondence(
    before: TextLayout, after: TextLayout
) -> tuple[GlyphPair, ...]:
    """Match repeated glyphs deterministically and pair each replacement block."""
    first, second = before.glyphs, after.glyphs
    pairs: list[GlyphPair] = []
    try:
        if len(first) == len(second):
            pairs.extend(
                GlyphPair(old, new)
                for old, new in zip(first, second, strict=True)
            )
            return tuple(pairs)
        matcher = SequenceMatcher(
            a=tuple(glyph.text for glyph in first),
            b=tuple(glyph.text for glyph in second),
            autojunk=False,
        )
        for opcode in matcher.get_opcodes():
            left, right, start, end = opcode[1:]
            pairs.extend(
                GlyphPair(old, new)
                for old, new in zip_longest(
                    first[left:right], second[start:end]
                )
            )
    except BaseException:
        for pair in pairs:
            pair.close()
        raise
    return tuple(pairs)


class TextAnimator:
    """Render one layer using two sentence layouts and one active morph only."""

    layer: TextLayer
    layouts: OrderedDict[str, TextLayout]
    pairs: tuple[GlyphPair, ...]
    active: tuple[str, str] | None
    padding: int

    def __init__(self, layer: TextLayer) -> None:
        """Prepare ownership without retaining rendered timeline frames."""
        self.layer = layer
        self.layouts = OrderedDict()
        self.pairs = ()
        self.active = None
        blur = max((cue.blur for cue in layer.text_cues), default=0)
        self.padding = (
            layer.stroke_width
            + max(abs(value) for value in layer.shadow_offset)
            + math.ceil(3 * blur)
            + math.ceil(layer.font_size * 0.15)
            + 3
        )

    def close(self) -> None:
        """Release all sentence and transition raster storage."""
        self.clear_pair()
        for layout in self.layouts.values():
            layout.close()
        self.layouts.clear()

    def clear_pair(self) -> None:
        """Drop the old morph before its source layouts can be evicted."""
        for pair in self.pairs:
            pair.close()
        self.pairs = ()
        self.active = None

    def layout(self, sentence: str) -> TextLayout:
        """Retain at most the two sentences that an active morph can require."""
        if sentence not in self.layouts:
            self.layouts[sentence] = TextLayout(
                self.layer, sentence, self.padding
            )
        self.layouts.move_to_end(sentence)
        while len(self.layouts) > 2:
            expired = self.layouts.popitem(last=False)[1]
            expired.close()
        return self.layouts[sentence]

    def transition(
        self, previous: str, cue: TextCue, progress: float
    ) -> Image.Image:
        """Draw staggered correspondences with a bounded preparation cache."""
        if previous == cue.text:
            return self.layout(previous).image.copy()
        self.prepare_pair(previous, cue)
        frame = Image.new("RGBA", self.layer.size.as_tuple)
        try:
            for index, pair in enumerate(self.pairs):
                phase = self.phase(progress, cue, index)
                pair.draw(frame, phase, cue, self.layer)
        except BaseException:
            frame.close()
            raise
        else:
            return frame

    def prepare_pair(self, previous: str, cue: TextCue) -> None:
        """Load one bounded correspondence shared by both rendering paths."""
        key = previous, cue.text
        if self.active != key:
            self.clear_pair()
            self.pairs = correspondence(
                self.layout(previous), self.layout(cue.text)
            )
            self.active = key

    def phase(self, progress: float, cue: TextCue, index: int) -> float:
        """Evaluate the canonical stagger independently of sampling order."""
        last = max(1, len(self.pairs) - 1)
        return smooth(
            (progress - cue.stagger * index / last) / (1 - cue.stagger)
        )

    def state(self, seconds: float) -> tuple[str, TextCue | None, float]:
        """Resolve exact holds and the active cue before sampling any glyphs."""
        if (
            not math.isfinite(seconds)
            or not 0 <= seconds <= self.layer.duration
        ):
            raise ValueError("Text time is outside the layer duration")
        previous = self.layer.text
        for cue in self.layer.text_cues:
            if seconds <= cue.at:
                return previous, None, 0
            if seconds < cue.at + cue.transition:
                return previous, cue, (seconds - cue.at) / cue.transition
            previous = cue.text
        return previous, None, 0

    def ink_frame(self, seconds: float) -> tuple[GlyphInk, ...]:
        """Return caller-owned raw geometry masks for every visible glyph."""
        previous, cue, progress = self.state(seconds)
        prepared: list[GlyphInk] = []
        try:
            if cue is None or previous == cue.text:
                prepared.extend(
                    GlyphInk(glyph.mask.copy(), (glyph.x, glyph.y))
                    for glyph in self.layout(previous).glyphs
                )
            else:
                self.prepare_pair(previous, cue)
                for index, pair in enumerate(self.pairs):
                    prepared.append(
                        pair.ink(
                            self.phase(progress, cue, index), cue, self.layer
                        )
                    )
        except BaseException:
            for sample in prepared:
                sample.close()
            raise
        else:
            return tuple(prepared)

    def frame(self, seconds: float) -> Image.Image:
        """Return an owned exact hold or a moving, reshaped sentence texture."""
        previous, cue, progress = self.state(seconds)
        if cue is None:
            return self.layout(previous).image.copy()
        return self.transition(previous, cue, progress)
