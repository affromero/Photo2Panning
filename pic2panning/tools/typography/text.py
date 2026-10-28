"""Canonical wrapped text textures and bounded grapheme layouts."""

import math
import unicodedata
from collections.abc import Iterator

from PIL import Image, ImageColor, ImageDraw, ImageFont

from pic2panning.schemas.creative.composition import TextLayer
from pic2panning.tools.io import require_file
from pic2panning.tools.text import wrap_text

type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def text_font(layer: TextLayer) -> Font:
    """Use the same declared font for whole sentences and moving glyphs."""
    return (
        ImageFont.truetype(require_file(layer.font), size=layer.font_size)
        if layer.font
        else ImageFont.load_default(size=layer.font_size)
    )


def wrapped_text(layer: TextLayer, text: str, font: Font) -> str:
    """Wrap through the existing title-text policy."""
    with Image.new("RGBA", (1, 1)) as image:
        return wrap_text(
            text,
            ImageDraw.Draw(image),
            font,
            max(1, layer.size.width - 2 * layer.stroke_width),
        )


def text_position(
    layer: TextLayer, text: str, draw: ImageDraw.ImageDraw, font: Font
) -> tuple[float, float]:
    """Retain the canonical block alignment and top-edge placement."""
    bounds = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        align=layer.align.value,
        spacing=layer.spacing,
        stroke_width=layer.stroke_width,
    )
    width = bounds[2] - bounds[0]
    x = 0.0
    if layer.align.value == "center":
        x = (layer.size.width - width) / 2
    elif layer.align.value == "right":
        x = layer.size.width - width
    return x - bounds[0], -bounds[1]


def text_image(layer: TextLayer, text: str | None = None) -> Image.Image:
    """Draw an owned, pixel-compatible static text rectangle."""
    image = Image.new("RGBA", layer.size.as_tuple)
    draw = ImageDraw.Draw(image)
    font = text_font(layer)
    wrapped = wrapped_text(layer, layer.text if text is None else text, font)
    position = text_position(layer, wrapped, draw, font)
    if layer.shadow_offset != (0, 0):
        draw.multiline_text(
            (
                position[0] + layer.shadow_offset[0],
                position[1] + layer.shadow_offset[1],
            ),
            wrapped,
            font=font,
            fill=ImageColor.getcolor(layer.shadow_color, "RGBA"),
            align=layer.align.value,
            spacing=layer.spacing,
        )
    draw.multiline_text(
        position,
        wrapped,
        font=font,
        fill=ImageColor.getcolor(layer.color, "RGBA"),
        align=layer.align.value,
        spacing=layer.spacing,
        stroke_width=layer.stroke_width,
        stroke_fill=ImageColor.getcolor(layer.stroke_color, "RGBA"),
    )
    return image


def graphemes(text: str) -> Iterator[str]:
    """Keep combining marks, emoji modifiers, ZWJ runs, and flags together."""
    current = ""
    regional = 0
    for character in text:
        point = ord(character)
        is_regional = 0x1F1E6 <= point <= 0x1F1FF
        attached = (
            character == "\u200d"
            or current.endswith("\u200d")
            or unicodedata.combining(character) != 0
            or 0xFE00 <= point <= 0xFE0F
            or 0xE0100 <= point <= 0xE01EF
            or 0x1F3FB <= point <= 0x1F3FF
            or (is_regional and regional % 2 == 1)
        )
        if current and not attached:
            yield current
            current = ""
        current += character
        regional = regional + 1 if is_regional else 0
    if current:
        yield current


class Glyph:
    """One owned, padded ink mask and its exact sentence-space position."""

    text: str
    mask: Image.Image
    image: Image.Image
    x: float
    y: float

    def __init__(
        self,
        text: str,
        mask: Image.Image,
        image: Image.Image,
        position: tuple[float, float],
    ) -> None:
        """Keep texture geometry independent from the animation timeline."""
        self.text = text
        self.mask = mask
        self.image = image
        self.x, self.y = position

    def close(self) -> None:
        """Release the glyph's only raster allocation."""
        self.mask.close()
        self.image.close()


class GlyphInk:
    """One owned unstyled ink mask at its canonical sentence-space origin."""

    mask: Image.Image
    x: float
    y: float

    def __init__(
        self, mask: Image.Image, position: tuple[float, float]
    ) -> None:
        """Transfer mask ownership to a sampled geometric glyph."""
        self.mask = mask
        self.x, self.y = position

    def close(self) -> None:
        """Release the only raster allocation owned by this glyph sample."""
        self.mask.close()


class TextLayout:
    """Own one whole-sentence texture and the matching visible glyph masks."""

    image: Image.Image
    glyphs: tuple[Glyph, ...]

    def __init__(self, layer: TextLayer, text: str, padding: int) -> None:
        """Prepare a single sentence once, without caching rendered frames."""
        self.image = text_image(layer, text)
        self.glyphs = ()
        prepared: list[Glyph] = []
        try:
            prepared.extend(layout_glyphs(layer, text, padding))
            self.glyphs = tuple(prepared)
        except BaseException:
            for glyph in prepared:
                glyph.close()
            self.close()
            raise

    def close(self) -> None:
        """Release this sentence and every padded ink tile."""
        self.image.close()
        for glyph in self.glyphs:
            glyph.close()


def layout_glyphs(
    layer: TextLayer, text: str, padding: int
) -> Iterator[Glyph]:
    """Locate grapheme tiles with canonical wrapping, alignment, and kerning."""
    font = text_font(layer)
    wrapped = wrapped_text(layer, text, font)
    with Image.new("L", (1, 1)) as measure:
        draw = ImageDraw.Draw(measure)
        origin = text_position(layer, wrapped, draw, font)
        line_spacing = (
            draw.textbbox(
                (0, 0), "A", font=font, stroke_width=layer.stroke_width
            )[3]
            + layer.stroke_width
            + layer.spacing
        )
        lines = wrapped.split("\n")
        widths = [draw.textlength(line, font=font) for line in lines]
        maximum = max(widths, default=0.0)
        for row, (line, width) in enumerate(zip(lines, widths, strict=True)):
            difference = maximum - width
            offset = (
                difference / 2
                if layer.align.value == "center"
                else difference
                if layer.align.value == "right"
                else 0
            )
            previous = ""
            for cluster in graphemes(line):
                advance = draw.textlength(
                    previous + cluster, font=font
                ) - draw.textlength(cluster, font=font)
                previous += cluster
                if cluster.isspace():
                    continue
                glyph = glyph_mask(cluster, font, padding)
                try:
                    sprite = glyph_image(cluster, font, layer, padding)
                except BaseException:
                    glyph.close()
                    raise
                left, top = draw.textbbox((0, 0), cluster, font=font)[:2]
                yield Glyph(
                    cluster,
                    glyph,
                    sprite,
                    (
                        origin[0]
                        + offset
                        + advance
                        + math.floor(left)
                        - padding,
                        origin[1]
                        + row * line_spacing
                        + math.floor(top)
                        - padding,
                    ),
                )


def glyph_mask(text: str, font: Font, padding: int) -> Image.Image:
    """Draw only the ink coverage, leaving room for deformation and effects."""
    with Image.new("L", (1, 1)) as measure:
        left, top, right, bottom = ImageDraw.Draw(measure).textbbox(
            (0, 0), text, font=font
        )
    left, top = math.floor(left), math.floor(top)
    width = max(1, math.ceil(right) - left + 2 * padding)
    height = max(1, math.ceil(bottom) - top + 2 * padding)
    image = Image.new("L", (width, height))
    ImageDraw.Draw(image).text(
        (padding - left, padding - top), text, font=font, fill=255
    )
    return image


def glyph_image(
    text: str, font: Font, layer: TextLayer, padding: int
) -> Image.Image:
    """Draw an exact owned glyph tile using the canonical font and style."""
    with Image.new("L", (1, 1)) as measure:
        left, top, right, bottom = ImageDraw.Draw(measure).textbbox(
            (0, 0), text, font=font
        )
    left, top = math.floor(left), math.floor(top)
    width = max(1, math.ceil(right) - left + 2 * padding)
    height = max(1, math.ceil(bottom) - top + 2 * padding)
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    position = padding - left, padding - top
    if layer.shadow_offset != (0, 0):
        draw.text(
            (
                position[0] + layer.shadow_offset[0],
                position[1] + layer.shadow_offset[1],
            ),
            text,
            font=font,
            fill=ImageColor.getcolor(layer.shadow_color, "RGBA"),
        )
    draw.text(
        position,
        text,
        font=font,
        fill=ImageColor.getcolor(layer.color, "RGBA"),
        stroke_width=layer.stroke_width,
        stroke_fill=ImageColor.getcolor(layer.stroke_color, "RGBA"),
    )
    return image
