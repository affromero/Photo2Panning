"""Render portable, wrapped title cards with Pillow."""

from PIL import Image, ImageColor, ImageDraw, ImageFont

from pic2panning.schemas.base import Size
from pic2panning.schemas.media import Cover


def wrap_text(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width: int,
) -> str:
    """Wrap paragraphs and split words that exceed the available width."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}" if line else word
            if draw.textlength(candidate, font=font) <= width:
                line = candidate
                continue
            if line:
                lines.append(line)
            line = ""
            for character in word:
                candidate = line + character
                if line and draw.textlength(candidate, font=font) > width:
                    lines.append(line)
                    line = character
                else:
                    line = candidate
        lines.append(line)
    return "\n".join(lines)


def title_card(cover: Cover, size: Size) -> Image.Image:
    """Fit a title within a card using a supplied font or Pillow's font."""
    image = Image.new(
        "RGB", size.as_tuple, ImageColor.getrgb(cover.background)
    )
    draw = ImageDraw.Draw(image)
    margin = max(1, min(size.as_tuple) // 20)
    available_width = max(1, size.width - 2 * margin)
    available_height = max(1, size.height - 2 * margin)
    maximum = max(1, min(size.height, size.width // 2))
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    wrapped = cover.text
    bounds = (0.0, 0.0, 0.0, 0.0)
    for font_size in range(maximum, 0, -1):
        if cover.font:
            font = ImageFont.truetype(cover.font, size=font_size)
        else:
            font = ImageFont.load_default(size=font_size)
        wrapped = wrap_text(cover.text, draw, font, available_width)
        bounds = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, align=cover.align.value, spacing=2
        )
        if bounds[3] - bounds[1] <= available_height:
            break
    text_width = bounds[2] - bounds[0]
    if cover.align.value == "left":
        left = float(margin)
    elif cover.align.value == "right":
        left = size.width - margin - text_width
    else:
        left = (size.width - text_width) / 2
    top = (size.height - bounds[3] + bounds[1]) / 2 - bounds[1]
    draw.multiline_text(
        (left - bounds[0], top),
        wrapped,
        font=font,
        fill=ImageColor.getrgb(cover.color),
        align=cover.align.value,
        spacing=2,
    )
    return image
