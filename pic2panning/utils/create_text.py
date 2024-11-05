"""Create a black image with wrapped text, adjusting the font size and justification."""

from PIL import Image, ImageDraw, ImageFont


def create_text_image(
    text: str, justification: str, image_size: tuple[int, int], font_path: str
) -> Image.Image:
    """Create a black image with wrapped text, adjusting the font size and justification.

    Args:
        text (str): Text to display on the image.
        justification (str): Justification for the text ('left', 'center', 'right').
        image_size (Tuple[int, int]): Width and height of the image.
        font_path (str): Path to the font file.
        font_size (int): Font size of the text.

    Returns:
        Image.Image: A PIL Image object with the text rendered on it.

    """
    # Create a black image
    img = Image.new("RGB", image_size, color="black")
    draw = ImageDraw.Draw(img)

    # Load the default font
    font_size = 10  # Initial font size - will be increase to fit the text
    font = ImageFont.truetype(font_path, size=font_size)

    # Estimate maximum font size
    last_font_size = font_size
    while True:
        test_size = draw.textsize(text, font=font)
        if test_size[0] > image_size[0] or test_size[1] > image_size[1]:
            break
        font_size = max(10, font_size + 1)  # Increase font size until it fits
        if font_size == last_font_size:
            break
        last_font_size = font_size
        font = ImageFont.truetype(font_path, size=font_size)

    font_size = int(
        font_size * 0.8
    )  # Slightly reduce font size for better fit
    font = ImageFont.truetype(font_path, size=font_size)

    # Text Wrapping
    lines = []
    for line in text.split("\n"):
        words = line.split()
        current_line: list[str] = []
        for word in words:
            test_line = " ".join(current_line + [word])
            width, _ = draw.textsize(test_line, font=font)
            if width <= image_size[0]:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))

    # Determine vertical position for central alignment
    total_height = len(lines) * font.getsize(lines[0])[1]
    current_y = (image_size[1] - total_height) // 2

    # Draw text onto the image
    for line in lines:
        width, height = draw.textsize(line, font=font)
        if justification == "center":
            x = (image_size[0] - width) // 2
        elif justification == "right":
            x = image_size[0] - width
        else:
            x = 0  # 'left' justification is default

        draw.text((x, current_y), line, font=font, fill="white")
        current_y += height

    return img
