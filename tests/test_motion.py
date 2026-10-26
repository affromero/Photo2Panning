"""Temporal exposure preserves photo geometry and stabilizes fine details."""

from contextlib import ExitStack

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pic2panning.schemas.base import Size
from pic2panning.schemas.media import ImageClip, Motion
from pic2panning.tools.frames import image_frames


def test_shutter_reduces_temporal_variation_of_fine_stripes() -> None:
    """Integrating a moving fine pattern reduces its frame-to-frame strobe."""
    with Image.new("RGB", (96, 16)) as source:
        draw = ImageDraw.Draw(source)
        for column in range(0, source.width, 2):
            draw.line((column, 0, column, source.height - 1), fill="white")
        variations = []
        for angle in (0, 180):
            values = []
            clip = ImageClip(
                source="stripes.png", duration=1, shutter_angle=angle
            )
            for frame in image_frames(
                source, clip, Size(width=16, height=16), 13
            ):
                with frame:
                    values.append(frame.getpixel((8, 8)))
            variations.append(float(np.std(np.asarray(values[1:-1]))))
    assert variations[0] > 20
    assert variations[1] < variations[0] * 0.4


@pytest.mark.parametrize("motion", [Motion.PAN_LR, Motion.PAN_UD])
def test_shutter_spreads_bright_detail_along_motion_and_preserves_energy(
    motion: Motion,
) -> None:
    """Bright lines form directional trails without gamma-average darkening."""
    horizontal = motion == Motion.PAN_LR
    dimensions = (112, 32) if horizontal else (32, 112)
    bounds = (54, 0, 57, 31) if horizontal else (0, 54, 31, 57)
    clip = ImageClip(
        source="lights.png", duration=1, motion=motion, shutter_angle=360
    )
    with Image.new("RGB", dimensions) as source, ExitStack() as frames:
        ImageDraw.Draw(source).rectangle(bounds, fill="white")
        rendered = [
            frames.enter_context(frame)
            for frame in image_frames(
                source, clip, Size(width=32, height=32), 11
            )
        ]
        pixels = np.asarray(rendered[5])
        profile = pixels[16, :, 0] if horizontal else pixels[:, 16, 0]
        perpendicular = pixels[:, 16, 0] if horizontal else pixels[16, :, 0]
        energy = sum(
            value / (255 * 12.92)
            if value <= 10
            else ((value / 255 + 0.055) / 1.055) ** 2.4
            for value in (int(level) for level in profile)
        )
        assert np.count_nonzero(profile > 20) > 4
        assert np.ptp(perpendicular) == 0
        assert 3.0 < energy < 4.5


@pytest.mark.parametrize(
    "color", [(0, 0, 0), (1, 13, 27), (64, 127, 220), (255, 255, 255)]
)
def test_shutter_keeps_uniform_colors_exact(
    color: tuple[int, int, int],
) -> None:
    """Linear-light integration does not shift constant dark or bright colors."""
    clip = ImageClip(source="color.png", duration=1, shutter_angle=180)
    with Image.new("RGB", (48, 16), color) as source:
        for frame in image_frames(source, clip, Size(width=16, height=16), 7):
            with frame:
                assert frame.mode == "RGB"
                assert frame.getextrema() == tuple(
                    (value, value) for value in color
                )
        assert source.getpixel((0, 0)) == color


@pytest.mark.parametrize("motion", list(Motion))
@pytest.mark.parametrize("focus_center", [False, True])
def test_shutter_keeps_exact_endpoints_and_a_palindromic_return(
    motion: Motion, *, focus_center: bool
) -> None:
    """Exposure changes interiors while endpoint joins and reverse stay exact."""
    clip = ImageClip(
        source="detail.png",
        duration=1,
        motion=motion,
        reverse=True,
        focus_center=focus_center,
        shutter_angle=180,
    )
    size = Size(width=24, height=24)
    with Image.new("RGB", (96, 64)) as source, ExitStack() as frames:
        source.putdata(
            [
                (column * 2, row * 3, (column + row) % 256)
                for row in range(source.height)
                for column in range(source.width)
            ]
        )
        exposed = [
            frames.enter_context(frame).tobytes()
            for frame in image_frames(source, clip, size, 5)
        ]
        sharp = [
            frames.enter_context(frame).tobytes()
            for frame in image_frames(
                source, clip.model_copy(update={"shutter_angle": 0}), size, 5
            )
        ]
        assert len(exposed) == 9
        assert exposed == exposed[::-1]
        assert exposed[0] == sharp[0]
        assert exposed[4] == sharp[4]
        assert source.getpixel((95, 63)) == (190, 189, 158)


def test_zero_shutter_keeps_each_instantaneous_source_crop() -> None:
    """Disabling exposure integration preserves the original sharp sampling."""
    clip = ImageClip(source="gradient.png", duration=1, shutter_angle=0)
    size = Size(width=32, height=32)
    with Image.new("RGB", (112, 32)) as source:
        source.putdata(
            [
                (column * 2, column, 255 - column * 2)
                for column in range(source.width)
            ]
            * source.height
        )
        for index, frame in enumerate(image_frames(source, clip, size, 11)):
            with (
                frame,
                source.crop((index * 8, 0, index * 8 + 32, 32)) as crop,
            ):
                assert frame.tobytes() == crop.tobytes()


def test_focus_center_shortens_the_exposure_trail_where_motion_slows() -> None:
    """Each shutter sample follows the eased path around the frame time."""
    trail_widths = []
    with Image.new("RGB", (112, 32)) as source:
        ImageDraw.Draw(source).rectangle((54, 0, 57, 31), fill="white")
        for focus in (False, True):
            clip = ImageClip(
                source="lights.png",
                duration=1,
                shutter_angle=360,
                focus_center=focus,
            )
            for index, frame in enumerate(
                image_frames(source, clip, Size(width=32, height=32), 11)
            ):
                with frame:
                    if index == 5:
                        profile = np.asarray(frame)[16, :, 0]
                        trail_widths.append(
                            int(np.count_nonzero(profile > 20))
                        )
    assert trail_widths[1] < trail_widths[0]


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("focus_center", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_short_clips_keep_their_frame_count_and_exact_endpoints(
    count: int, *, focus_center: bool, reverse: bool
) -> None:
    """One or two forward frames remain safe at maximum shutter and focus."""
    clip = ImageClip(
        source="short.png",
        duration=count / 2,
        shutter_angle=360,
        focus_center=focus_center,
        reverse=reverse,
    )
    with Image.new("RGB", (48, 16), "red") as source:
        ImageDraw.Draw(source).rectangle((32, 0, 47, 15), fill="blue")
        colors = []
        for frame in image_frames(source, clip, Size(width=16, height=16), 2):
            with frame:
                colors.append(frame.getpixel((8, 8)))
    expected = [(255, 0, 0)] if count == 1 else [(255, 0, 0), (0, 0, 255)]
    if reverse and count == 2:
        expected.append((255, 0, 0))
    assert colors == expected
