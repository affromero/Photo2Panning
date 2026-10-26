"""Image edits preserve input files and apply operations in declared order."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, UnidentifiedImageError

from pic2panning import (
    Adjust,
    Axis,
    Crop,
    EditRequest,
    Flip,
    Resize,
    Rotate,
    Size,
    edit,
)


def test_ordered_edits_change_pixels_and_dimensions_without_changing_input(
    tmp_path: Path,
) -> None:
    """Crop, resize, rotate, and brightness affect the intended region."""
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    with Image.new("RGB", (4, 2), "red") as image:
        ImageDraw.Draw(image).rectangle((2, 0, 3, 1), fill="blue")
        image.save(source)
    original = source.read_bytes()
    result = edit(
        EditRequest(
            source=str(source),
            output=str(output),
            operations=(
                Crop(left=2, top=0, right=4, bottom=2),
                Resize(size=Size(width=4, height=2)),
                Rotate(degrees=90),
                Adjust(brightness=0.5),
            ),
        )
    )
    with Image.open(output) as image:
        assert image.size == (2, 4)
        assert image.getpixel((0, 0)) == (0, 0, 127)
    assert (result.width, result.height) == (2, 4)
    assert source.read_bytes() == original


@pytest.mark.parametrize("axis", [Axis.HORIZONTAL, Axis.VERTICAL])
def test_reflection_moves_the_selected_corner_to_the_expected_edge(
    tmp_path: Path, axis: Axis
) -> None:
    """Horizontal and vertical reflection retain orientation and colors."""
    source = tmp_path / "source.png"
    output = tmp_path / "flipped.png"
    with Image.new("RGB", (3, 3), "black") as image:
        image.putpixel((0, 0), (255, 0, 0))
        image.save(source)
    edit(
        EditRequest(
            source=str(source),
            output=str(output),
            operations=(Flip(axis=axis),),
        )
    )
    expected = (2, 0) if axis == Axis.HORIZONTAL else (0, 2)
    with Image.open(output) as image:
        assert image.getpixel(expected) == (255, 0, 0)
        assert image.getpixel((0, 0)) == (0, 0, 0)


def test_out_of_bounds_crop_does_not_publish_an_output(tmp_path: Path) -> None:
    """A crop fails instead of inserting black padding outside the photo."""
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    Image.new("RGB", (4, 4)).save(source)
    with pytest.raises(ValueError, match="bounds"):
        edit(
            EditRequest(
                source=str(source),
                output=str(output),
                operations=(Crop(left=0, top=0, right=5, bottom=4),),
            )
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source.png"]


def test_failed_edit_preserves_an_existing_output(tmp_path: Path) -> None:
    """Explicit overwrite still publishes only complete successful work."""
    source = tmp_path / "broken.png"
    output = tmp_path / "result.png"
    source.write_text("invalid image")
    output.write_bytes(b"previous result")
    with pytest.raises(UnidentifiedImageError, match="identify image"):
        edit(
            EditRequest(
                source=str(source),
                output=str(output),
                operations=(Rotate(degrees=90),),
                overwrite=True,
            )
        )
    assert output.read_bytes() == b"previous result"
