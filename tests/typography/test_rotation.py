"""Free X/Y/Z perspective retains camera placement and complete glyph bounds."""

import pytest
from PIL import Image, ImageChops

from pic2panning.schemas.base import Size
from pic2panning.tools.typography.rotation import (
    TextOrientation,
    TextPlacement,
    project_text,
)


def placement() -> TextPlacement:
    """Center a 200 by 80 authored rectangle in a 400 by 240 output."""
    return TextPlacement(
        output=Size(width=400, height=240),
        authored=Size(width=200, height=80),
        inverse=(1, 0, -100, 0, 1, -80),
    )


def test_flat_rotation_matches_the_existing_affine_pixel_path() -> None:
    """Zero angles preserve the canonical transform without a perspective seam."""
    options = placement()
    with (
        Image.new("RGBA", (200, 80), (200, 30, 70, 128)) as bitmap,
        bitmap.transform(
            options.output.as_tuple,
            Image.Transform.AFFINE,
            options.inverse,
            resample=Image.Resampling.BICUBIC,
        ) as expected,
        project_text(bitmap, TextOrientation(), options) as actual,
        ImageChops.difference(actual, expected) as difference,
    ):
        assert difference.getbbox(alpha_only=False) is None


def test_y_rotation_retains_letters_beyond_the_original_texture_box() -> None:
    """The near edge expands into the output instead of clipping at texture edges."""
    with (
        Image.new("RGBA", (200, 80), "white") as bitmap,
        project_text(
            bitmap,
            TextOrientation(rotation_y=45, perspective=130),
            placement(),
        ) as actual,
        actual.getchannel("A") as alpha,
    ):
        bounds = alpha.getbbox()
        assert bounds is not None
        assert bounds[2] > 330
        assert bounds[0] > 140
        assert bounds[1] < 50
        assert actual.getpixel((340, 120)) == (255, 255, 255, 255)


def test_existing_layer_mapping_retains_translation_under_3d_rotation() -> (
    None
):
    """A camera/layer translation moves the projected ink by the declared pixels."""
    original = placement()
    shifted = original.model_copy(update={"inverse": (1, 0, -125, 0, 1, -90)})
    with (
        Image.new("RGBA", (200, 80), "white") as bitmap,
        project_text(
            bitmap, TextOrientation(rotation_x=25, rotation_y=-20), original
        ) as first,
        project_text(
            bitmap, TextOrientation(rotation_x=25, rotation_y=-20), shifted
        ) as second,
        first.getchannel("A") as first_alpha,
        second.getchannel("A") as second_alpha,
    ):
        first_bounds, second_bounds = (
            first_alpha.getbbox(),
            second_alpha.getbbox(),
        )
        assert first_bounds is not None
        assert second_bounds is not None
        assert second_bounds[0] - first_bounds[0] == 25
        assert second_bounds[1] - first_bounds[1] == 10
        assert second_bounds[2] - first_bounds[2] == 25
        assert second_bounds[3] - first_bounds[3] == 10


def test_extreme_perspective_rejects_a_plane_crossing_the_camera() -> None:
    """Invalid perspective cannot turn a behind-camera plane into mirrored text."""
    with (
        Image.new("RGBA", (200, 80), "white") as bitmap,
        pytest.raises(ValueError, match="behind the virtual camera"),
    ):
        project_text(
            bitmap,
            TextOrientation(rotation_y=80, perspective=40),
            placement(),
        )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_half_turn_renders_through_an_invisible_edge_on_frame(
    axis: str,
) -> None:
    """A two-second 30 fps flip passes 90 degrees and ends on the mirrored texture."""
    options = placement()
    with Image.new("RGBA", (200, 80), "red") as bitmap:
        bitmap.paste("blue", (0, 0, 70, 30))
        for frame in range(61):
            angle = frame * 3
            with (
                project_text(
                    bitmap,
                    TextOrientation(
                        rotation_x=angle if axis == "x" else 0,
                        rotation_y=angle if axis == "y" else 0,
                    ),
                    options,
                ) as actual,
                actual.getchannel("A") as alpha,
            ):
                if frame == 30:
                    assert alpha.getbbox() is None
                else:
                    assert alpha.getbbox() is not None
                if frame == 60:
                    flip = (
                        Image.Transpose.FLIP_TOP_BOTTOM
                        if axis == "x"
                        else Image.Transpose.FLIP_LEFT_RIGHT
                    )
                    with (
                        bitmap.transpose(flip) as mirrored,
                        mirrored.transform(
                            options.output.as_tuple,
                            Image.Transform.AFFINE,
                            options.inverse,
                            resample=Image.Resampling.BICUBIC,
                        ) as expected,
                        ImageChops.difference(actual, expected) as difference,
                    ):
                        assert difference.getbbox(alpha_only=False) is None


def test_edge_on_rotation_does_not_hide_an_invalid_placement() -> None:
    """A singular camera/layer transform still fails at a zero-area angle."""
    options = placement().model_copy(update={"inverse": (0, 0, 0, 0, 1, 0)})
    with (
        Image.new("RGBA", (200, 80), "white") as bitmap,
        pytest.raises(ValueError, match="placement is degenerate"),
    ):
        project_text(bitmap, TextOrientation(rotation_y=90), options)
