"""Camera and animated layer pixel outcomes through the canonical frame API."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
    PreviewRequest,
    TextLayer,
)
from pic2panning.tools.creative.composition import preview
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)


def test_world_camera_preserves_positions_and_zoom(
    composition: ComposeRequest,
) -> None:
    """A centered crop retains source proportions; moving the camera reveals blue."""
    with (
        CompositionSession(composition) as session,
        composition_frame(composition, 0, session) as image,
    ):
        assert image.getpixel((10, 50)) == (255, 0, 0)
        assert image.getpixel((90, 50)) == (0, 0, 255)
    moved = composition.model_copy(
        update={"camera": (CameraKeyframe(x=150, y=50, zoom=2),)}
    )
    with (
        CompositionSession(moved) as session,
        composition_frame(moved, 0, session) as image,
    ):
        assert image.getpixel((10, 50)) == (0, 0, 255)
        assert image.getpixel((90, 50)) == (0, 0, 255)


def test_animated_viewport_overlay_fades_over_world(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Viewport graphics ignore camera placement and interpolate their opacity."""
    source = tmp_path / "green.png"
    with Image.new("RGB", (20, 20), "#00ff00") as image:
        image.save(source)
    overlay = ImageLayer(
        id="graphic",
        source=str(source),
        size=Size(width=20, height=20),
        duration=1,
        space="viewport",
        keyframes=(
            LayerKeyframe(x=20, y=20, opacity=0),
            LayerKeyframe(at=1, x=20, y=20, opacity=1),
        ),
    )
    request = composition.model_copy(
        update={"layers": (*composition.layers, overlay)}
    )
    with (
        CompositionSession(request) as session,
        composition_frame(request, 3, session) as image,
    ):
        assert image.getpixel((20, 20)) == (127, 128, 0)
        assert image.getpixel((80, 20)) == (0, 0, 255)


def test_preview_is_exact_canonical_frame_and_keeps_source(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A PNG preview equals the selected camera frame and cannot overwrite inputs."""
    output = tmp_path / "preview.png"
    preview(
        PreviewRequest(composition=composition, output=str(output), frame=3)
    )
    with (
        CompositionSession(composition) as session,
        composition_frame(composition, 3, session) as expected,
        Image.open(output) as actual,
    ):
        assert actual.tobytes() == expected.tobytes()
    source = composition.layers[0]
    assert isinstance(source, ImageLayer)
    original = Path(source.source).read_bytes()
    with pytest.raises(ValueError, match="aliases an input"):
        preview(
            PreviewRequest(
                composition=composition, output=source.source, overwrite=True
            )
        )
    assert Path(source.source).read_bytes() == original


def test_transparent_text_retains_scene_around_glyphs(
    composition: ComposeRequest,
) -> None:
    """In-frame text draws visible glyphs while the surrounding scene stays intact."""
    text = TextLayer(
        id="caption",
        text="Hello",
        font_size=18,
        size=Size(width=80, height=40),
        duration=1,
        space="viewport",
        keyframes=(LayerKeyframe(x=50, y=50),),
        stroke_width=1,
    )
    request = composition.model_copy(
        update={"layers": (*composition.layers, text)}
    )
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as image,
    ):
        assert image.getpixel((5, 5)) == (255, 0, 0)
        assert any(
            all(
                channel > 230
                for channel in cast("tuple[int, int, int]", pixel)
            )
            for pixel in image.get_flattened_data()
        )


def test_preview_rejects_out_of_range_frame(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A bad frame selection publishes no output."""
    output = tmp_path / "bad.png"
    with pytest.raises(ValueError, match="outside"):
        preview(
            PreviewRequest(
                composition=composition, output=str(output), frame=6
            )
        )
    assert not output.exists()


def test_preview_at_late_time_selects_final_visible_frame(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Every time within the timeline maps to a visible frame, including its tail."""
    output = tmp_path / "last.png"
    preview(
        PreviewRequest(composition=composition, output=str(output), at=0.999)
    )
    with (
        CompositionSession(composition) as session,
        composition_frame(composition, 5, session) as expected,
        Image.open(output) as actual,
    ):
        assert actual.tobytes() == expected.tobytes()


def test_shutter_softens_moving_edges_in_linear_light_and_keeps_endpoints(
    composition: ComposeRequest,
) -> None:
    """Exposure creates intermediate edge brightness while endpoints stay sharp."""
    source = composition.layers[0]
    assert isinstance(source, ImageLayer)
    with Image.new("RGB", (200, 100)) as image:
        ImageDraw.Draw(image).rectangle((100, 0, 199, 99), fill="white")
        image.save(source.source)
    sharp = composition.model_copy(
        update={
            "fps": 5,
            "camera": (
                CameraKeyframe(x=40, y=50),
                CameraKeyframe(at=1, x=160, y=50),
            ),
        }
    )
    exposed = sharp.model_copy(
        update={"shutter_angle": 360, "shutter_samples": 8}
    )
    with (
        CompositionSession(sharp) as sharp_session,
        CompositionSession(exposed) as exposed_session,
    ):
        for index in range(5):
            with (
                composition_frame(sharp, index, sharp_session) as instant,
                composition_frame(
                    exposed, index, exposed_session
                ) as integrated,
            ):
                if index in {0, 4}:
                    assert integrated.tobytes() == instant.tobytes()
                    continue
                if index == 2:
                    with (
                        instant.convert("L") as sharp_gray,
                        integrated.convert("L") as soft_gray,
                        sharp_gray.crop((0, 50, 100, 51)) as sharp_band,
                        soft_gray.crop((0, 50, 100, 51)) as soft_band,
                    ):
                        sharp_row = sharp_band.tobytes()
                        soft_row = soft_band.tobytes()
                    assert (
                        sum(0 < value < 255 for value in soft_row)
                        > sum(0 < value < 255 for value in sharp_row) + 8
                    )
                    assert (
                        175
                        < cast(
                            "tuple[int, int, int]",
                            integrated.getpixel((62, 50)),
                        )[0]
                        < 210
                    )
