"""Seekable falling glyphs, physical depth visibility, and exact settled pixels."""

from PIL import Image, ImageChops

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import LayerKeyframe, TextLayer
from pic2panning.schemas.creative.surface import LetterDrop, SurfaceAttachment
from pic2panning.tools.creative.depth import DepthAsset
from pic2panning.tools.typography.drop import (
    LetterDropRenderer,
    SurfaceContext,
)
from pic2panning.tools.typography.projection import render_surface
from pic2panning.tools.typography.rotation import (
    TextOrientation,
    TextPlacement,
    project_text,
)
from pic2panning.tools.typography.surface import fit_surface
from pic2panning.tools.typography.text import TextLayout


def caption(text: str = "AB") -> TextLayer:
    """Two visible glyphs land at different times inside an ordinary text layer."""
    return TextLayer(
        id="caption",
        text=text,
        font_size=48,
        size=Size(width=240, height=80),
        duration=3,
        keyframes=(LayerKeyframe(x=200, y=200),),
        letter_drop=LetterDrop(height=1, rotation=(0, 0, 0)),
    )


def placement(layer: TextLayer) -> TextPlacement:
    """Place the canonical text texture in a larger output without clipping."""
    return TextPlacement(
        authored=layer.size,
        output=Size(width=400, height=320),
        inverse=(1, 0, -80, 0, 1, -160),
    )


def test_glyphs_fall_in_reading_order_and_settle_to_exact_sentence_pixels() -> (
    None
):
    """The initial frame is clear, the first letter enters alone, and landing is exact."""
    layer = caption()
    layout = TextLayout(layer, layer.text, padding=3)
    renderer = LetterDropRenderer(layer, layout)
    try:
        with (
            renderer.free_frame(
                0, layer.keyframes[0], placement(layer)
            ) as empty,
            renderer.free_frame(
                0.2, layer.keyframes[0], placement(layer)
            ) as entering,
            renderer.free_frame(
                2, layer.keyframes[0], placement(layer)
            ) as settled,
            project_text(
                layout.image, TextOrientation(), placement(layer)
            ) as expected,
            entering.getchannel("A") as first_alpha,
            settled.getchannel("A") as final_alpha,
            ImageChops.difference(settled, expected) as difference,
        ):
            assert empty.getbbox() is None
            first_bounds, final_bounds = (
                first_alpha.getbbox(),
                final_alpha.getbbox(),
            )
            assert first_bounds is not None
            assert final_bounds is not None
            assert (
                first_bounds[2] - first_bounds[0]
                < (final_bounds[2] - final_bounds[0]) * 0.7
            )
            assert first_bounds[1] < final_bounds[1] - 20
            assert difference.getbbox() is None
    finally:
        renderer.close()
        layout.close()


def test_backwards_preview_seek_reproduces_the_same_falling_letters() -> None:
    """A cached settled prefix never survives when the playhead returns earlier."""
    layer = caption()
    layout = TextLayout(layer, layer.text, padding=3)
    renderer = LetterDropRenderer(layer, layout)
    try:
        with renderer.free_frame(
            0.2, layer.keyframes[0], placement(layer)
        ) as first:
            original = first.tobytes()
        with renderer.free_frame(
            1.9, layer.keyframes[0], placement(layer)
        ) as later:
            assert later.getbbox() is not None
        with renderer.free_frame(
            0.2, layer.keyframes[0], placement(layer)
        ) as returned:
            assert returned.tobytes() == original
    finally:
        renderer.close()
        layout.close()


def test_source_mask_applies_to_both_moving_and_settled_glyphs() -> None:
    """A text mask is respected before 3D projection for every entrance phase."""
    layer = caption()
    layout = TextLayout(layer, layer.text, padding=3)
    with Image.new("L", layer.size.as_tuple) as mask:
        renderer = LetterDropRenderer(layer, layout, mask)
        try:
            for seconds in (0.2, 1.5, 2.2):
                with renderer.free_frame(
                    seconds, layer.keyframes[0], placement(layer)
                ) as image:
                    assert image.getbbox() is None
        finally:
            renderer.close()
    assert layout.image.getbbox() is not None
    layout.close()


def test_falling_surface_glyph_has_its_own_depth_before_landing(
    table: DepthAsset,
) -> None:
    """A lifted glyph passes in front of an object, then hides behind it on the table."""
    layer = caption("A").model_copy(
        update={
            "size": Size(width=160, height=80),
            "letter_drop": LetterDrop(height=3, rotation=(0, 0, 0)),
        }
    )
    options = SurfaceAttachment(layer_id="table", region=(40, 40, 120, 80))
    context = SurfaceContext(
        asset=table, attachment=options, plane=fit_surface(table, options)
    )
    table.depth.paste(3, (40, 20, 120, 90))
    layout = TextLayout(layer, layer.text, padding=3)
    renderer = LetterDropRenderer(layer, layout)
    pose = LayerKeyframe(x=80, y=60)
    try:
        with (
            renderer.surface_frame(0.01, pose, context) as falling,
            renderer.surface_frame(2, pose, context) as landed,
            render_surface(
                layout.image, table, options, context.plane, pose
            ) as expected,
            falling.crop((50, 30, 110, 70)) as near_object,
            landed.crop((50, 30, 110, 70)) as behind_object,
            ImageChops.difference(landed, expected) as difference,
        ):
            assert near_object.getbbox() is not None
            assert behind_object.getbbox() is None
            assert difference.getbbox() is None
    finally:
        renderer.close()
        layout.close()


def test_initial_xyz_angles_and_camera_depth_change_the_entrance() -> None:
    """A tumbling glyph differs visibly from the same free fall without rotation."""
    simple = caption("A")
    tumbling = simple.model_copy(
        update={"letter_drop": LetterDrop(height=1, rotation=(60, -35, -25))}
    )
    layout = TextLayout(simple, simple.text, padding=3)
    straight = LetterDropRenderer(simple, layout)
    tilted = LetterDropRenderer(tumbling, layout)
    try:
        with (
            straight.free_frame(
                0.5, simple.keyframes[0], placement(simple)
            ) as first,
            tilted.free_frame(
                0.5, simple.keyframes[0], placement(simple)
            ) as second,
            ImageChops.difference(first, second) as difference,
        ):
            assert first.getbbox() is not None
            assert second.getbbox() is not None
            assert difference.getbbox() is not None
    finally:
        straight.close()
        tilted.close()
        layout.close()
