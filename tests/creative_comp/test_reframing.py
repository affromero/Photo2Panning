"""Viewport graphics retain their authored placement across output sizes."""

from contextlib import closing
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
    TextLayer,
)
from pic2panning.schemas.creative.social import (
    SocialPackRequest,
    SocialVariant,
)
from pic2panning.tools.creative.composition import reframe_composition
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)
from pic2panning.tools.creative.social import social_pack
from pic2panning.tools.media import decode_frames, probe


def color_bounds(
    image: Image.Image, color: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    """Locate a semantic graphic color while allowing resampling at its edge."""
    with (
        Image.new("RGB", image.size, color) as target,
        ImageChops.difference(image, target) as difference,
        difference.getchannel("R") as red,
        difference.getchannel("G") as green,
        difference.getchannel("B") as blue,
        ImageChops.lighter(red, green) as first,
        ImageChops.lighter(first, blue) as maximum,
        maximum.point(
            [255 if value < 20 else 0 for value in range(256)]
        ) as selected,
    ):
        bounds = selected.getbbox()
    assert bounds is not None, f"Graphic color {color} disappeared"
    return bounds


def authored_graphics(
    composition: ComposeRequest, tmp_path: Path
) -> ComposeRequest:
    """Create a lower headline and upper logo in a portrait pixel viewport."""
    source = composition.layers[0]
    assert isinstance(source, ImageLayer)
    with Image.open(source.source) as image:
        ImageDraw.Draw(image).rectangle((110, 20, 119, 29), fill="yellow")
        image.save(source.source)
    logo = tmp_path / "logo.png"
    with Image.new("RGB", (60, 20), "lime") as image:
        image.save(logo)
    headline = TextLayer(
        id="headline",
        text="HELLO",
        font_size=32,
        color="white",
        size=Size(width=160, height=60),
        duration=1,
        space="viewport",
        keyframes=(LayerKeyframe(x=100, y=300),),
    )
    graphic = ImageLayer(
        id="logo",
        source=str(logo),
        size=Size(width=60, height=20),
        duration=1,
        space="viewport",
        keyframes=(
            LayerKeyframe(x=50, y=100),
            LayerKeyframe(at=1, x=150, y=300),
        ),
    )
    return composition.model_copy(
        update={
            "size": Size(width=200, height=400),
            "layers": (*composition.layers, headline, graphic),
        }
    )


def test_reframing_keeps_headline_logo_and_world_photo_consistent(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Portrait, square, and small drafts retain text and unstretched world detail."""
    authored = authored_graphics(composition, tmp_path)
    widths: list[float] = []
    for size in (
        Size(width=200, height=400),
        Size(width=200, height=200),
        Size(width=100, height=200),
    ):
        request = reframe_composition(authored, size)
        assert request.camera == authored.camera
        assert request.canvas == authored.canvas
        assert request.layers[0] == authored.layers[0]
        with (
            CompositionSession(request) as session,
            composition_frame(request, 0, session) as image,
        ):
            text = color_bounds(image, (255, 255, 255))
            assert abs((text[0] + text[2]) / 2 - size.width / 2) < 3
            assert size.height / 2 < text[1] < text[3] < size.height
            widths.append((text[2] - text[0]) / size.width)
            assert image.getpixel((size.width // 4, size.height // 4)) == (
                0,
                255,
                0,
            )
            marker = color_bounds(image, (255, 255, 0))
            assert abs((marker[2] - marker[0]) - (marker[3] - marker[1])) <= 1
            assert image.getpixel((0, size.height - 1)) == (255, 0, 0)
            assert image.getpixel((size.width - 1, size.height - 1)) == (
                0,
                0,
                255,
            )
            with composition_frame(request, 3, session) as moved:
                assert moved.getpixel((size.width // 2, size.height // 2)) == (
                    0,
                    255,
                    0,
                )
    assert max(widths) - min(widths) < 0.02
    assert authored.size == Size(width=200, height=400)


def test_social_videos_covers_and_carousels_reframe_viewport_titles(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Every artifact retains a visible headline when aspect or draft size changes."""
    authored = authored_graphics(composition, tmp_path)
    variants = (
        SocialVariant(name="portrait", size=Size(width=200, height=400)),
        SocialVariant(name="square", size=Size(width=200, height=200)),
        SocialVariant(name="draft", size=Size(width=100, height=200)),
    )
    directory = tmp_path / "pack"
    social_pack(
        SocialPackRequest(
            composition=authored,
            output_directory=str(directory),
            variants=variants,
            carousel_at=(0.5,),
        )
    )
    for variant in variants:
        for suffix in ("-cover.png", "-carousel-001.png"):
            with Image.open(directory / f"{variant.name}{suffix}") as image:
                text = color_bounds(image, (255, 255, 255))
                assert (
                    variant.size.height / 2
                    < text[1]
                    < text[3]
                    < variant.size.height
                )
        video = str(directory / f"{variant.name}.mp4")
        with (
            closing(
                decode_frames(video, probe(video), authored.fps)
            ) as decoded,
            next(decoded) as frame,
        ):
            assert (
                color_bounds(frame, (255, 255, 255))[1]
                > variant.size.height / 2
            )
