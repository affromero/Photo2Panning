"""Surface type and 3D entrances survive the complete compositor pipeline."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageChops, ImageDraw
from pydantic import ValidationError

from pic2panning.api import run_operation
from pic2panning.backends.creative.depth import DepthPrediction
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
    PreviewRequest,
    TextLayer,
)
from pic2panning.schemas.creative.depth import (
    DepthOcclusion,
    DepthRequest,
    DepthStyle,
)
from pic2panning.schemas.creative.surface import (
    LetterDrop,
    SolidText,
    SurfaceAttachment,
)
from pic2panning.schemas.creative.typography import TextCue
from pic2panning.tools.creative.depth import extract_depth
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)


class TableWithCup:
    """Supply known metric geometry at the external model boundary."""

    table_depth: float

    def __init__(self, table_depth: float = 5) -> None:
        """Optionally make the table as near as its foreground cup."""
        self.table_depth = table_depth

    def predict(
        self, request: DepthRequest, image: Image.Image
    ) -> DepthPrediction:
        """Keep a nearer cup inside a flat five-metre background plane."""
        if request.device != "cpu":
            raise ValueError("The fixture requires an explicit CPU request")
        depth = Image.new("F", image.size, self.table_depth)
        depth.paste(3, (24, 15, 40, 32))
        return DepthPrediction(
            depth=depth,
            mask=Image.new("L", image.size, 255),
            intrinsics=((64, 0, 32), (0, 64, 24), (0, 0, 1)),
        )


@pytest.fixture
def scene(tmp_path: Path) -> ComposeRequest:
    """Build a real source-bound bundle and attach white text to its table."""
    source = tmp_path / "table.png"
    with Image.new("RGB", (64, 48), "navy") as image:
        ImageDraw.Draw(image).rectangle((24, 15, 39, 31), fill="#800000")
        image.save(source)
    asset = extract_depth(
        DepthRequest(
            source=str(source),
            output=str(tmp_path / "table.tiff"),
            device="cpu",
        ),
        TableWithCup(),
    )
    host = ImageLayer(
        id="table",
        source=str(source),
        size=Size(width=64, height=48),
        duration=2,
        keyframes=(LayerKeyframe(x=32, y=24),),
        depth=DepthStyle(metadata=asset.metadata),
    )
    text = TextLayer(
        id="type",
        text="MMMM",
        font_size=18,
        duration=2,
        size=Size(width=48, height=20),
        keyframes=(LayerKeyframe(x=32, y=24),),
        surface=SurfaceAttachment(layer_id="table", region=(8, 34, 56, 46)),
    )
    return ComposeRequest(
        canvas=Size(width=64, height=48),
        duration=2,
        camera=(CameraKeyframe(x=32, y=24),),
        layers=(host, text),
        output=str(tmp_path / "film.mp4"),
        size=Size(width=64, height=48),
        fps=10,
    )


def test_attached_text_is_visible_on_table_and_hidden_behind_cup(
    scene: ComposeRequest,
) -> None:
    """The complete frame uses source depth rather than covering the nearer cup."""
    baseline = scene.model_copy(update={"layers": scene.layers[:1]})
    with (
        CompositionSession(baseline) as background,
        composition_frame(baseline, 0, background) as original,
        CompositionSession(scene) as session,
        composition_frame(scene, 0, session) as result,
        ImageChops.difference(original, result) as changed,
        changed.crop((24, 15, 40, 32)) as cup,
    ):
        assert changed.getbbox() is not None
        assert cup.getbbox() is None
        assert any(
            cast("tuple[int, ...]", result.getpixel((x, y)))[0] > 220
            and cast("tuple[int, ...]", result.getpixel((x, y)))[1] > 220
            for y in range(result.height)
            for x in range(result.width)
        )


@pytest.mark.parametrize("operation", ["compose", "preview"])
def test_native_executable_output_alias_is_rejected_before_worker_launch(
    scene: ComposeRequest, tmp_path: Path, operation: str
) -> None:
    """Standalone outputs preserve an explicit native executable's source bytes."""
    executable = tmp_path / (
        "native.mp4" if operation == "compose" else "native.png"
    )
    original = b"protected native executable bytes"
    executable.write_bytes(original)
    text = scene.layers[-1]
    assert isinstance(text, TextLayer)
    solid = text.model_copy(
        update={"solid": SolidText(executable=str(executable))}
    )
    request = scene.model_copy(update={"layers": (scene.layers[0], solid)})
    job = (
        request.model_copy(
            update={"output": str(executable), "overwrite": True}
        )
        if operation == "compose"
        else PreviewRequest(
            composition=request, output=str(executable), overwrite=True
        )
    )
    with pytest.raises(ValueError, match="aliases an input"):
        run_operation(job)
    assert executable.read_bytes() == original


def test_surface_text_multiplies_host_and_own_opacity_once(
    scene: ComposeRequest,
) -> None:
    """Two half-opacity layers leave quarter-strength text over the background."""
    layers = tuple(
        layer.model_copy(
            update={
                "keyframes": (
                    layer.keyframes[0].model_copy(update={"opacity": 0.5}),
                )
            }
        )
        for layer in scene.layers
    )
    request = scene.model_copy(update={"layers": layers})
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as image,
    ):
        whites = [
            pixel[0]
            for pixel in cast(
                "list[tuple[int, int, int]]", list(image.get_flattened_data())
            )
            if pixel[0] == pixel[1] and pixel[0] > 10
        ]
        assert whites
        assert 62 <= max(whites) <= 65


def test_surface_drop_can_seek_past_morph_and_back_to_the_entrance(
    scene: ComposeRequest,
) -> None:
    """Initial glyph storage survives later cue eviction and backwards previews."""
    original = scene.layers[-1]
    assert isinstance(original, TextLayer)
    text = original.model_copy(
        update={
            "letter_drop": LetterDrop(duration=0.5, height=0.5),
            "text_cues": (
                TextCue(at=0.8, text="TIME", transition=0.2),
                TextCue(at=1.4, text="ROOM", transition=0.2),
            ),
        }
    )
    request = scene.model_copy(update={"layers": (scene.layers[0], text)})
    with CompositionSession(request) as session:
        with composition_frame(request, 2, session) as first:
            before = first.tobytes()
        for frame in (10, 16, 19):
            with composition_frame(request, frame, session) as image:
                assert image.size == (64, 48)
        with composition_frame(request, 2, session) as again:
            assert again.tobytes() == before


def test_attached_text_inherits_another_depth_layer_hiding_its_host(
    scene: ComposeRequest, tmp_path: Path
) -> None:
    """An occluded table cannot leak surface text over nearer geometry."""
    source = tmp_path / "near.png"
    with Image.new("RGB", (64, 48), "black") as image:
        image.save(source)
    asset = extract_depth(
        DepthRequest(
            source=str(source),
            output=str(tmp_path / "near.tiff"),
            device="cpu",
        ),
        TableWithCup(table_depth=3),
    )
    near = ImageLayer(
        id="near",
        source=str(source),
        size=Size(width=64, height=48),
        duration=2,
        keyframes=(LayerKeyframe(x=32, y=24),),
        depth=DepthStyle(metadata=asset.metadata),
    )
    table = scene.layers[0].model_copy(
        update={"occlusion": DepthOcclusion(layer_id="near", plane_metres=4)}
    )
    request = scene.model_copy(
        update={"layers": (near, table, scene.layers[1])}
    )
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as image,
    ):
        assert image.getbbox() is None


@pytest.mark.parametrize("invalid", ["order", "depth", "interval", "space"])
def test_invalid_surface_hosts_fail_before_media_processing(
    scene: ComposeRequest, invalid: str
) -> None:
    """A surface attachment always has a visible, earlier calibrated image."""
    value = scene.model_dump()
    if invalid == "order":
        value["layers"] = list(reversed(value["layers"]))
    elif invalid == "depth":
        value["layers"][0]["depth"] = None
    elif invalid == "interval":
        value["layers"][0]["start"] = 0.5
        value["layers"][0]["duration"] = 1.5
    else:
        value["layers"][1]["space"] = "viewport"
    with pytest.raises(ValidationError, match="Surface"):
        ComposeRequest.model_validate(value)
