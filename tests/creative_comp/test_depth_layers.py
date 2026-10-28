"""Depth occlusion follows source placement and the world camera together."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from pic2panning.backends.creative.depth import DepthPrediction
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
)
from pic2panning.schemas.creative.depth import (
    DepthOcclusion,
    DepthRequest,
    DepthStyle,
)
from pic2panning.tools.creative.depth import extract_depth
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)


class TwoPlaneBoundary:
    """Produce a near left plane and far right plane at the model boundary."""

    far: float
    sky_right: bool

    def __init__(self, far: float = 10, *, sky_right: bool = False) -> None:
        """Optionally make both halves the same near plane."""
        self.far = far
        self.sky_right = sky_right

    def predict(
        self, request: DepthRequest, image: Image.Image
    ) -> DepthPrediction:
        """Return analytical geometry while retaining real artifact validation."""
        if request.device != "cpu":
            raise ValueError("Synthetic depth requires CPU")
        depth = Image.new("F", image.size)
        depth.putdata(
            [
                1 if column < image.width // 2 else self.far
                for _ in range(image.height)
                for column in range(image.width)
            ]
        )
        mask = Image.new("L", image.size, 255)
        if self.sky_right:
            mask.paste(0, (image.width // 2, 0, image.width, image.height))
        return DepthPrediction(
            depth=depth,
            mask=mask,
            intrinsics=(
                (float(image.width), 0, image.width / 2),
                (0, float(image.height), image.height / 2),
                (0, 0, 1),
            ),
        )


def test_occlusion_tracks_depth_source_camera_and_overlay(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A green graphic hides behind near red geometry and covers far blue pixels."""
    original = composition.layers[0]
    assert isinstance(original, ImageLayer)
    asset = extract_depth(
        DepthRequest(
            source=original.source,
            output=str(tmp_path / "photo.tiff"),
            device="cpu",
        ),
        TwoPlaneBoundary(),
    )
    photo = original.model_copy(
        update={"depth": DepthStyle(metadata=asset.metadata)}
    )
    green = tmp_path / "green.png"
    with Image.new("RGB", (200, 100), "lime") as image:
        image.save(green)
    graphic = ImageLayer(
        id="graphic",
        source=str(green),
        duration=1,
        size=Size(width=200, height=100),
        keyframes=(LayerKeyframe(x=100, y=50),),
        occlusion=DepthOcclusion(layer_id=photo.id, plane_metres=5),
    )
    request = composition.model_copy(
        update={
            "layers": (photo, graphic),
            "camera": (
                CameraKeyframe(x=100, y=50),
                CameraKeyframe(at=0.5, x=150, y=50),
            ),
        }
    )
    with CompositionSession(request) as session:
        with composition_frame(request, 0, session) as first:
            assert first.getpixel((10, 50)) == (255, 0, 0)
            assert first.getpixel((90, 50)) == (0, 255, 0)
        with composition_frame(request, 3, session) as moved:
            assert moved.getpixel((10, 50)) == (0, 255, 0)
            assert moved.getpixel((90, 50)) == (0, 255, 0)


def test_rotated_contain_padding_does_not_occlude_graphics(
    tmp_path: Path,
) -> None:
    """Transparent antialiased source borders leave the graphic behind visible."""
    source, green = tmp_path / "white.png", tmp_path / "green.png"
    with Image.new("RGB", (32, 16), "white") as image:
        image.save(source)
    with Image.new("RGB", (64, 64), "lime") as image:
        image.save(green)
    asset = extract_depth(
        DepthRequest(
            source=str(source),
            output=str(tmp_path / "white.tiff"),
            device="cpu",
        ),
        TwoPlaneBoundary(far=1),
    )
    photo = ImageLayer(
        id="photo",
        source=str(source),
        size=Size(width=32, height=32),
        duration=1,
        keyframes=(LayerKeyframe(x=32, y=32, rotation=17, scale=1.2),),
        depth=DepthStyle(metadata=asset.metadata),
    )
    request = ComposeRequest(
        canvas=Size(width=64, height=64),
        duration=1,
        camera=(CameraKeyframe(x=32, y=32),),
        layers=(photo,),
        size=Size(width=64, height=64),
        output=str(tmp_path / "result.mp4"),
    )
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as baseline,
    ):
        edges = [
            (x, y)
            for y in range(64)
            for x in range(64)
            if 0
            < cast("tuple[int, int, int]", baseline.getpixel((x, y)))[0]
            < 50
        ]
    assert edges
    graphic = ImageLayer(
        id="graphic",
        source=str(green),
        size=Size(width=64, height=64),
        duration=1,
        keyframes=(LayerKeyframe(x=32, y=32),),
        occlusion=DepthOcclusion(layer_id="photo", plane_metres=2),
    )
    request = request.model_copy(update={"layers": (photo, graphic)})
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as result,
    ):
        assert all(
            cast("tuple[int, int, int]", result.getpixel(position))[1] > 200
            for position in edges
        )


@pytest.mark.parametrize(
    ("dimension", "red_x", "blue_x"), [(50, 27, 34), (100, 55, 65)]
)
def test_depth_parallax_strength_tracks_projected_output_density(
    composition: ComposeRequest,
    tmp_path: Path,
    dimension: int,
    red_x: int,
    blue_x: int,
) -> None:
    """A source-pixel warp scales consistently at reduced and native pixel density."""
    original = composition.layers[0]
    assert isinstance(original, ImageLayer)
    asset = extract_depth(
        DepthRequest(
            source=original.source,
            output=str(tmp_path / "flat.tiff"),
            device="cpu",
        ),
        TwoPlaneBoundary(far=1),
    )
    style = DepthStyle(metadata=asset.metadata, focus_metres=2, parallax_x=20)
    photo = original.model_copy(update={"depth": style})
    request = composition.model_copy(
        update={
            "layers": (photo,),
            "size": Size(width=dimension, height=dimension),
        }
    )
    with (
        CompositionSession(request) as session,
        composition_frame(request, 3, session) as frame,
    ):
        red = cast(
            "tuple[int, int, int]", frame.getpixel((red_x, dimension // 2))
        )
        blue = cast(
            "tuple[int, int, int]", frame.getpixel((blue_x, dimension // 2))
        )
        assert red[0] > 250
        assert red[2] < 5
        assert blue[0] < 5
        assert blue[2] > 250


def test_retained_sky_depth_does_not_occlude_graphics(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Foreground hides a graphic while sky with finite near depth stays open."""
    original = composition.layers[0]
    assert isinstance(original, ImageLayer)
    asset = extract_depth(
        DepthRequest(
            source=original.source,
            output=str(tmp_path / "sky.tiff"),
            device="cpu",
        ),
        TwoPlaneBoundary(far=1, sky_right=True),
    )
    photo = original.model_copy(
        update={"depth": DepthStyle(metadata=asset.metadata)}
    )
    source = tmp_path / "green.png"
    with Image.new("RGB", (200, 100), "lime") as image:
        image.save(source)
    graphic = ImageLayer(
        id="graphic",
        source=str(source),
        duration=1,
        size=Size(width=200, height=100),
        keyframes=(LayerKeyframe(x=100, y=50),),
        occlusion=DepthOcclusion(layer_id=photo.id, plane_metres=2),
    )
    request = composition.model_copy(update={"layers": (photo, graphic)})
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as frame,
    ):
        assert frame.getpixel((10, 50)) == (255, 0, 0)
        assert frame.getpixel((90, 50)) == (0, 255, 0)
