"""Small deterministic media used by composition behavior tests."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
)


@pytest.fixture
def composition(tmp_path: Path) -> ComposeRequest:
    """Create a red/blue world larger than its centered square viewport."""
    source = tmp_path / "photo.png"
    with Image.new("RGB", (200, 100), "red") as image:
        ImageDraw.Draw(image).rectangle((100, 0, 199, 99), fill="blue")
        image.save(source)
    return ComposeRequest(
        canvas=Size(width=200, height=100),
        duration=1,
        camera=(CameraKeyframe(x=100, y=50),),
        layers=(
            ImageLayer(
                id="photo",
                source=str(source),
                duration=1,
                size=Size(width=200, height=100),
                keyframes=(LayerKeyframe(x=100, y=50),),
            ),
        ),
        size=Size(width=100, height=100),
        fps=6,
        output=str(tmp_path / "composition.mp4"),
    )
