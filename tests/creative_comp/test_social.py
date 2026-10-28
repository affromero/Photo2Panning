"""Social aspect exports and complete derived-output source protection."""

from pathlib import Path

import pytest
from PIL import Image, ImageChops

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import ComposeRequest, ImageLayer
from pic2panning.schemas.creative.social import (
    SocialPackRequest,
    SocialVariant,
)
from pic2panning.tools.creative.social import social_outputs, social_pack
from pic2panning.tools.media import probe


def test_social_pack_outputs_aspects_covers_and_separate_guides(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Videos and selected stills use their declared viewports and clean covers."""
    request = SocialPackRequest(
        composition=composition,
        output_directory=str(tmp_path / "social"),
        variants=(
            SocialVariant(name="portrait", size=Size(width=100, height=200)),
            SocialVariant(name="square", size=Size(width=100, height=100)),
        ),
        carousel_at=(0.5,),
        cover_at=0.999,
        preview_guides=True,
    )
    result = social_pack(request)
    assert result.outputs == social_outputs(request)
    assert len(result.outputs) == 8
    for variant in request.variants:
        video = probe(str(tmp_path / "social" / f"{variant.name}.mp4"))
        assert (video.width, video.height) == variant.size.as_tuple
        with (
            Image.open(
                tmp_path / "social" / f"{variant.name}-cover.png"
            ) as clean,
            Image.open(
                tmp_path / "social" / f"{variant.name}-guides.png"
            ) as guides,
            ImageChops.difference(clean, guides) as difference,
        ):
            assert difference.getbbox() is not None
            assert clean.getpixel((0, 0)) == guides.getpixel((0, 0))


def test_derived_cover_alias_is_rejected_before_any_export(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A cover hardlink to a source blocks the entire pack before video output."""
    source = composition.layers[0]
    directory = tmp_path / "social"
    directory.mkdir()
    cover = directory / "square-cover.png"
    assert isinstance(source, ImageLayer)
    cover.hardlink_to(source.source)
    original = cover.read_bytes()
    request = SocialPackRequest(
        composition=composition,
        output_directory=str(directory),
        variants=(
            SocialVariant(name="square", size=Size(width=100, height=100)),
        ),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="aliases an input"):
        social_pack(request)
    assert not (directory / "square.mp4").exists()
    assert cover.read_bytes() == original


def test_social_duration_reports_encoded_whole_frames(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Pack metadata uses encoded frame duration after rounding a short timeline."""
    short = composition.model_copy(
        update={
            "duration": 0.2,
            "layers": tuple(
                layer.model_copy(update={"duration": 0.2})
                for layer in composition.layers
            ),
        }
    )
    request = SocialPackRequest(
        composition=short,
        output_directory=str(tmp_path / "short"),
        variants=(
            SocialVariant(name="square", size=Size(width=100, height=100)),
        ),
    )
    result = social_pack(request)
    assert result.duration == pytest.approx(2 / 6)
    assert probe(result.outputs[0]).frames == 2
