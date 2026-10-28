"""Encoded timeline content, source-time ramps, and selective motion."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw

from pic2panning.errors import MediaError
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    Cinemagraph,
    ComposeRequest,
    LayerKeyframe,
    PreviewRequest,
    SourceTime,
    VideoLayer,
)
from pic2panning.tools.creative.composition import compose, preview
from pic2panning.tools.creative.frames import (
    CompositionSession,
    composition_frame,
)
from pic2panning.tools.media import decode_frames, encode_frames, ffmpeg, probe


def colors() -> Iterator[Image.Image]:
    """Return a six-frame source with distinguishable temporal regions."""
    for color in ("red", "red", "lime", "lime", "blue", "blue"):
        yield Image.new("RGB", (100, 100), color)


def test_composition_encodes_exact_frames_and_silent_audio(
    composition: ComposeRequest,
) -> None:
    """Explicit duration and fps produce six square, silent frames."""
    result = compose(composition)
    metadata = probe(composition.output)
    assert result.frames == 6
    assert result.duration == 1
    assert metadata.frames == 6
    assert metadata.width == metadata.height == 100
    assert not metadata.has_audio
    with (
        closing(decode_frames(composition.output, metadata, 6)) as decoded,
        next(decoded) as image,
    ):
        red, green, blue = cast(
            "tuple[int, int, int]", image.getpixel((10, 50))
        )
        assert red > 240
        assert green < 10
        assert blue < 10


def test_video_ramp_and_hold_match_preview(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A source-time hold retains red before a ramp reaches later source colors."""
    source = tmp_path / "source.mp4"
    encode_frames(colors(), str(source), Size(width=100, height=100), 6)
    video = VideoLayer(
        id="video",
        source=str(source),
        duration=1,
        size=Size(width=100, height=100),
        keyframes=(LayerKeyframe(x=100, y=50),),
        source_times=(
            SourceTime(at=0, source_time=0),
            SourceTime(at=0.5, source_time=0),
            SourceTime(at=1, source_time=1),
        ),
    )
    request = composition.model_copy(update={"layers": (video,)})
    with CompositionSession(request) as session:
        with composition_frame(request, 3, session) as held:
            assert (
                cast("tuple[int, int, int]", held.getpixel((50, 50)))[0] > 240
            )
        with composition_frame(request, 5, session) as exported:
            expected = exported.tobytes()
            assert (
                cast("tuple[int, int, int]", exported.getpixel((50, 50)))[2]
                > 240
            )
    destination = tmp_path / "video-preview.png"
    preview(
        PreviewRequest(composition=request, output=str(destination), frame=5)
    )
    with Image.open(destination) as actual:
        assert actual.tobytes() == expected


def test_cinemagraph_moves_only_white_mask_pixels(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A clean plate remains red outside the explicitly selected moving half."""
    source = tmp_path / "motion.mp4"
    encode_frames(
        (Image.new("RGB", (100, 100), "blue") for _ in range(6)),
        str(source),
        Size(width=100, height=100),
        6,
    )
    plate = tmp_path / "plate.png"
    mask = tmp_path / "mask.png"
    with Image.new("RGB", (100, 100), "red") as image:
        image.save(plate)
    with Image.new("L", (100, 100)) as image:
        ImageDraw.Draw(image).rectangle((0, 0, 49, 99), fill=255)
        image.save(mask)
    video = VideoLayer(
        id="motion",
        source=str(source),
        duration=1,
        size=Size(width=100, height=100),
        keyframes=(LayerKeyframe(x=100, y=50),),
        cinemagraph=Cinemagraph(mask=str(mask), reference=str(plate)),
    )
    request = composition.model_copy(update={"layers": (video,)})
    with (
        CompositionSession(request) as session,
        composition_frame(request, 0, session) as image,
    ):
        assert cast("tuple[int, int, int]", image.getpixel((10, 50)))[2] > 240
        assert image.getpixel((90, 50)) == (255, 0, 0)
    with Image.new("L", (20, 20), 255) as image:
        image.save(mask)
    with (
        pytest.raises(MediaError, match="must match"),
        CompositionSession(request),
    ):
        pass


def test_fractional_video_sampling_matches_export_preview(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Fractional frame timestamps use the same CFR decisions in both paths."""
    original, fractional = (
        tmp_path / "original.mp4",
        tmp_path / "fractional.mp4",
    )
    encode_frames(colors(), str(original), Size(width=100, height=100), 6)
    ffmpeg(
        [
            "-nostdin",
            "-y",
            "-i",
            str(original),
            "-vf",
            "setpts=PTS*1.001",
            "-r",
            "6000/1001",
            str(fractional),
        ]
    )
    layer = VideoLayer(
        id="fractional",
        source=str(fractional),
        duration=1,
        size=Size(width=100, height=100),
        keyframes=(LayerKeyframe(x=100, y=50),),
    )
    request = composition.model_copy(update={"layers": (layer,)})
    with (
        CompositionSession(request) as session,
        composition_frame(request, 2, session) as image,
    ):
        expected = image.tobytes()
    output = tmp_path / "fractional-preview.png"
    preview(PreviewRequest(composition=request, output=str(output), frame=2))
    with Image.open(output) as actual:
        assert actual.tobytes() == expected


def test_video_shutter_preview_matches_exported_exposure(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A source-time ramp integrates red and green in matching preview/export frames."""
    source = tmp_path / "exposure-source.mp4"
    encode_frames(colors(), str(source), Size(width=100, height=100), 6)
    layer = VideoLayer(
        id="video",
        source=str(source),
        duration=1,
        size=Size(width=100, height=100),
        keyframes=(LayerKeyframe(x=100, y=50),),
        source_times=(
            SourceTime(at=0, source_time=0),
            SourceTime(at=0.5, source_time=0),
            SourceTime(at=1, source_time=1),
        ),
    )
    request = composition.model_copy(
        update={"layers": (layer,), "shutter_angle": 360}
    )
    compose(request)
    output = tmp_path / "exposure-preview.png"
    preview(PreviewRequest(composition=request, output=str(output), frame=4))
    with Image.open(output) as image:
        expected = cast("tuple[int, int, int]", image.getpixel((50, 50)))
        assert 170 < expected[0] < 210
        assert 170 < expected[1] < 210
        assert expected[2] < 10
    with closing(
        decode_frames(request.output, probe(request.output), request.fps)
    ) as decoded:
        for index, frame in enumerate(decoded):
            with frame:
                if index == 4:
                    actual = cast(
                        "tuple[int, int, int]", frame.getpixel((50, 50))
                    )
                    assert all(
                        abs(left - right) < 10
                        for left, right in zip(actual, expected, strict=True)
                    )
    with CompositionSession(request, preview=True) as session:
        with composition_frame(request, 4, session) as forward:
            expected_pixels = forward.tobytes()
        with composition_frame(request, 1, session) as earlier:
            assert (
                cast("tuple[int, int, int]", earlier.getpixel((50, 50)))[0]
                > 240
            )
        with composition_frame(request, 4, session) as repeated:
            assert repeated.tobytes() == expected_pixels
