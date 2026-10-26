"""Behavioral coverage for geometry and real FFmpeg timelines."""

import array
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import closing
from typing import cast

import pytest
from klogr.path import path_exists, path_join
from PIL import Image, ImageChops, ImageDraw, ImageSequence, ImageStat

from pic2panning.errors import MediaError
from pic2panning.schemas.base import Size
from pic2panning.schemas.media import (
    AudioTrack,
    Cover,
    CoverPosition,
    Encoding,
    EncodingPreset,
    ImageClip,
    Motion,
    RenderRequest,
    VideoClip,
)
from pic2panning.tools.frames import (
    center_progress,
    image_frames,
    transform_frame,
    video_frame,
)
from pic2panning.tools.media import decode_frames, encode_frames, ffmpeg, probe
from pic2panning.tools.render import render
from pic2panning.tools.text import title_card


def test_large_photo_crop_stays_inside_source_at_both_pan_edges() -> None:
    """Fractional 24-megapixel scaling never creates a negative crop offset."""
    with Image.new("RGB", (6000, 4000), "blue") as source:
        for progress in (0.0, 1.0):
            with transform_frame(
                source, Size(width=1080, height=1920), Motion.PAN_LR, progress
            ) as frame:
                assert frame.size == (1080, 1920)
                assert color_at(frame) == (0, 0, 255)


@pytest.fixture
def workspace() -> Iterator[str]:
    """Give every test a separate, automatically cleaned media directory."""
    with tempfile.TemporaryDirectory(prefix="pic2panning-test-") as directory:
        yield directory


@pytest.fixture
def media_workspace(workspace: str) -> str:
    """Require real media executables for integration scenarios."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and FFprobe are required for media integration")
    return workspace


def color_at(
    image: Image.Image, x: int = 20, y: int = 20
) -> tuple[int, int, int]:
    """Read an RGB test pixel."""
    return cast("tuple[int, int, int]", image.getpixel((x, y)))


def striped_image(*, vertical: bool = False) -> Image.Image:
    """Make red, green, and blue bands with unambiguous motion endpoints."""
    image = Image.new("RGB", (120, 40))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(("red", "lime", "blue")):
        draw.rectangle((index * 40, 0, index * 40 + 39, 39), fill=color)
    if vertical:
        transposed = image.transpose(Image.Transpose.TRANSPOSE)
        image.close()
        return transposed
    return image


@pytest.mark.parametrize(
    ("motion", "vertical", "first", "last"),
    [
        (Motion.PAN_LR, False, (255, 0, 0), (0, 0, 255)),
        (Motion.PAN_RL, False, (0, 0, 255), (255, 0, 0)),
        (Motion.PAN_UD, True, (255, 0, 0), (0, 0, 255)),
        (Motion.PAN_DU, True, (0, 0, 255), (255, 0, 0)),
    ],
)
def test_pan_reaches_both_edges_in_requested_direction(
    motion: Motion,
    *,
    vertical: bool,
    first: tuple[int, int, int],
    last: tuple[int, int, int],
) -> None:
    """All pan directions reach the source edges without stretching."""
    clip = ImageClip(source="unused.png", duration=1, motion=motion)
    with striped_image(vertical=vertical) as source:
        colors = []
        for frame in image_frames(source, clip, Size(width=40, height=40), 3):
            with frame:
                colors.append(color_at(frame))
    assert colors == [first, (0, 255, 0), last]


def test_reverse_returns_to_start_without_repeating_turnaround() -> None:
    """Reversing three forward frames yields five continuously ordered frames."""
    clip = ImageClip(source="unused.png", duration=1, reverse=True)
    with striped_image() as source:
        colors = []
        for frame in image_frames(source, clip, Size(width=40, height=40), 3):
            with frame:
                colors.append(color_at(frame))
    assert colors == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]


@pytest.mark.parametrize("motion", [Motion.ZOOM_IN, Motion.ZOOM_OUT])
def test_zoom_changes_crop_and_preserves_output_dimensions(
    motion: Motion,
) -> None:
    """Zoom moves between the full image and its central region."""
    with Image.new("RGB", (80, 80), "red") as source:
        ImageDraw.Draw(source).rectangle((20, 20, 59, 59), fill="blue")
        clip = ImageClip(
            source="unused.png", duration=1, motion=motion, zoom=2
        )
        colors = []
        for frame in image_frames(source, clip, Size(width=40, height=40), 2):
            with frame:
                assert frame.size == (40, 40)
                colors.append(color_at(frame, 5, 5))
    if motion == Motion.ZOOM_IN:
        assert colors == [(255, 0, 0), (0, 0, 255)]
    else:
        assert colors == [(0, 0, 255), (255, 0, 0)]


def test_cover_crop_preserves_square_geometry() -> None:
    """A square feature stays square when the source has a different aspect."""
    with Image.new("RGB", (80, 40), "black") as source:
        ImageDraw.Draw(source).rectangle((30, 10, 49, 29), fill="red")
        with transform_frame(
            source, Size(width=40, height=40), None, 0
        ) as frame:
            bounds = frame.getbbox()
            assert bounds == (10, 10, 30, 30)


def test_focus_center_slows_the_middle_and_retains_endpoints() -> None:
    """The center covers less distance than equal intervals near the edges."""
    assert center_progress(0) == pytest.approx(0)
    assert center_progress(1) == pytest.approx(1)
    middle = center_progress(0.55) - center_progress(0.45)
    edge = center_progress(0.1) - center_progress(0)
    assert 0 < middle < edge


def test_video_pan_bounces_after_reaching_an_edge() -> None:
    """A pan continues back through the source instead of leaving its bounds."""
    clip = VideoClip(source="unused.mp4", motion=Motion.PAN_LR, pan_speed=40)
    size = Size(width=40, height=40)
    with striped_image() as source:
        colors = []
        for index in range(5):
            with video_frame(source, clip, size, index, 5) as frame:
                colors.append(color_at(frame))
    assert colors == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]


def test_title_card_renders_default_font_and_wraps_long_words() -> None:
    """Portable default font and long words produce visible bounded text."""
    cover = Cover(text="Averylongwordwithoutspaces\nNext line", color="yellow")
    with title_card(cover, Size(width=80, height=60)) as image:
        assert image.size == (80, 60)
        assert image.getbbox() is not None
        assert color_at(image, 0, 0) == (0, 0, 0)


def save_stripes(directory: str) -> str:
    """Write a deterministic motion source."""
    path = path_join(directory, "source.png")
    with striped_image() as image:
        image.save(path)
    return path


def decoded_colors(path: str, fps: int) -> list[tuple[int, int, int]]:
    """Observe output pixels after real container decoding."""
    colors = []
    with closing(decode_frames(path, probe(path), fps)) as frames:
        for frame in frames:
            with frame:
                colors.append(color_at(frame))
    return colors


def audio_energy(path: str, start: float, duration: float) -> float:
    """Measure decoded PCM energy within a specific output time interval."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        message = "FFmpeg is required to measure audio."
        raise RuntimeError(message)
    result = subprocess.run(
        [
            binary,
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            path,
            "-t",
            str(duration),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-ar",
            "8000",
            "-ac",
            "1",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = array.array("h")
    samples.frombytes(result.stdout)
    return sum(abs(value) for value in samples) / max(1, len(samples))


def test_rendered_reverse_has_exact_frames_colors_and_dimensions(
    media_workspace: str,
) -> None:
    """A real MP4 retains the requested timeline and pan endpoints."""
    source = save_stripes(media_workspace)
    output = path_join(media_workspace, "result.mp4")
    result = render(
        RenderRequest(
            clips=(ImageClip(source=source, duration=1, reverse=True),),
            output=output,
            size=Size(width=40, height=40),
            fps=3,
        )
    )
    info = probe(output)
    assert (info.width, info.height, info.frames) == (40, 40, 5)
    assert not info.has_audio
    assert result.frames == 5
    assert result.duration == pytest.approx(5 / 3)
    colors = decoded_colors(output, 3)
    assert len(colors) == 5
    assert colors[0][0] > 240
    assert colors[2][2] > 240
    assert colors[-1][0] > 240
    assert path_exists(source)


def test_speed_preserves_audio_and_cover_has_independent_duration(
    media_workspace: str,
) -> None:
    """A silent cover precedes speed-adjusted video with synchronized audio."""
    source = path_join(media_workspace, "source.mp4")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=80x40:r=10:d=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            source,
        ]
    )
    output = path_join(media_workspace, "result.mp4")
    result = render(
        RenderRequest(
            clips=(VideoClip(source=source, speed=2),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
            cover=Cover(text="Title", duration=0.4),
        )
    )
    assert result.frames == 10
    assert result.duration == pytest.approx(1)
    info = probe(output)
    assert info.has_audio
    assert info.duration == pytest.approx(1, abs=0.001)
    colors = decoded_colors(output, 10)
    assert len(colors) == 10
    assert colors[-1][0] > 240
    assert audio_energy(output, 0.02, 0.15) < 100
    assert audio_energy(output, 0.6, 0.2) > 1000
    assert path_exists(source)


@pytest.mark.parametrize(("speed", "expected"), [(4.0, 1), (0.25, 16)])
def test_video_speed_outside_single_audio_filter_range(
    media_workspace: str, speed: float, expected: int
) -> None:
    """Large speed changes retain a playable frame sequence and audible sound."""
    source = path_join(media_workspace, "source.mp4")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=40x40:r=10:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            source,
        ]
    )
    output = path_join(media_workspace, "result.mp4")
    result = render(
        RenderRequest(
            clips=(VideoClip(source=source, speed=speed),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
        )
    )
    assert result.frames == expected
    assert probe(output).frames == expected
    assert audio_energy(output, 0, min(0.1, expected / 10)) > 1000


def test_very_short_sped_up_video_still_produces_one_frame(
    media_workspace: str,
) -> None:
    """A positive video timeline shorter than one output frame stays playable."""
    source = path_join(media_workspace, "source.mp4")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=40x40:r=10:d=0.1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            source,
        ]
    )
    output = path_join(media_workspace, "result.mp4")
    result = render(
        RenderRequest(
            clips=(VideoClip(source=source, speed=20),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
        )
    )
    assert result.frames == 1
    assert probe(output).frames == 1
    assert not probe(output).has_audio


@pytest.mark.parametrize("loop", [False, True])
def test_audio_offset_pads_or_loops_short_track(
    media_workspace: str, *, loop: bool
) -> None:
    """Replacement audio applies its source offset and explicit loop policy."""
    source = save_stripes(media_workspace)
    audio = path_join(media_workspace, "tone.wav")
    ffmpeg(
        ["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.6", audio]
    )
    output = path_join(media_workspace, "result.mp4")
    render(
        RenderRequest(
            clips=(ImageClip(source=source, duration=1),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
            audio=AudioTrack(source=audio, start=0.3, loop=loop),
        )
    )
    assert audio_energy(output, 0.05, 0.1) > 1000
    ending = audio_energy(output, 0.8, 0.1)
    if loop:
        assert ending > 1000
    else:
        assert ending < 100


def test_looped_audio_never_reintroduces_the_skipped_prefix(
    media_workspace: str,
) -> None:
    """Repeating a selected tone excludes the silent prefix every cycle."""
    source = save_stripes(media_workspace)
    audio = path_join(media_workspace, "prefixed-tone.wav")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-af",
            "volume=volume=0:enable='lt(t,0.5)'",
            audio,
        ]
    )
    output = path_join(media_workspace, "looped.mp4")
    render(
        RenderRequest(
            clips=(ImageClip(source=source, duration=2),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
            audio=AudioTrack(source=audio, start=0.5, loop=True),
        )
    )
    for start in (0.05, 0.55, 1.05, 1.55):
        assert audio_energy(output, start, 0.1) > 1000


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("suffix", ["wav", "m4a"])
def test_audio_offset_at_the_end_rejects_an_empty_selection(
    media_workspace: str, suffix: str, *, loop: bool
) -> None:
    """An unusable soundtrack offset fails without publishing the video."""
    source = save_stripes(media_workspace)
    audio = path_join(media_workspace, f"short.{suffix}")
    ffmpeg(["-y", "-f", "lavfi", "-i", "sine=duration=0.4", audio])
    output = path_join(media_workspace, "empty-selection.mp4")
    with pytest.raises(MediaError, match=r"offset .* leaves no audio"):
        render(
            RenderRequest(
                clips=(ImageClip(source=source, duration=0.2),),
                output=output,
                size=Size(width=40, height=40),
                fps=10,
                audio=AudioTrack(source=audio, start=0.4, loop=loop),
            )
        )
    assert not path_exists(output)


def test_multi_clip_order_and_end_cover_survive_concatenation(
    media_workspace: str,
) -> None:
    """Concatenation preserves clip order and includes a silent final card."""
    red = path_join(media_workspace, "red.png")
    blue = path_join(media_workspace, "blue.png")
    for path, color in ((red, "red"), (blue, "blue")):
        with Image.new("RGB", (40, 40), color) as image:
            image.save(path)
    output = path_join(media_workspace, "result.mp4")
    result = render(
        RenderRequest(
            clips=(
                ImageClip(source=red, duration=0.2),
                ImageClip(source=blue, duration=0.2),
            ),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
            cover=Cover(text="End", duration=0.2, position=CoverPosition.END),
        )
    )
    assert result.frames == 6
    assert not probe(output).has_audio
    colors = decoded_colors(output, 10)
    assert len(colors) == 6
    assert colors[0][0] > 240
    assert colors[2][2] > 240
    assert path_exists(red)
    assert path_exists(blue)


def test_gif_supports_odd_dimensions_and_correct_frame_timing(
    media_workspace: str,
) -> None:
    """An animated GIF preserves its odd-sized viewport and timeline."""
    source = save_stripes(media_workspace)
    output = path_join(media_workspace, "result.gif")
    result = render(
        RenderRequest(
            clips=(ImageClip(source=source, duration=0.6),),
            output=output,
            size=Size(width=31, height=25),
            fps=5,
        )
    )
    assert result.frames == 3
    with Image.open(output) as image:
        assert image.size == (31, 25)
        durations = [
            frame.info["duration"] for frame in ImageSequence.Iterator(image)
        ]
    assert sum(durations) == 600


def test_gif_at_maximum_representable_rate_has_exact_duration(
    media_workspace: str,
) -> None:
    """One hundred GIF frames per second use exact centisecond delays."""
    source = save_stripes(media_workspace)
    output = path_join(media_workspace, "fast.gif")
    result = render(
        RenderRequest(
            clips=(ImageClip(source=source, duration=0.2),),
            output=output,
            size=Size(width=40, height=40),
            fps=100,
        )
    )
    with Image.open(output) as image:
        durations = [
            frame.info["duration"] for frame in ImageSequence.Iterator(image)
        ]
    assert result.frames == len(durations) == 20
    assert result.duration is not None
    assert sum(durations) == result.duration * 1000 == 200


def test_failed_video_decode_does_not_publish_output(
    media_workspace: str,
) -> None:
    """A corrupt source reports FFmpeg diagnostics and leaves no output file."""
    source = path_join(media_workspace, "corrupt.png")
    with Image.new("RGB", (40, 40), "red") as image:
        image.save(source)
    output = path_join(media_workspace, "result.mp4")
    with pytest.raises(MediaError, match=r"No video|duration|Media command"):
        render(
            RenderRequest(
                clips=(VideoClip(source=source),),
                output=output,
                size=Size(width=40, height=40),
                fps=5,
            )
        )
    assert not path_exists(output)
    assert path_exists(source)


def test_interrupted_frame_source_preserves_original_error(
    media_workspace: str,
) -> None:
    """Closing a stopped encoder cannot hide the frame provider's failure."""

    def interrupted_frames() -> Iterator[Image.Image]:
        yield Image.new("RGB", (20, 20), "red")
        raise ValueError("Original frame source failed")

    output = path_join(media_workspace, "incomplete.mp4")
    with pytest.raises(ValueError, match="Original frame source failed"):
        encode_frames(
            interrupted_frames(), output, Size(width=20, height=20), 5
        )


def test_muted_source_video_has_no_artificial_audio_stream(
    media_workspace: str,
) -> None:
    """Explicit source muting produces a video-only output stream."""
    source = path_join(media_workspace, "source.mp4")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=40x40:r=10:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            source,
        ]
    )
    output = path_join(media_workspace, "result.mp4")
    result = render(
        RenderRequest(
            clips=(VideoClip(source=source, keep_audio=False),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
        )
    )
    info = probe(output)
    assert not info.has_audio
    assert info.frames == result.frames == 4


def test_per_clip_audio_retains_sound_and_silent_timeline_gaps(
    media_workspace: str,
) -> None:
    """A replacement track stays attached to its clip after concatenation."""
    source = save_stripes(media_workspace)
    audio = path_join(media_workspace, "tone.wav")
    ffmpeg(
        ["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4", audio]
    )
    output = path_join(media_workspace, "result.mp4")
    render(
        RenderRequest(
            clips=(
                ImageClip(source=source, duration=0.2),
                ImageClip(
                    source=source, duration=0.2, audio=AudioTrack(source=audio)
                ),
            ),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
        )
    )
    assert probe(output).has_audio
    assert audio_energy(output, 0.02, 0.1) < 100
    assert audio_energy(output, 0.25, 0.1) > 1000


def test_longer_audio_does_not_extend_video_zoom_timeline(
    media_workspace: str,
) -> None:
    """A zoom reaches its endpoint when the audio outlasts its source video."""
    source = path_join(media_workspace, "source.mp4")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=80x80:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-vf",
            "drawbox=x=20:y=20:w=40:h=40:color=blue:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            source,
        ]
    )
    source_info = probe(source)
    assert source_info.duration == pytest.approx(1)
    assert source_info.has_audio
    assert source_info.frames == 10
    output = path_join(media_workspace, "zoomed.mp4")
    result = render(
        RenderRequest(
            clips=(VideoClip(source=source, motion=Motion.ZOOM_IN, zoom=2),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
        )
    )
    assert result.frames == 10
    assert result.duration == pytest.approx(1)
    colors = []
    with closing(decode_frames(output, probe(output), 10)) as frames:
        for frame in frames:
            with frame:
                colors.append(color_at(frame, 3, 3))
    assert len(colors) == 10
    assert colors[0][0] > 240
    assert colors[-1][2] > 240
    assert colors[-1][0] < 10


def test_video_without_stream_duration_uses_container_timeline(
    media_workspace: str,
) -> None:
    """A container duration remains usable when its stream omits duration."""
    source = path_join(media_workspace, "source.mkv")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=40x40:r=10:d=0.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            source,
        ]
    )
    info = probe(source)
    assert info.duration == pytest.approx(0.4)
    assert (info.width, info.height) == (40, 40)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_anamorphic_video_preserves_display_geometry_and_rotation(
    media_workspace: str, rotation: int
) -> None:
    """Pixel aspect and display rotation are each applied once before crop."""
    source = path_join(media_workspace, "anamorphic.mp4")
    ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=120x40:r=10:d=0.2",
            "-vf",
            (
                "drawbox=x=50:y=10:w=20:h=20:color=blue:t=fill,"
                "drawbox=x=0:y=0:w=10:h=40:color=lime:t=fill,setsar=2"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            source,
        ]
    )
    rotated = path_join(media_workspace, "oriented.mp4")
    ffmpeg(
        [
            "-y",
            "-display_rotation",
            str(rotation),
            "-i",
            source,
            "-c",
            "copy",
            rotated,
        ]
    )
    info = probe(rotated)
    dimensions = (40, 240) if rotation % 180 else (240, 40)
    assert (info.width, info.height) == dimensions
    green_position = {
        0: (5, 20),
        90: (20, 234),
        180: (234, 20),
        270: (20, 5),
    }[rotation]
    with (
        closing(decode_frames(rotated, info, 10)) as frames,
        next(frames) as frame,
    ):
        assert frame.size == dimensions
        red, green, blue_value = color_at(frame, *green_position)
        assert green > 240
        assert red < 10
        assert blue_value < 10
    output = path_join(media_workspace, "square.mp4")
    render(
        RenderRequest(
            clips=(VideoClip(source=rotated),),
            output=output,
            size=Size(width=40, height=40),
            fps=10,
        )
    )
    with (
        closing(decode_frames(output, probe(output), 10)) as frames,
        next(frames) as frame,
        frame.getchannel("B") as blue,
        blue.point(lambda value: 255 if value > 150 else 0) as mask,
    ):
        bounds = mask.getbbox()
    assert bounds is not None
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    expected = (20, 40) if rotation % 180 else (40, 20)
    assert (width, height) == expected


def test_encoding_quality_controls_decoded_detail(
    media_workspace: str,
) -> None:
    """The selected CRF changes retained detail in a real encoded video."""
    size = Size(width=33, height=31)
    errors = []
    with Image.new("RGB", size.as_tuple) as source:
        source.putdata(
            [
                ((index * 73) % 256, (index * 37) % 256, (index * 19) % 256)
                for index in range(size.width * size.height)
            ]
        )
        for crf in (0, 45):
            output = path_join(media_workspace, f"quality-{crf}.mp4")
            count = encode_frames(
                (source.copy() for _ in range(2)),
                output,
                size,
                10,
                Encoding(crf=crf, preset=EncodingPreset.SLOW),
            )
            assert count == 2
            with (
                closing(decode_frames(output, probe(output), 10)) as frames,
                next(frames) as frame,
                ImageChops.difference(source, frame) as difference,
            ):
                errors.append(sum(ImageStat.Stat(difference).mean))
    assert errors[0] < errors[1] / 4
