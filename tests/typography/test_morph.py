"""Sentence morphs retain readable endpoints and move actual glyph shapes."""

import gc
import weakref
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageStat
from pydantic import ValidationError

from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    LayerKeyframe,
    TextLayer,
)
from pic2panning.schemas.creative.typography import TextCue
from pic2panning.tools.typography.morph import TextAnimator
from pic2panning.tools.typography.text import graphemes, text_image


def text_layer(
    text: str = "KEEP THE COAST",
    cues: tuple[TextCue, ...] | None = None,
) -> TextLayer:
    """Create a text rectangle large enough for readable changing sentences."""
    return TextLayer(
        id="headline",
        text=text,
        text_cues=cues
        if cues is not None
        else (TextCue(at=1, text="KEEP THE LIGHT", transition=1),),
        duration=6,
        size=Size(width=900, height=240),
        keyframes=(LayerKeyframe(x=450, y=120),),
        font_size=76,
        color="#edf2d0",
        stroke_width=1,
        stroke_color="#edf2d0",
    )


def test_sentence_holds_and_transition_endpoints_match_static_text() -> None:
    """Both complete sentences stay exact and readable outside their transition."""
    layer = text_layer()
    animator = TextAnimator(layer)
    try:
        for seconds, sentence in (
            (0, layer.text),
            (0.8, layer.text),
            (1, layer.text),
            (2, "KEEP THE LIGHT"),
            (5.99, "KEEP THE LIGHT"),
        ):
            with (
                animator.frame(seconds) as actual,
                text_image(layer, sentence) as expected,
            ):
                assert actual.mode == "RGBA"
                assert actual.size == layer.size.as_tuple
                assert actual.tobytes() == expected.tobytes()
    finally:
        animator.close()


def test_intermediate_glyph_shapes_change_beyond_a_sentence_dissolve() -> None:
    """The transition changes spatial coverage instead of fading two bitmaps."""
    layer = text_layer("COAST", (TextCue(at=1, text="LIGHT", transition=1),))
    animator = TextAnimator(layer)
    try:
        with (
            animator.frame(1) as first,
            animator.frame(1.5) as middle,
            animator.frame(2) as last,
            Image.blend(first, last, 0.5) as dissolve,
            ImageChops.difference(middle, dissolve) as different,
        ):
            assert middle.tobytes() not in (first.tobytes(), last.tobytes())
            assert max(ImageStat.Stat(different).mean) > 2
            assert middle.getbbox() is not None
    finally:
        animator.close()


def test_a_repeated_sentence_stays_pixel_stable_through_its_cue() -> None:
    """Declaring the same sentence twice cannot introduce a rendering pulse."""
    layer = text_layer(cues=(TextCue(at=1, text="KEEP THE COAST"),))
    animator = TextAnimator(layer)
    try:
        with text_image(layer) as expected:
            for moment in (1, 1.1, 1.4, 1.7, 2):
                with animator.frame(moment) as actual:
                    assert actual.tobytes() == expected.tobytes()
    finally:
        animator.close()


def test_changed_glyphs_preserve_visible_ink_through_the_midpoint() -> None:
    """An O becoming an I remains a solid moving letter instead of fragments."""
    layer = text_layer(
        "O",
        (TextCue(at=1, text="I", transition=1, stagger=0, blur=0, travel=0),),
    ).model_copy(update={"stroke_width": 0})
    animator = TextAnimator(layer)
    try:
        with (
            animator.frame(1) as first,
            animator.frame(1.5) as middle,
            animator.frame(2) as last,
        ):
            masses = []
            for image in (first, middle, last):
                with image.getchannel("A") as alpha:
                    masses.append(sum(ImageStat.Stat(alpha).sum) / 255)
            expected = (masses[0] + masses[2]) / 2
            assert expected * 0.85 < masses[1] < expected * 1.2
    finally:
        animator.close()


def test_nonmonotone_preview_times_reproduce_the_same_morph_pixels() -> None:
    """Scrubbing across several sentences does not alter a repeated frame."""
    layer = text_layer(
        cues=(
            TextCue(at=1, text="KEEP THE LIGHT", transition=0.8),
            TextCue(at=3, text="TAKE THE LONG WAY HOME", transition=1),
        )
    )
    animator = TextAnimator(layer)
    try:
        with animator.frame(1.4) as original:
            pixels = original.tobytes()
        for seconds in (5, 0, 3.5, 2, 1.01, 5.9):
            with animator.frame(seconds) as frame:
                assert frame.size == layer.size.as_tuple
        with animator.frame(1.4) as repeated:
            assert repeated.tobytes() == pixels
    finally:
        animator.close()


@pytest.mark.parametrize(
    ("first", "last"),
    [
        ("A", "A!"),
        ("STAY HERE", "GO"),
        ("MOMENT", "MOVEMENT"),
        ("ÉTÉ", "CAFÉ"),
        ("WAIT...", "GO!"),
    ],
)
def test_insertions_deletions_punctuation_and_accents_finish_readably(
    first: str, last: str
) -> None:
    """Different sentence lengths and repeated letters retain exact targets."""
    layer = text_layer(first, (TextCue(at=1, text=last, transition=1),))
    animator = TextAnimator(layer)
    try:
        for seconds in (1.1, 1.4, 1.7, 1.9):
            with animator.frame(seconds) as image:
                assert image.getbbox() is not None
        with animator.frame(2) as image, text_image(layer, last) as expected:
            assert image.tobytes() == expected.tobytes()
    finally:
        animator.close()


def test_grapheme_groups_keep_accents_emoji_and_flags_intact() -> None:
    """A moving glyph never splits a combining mark, modifier, or ZWJ run."""
    assert tuple(graphemes("E\u0301 👩🏽\u200d🚀 🇨🇴")) == (
        "E\u0301",
        " ",
        "👩🏽\u200d🚀",
        " ",
        "🇨🇴",
    )


def test_returned_text_frames_survive_animator_cleanup_without_retention() -> (
    None
):
    """Each returned frame belongs to its caller and no timeline frame is cached."""
    animator = TextAnimator(text_layer())
    frames: list[weakref.ReferenceType[Image.Image]] = []
    for index in range(20):
        frame = animator.frame(1.05 + index / 25)
        frames.append(weakref.ref(frame))
        frame.close()
    del frame
    gc.collect()
    assert all(reference() is None for reference in frames)
    held = animator.frame(3)
    animator.close()
    try:
        assert held.getbbox() is not None
    finally:
        held.close()


@pytest.mark.parametrize("seconds", [-1, 6.01, float("nan"), float("inf")])
def test_text_sampling_rejects_times_outside_its_finite_duration(
    seconds: float,
) -> None:
    """Invalid sample times cannot trigger unbounded glyph work."""
    animator = TextAnimator(text_layer())
    try:
        with pytest.raises(ValueError, match="time"):
            animator.frame(seconds)
    finally:
        animator.close()


@pytest.mark.parametrize(
    "cues",
    [
        (
            TextCue(at=1, text="SECOND", transition=2),
            TextCue(at=2, text="THIRD", transition=1),
        ),
        (TextCue(at=5.5, text="TOO LATE", transition=1),),
        (
            TextCue(at=3, text="THIRD"),
            TextCue(at=1, text="SECOND"),
        ),
    ],
)
def test_overlapping_unordered_and_out_of_bounds_cues_fail_validation(
    cues: tuple[TextCue, ...],
) -> None:
    """A sentence's readable interval is determined before a renderer starts."""
    with pytest.raises(ValidationError):
        text_layer(cues=cues)


def test_missing_explicit_font_surfaces_before_sentence_rendering(
    tmp_path: Path,
) -> None:
    """A missing explicit font surfaces before an empty replacement can render."""
    layer = text_layer().model_copy(
        update={"font": str(tmp_path / "missing.ttf")}
    )
    animator = TextAnimator(layer)
    try:
        with pytest.raises(FileNotFoundError):
            animator.frame(1.5)
    finally:
        animator.close()
