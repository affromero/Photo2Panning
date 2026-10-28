"""Composition validation fails before filesystem or media processing begins."""

import pytest
from pydantic import ValidationError

from pic2panning.schemas.creative.audio import MixTrack
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    Cinemagraph,
    ComposeRequest,
)


def test_camera_times_and_duplicate_layer_ids_are_rejected(
    composition: ComposeRequest,
) -> None:
    """Ambiguous keyframe order and occlusion IDs cannot reach rendering."""
    document = composition.model_dump()
    document["camera"] = [CameraKeyframe(at=0.5, x=100, y=50).model_dump()]
    with pytest.raises(ValidationError, match="Camera times"):
        ComposeRequest.model_validate(document)
    document = composition.model_dump()
    document["layers"] = [document["layers"][0], document["layers"][0]]
    with pytest.raises(ValidationError, match="unique"):
        ComposeRequest.model_validate(document)


def test_cinemagraph_requires_explicit_reference_and_fades_fit() -> None:
    """Selective motion never invents a clean plate or truncates an envelope."""
    with pytest.raises(ValidationError, match="reference"):
        Cinemagraph(mask="mask.png")
    with pytest.raises(ValidationError, match="fades"):
        MixTrack(source="audio.wav", duration=1, fade_in=0.7, fade_out=0.7)


def test_mp4_composition_requires_opaque_background(
    composition: ComposeRequest,
) -> None:
    """A video export cannot silently discard background transparency."""
    document = composition.model_dump()
    document["background"] = "#00000080"
    with pytest.raises(ValidationError, match="opaque"):
        ComposeRequest.model_validate(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shutter_angle", -1),
        ("shutter_angle", 361),
        ("shutter_samples", 0),
        ("shutter_samples", 17),
    ],
)
def test_composition_shutter_controls_are_bounded(
    composition: ComposeRequest, field: str, value: int
) -> None:
    """Invalid exposure settings cannot start media decoding or rendering."""
    document = composition.model_dump()
    document[field] = value
    with pytest.raises(ValidationError):
        ComposeRequest.model_validate(document)
