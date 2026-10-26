"""Requests reject invalid jobs before any file or model is accessed."""

import json

import pytest
from pydantic import JsonValue, ValidationError

from pic2panning import (
    JOB_ADAPTER,
    AudioTrack,
    GenerateRequest,
    GenerationModel,
    ImageClip,
    MaskRequest,
    RenderRequest,
    Size,
    job_schema,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "render", "clips": [], "output": "out.mp4"},
        {
            "kind": "render",
            "clips": [{"source": "photo.png", "kind": "image", "duration": 0}],
            "output": "out.mp4",
        },
        {
            "kind": "render",
            "clips": [
                {"source": "photo.png", "kind": "image", "motion": "diagonal"}
            ],
            "output": "out.mp4",
        },
        {
            "kind": "render",
            "clips": [{"source": "photo.png", "kind": "image", "duraton": 2}],
            "output": "out.mp4",
        },
        {
            "kind": "render",
            "clips": [{"source": "photo.png", "kind": "image"}],
            "output": "out.mp4",
            "fps": 0,
        },
        {
            "kind": "render",
            "clips": [{"source": "photo.png", "kind": "image"}],
            "output": "out.mp4",
            "size": {"width": 101, "height": 100},
        },
        {
            "kind": "edit",
            "source": "photo.png",
            "output": "out.png",
            "operations": [
                {"op": "crop", "left": 10, "top": 0, "right": 5, "bottom": 5}
            ],
        },
        {
            "kind": "generate",
            "output": "out.png",
            "prompt": "coast",
            "size": {"width": 100, "height": 100},
        },
        {
            "kind": "mask",
            "source": "photo.png",
            "output": "out.png",
            "prompt": "person",
            "device": "cuda",
        },
    ],
)
def test_invalid_jobs_fail_without_loading_inputs(
    payload: dict[str, JsonValue],
) -> None:
    """Malformed options cannot reach a media or model operation."""
    with pytest.raises(ValidationError):
        JOB_ADAPTER.validate_python(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_durations_are_rejected(value: float) -> None:
    """Invalid time values cannot become infinite loops or frame counts."""
    with pytest.raises(ValidationError):
        ImageClip(source="photo.png", duration=value)


def test_json_round_trip_preserves_the_complete_timeline() -> None:
    """An agent can serialize a job and recover its validated semantics."""
    request = RenderRequest(
        output="out.mp4",
        clips=(ImageClip(source="photo.png", duration=1.5, reverse=True),),
        size=Size(width=1920, height=1080),
        fps=60,
        audio=AudioTrack(source="track.wav", start=2.5, loop=True),
    )
    recovered = JOB_ADAPTER.validate_json(request.model_dump_json())
    assert recovered == request


def test_global_and_per_clip_audio_cannot_conflict() -> None:
    """Audio ownership must be explicit for the whole timeline."""
    with pytest.raises(ValidationError, match="global audio"):
        RenderRequest(
            output="out.mp4",
            clips=(
                ImageClip(
                    source="photo.png", audio=AudioTrack(source="first.wav")
                ),
            ),
            audio=AudioTrack(source="second.wav"),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"guidance": 3.5},
        {"strength": 0.5},
        {"mask": "mask.png"},
    ],
)
def test_unsupported_generation_options_fail_before_loading_models(
    options: dict[str, JsonValue],
) -> None:
    """An option is never silently ignored by the selected model."""
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate(
            {"output": "out.png", "prompt": "a coast", **options}
        )


def test_fill_requires_source_and_mask() -> None:
    """The conditioned fill model cannot run without its inputs."""
    with pytest.raises(ValidationError, match="source and mask"):
        GenerateRequest(
            output="out.png", prompt="coast", model=GenerationModel.FLUX1_FILL
        )


@pytest.mark.parametrize("kind", ["edit", "generate", "mask"])
def test_unsupported_image_outputs_are_rejected(kind: str) -> None:
    """Bad output formats fail before image reading or inference."""
    payload: dict[str, JsonValue] = {
        "kind": kind,
        "source": "photo.png",
        "output": "out.mp4",
    }
    if kind == "edit":
        payload["operations"] = [{"op": "rotate", "degrees": 90}]
    else:
        payload["prompt"] = "a coast"
    with pytest.raises(ValidationError):
        JOB_ADAPTER.validate_python(payload)


def test_masks_require_a_lossless_output() -> None:
    """Mask pixels survive saving without JPEG artifacts."""
    with pytest.raises(ValidationError, match="png"):
        MaskRequest(source="photo.png", output="mask.jpg", prompt="person")


@pytest.mark.parametrize("output", ["mp4", "gif", ".mp4", "out.avi"])
def test_motion_outputs_require_a_real_supported_file_suffix(
    output: str,
) -> None:
    """A video filename cannot be mistaken for a different encoding format."""
    with pytest.raises(ValidationError, match="end in"):
        RenderRequest(output=output, clips=(ImageClip(source="photo.png"),))


@pytest.mark.parametrize("output", ["out.jpg", "out.webp"])
def test_masked_generation_requires_lossless_png(output: str) -> None:
    """Unselected source pixels cannot be recompressed during an edit."""
    with pytest.raises(ValidationError, match="png"):
        GenerateRequest(
            source="photo.png", mask="mask.png", output=output, prompt="coast"
        )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (GenerationModel.FLUX3, 16),
        (GenerationModel.FLUX1_SCHNELL, 4),
        (GenerationModel.FLUX1_DEV, 28),
        (GenerationModel.FLUX1_FILL, 28),
    ],
)
def test_generation_uses_model_appropriate_steps_unless_explicit(
    model: GenerationModel, expected: int
) -> None:
    """Selecting a model selects useful defaults and retains explicit settings."""
    request = GenerateRequest(
        source="photo.png",
        mask="mask.png",
        output="out.png",
        prompt="coast",
        model=model,
    )
    assert request.inference_steps == expected
    explicit = GenerateRequest(
        source="photo.png",
        mask="mask.png",
        output="out.png",
        prompt="coast",
        model=model,
        steps=7,
    )
    assert explicit.inference_steps == 7


def test_schema_contains_the_discriminator_and_field_meanings() -> None:
    """Schema discovery gives an agent choices, constraints, and descriptions."""
    encoded = json.dumps(job_schema())
    for semantic in (
        "discriminator",
        "render",
        "generate",
        "minimum",
        "description",
        "quantization",
        "overwrite",
    ):
        assert semantic in encoded
