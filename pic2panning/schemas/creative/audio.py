"""Explicit soundtrack mixing and CPU beat analysis contracts."""

from typing import Annotated, Literal, Self

from klogr.path import get_suffix
from pydantic import Field, model_validator

from pic2panning.schemas.base import (
    NonEmptyString,
    NonNegativeFloat,
    PositiveFloat,
    PydanticBaseModel,
)


class Ducking(PydanticBaseModel):
    """Compress a music or ambience track using the audible voice mix."""

    threshold_db: Annotated[float, Field(ge=-60, le=0)] = -24
    ratio: Annotated[float, Field(ge=1, le=20)] = 6
    attack_ms: Annotated[float, Field(ge=0.01, le=2000)] = 20
    release_ms: Annotated[float, Field(ge=0.01, le=9000)] = 250


class MixTrack(PydanticBaseModel):
    """One explicitly selected audio source, independent of video layers."""

    source: NonEmptyString
    role: Literal["music", "voice", "ambience"] = "music"
    start: NonNegativeFloat = 0
    source_start: NonNegativeFloat = 0
    duration: PositiveFloat
    loop: bool = False
    gain_db: Annotated[float, Field(ge=-120, le=36)] = 0
    fade_in: NonNegativeFloat = 0
    fade_out: NonNegativeFloat = 0
    ducking: Ducking | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        """Keep fades within the track and voice out of its own sidechain."""
        if self.fade_in + self.fade_out > self.duration:
            raise ValueError("Audio fades must fit within track duration")
        if self.role == "voice" and self.ducking is not None:
            raise ValueError("Voice tracks cannot duck themselves")
        return self


class BeatRequest(PydanticBaseModel):
    """Analyze audio on the CPU and publish onset times in a JSON file."""

    kind: Literal["beats"] = "beats"
    source: NonEmptyString
    output: NonEmptyString
    start: NonNegativeFloat = 0
    duration: PositiveFloat | None = None
    sensitivity: Annotated[float, Field(gt=0, le=10)] = 1.5
    minimum_gap: Annotated[float, Field(ge=0.05, le=10)] = 0.2
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        """Require a JSON destination before decoding audio."""
        if get_suffix(self.output).lower() != ".json":
            raise ValueError("Beat output must end in .json")
        return self


class BeatAnalysis(PydanticBaseModel):
    """Seconds relative to the analyzed selection, with an estimated tempo."""

    markers: tuple[NonNegativeFloat, ...]
    duration: NonNegativeFloat
    bpm: PositiveFloat | None = None
