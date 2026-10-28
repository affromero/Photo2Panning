"""Timed sentence transitions with explicit motion and readable holds."""

from typing import Annotated

from pydantic import Field

from pic2panning.schemas.base import PositiveFloat, PydanticBaseModel


class TextCue(PydanticBaseModel):
    """Morph from the preceding sentence at a layer-local time in seconds."""

    at: PositiveFloat
    """Start changing the preceding sentence; the first is TextLayer.text."""

    text: Annotated[str, Field(min_length=1, max_length=1024)]
    """Exact target sentence, wrapped using the layer's font and rectangle."""

    transition: Annotated[float, Field(gt=0, le=10)] = 0.8
    """Complete transition interval, including the stagger, in seconds."""

    stagger: Annotated[float, Field(ge=0, le=0.8)] = 0.25
    """Fraction of the transition used to stagger glyph starts left to right."""

    blur: Annotated[float, Field(ge=0, le=32)] = 1.5
    """Maximum intermediate glyph blur in source text pixels."""

    travel: Annotated[float, Field(ge=0, le=4)] = 0.6
    """Curved movement and entrance distance, measured in font sizes."""
