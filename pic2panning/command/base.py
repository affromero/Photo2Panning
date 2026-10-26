"""Common explicit output options for typed terminal commands."""

from pic2panning.schemas.base import PydanticBaseModel


class OutputCommand(PydanticBaseModel):
    """Require an output path and explicit consent to replace an artifact."""

    output: str
    """Destination file; its suffix selects the output format."""

    overwrite: bool = False
    """Replace an existing output while preserving every input file."""
