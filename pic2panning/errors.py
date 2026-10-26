"""Actionable failures at media and model boundaries."""


class MediaError(RuntimeError):
    """A media decoder, encoder, or processing backend failed."""


class DependencyError(RuntimeError):
    """An optional dependency or required system executable is missing."""
