"""Owned analytical depth assets for typography projection behavior."""

from collections.abc import Generator

import pytest
from PIL import Image

from pic2panning.schemas.creative.depth import DepthArtifact, DepthMetadata
from pic2panning.tools.creative.depth import DepthAsset


@pytest.fixture
def table() -> Generator[DepthAsset]:
    """Provide a front-facing five-metre plane with calibrated pixel centers."""
    artifact = DepthArtifact(path="asset.png", sha256="0" * 64)
    document = DepthMetadata(
        source_sha256="0" * 64,
        width=160,
        height=120,
        grid_width=160,
        grid_height=120,
        intrinsics=((160, 0, 80), (0, 160, 60), (0, 0, 1)),
        depth=artifact,
        preview=artifact,
        normals=artifact,
        mask=artifact,
        preview_near_metres=5,
        preview_far_metres=5,
        device="cpu",
        num_tokens=1800,
        max_edge=256,
    )
    asset = DepthAsset(
        depth=Image.new("F", (160, 120), 5),
        mask=Image.new("L", (160, 120), 255),
        normals=Image.new("RGB", (160, 120), (128, 128, 0)),
        metadata=document,
    )
    try:
        yield asset
    finally:
        asset.close()
