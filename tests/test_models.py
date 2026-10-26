"""Behavioral checks for model jobs without downloading model weights."""

import os
import subprocess
import sys
from pathlib import Path
from typing import override

import pytest
from PIL import Image

from pic2panning.backends.contracts import ImageBackend, MaskBackend
from pic2panning.errors import MediaError
from pic2panning.schemas.base import Size
from pic2panning.schemas.models import GenerateRequest, MaskRequest
from pic2panning.tools.models import generate, mask


class SolidImageBackend(ImageBackend):
    """A local inference boundary that produces a predictable red image."""

    @override
    def generate(
        self,
        request: GenerateRequest,
        image: Image.Image | None,
        mask: Image.Image | None,
    ) -> Image.Image:
        """Return the requested dimensions for output verification."""
        return Image.new("RGB", request.size.as_tuple, "red")


class FailingImageBackend(ImageBackend):
    """A failed inference boundary used to check error propagation."""

    @override
    def generate(
        self,
        request: GenerateRequest,
        image: Image.Image | None,
        mask: Image.Image | None,
    ) -> Image.Image:
        """Surface the actual provider error."""
        message = "Inference device disconnected"
        raise OSError(message)


class SeparateMaskBackend(MaskBackend):
    """Two detected objects at opposite image corners."""

    @override
    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Return separate masks with known selected pixels."""
        first = Image.new("L", image.size)
        first.putpixel((0, 0), 255)
        second = Image.new("L", image.size)
        second.putpixel((image.width - 1, image.height - 1), 255)
        return first, second


class EmptyMaskBackend(MaskBackend):
    """An inference boundary finding no matching objects."""

    @override
    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Report an empty detection result."""
        return ()


class BlankMaskBackend(MaskBackend):
    """An invalid inference result selecting no pixels for an object."""

    @override
    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Return a blank mask to verify detection failures are surfaced."""
        return (Image.new("L", image.size),)


def test_generation_saves_model_output_at_requested_dimensions(
    tmp_path: Path,
) -> None:
    """Model results become a saved image with meaningful metadata."""
    output = tmp_path / "generated.png"
    result = generate(
        GenerateRequest(
            output=str(output),
            prompt="A red square",
            size=Size(width=32, height=16),
        ),
        SolidImageBackend(),
    )
    with Image.open(output) as image:
        assert image.size == (32, 16)
        assert image.getpixel((0, 0)) == (255, 0, 0)
    assert result.outputs == (str(output),)
    assert result.width == 32


def test_masked_edit_preserves_every_unselected_pixel(tmp_path: Path) -> None:
    """An edit changes only pixels selected by the input mask."""
    source = tmp_path / "source.png"
    selection = tmp_path / "mask.png"
    output = tmp_path / "edited.png"
    Image.new("RGB", (32, 16), "blue").save(source)
    selected = Image.new("L", (32, 16))
    selected.paste(255, (0, 0, 16, 16))
    selected.save(selection)
    original = source.read_bytes()
    generate(
        GenerateRequest(
            source=str(source),
            mask=str(selection),
            output=str(output),
            prompt="Make the selected half red",
            size=Size(width=32, height=16),
        ),
        SolidImageBackend(),
    )
    with Image.open(output) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((31, 15)) == (0, 0, 255)
    assert source.read_bytes() == original


def test_masked_edit_rejects_geometry_changes(tmp_path: Path) -> None:
    """Unselected pixels are never resized during a masked edit."""
    source = tmp_path / "source.png"
    selection = tmp_path / "mask.png"
    Image.new("RGB", (32, 16)).save(source)
    Image.new("L", (32, 16)).save(selection)
    with pytest.raises(MediaError, match="dimensions"):
        generate(
            GenerateRequest(
                source=str(source),
                mask=str(selection),
                output=str(tmp_path / "out.png"),
                prompt="Edit",
                size=Size(width=16, height=16),
            ),
            FailingImageBackend(),
        )


def test_provider_failure_leaves_no_output(tmp_path: Path) -> None:
    """A failed model call preserves the provider error and cleans up files."""
    output = tmp_path / "output.png"
    with pytest.raises(OSError, match="disconnected"):
        generate(
            GenerateRequest(output=str(output), prompt="An image"),
            FailingImageBackend(),
        )
    assert list(tmp_path.iterdir()) == []


def test_mask_job_merges_matching_objects(tmp_path: Path) -> None:
    """A merged mask selects pixels from every detected object."""
    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    Image.new("RGB", (32, 16)).save(source)
    result = mask(
        MaskRequest(source=str(source), output=str(output), prompt="objects"),
        SeparateMaskBackend(),
    )
    with Image.open(output) as image:
        assert image.getpixel((0, 0)) == 255
        assert image.getpixel((31, 15)) == 255
        assert image.getpixel((16, 8)) == 0
    assert result.outputs == (str(output),)


def test_mask_job_saves_separate_objects_without_merging(
    tmp_path: Path,
) -> None:
    """Separate output files retain each detected object's selection."""
    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    output.write_bytes(b"An existing unrelated artifact")
    Image.new("RGB", (32, 16)).save(source)
    result = mask(
        MaskRequest(
            source=str(source),
            output=str(output),
            prompt="objects",
            merge=False,
        ),
        SeparateMaskBackend(),
    )
    assert len(result.outputs) == 2
    assert output.read_bytes() == b"An existing unrelated artifact"
    with (
        Image.open(result.outputs[0]) as first,
        Image.open(result.outputs[1]) as second,
    ):
        assert first.getpixel((0, 0)) == 255
        assert first.getpixel((31, 15)) == 0
        assert second.getpixel((0, 0)) == 0
        assert second.getpixel((31, 15)) == 255


def test_no_detections_explains_failure_without_writing_mask(
    tmp_path: Path,
) -> None:
    """An empty model result is a clear error instead of a blank mask."""
    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    Image.new("RGB", (32, 16)).save(source)
    with pytest.raises(MediaError, match="No objects"):
        mask(
            MaskRequest(
                source=str(source), output=str(output), prompt="objects"
            ),
            EmptyMaskBackend(),
        )
    assert not output.exists()


def test_empty_predicted_mask_is_reported_as_model_failure(
    tmp_path: Path,
) -> None:
    """A detected object with no selected pixels never reports success."""
    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    Image.new("RGB", (32, 16)).save(source)
    with pytest.raises(MediaError, match="empty mask"):
        mask(
            MaskRequest(
                source=str(source), output=str(output), prompt="truck"
            ),
            BlankMaskBackend(),
        )
    assert not output.exists()


@pytest.mark.skipif(
    os.environ.get("PIC2PANNING_TEST_MODELS") != "1",
    reason="Set PIC2PANNING_TEST_MODELS=1 to check the optional model runtime",
)
def test_model_runtime_handles_cpu_tensors_in_a_background_worker() -> None:
    """Model jobs can safely run in the worker used by an internal LLM."""
    program = """
from concurrent.futures import ThreadPoolExecutor
import torch
value = torch.arange(10)
with ThreadPoolExecutor(max_workers=1) as executor:
    result = executor.submit(value.to, device="cpu").result()
assert result.tolist() == list(range(10))
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
