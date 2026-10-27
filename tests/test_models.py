"""Behavioral checks for model jobs without downloading model weights."""

import os
import subprocess
import sys
from pathlib import Path
from typing import override

import pytest
from PIL import Image
from pydantic import ValidationError

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
        if image is not None and image.mode != "RGB":
            raise ValueError("Model reference images must use RGB")
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


class CenterMaskBackend(MaskBackend):
    """A single detected pixel at the image center."""

    @override
    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Return a small selection with room to expand on every side."""
        selection = Image.new("L", image.size)
        selection.putpixel((image.width // 2, image.height // 2), 255)
        return (selection,)


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


@pytest.mark.parametrize("mode", ["RGBA", "P"])
@pytest.mark.parametrize("select_half", [False, True])
def test_masked_edit_retains_unselected_transparency_and_color(
    tmp_path: Path, mode: str, *, select_half: bool
) -> None:
    """Black masks retain every RGBA channel, including palette alpha."""
    source = tmp_path / "source.png"
    selection = tmp_path / "mask.png"
    output = tmp_path / "edited.png"
    with Image.new(mode, (32, 16)) as image:
        if mode == "P":
            image.putpalette([10, 20, 30, 40, 50, 60])
            image.info["transparency"] = bytes([0, 128])
            image.putpixel((31, 15), 1)
        else:
            image.paste((10, 20, 30, 0), (0, 0, 32, 16))
            image.putpixel((31, 15), (40, 50, 60, 128))
        image.save(source)
    with Image.new("L", (32, 16)) as selected:
        if select_half:
            selected.paste(255, (0, 0, 16, 16))
        selected.save(selection)
    original = source.read_bytes()
    generate(
        GenerateRequest(
            source=str(source),
            mask=str(selection),
            output=str(output),
            prompt="Make selected pixels red",
            size=Size(width=32, height=16),
        ),
        SolidImageBackend(),
    )
    with Image.open(output) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((31, 15)) == (40, 50, 60, 128)
        assert image.getpixel((16, 0)) == (10, 20, 30, 0)
        if select_half:
            assert image.getpixel((0, 0)) == (255, 0, 0, 255)
        else:
            with (
                Image.open(source) as original_image,
                original_image.convert("RGBA") as rgba,
            ):
                assert image.tobytes() == rgba.tobytes()
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
        assert image.getpixel((1, 0)) == 0
        assert image.getpixel((30, 15)) == 0
    assert result.outputs == (str(output),)


@pytest.mark.parametrize(
    ("source_size", "padding", "bounds", "selected_count"),
    [
        ((9, 9), 2, (2, 2, 7, 7), 25),
        ((5, 5), 128, (0, 0, 5, 5), 25),
    ],
)
def test_mask_padding_expands_the_selected_area_without_resizing(
    tmp_path: Path,
    source_size: tuple[int, int],
    padding: int,
    bounds: tuple[int, int, int, int],
    selected_count: int,
) -> None:
    """Padding includes surrounding pixels while retaining binary geometry."""
    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    with Image.new("RGB", source_size, "blue") as image:
        image.save(source)
    original = source.read_bytes()
    mask(
        MaskRequest(
            source=str(source),
            output=str(output),
            prompt="object",
            padding=padding,
        ),
        CenterMaskBackend(),
    )
    with Image.open(output) as selection:
        assert selection.mode == "L"
        assert selection.size == source_size
        assert selection.getbbox() == bounds
        assert selection.tobytes().count(255) == selected_count
        assert set(selection.tobytes()) <= {0, 255}
    assert source.read_bytes() == original


@pytest.mark.parametrize("merge", [False, True])
def test_padded_masks_clip_at_image_edges_in_both_output_modes(
    tmp_path: Path, *, merge: bool
) -> None:
    """Corner selections expand inward without adding image pixels."""
    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    with Image.new("RGB", (9, 9)) as image:
        image.save(source)
    result = mask(
        MaskRequest(
            source=str(source),
            output=str(output),
            prompt="objects",
            padding=2,
            merge=merge,
        ),
        SeparateMaskBackend(),
    )
    selected_counts = []
    for path in result.outputs:
        with Image.open(path) as selection:
            assert selection.size == (9, 9)
            assert set(selection.tobytes()) == {0, 255}
            assert selection.getpixel((4, 4)) == 0
            selected_counts.append(selection.tobytes().count(255))
    assert selected_counts == ([18] if merge else [9, 9])


@pytest.mark.parametrize("padding", [-1, 129, 1.5, True, "8"])
def test_mask_padding_rejects_invalid_pixel_expansion(padding: object) -> None:
    """Only bounded integer padding values reach the model tool."""
    with pytest.raises(ValidationError, match="padding"):
        MaskRequest.model_validate(
            {
                "source": "source.png",
                "output": "mask.png",
                "prompt": "object",
                "padding": padding,
            }
        )


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
