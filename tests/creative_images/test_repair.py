"""Verify contextual repairs through real processing and model boundaries."""

from pathlib import Path
from typing import override

import pytest
from PIL import Image
from pydantic import ValidationError

from pic2panning.backends.contracts import ImageBackend
from pic2panning.schemas.creative.repair import RepairRequest
from pic2panning.schemas.editing import Crop
from pic2panning.schemas.models import GenerateRequest
from pic2panning.tools.creative.repair import repair


class RepairBackend(ImageBackend):
    """Observe contextual model conditioning and return a known repair."""

    conditioning: Image.Image | None
    options: GenerateRequest | None
    fail: bool

    def __init__(self, *, fail: bool = False) -> None:
        """Keep observable inputs at the external inference boundary."""
        self.conditioning = None
        self.options = None
        self.fail = fail

    @override
    def generate(
        self,
        request: GenerateRequest,
        image: Image.Image | None,
        mask: Image.Image | None,
    ) -> Image.Image:
        """Return red pixels independently of selection compositing."""
        if self.fail:
            raise RuntimeError("Repair device failed")
        if image is None:
            raise ValueError("Repair requires a contextual reference")
        self.conditioning = image.copy()
        self.options = request
        return Image.new("RGB", request.size.as_tuple, "red")


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_repair_enlarges_context_and_preserves_every_pixel_outside_selection(
    tmp_path: Path, mode: str
) -> None:
    """Repair uses nearby content and retains source geometry and alpha."""
    source = tmp_path / "source.png"
    output = tmp_path / "repaired.png"
    with Image.new(mode, (80, 40), (10, 20, 30)) as original:
        if mode == "RGBA":
            original.putalpha(128)
        original.save(source)
        before = original.tobytes()
    backend = RepairBackend()
    result = repair(
        RepairRequest(
            source=str(source),
            output=str(output),
            region=Crop(left=30, top=10, right=50, bottom=30),
            prompt="Repair the selected area",
            context=10,
            max_edge=128,
            seed=19,
            steps=3,
        ),
        backend,
    )
    with Image.open(output) as image:
        assert image.size == (80, 40)
        assert image.mode == mode
        assert image.getpixel((40, 20)) == (
            (255, 0, 0, 255) if mode == "RGBA" else (255, 0, 0)
        )
        with Image.open(source) as original:
            assert original.tobytes() == before
            for y in range(image.height):
                for x in range(image.width):
                    if not (30 <= x < 50 and 10 <= y < 30):
                        assert image.getpixel((x, y)) == original.getpixel(
                            (x, y)
                        )
    assert result.outputs == (str(output),)
    assert backend.options is not None
    assert backend.options.seed == 19
    assert backend.options.steps == 3
    assert backend.options.size.as_tuple == (128, 128)
    assert backend.conditioning is not None
    with backend.conditioning as reference:
        assert reference.getpixel((0, 0)) == (10, 20, 30)
        assert reference.size == (128, 128)


def test_repair_pads_model_image_without_stretching_context(
    tmp_path: Path,
) -> None:
    """A narrow contextual crop receives padding after proportional resize."""
    source = tmp_path / "source.png"
    with Image.new("RGB", (100, 37), "blue") as image:
        image.save(source)
    backend = RepairBackend()
    repair(
        RepairRequest(
            source=str(source),
            output=str(tmp_path / "result.png"),
            region=Crop(left=0, top=0, right=100, bottom=37),
            prompt="Repair",
            context=0,
            max_edge=128,
        ),
        backend,
    )
    assert backend.conditioning is not None
    with backend.conditioning as reference:
        assert reference.size == (128, 48)
        assert reference.getpixel((100, 46)) == (0, 0, 255)
        assert reference.getpixel((100, 47)) == (0, 0, 0)
    with Image.open(tmp_path / "result.png") as image:
        assert image.size == (100, 37)
        assert image.getpixel((99, 36)) == (255, 0, 0)


def test_repair_feathers_only_inside_the_selected_mask(tmp_path: Path) -> None:
    """A full-source mask limits repair and feathering never grows its extent."""
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    output = tmp_path / "repaired.png"
    with Image.new("RGB", (64, 64), "blue") as image:
        image.save(source)
    with Image.new("L", (64, 64)) as selection:
        selection.paste(255, (20, 20, 44, 44))
        selection.save(mask)
    repair(
        RepairRequest(
            source=str(source),
            output=str(output),
            mask=str(mask),
            region=Crop(left=10, top=10, right=54, bottom=54),
            prompt="Repair",
            feather=2,
            max_edge=64,
        ),
        RepairBackend(),
    )
    with Image.open(output) as image:
        assert image.getpixel((32, 32)) == (255, 0, 0)
        edge = image.getpixel((21, 32))
        assert isinstance(edge, tuple)
        assert 0 < edge[0] < 255
        for y in range(64):
            for x in range(64):
                if not (20 <= x < 44 and 20 <= y < 44):
                    assert image.getpixel((x, y)) == (0, 0, 255)


@pytest.mark.parametrize("opacity", [64, 128])
def test_feather_preserves_partial_mask_strength_away_from_its_edges(
    tmp_path: Path, opacity: int
) -> None:
    """Grayscale weights retain their strength while geometric edges soften."""
    source, mask = tmp_path / "source.png", tmp_path / "mask.png"
    output = tmp_path / "repaired.png"
    with Image.new("RGB", (64, 64), "blue") as image:
        image.save(source)
    with Image.new("L", (64, 64)) as selection:
        selection.paste(opacity, (16, 16, 48, 48))
        selection.save(mask)
    repair(
        RepairRequest(
            source=str(source),
            output=str(output),
            mask=str(mask),
            region=Crop(left=0, top=0, right=64, bottom=64),
            prompt="Repair",
            feather=2,
            max_edge=64,
        ),
        RepairBackend(),
    )
    with Image.open(output) as image:
        assert image.getpixel((32, 32)) == (opacity, 0, 255 - opacity)
        boundary = image.getpixel((17, 32))
        assert isinstance(boundary, tuple)
        assert 0 < boundary[0] < opacity
        assert image.getpixel((15, 32)) == (0, 0, 255)


@pytest.mark.parametrize("problem", ["bounds", "mask-size", "empty-mask"])
def test_invalid_repair_geometry_fails_before_model_work(
    tmp_path: Path, problem: str
) -> None:
    """Invalid selections do not reach inference or leave partial files."""
    source = tmp_path / "source.png"
    selection = tmp_path / "mask.png"
    output = tmp_path / "output.png"
    with Image.new("RGB", (32, 16), "blue") as image:
        image.save(source)
    with Image.new(
        "L", (16, 16) if problem == "mask-size" else (32, 16)
    ) as mask:
        mask.save(selection)
    with pytest.raises(ValueError, match=r"region|mask|selected"):
        repair(
            RepairRequest(
                source=str(source),
                output=str(output),
                mask=str(selection) if problem != "bounds" else None,
                region=Crop(
                    left=0,
                    top=0,
                    right=33 if problem == "bounds" else 32,
                    bottom=16,
                ),
                prompt="Repair",
            ),
            RepairBackend(fail=True),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".pic2panning-*"))


def test_repair_provider_failure_preserves_existing_output(
    tmp_path: Path,
) -> None:
    """An inference error leaves an existing destination and source intact."""
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    output.write_bytes(b"Existing output")
    with Image.new("RGB", (32, 16), "blue") as image:
        image.save(source)
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="device failed"):
        repair(
            RepairRequest(
                source=str(source),
                output=str(output),
                region=Crop(left=2, top=2, right=20, bottom=12),
                prompt="Repair",
                overwrite=True,
            ),
            RepairBackend(fail=True),
        )
    assert source.read_bytes() == original
    assert output.read_bytes() == b"Existing output"
    assert not list(tmp_path.glob(".pic2panning-*"))


def test_repair_rejects_source_alias_before_inference(tmp_path: Path) -> None:
    """Even explicit overwrite cannot replace the original image."""
    source = tmp_path / "source.png"
    with Image.new("RGB", (32, 16), "blue") as image:
        image.save(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="aliases"):
        repair(
            RepairRequest(
                source=str(source),
                output=str(source),
                region=Crop(left=2, top=2, right=20, bottom=12),
                prompt="Repair",
                overwrite=True,
            ),
            RepairBackend(fail=True),
        )
    assert source.read_bytes() == original


@pytest.mark.parametrize("options", [{"output": "out.jpg"}, {"max_edge": 31}])
def test_repair_rejects_lossy_outputs_and_unaligned_model_limits(
    options: dict[str, object],
) -> None:
    """Invalid format and inference dimensions fail during validation."""
    with pytest.raises(ValidationError):
        RepairRequest.model_validate(
            {
                "source": "source.png",
                "output": "out.png",
                "region": {"left": 0, "top": 0, "right": 20, "bottom": 10},
                "prompt": "Repair",
                **options,
            }
        )
