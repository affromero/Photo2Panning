"""Cached metric depth drives visible effects without model execution."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from pic2panning.backends.creative.depth import DepthPrediction
from pic2panning.schemas.creative.depth import DepthRequest, DepthStyle
from pic2panning.tools.creative.depth import extract_depth, load_depth
from pic2panning.tools.creative.effects import (
    apply_depth_style,
    depth_mask,
    rescale_depth_style,
)


class EffectBoundary:
    """Supply a constant or two-plane metric field at the model boundary."""

    near: float
    far: float
    sky: bool

    def __init__(
        self, near: float = 3, far: float = 3, *, sky: bool = False
    ) -> None:
        """Select analytical depths without replacing effect processing."""
        self.near = near
        self.far = far
        self.sky = sky

    def predict(
        self, request: DepthRequest, image: Image.Image
    ) -> DepthPrediction:
        """Return a front-facing plane or one vertical depth discontinuity."""
        if request.device != "cpu":
            raise ValueError("Synthetic boundary requires explicit CPU work")
        width, height = image.size
        depth = Image.new("F", image.size)
        depth.putdata(
            [
                self.near if column < width // 2 else self.far
                for _ in range(height)
                for column in range(width)
            ]
        )
        mask = Image.new("L", image.size, "white")
        if self.sky:
            mask.paste(0, (width // 2, 0, width, height))
        return DepthPrediction(
            depth=depth,
            mask=mask,
            intrinsics=(
                (float(width), 0, width / 2),
                (0, float(height), height / 2),
                (0, 0, 1),
            ),
        )


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """Write a checkerboard whose fine detail makes focus changes visible."""
    source = tmp_path / "checker.png"
    with Image.new("RGB", (64, 32)) as image:
        image.putdata(
            [
                ((255,) * 3 if (column // 2 + row // 2) % 2 else (0,) * 3)
                for row in range(image.height)
                for column in range(image.width)
            ]
        )
        image.save(source)
    return source


def request_for(source: Path) -> DepthRequest:
    """Extract alongside a synthetic test source with CPU explicitly set."""
    return DepthRequest(
        source=str(source),
        output=str(source.with_suffix(".tiff")),
        device="cpu",
    )


def test_depth_mask_selects_nearer_geometry_with_metric_feather() -> None:
    """White pixels occlude an overlay and the feather crosses its plane."""
    with Image.new("F", (5, 1)) as depth:
        depth.putdata([1, 2, 3, 4, 5])
        with depth_mask(depth, 3, 0) as hard:
            np.testing.assert_array_equal(
                np.asarray(hard)[0], [255, 255, 0, 0, 0]
            )
        with depth_mask(depth, 3, 2) as feathered:
            values = np.asarray(feathered)[0]
            assert values[0] == 255
            assert values[-1] == 0
            assert 110 <= values[2] <= 145
            assert all(
                int(left) >= int(right) for left, right in pairwise(values)
            )


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_effects_reject_invalid_metric_pixels(invalid: float) -> None:
    """Bad depth cannot silently become a valid-looking occlusion mask."""
    with (
        Image.new("F", (2, 2), invalid) as depth,
        pytest.raises(ValueError, match=r"depth|finite|positive"),
    ):
        depth_mask(depth, 3)


def test_flat_depth_exports_normals_facing_the_camera(photo: Path) -> None:
    """A frontoparallel plane encodes negative camera Z in the blue channel."""
    result = extract_depth(request_for(photo), EffectBoundary())
    with Image.open(result.normals) as normals:
        values = np.asarray(normals)
        assert bool(np.all(np.abs(values[..., :2].astype(int) - 128) <= 1))
        assert int(values[..., 2].max()) == 0


def test_neutral_style_returns_independent_unchanged_pixels(
    photo: Path,
) -> None:
    """A cached asset can be previewed repeatedly without mutating inputs."""
    result = extract_depth(request_for(photo), EffectBoundary())
    with (
        load_depth(result.metadata, str(photo)) as asset,
        Image.open(photo) as source,
    ):
        original = np.asarray(source).copy()
        with apply_depth_style(
            source, asset, DepthStyle(metadata=result.metadata), 0.5
        ) as output:
            assert output.mode == "RGB"
            assert output is not source
            np.testing.assert_array_equal(np.asarray(output), original)
        np.testing.assert_array_equal(np.asarray(source), original)


def test_fog_increases_with_metric_distance(photo: Path) -> None:
    """Identical black pixels fade more strongly at the farther plane."""
    result = extract_depth(request_for(photo), EffectBoundary(1, 4))
    style = DepthStyle(metadata=result.metadata, fog_density=0.4)
    with (
        load_depth(result.metadata, str(photo)) as asset,
        Image.new("RGB", (64, 32), "black") as source,
        apply_depth_style(source, asset, style, 0) as output,
    ):
        pixels = np.asarray(output)
        assert 0 < int(pixels[16, 8, 0]) < int(pixels[16, 56, 0]) < 255
        np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
        np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])


def test_rack_focus_moves_detail_between_two_depth_planes(photo: Path) -> None:
    """The selected focus plane remains sharp as the other plane blurs."""
    result = extract_depth(request_for(photo), EffectBoundary(1, 4))
    style = DepthStyle(
        metadata=result.metadata,
        focus_metres=1,
        focus_end_metres=4,
        blur_radius=8,
    )
    with (
        load_depth(result.metadata, str(photo)) as asset,
        Image.open(photo) as source,
        apply_depth_style(source, asset, style, 0) as first,
        apply_depth_style(source, asset, style, 1) as last,
    ):
        first_pixels = np.asarray(first).astype(float)
        last_pixels = np.asarray(last).astype(float)
        assert first_pixels[8:24, 8:24].var() > first_pixels[8:24, 40:56].var()
        assert last_pixels[8:24, 8:24].var() < last_pixels[8:24, 40:56].var()
        np.testing.assert_array_equal(
            first_pixels[8:24, 8:24], np.asarray(source)[8:24, 8:24]
        )
        np.testing.assert_array_equal(
            last_pixels[8:24, 40:56], np.asarray(source)[8:24, 40:56]
        )


def test_directional_relighting_uses_camera_facing_normals(
    photo: Path,
) -> None:
    """Turning the light behind a flat surface removes its diffuse light."""
    result = extract_depth(request_for(photo), EffectBoundary())
    front = DepthStyle(
        metadata=result.metadata,
        ambient=0,
        relight_strength=1,
        light_direction=(0, 0, -1),
    )
    back = front.model_copy(update={"light_direction": (0, 0, 1)})
    with (
        load_depth(result.metadata, str(photo)) as asset,
        Image.new("RGB", (64, 32), (120, 120, 120)) as source,
        apply_depth_style(source, asset, front, 0) as lit,
        apply_depth_style(source, asset, back, 0) as dark,
    ):
        assert float(np.asarray(lit).mean()) > 100
        assert float(np.asarray(dark).mean()) < 5


def test_parallax_repeats_edges_instead_of_exposing_empty_pixels(
    photo: Path,
) -> None:
    """The explicit edge-repeat warp never inserts a blank border."""
    result = extract_depth(request_for(photo), EffectBoundary(2, 2))
    style = DepthStyle(
        metadata=result.metadata,
        focus_metres=4,
        parallax_x=16,
        parallax_y=8,
        exposed_pixels="edge-repeat",
    )
    with (
        load_depth(result.metadata, str(photo)) as asset,
        Image.new("RGB", (64, 32)) as source,
    ):
        source.putdata(
            [
                (20 + column, 40 + row, 90)
                for row in range(source.height)
                for column in range(source.width)
            ]
        )
        with apply_depth_style(source, asset, style, 1) as output:
            values = np.asarray(output)
            assert output.size == source.size
            assert int(values[..., 0].min()) >= 20
            assert int(values[..., 1].min()) >= 40
            assert bool(np.all(values[..., 2] == 90))
            assert not np.array_equal(values, np.asarray(source))


def test_relighting_preserves_sky_without_surface_geometry(
    photo: Path,
) -> None:
    """Sky pixels retain photographed light instead of noisy derived normals."""
    result = extract_depth(request_for(photo), EffectBoundary(sky=True))
    style = DepthStyle(
        metadata=result.metadata,
        ambient=0,
        relight_strength=1,
        light_direction=(0, 0, 1),
    )
    with (
        load_depth(result.metadata, str(photo)) as asset,
        Image.new("RGB", (64, 32), (120, 120, 120)) as source,
        apply_depth_style(source, asset, style, 0) as output,
    ):
        pixels = np.asarray(output)
        assert float(pixels[:, :32].mean()) < 5
        np.testing.assert_array_equal(
            pixels[:, 32:], np.asarray(source)[:, 32:]
        )


def test_working_grid_keeps_source_pixel_effect_distances() -> None:
    """Half-density rendering scales pixel effects and retains metric effects."""
    original = DepthStyle(
        metadata="photo.depth.json",
        focus_metres=2,
        focus_end_metres=7,
        blur_radius=20,
        parallax_x=8,
        parallax_y=-4,
        fog_density=0.2,
        relight_strength=0.6,
    )
    scaled = rescale_depth_style(original, (200, 100), (100, 50))
    assert scaled.blur_radius == 10
    assert (scaled.parallax_x, scaled.parallax_y) == (4, -2)
    assert scaled.focus_metres == original.focus_metres
    assert scaled.focus_end_metres == original.focus_end_metres
    assert scaled.fog_density == original.fog_density
    assert scaled.relight_strength == original.relight_strength
    assert (original.blur_radius, original.parallax_x) == (20, 8)


@pytest.mark.parametrize("size", [(0, 10), (201, 100), (100, 100)])
def test_working_grid_rejects_invalid_sizes(size: tuple[int, int]) -> None:
    """Working grids preserve source aspect and never invent native pixels."""
    with pytest.raises(ValueError, match=r"dimensions|aspect"):
        rescale_depth_style(
            DepthStyle(metadata="photo.depth.json"), (200, 100), size
        )


@pytest.mark.parametrize(
    "options",
    [
        {"parallax_x": 33},
        {"parallax_y": -33},
        {"blur_radius": 65},
        {"light_direction": (0, 0, 0)},
        {"fog_color": "not-a-color"},
        {"exposed_pixels": "fill"},
    ],
)
def test_unsupported_effect_settings_fail_before_rendering(
    options: dict[str, object],
) -> None:
    """Device-free validation rejects unsafe bounds and invented fill modes."""
    with pytest.raises(ValidationError):
        DepthStyle.model_validate({"metadata": "cached.depth.json", **options})
