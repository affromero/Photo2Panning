"""Check joint endpoint generation through external model boundaries."""

import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Self

import pytest
from PIL import Image, ImageDraw
from pydantic import TypeAdapter, ValidationError

from pic2panning.backends.ltx import LtxBackend
from pic2panning.errors import MediaError
from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.video import GenerateVideoRequest
from pic2panning.tools.generation import generate_video
from pic2panning.tools.media import (
    decode_frames,
    encode_frames,
    normalize_audio,
    probe,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Generated video verification requires FFmpeg and FFprobe",
)


class Guide(PydanticBaseModel):
    """Expose arguments received by the external conditioning factory."""

    path: str
    frame_idx: int
    strength: float
    crf: int

    @classmethod
    def guide(
        cls, path: str, frame_idx: int, strength: float, *, crf: int
    ) -> Guide:
        """Keep the guide kind and endpoint positions visible to assertions."""
        return cls(path=path, frame_idx=frame_idx, strength=strength, crf=crf)


class Pipeline:
    """An external inference boundary writing actual measurable MP4 frames."""

    generate_audio: bool
    constructor: dict[str, object]
    parameters: dict[str, object]
    guides: list[Guide]
    reference_modes: list[str]
    reference_sizes: list[tuple[int, int]]
    reference_pixels: list[bytes]
    problem: str | None

    def __init__(self) -> None:
        """Record provider arguments without replacing application helpers."""
        self.generate_audio = True
        self.constructor = {}
        self.parameters = {}
        self.guides = []
        self.reference_modes = []
        self.reference_sizes = []
        self.reference_pixels = []
        self.problem = None

    def __call__(self, **parameters: object) -> Self:
        """Observe the external pipeline's explicit constructor arguments."""
        self.constructor = parameters
        return self

    def frames(self, size: Size, count: int) -> Iterator[Image.Image]:
        """Return a recognizable generated middle between both real guides."""
        for index in range(count):
            if index in {0, count - 1}:
                guide = self.guides[0 if index == 0 else 1]
                with (
                    Image.open(guide.path) as image,
                    image.convert("RGB") as rgb,
                ):
                    yield rgb.resize(size.as_tuple)
            else:
                yield Image.new("RGB", size.as_tuple, "red")

    def generate_and_save(self, **parameters: object) -> str:
        """Honor requested dimensions and timing unless simulating failure."""
        self.parameters = parameters
        self.guides = TypeAdapter(list[Guide]).validate_python(
            parameters["images"]
        )
        for guide in self.guides:
            with Image.open(guide.path) as reference:
                self.reference_modes.append(reference.mode)
                self.reference_sizes.append(reference.size)
                self.reference_pixels.append(reference.tobytes())
        output = TypeAdapter(str).validate_python(parameters["output_path"])
        if self.problem in {"provider", "corrupt"}:
            Path(output).write_bytes(b"Incomplete model output")
            if self.problem == "provider":
                raise OSError("Metal inference failed")
            return output
        size = Size(
            width=TypeAdapter(int).validate_python(parameters["width"]),
            height=TypeAdapter(int).validate_python(parameters["height"]),
        )
        count = TypeAdapter(int).validate_python(parameters["num_frames"])
        fps = TypeAdapter(int).validate_python(parameters["frame_rate"])
        if self.problem == "dimensions":
            size = Size(width=128, height=64)
        elif self.problem == "frame count":
            count -= 1
        elif self.problem == "frame rate":
            fps *= 2
        encode_frames(self.frames(size, count), output, size, fps)
        if self.problem == "audio":
            with TemporaryDirectory(
                prefix="test-generated-audio-"
            ) as directory:
                scored = str(Path(directory) / "scored.mp4")
                normalize_audio(output, scored, count / fps)
                shutil.copyfile(scored, output)
        return output


class PipelinesModule(ModuleType):
    """Export a callable external pipeline constructor."""

    TI2VidTwoStagesPipeline: Pipeline

    def __init__(self, pipeline: Pipeline) -> None:
        """Supply the one pipeline explicitly supported by the adapter."""
        super().__init__("ltx_pipelines_mlx")
        self.TI2VidTwoStagesPipeline = pipeline


class ConditioningModule(ModuleType):
    """Export the canonical external guide type."""

    ImageConditioningInput: type[Guide] = Guide


class HubModule(ModuleType):
    """Keep model downloads at an observable network boundary."""

    directories: dict[str, str]
    snapshots: dict[str, dict[str, object]]
    available: bool

    def __init__(self, directory: Path) -> None:
        """Provide local checkpoints without requesting model weights."""
        super().__init__("huggingface_hub")
        self.directories = {
            "dgrauet/ltx-2.3-mlx-q4": str(directory / "video"),
            "mlx-community/gemma-3-12b-it-4bit": str(directory / "text"),
        }
        for checkpoint in self.directories.values():
            Path(checkpoint).mkdir()
        self.snapshots = {}
        self.available = True

    def snapshot_download(
        self, repo_id: str, *, revision: str, allow_patterns: list[str]
    ) -> str:
        """Capture revisions and file filters, or expose download failure."""
        if not self.available:
            raise OSError("Hugging Face download unavailable")
        self.snapshots[repo_id] = {
            "revision": revision,
            "allow_patterns": allow_patterns,
        }
        return self.directories[repo_id]


@pytest.fixture
def model_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Pipeline, HubModule]:
    """Replace external imports while leaving production loaders untouched."""
    pipeline = Pipeline()
    hub = HubModule(tmp_path)
    modules = (
        PipelinesModule(pipeline),
        ConditioningModule("ltx_pipelines_mlx.utils.types"),
        hub,
    )
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return pipeline, hub


def source_images(directory: Path) -> tuple[str, str]:
    """Make an oriented, banded start and a transparent blue destination."""
    first = directory / "first.png"
    last = directory / "last.png"
    with Image.new("RGB", (64, 128), "lime") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 63, 31), fill="red")
        draw.rectangle((0, 96, 63, 127), fill="blue")
        orientation = Image.Exif()
        orientation[274] = 6
        image.save(first, exif=orientation)
    with Image.new("RGBA", (128, 64), (0, 0, 255, 128)) as image:
        image.save(last)
    return str(first), str(last)


@pytest.mark.parametrize(
    "options",
    [
        {"size": {"width": 576, "height": 1000}},
        {"frames": 64},
        {"frames": 1},
        {"frames": True},
        {"fps": 0},
        {"fps": 61},
        {"steps": 0},
        {"refine_steps": 4},
        {"seed": -1},
        {"output": "out.gif"},
        {"prompt": "   "},
        {"negative_prompt": "   "},
        {"unknown": "option"},
    ],
)
def test_invalid_generation_options_fail_before_model_or_source_access(
    options: dict[str, object],
) -> None:
    """Model-incompatible jobs cannot trigger resource loading or rounding."""
    with pytest.raises(ValidationError):
        GenerateVideoRequest.model_validate(
            {
                "first_frame": "missing-first.png",
                "last_frame": "missing-last.png",
                "prompt": "Continuous camera pan right",
                "output": "out.mp4",
                **options,
            }
        )


def test_joint_endpoint_generation_preserves_guides_and_measured_timeline(
    model_boundary: tuple[Pipeline, HubModule], tmp_path: Path
) -> None:
    """Production orchestration publishes checked video with both image guides."""
    pipeline, hub = model_boundary
    first, last = source_images(tmp_path)
    originals = {path: Path(path).read_bytes() for path in (first, last)}
    output = tmp_path / "generated.mp4"
    result = generate_video(
        GenerateVideoRequest(
            first_frame=first,
            last_frame=last,
            prompt="A continuous rightward camera glide across the skyline",
            output=str(output),
            size=Size(width=64, height=64),
            frames=9,
            fps=6,
            seed=17,
            steps=7,
            refine_steps=2,
        )
    )
    assert result.outputs == (str(output),)
    assert (result.width, result.height, result.frames) == (64, 64, 9)
    assert result.duration == pytest.approx(1.5)
    assert not probe(str(output)).has_audio
    assert pipeline.constructor == {
        "model_dir": hub.directories["dgrauet/ltx-2.3-mlx-q4"],
        "gemma_model_id": hub.directories["mlx-community/gemma-3-12b-it-4bit"],
        "low_memory": True,
        "low_ram_streaming": True,
    }
    assert pipeline.generate_audio is False
    assert pipeline.reference_modes == ["RGB", "RGB"]
    assert pipeline.reference_sizes == [(64, 64), (64, 64)]
    assert pipeline.reference_pixels == [
        bytes([0, 255, 0]) * 64**2,
        bytes([0, 0, 255]) * 64**2,
    ]
    assert [guide.frame_idx for guide in pipeline.guides] == [0, 8]
    assert all(
        guide.strength == 1 and guide.crf == 0 for guide in pipeline.guides
    )
    assert all(not Path(guide.path).exists() for guide in pipeline.guides)
    parameters = pipeline.parameters
    assert parameters["prompt"] == (
        "A continuous rightward camera glide across the skyline"
    )
    assert parameters["seed"] == 17
    assert (parameters["height"], parameters["width"]) == (64, 64)
    assert parameters["num_frames"] == 9
    assert parameters["frame_rate"] == 6.0
    assert parameters["stage1_steps"] == 7
    assert parameters["stage2_steps"] == 2
    assert parameters["stg_scale"] == 0.0
    assert parameters["refine_audio"] is True
    assert "negative_prompt" not in parameters
    assert hub.snapshots["dgrauet/ltx-2.3-mlx-q4"]["revision"] == (
        "56a5866d638ecfe37c54d348e88938235185c2d4"
    )
    files = TypeAdapter(list[str]).validate_python(
        hub.snapshots["dgrauet/ltx-2.3-mlx-q4"]["allow_patterns"]
    )
    assert set(files) == {
        "config.json",
        "embedded_config.json",
        "quantize_config.json",
        "split_model.json",
        "transformer-dev.safetensors",
        "transformer-distilled.safetensors",
        "connector.safetensors",
        "vae_encoder.safetensors",
        "vae_decoder.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json",
    }
    assert hub.snapshots["mlx-community/gemma-3-12b-it-4bit"] == {
        "revision": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
        "allow_patterns": ["*.json", "*.safetensors", "tokenizer.model"],
    }
    assert all(
        Path(path).read_bytes() == original
        for path, original in originals.items()
    )
    with closing(decode_frames(str(output), probe(str(output)), 6)) as frames:
        colors = []
        for frame in frames:
            with frame:
                colors.append(
                    TypeAdapter(tuple[int, int, int]).validate_python(
                        frame.getpixel((32, 32))
                    )
                )
    assert len(colors) == 9
    for color, channel in ((colors[0], 1), (colors[4], 0), (colors[-1], 2)):
        assert color[channel] > 240
        assert sum(color) - color[channel] < 20


def test_explicit_local_models_do_not_request_downloads(
    model_boundary: tuple[Pipeline, HubModule], tmp_path: Path
) -> None:
    """Local checkpoint choices and memory settings reach the real adapter."""
    pipeline, hub = model_boundary
    hub.available = False
    first, last = source_images(tmp_path)
    output = tmp_path / "generated.mp4"
    LtxBackend().generate(
        GenerateVideoRequest(
            first_frame=first,
            last_frame=last,
            output=str(output),
            prompt="Pan right",
            negative_prompt="Morphing buildings, dissolves, ghosting",
            size=Size(width=64, height=64),
            frames=9,
            weights=hub.directories["dgrauet/ltx-2.3-mlx-q4"],
            text_weights=hub.directories["mlx-community/gemma-3-12b-it-4bit"],
            low_memory=False,
        ),
        first,
        last,
        str(output),
    )
    assert output.is_file()
    assert probe(str(output)).frames == 9
    assert pipeline.constructor["low_memory"] is False
    assert pipeline.constructor["low_ram_streaming"] is True
    assert pipeline.parameters["negative_prompt"] == (
        "Morphing buildings, dissolves, ghosting"
    )
    assert hub.snapshots == {}


@pytest.mark.parametrize(
    "problem", ["dimensions", "frame count", "frame rate", "corrupt", "audio"]
)
def test_invalid_model_video_is_never_published(
    model_boundary: tuple[Pipeline, HubModule], tmp_path: Path, problem: str
) -> None:
    """Bad model output cannot replace an existing completed artifact."""
    pipeline, hub = model_boundary
    pipeline.problem = problem
    first, last = source_images(tmp_path)
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"Previous completed artifact")
    request = GenerateVideoRequest(
        first_frame=first,
        last_frame=last,
        output=str(output),
        prompt="Pan right",
        size=Size(width=64, height=64),
        frames=9,
        fps=6,
        weights=hub.directories["dgrauet/ltx-2.3-mlx-q4"],
        text_weights=hub.directories["mlx-community/gemma-3-12b-it-4bit"],
        overwrite=True,
    )
    with pytest.raises(MediaError):
        generate_video(request)
    assert output.read_bytes() == b"Previous completed artifact"
    assert not list(tmp_path.glob(".pic2panning-*"))
    assert all(not Path(guide.path).exists() for guide in pipeline.guides)


@pytest.mark.parametrize("failure", ["provider", "download"])
def test_inference_and_download_errors_propagate_without_fallback(
    model_boundary: tuple[Pipeline, HubModule], tmp_path: Path, failure: str
) -> None:
    """The actual failed model or network boundary supplies the error."""
    pipeline, hub = model_boundary
    pipeline.problem = "provider" if failure == "provider" else None
    hub.available = failure != "download"
    first, last = source_images(tmp_path)
    output = tmp_path / "generated.mp4"
    with pytest.raises(OSError, match=r"Metal|Hugging Face"):
        generate_video(
            GenerateVideoRequest(
                first_frame=first,
                last_frame=last,
                output=str(output),
                prompt="Pan right",
                size=Size(width=64, height=64),
                frames=9,
            )
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".pic2panning-*"))


@pytest.mark.parametrize("source", ["first", "last", "weights", "text"])
def test_overwrite_protects_both_guides_and_explicit_model_files(
    tmp_path: Path, source: str
) -> None:
    """Hardlink aliases cannot overwrite any declared generation input."""
    first, last = source_images(tmp_path)
    weights = tmp_path / "weights.safetensors"
    text = tmp_path / "text.safetensors"
    weights.write_bytes(b"Unique model weights")
    text.write_bytes(b"Unique text weights")
    paths = {
        "first": Path(first),
        "last": Path(last),
        "weights": weights,
        "text": text,
    }
    protected = paths[source]
    original = protected.read_bytes()
    output = tmp_path / "generated.mp4"
    os.link(protected, output)
    with pytest.raises(ValueError, match="aliases"):
        generate_video(
            GenerateVideoRequest(
                first_frame=first,
                last_frame=last,
                output=str(output),
                prompt="Pan right",
                weights=str(weights),
                text_weights=str(text),
                overwrite=True,
            )
        )
    assert protected.read_bytes() == original
    assert output.read_bytes() == original
