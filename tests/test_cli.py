"""Exercise public command lines through fresh Python processes."""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw
from pydantic import JsonValue, TypeAdapter

from pic2panning.command.media import RenderCommand
from pic2panning.command.models import (
    GenerateCommand,
    GenerateVideoCommand,
    MaskCommand,
)
from pic2panning.command.parse import parse_command
from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.editing import EditRequest, Resize
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.media import ImageClip, RenderRequest
from pic2panning.tools.media import decode_frames, ffmpeg, probe


class CliError(PydanticBaseModel):
    """The public error envelope written after any diagnostic messages."""

    error: str
    message: str


def run_cli(
    arguments: Sequence[str], module: str = "pic2panning"
) -> subprocess.CompletedProcess[str]:
    """Run the installed module without sharing imported application state."""
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def successful_job(result: subprocess.CompletedProcess[str]) -> JobResult:
    """Require a successful command with exactly one JSON result on stdout."""
    assert result.returncode == 0, result.stdout + result.stderr
    return JobResult.model_validate_json(result.stdout)


def failed_job(result: subprocess.CompletedProcess[str]) -> CliError:
    """Require a failed command and parse its final structured diagnostic."""
    assert result.returncode != 0, result.stdout + result.stderr
    assert result.stdout == ""
    return CliError.model_validate_json(result.stderr.splitlines()[-1])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Rendering requires FFmpeg and FFprobe",
)
def test_render_keeps_progress_logs_out_of_json_stdout(tmp_path: Path) -> None:
    """A real render produces parseable JSON while retaining progress logs."""
    source = tmp_path / "source.png"
    output = tmp_path / "rendered.mp4"
    Image.new("RGB", (64, 32), "red").save(source)
    process = run_cli(
        [
            "render",
            str(source),
            "--output",
            str(output),
            "--duration",
            "0.2",
            "--fps",
            "5",
            "--size",
            "32",
            "32",
        ]
    )
    result = successful_job(process)
    assert result.outputs == (str(output),)
    assert result.frames == 1
    assert result.duration == pytest.approx(0.2)
    assert "Rendering clip" in process.stderr
    assert output.is_file()
    assert source.is_file()


def test_edit_command_applies_requested_output_size(tmp_path: Path) -> None:
    """Terminal edit options produce the same useful image result as jobs."""
    source = tmp_path / "source.png"
    output = tmp_path / "edited.png"
    Image.new("RGB", (12, 8), "blue").save(source)
    result = successful_job(
        run_cli(
            [
                "edit",
                str(source),
                "--output",
                str(output),
                "--crop",
                "2",
                "1",
                "10",
                "7",
                "--resize",
                "4",
                "2",
                "--rotate",
                "90",
            ]
        )
    )
    assert (result.width, result.height) == (2, 4)
    with Image.open(output) as image:
        assert image.size == (2, 4)
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_run_executes_serialized_edit_and_preserves_source(
    tmp_path: Path,
) -> None:
    """A saved JSON job can be executed directly by an external agent."""
    source = tmp_path / "source.png"
    output = tmp_path / "edited.png"
    job = tmp_path / "job.json"
    Image.new("RGB", (12, 8), "lime").save(source)
    original = source.read_bytes()
    request = EditRequest(
        source=str(source),
        output=str(output),
        operations=(Resize(size=Size(width=3, height=5)),),
    )
    job.write_text(request.model_dump_json(), encoding="utf-8")
    result = successful_job(run_cli(["run", str(job)]))
    assert result.outputs == (str(output),)
    assert (result.width, result.height) == (3, 5)
    assert source.read_bytes() == original
    with Image.open(output) as image:
        assert image.size == (3, 5)
        assert image.getpixel((0, 0)) == (0, 255, 0)


def test_validate_does_not_read_inputs_or_create_output_directory(
    tmp_path: Path,
) -> None:
    """Validation accepts valid jobs with unavailable local and remote media."""
    output = tmp_path / "uncreated" / "video.mp4"
    request = RenderRequest(
        clips=(
            ImageClip(source=str(tmp_path / "missing.png")),
            ImageClip(source="https://127.0.0.1:1/never-downloaded.png"),
        ),
        output=str(output),
        size=Size(width=32, height=32),
    )
    job = tmp_path / "job.json"
    job.write_text(request.model_dump_json(), encoding="utf-8")
    result = run_cli(["validate", str(job)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert RenderRequest.model_validate_json(result.stdout) == request
    assert {path.name for path in tmp_path.iterdir()} == {"job.json"}


@pytest.mark.parametrize("module", ["pic2panning", "pic2panning.main"])
def test_module_entry_points_emit_discoverable_schema(module: str) -> None:
    """Both documented module entry points expose the complete job schema."""
    result = run_cli(["schema"], module=module)
    assert result.returncode == 0, result.stdout + result.stderr
    schema = TypeAdapter(dict[str, JsonValue]).validate_json(result.stdout)
    discriminator = schema["discriminator"]
    assert isinstance(discriminator, dict)
    assert discriminator["propertyName"] == "kind"
    assert isinstance(discriminator["mapping"], dict)
    assert set(discriminator["mapping"]) == {
        "render",
        "edit",
        "generate",
        "mask",
        "generate-video",
    }


def test_schema_discovery_does_not_import_optional_model_frameworks() -> None:
    """Schema discovery stays usable without loading optional native code."""
    program = """
import json
import sys
from pic2panning.cli import main
status = main(["schema"])
frameworks = {"torch", "transformers", "mflux", "mlx", "diffusers", "sam2", "ltx_pipelines_mlx", "ltx_core_mlx"}
loaded = sorted(name for name in frameworks if name in sys.modules)
sys.stderr.write(json.dumps(loaded) + "\\n")
raise SystemExit(status)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    schema = TypeAdapter(dict[str, JsonValue]).validate_json(result.stdout)
    assert "oneOf" in schema
    assert TypeAdapter(list[str]).validate_json(result.stderr) == []


def test_unknown_command_returns_structured_error() -> None:
    """An agent can parse an unsupported-command failure without a traceback."""
    error = failed_job(run_cli(["does-not-exist"]))
    assert error.error == "ValueError"
    assert "does-not-exist" in error.message
    assert "subcommands" in error.message.lower()


def test_invalid_render_request_fails_before_reading_source(
    tmp_path: Path,
) -> None:
    """Invalid timing is reported as validation instead of a missing file."""
    error = failed_job(
        run_cli(
            [
                "render",
                str(tmp_path / "missing.png"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--duration",
                "0",
            ]
        )
    )
    assert error.error == "ValidationError"
    assert "duration" in error.message
    assert list(tmp_path.iterdir()) == []


def test_missing_image_returns_structured_file_error(tmp_path: Path) -> None:
    """A missing source names the failed file and leaves no output."""
    source = tmp_path / "missing.png"
    error = failed_job(
        run_cli(
            [
                "edit",
                str(source),
                "--output",
                str(tmp_path / "out.png"),
                "--resize",
                "4",
                "4",
            ]
        )
    )
    assert error.error == "FileNotFoundError"
    assert str(source) in error.message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "options",
    [["--reverse"], ["--focus-center"], ["--shutter-angle", "180"]],
)
def test_image_only_motion_options_reject_video_inputs(
    tmp_path: Path,
    options: list[str],
) -> None:
    """Video jobs reject incompatible flags before attempting media I/O."""
    error = failed_job(
        run_cli(
            [
                "render",
                str(tmp_path / "missing.mp4"),
                "--video",
                *options,
                "--output",
                str(tmp_path / "out.mp4"),
            ]
        )
    )
    assert error.error == "ValueError"
    assert "image inputs" in error.message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("angle", ["-1", "361", "nan", "inf"])
def test_invalid_shutter_angles_fail_before_reading_a_photo(
    tmp_path: Path, angle: str
) -> None:
    """Exposure bounds and finite-number validation precede media processing."""
    error = failed_job(
        run_cli(
            [
                "render",
                str(tmp_path / "missing.png"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--shutter-angle",
                angle,
            ]
        )
    )
    assert "shutter" in error.message.lower()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Video rendering requires FFmpeg and FFprobe",
)
def test_video_speed_command_keeps_default_crop_stationary(
    tmp_path: Path,
) -> None:
    """Changing video speed does not implicitly enable photo panning."""
    image_path = tmp_path / "bands.png"
    source = tmp_path / "source.mp4"
    output = tmp_path / "result.mp4"
    with Image.new("RGB", (120, 40)) as image:
        draw = ImageDraw.Draw(image)
        for index, color in enumerate(("red", "lime", "blue")):
            draw.rectangle((index * 40, 0, index * 40 + 39, 39), fill=color)
        image.save(image_path)
    ffmpeg(
        [
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-t",
            "0.8",
            "-r",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
    )
    result = successful_job(
        run_cli(
            [
                "render",
                str(source),
                "--video",
                "--speed",
                "2",
                "--output",
                str(output),
                "--fps",
                "5",
                "--size",
                "40",
                "40",
            ]
        )
    )
    assert result.frames == 2
    with closing(decode_frames(str(output), probe(str(output)), 5)) as frames:
        for frame in frames:
            with frame:
                red, green, blue = cast(
                    "tuple[int, int, int]", frame.getpixel((20, 20))
                )
                assert green > 240
                assert red < 10
                assert blue < 10
    assert source.is_file()


@pytest.mark.parametrize(
    "command",
    [
        "render",
        "edit",
        "generate",
        "mask",
        "generate-video",
        "run",
        "validate",
        "schema",
        "inspect",
    ],
)
def test_each_command_exposes_help_without_required_inputs(
    command: str,
) -> None:
    """Every Tyro subcommand explains its options without media or models."""
    result = run_cli([command, "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"pic2panning {command}" in result.stdout
    assert "--help" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "demo",
    [
        "panning",
        "zoom",
        "panning_video",
        "speed",
        "add_audio_to_video",
        "img2img",
        "inpaint",
        "sam",
    ],
)
def test_every_demo_uses_the_same_typed_cli_help(demo: str) -> None:
    """Demo defaults remain compatible with Tyro and need no model imports."""
    result = run_cli(["--help"], module=f"pic2panning.demos.{demo}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "options",
    [
        ["--fps", "not-a-number"],
        ["--size", "32"],
        ["--motion", "diagonal"],
    ],
)
def test_invalid_terminal_values_return_json_without_reading_media(
    tmp_path: Path, options: list[str]
) -> None:
    """Numeric, tuple, and enum parse failures keep stdout empty and structured."""
    error = failed_job(
        run_cli(
            [
                "render",
                str(tmp_path / "missing.png"),
                "--output",
                str(tmp_path / "out.mp4"),
                *options,
            ]
        )
    )
    assert error.error == "ValueError"
    assert options[0] in error.message
    assert list(tmp_path.iterdir()) == []


def test_render_options_preserve_timeline_sound_and_export_quality() -> None:
    """Flat terminal flags construct the complete canonical render request."""
    command = parse_command(
        [
            "render",
            "coast.png",
            "forest.png",
            "--output",
            "trip.mp4",
            "--size",
            "640",
            "360",
            "--fps",
            "24",
            "--duration",
            "2.5",
            "--motion",
            "zoom-in",
            "--zoom",
            "1.7",
            "--reverse",
            "--focus-center",
            "--shutter-angle",
            "180",
            "--audio",
            "music.wav",
            "--audio-start",
            "2",
            "--loop-audio",
            "--cover",
            "A weekend away",
            "--crf",
            "16",
            "--preset",
            "slow",
            "--overwrite",
        ]
    )
    assert isinstance(command, RenderCommand)
    request = command.to_job()
    assert tuple(clip.source for clip in request.clips) == (
        "coast.png",
        "forest.png",
    )
    for clip in request.clips:
        assert isinstance(clip, ImageClip)
        assert clip.duration == 2.5
        assert clip.motion == "zoom-in"
        assert clip.zoom == 1.7
        assert clip.reverse
        assert clip.focus_center
        assert clip.shutter_angle == 180
    assert request.size.as_tuple == (640, 360)
    assert request.fps == 24
    assert request.encoding.crf == 16
    assert request.encoding.preset == "slow"
    assert request.audio is not None
    assert request.audio.source == "music.wav"
    assert request.audio.start == 2
    assert request.audio.loop
    assert request.cover is not None
    assert request.cover.text == "A weekend away"
    assert request.overwrite


def test_generation_options_use_model_values_and_preserve_controls() -> None:
    """Model enum values and quantization bits keep their documented spelling."""
    command = parse_command(
        [
            "generate",
            "--source",
            "coast.png",
            "--mask",
            "mask.png",
            "--output",
            "evening.png",
            "--prompt",
            "A rocky coast at dusk",
            "--model",
            "flux1-dev",
            "--quantization",
            "6",
            "--weights",
            "local/weights",
            "--size",
            "32",
            "16",
            "--seed",
            "7",
            "--steps",
            "12",
            "--strength",
            "0.5",
            "--guidance",
            "4.5",
        ]
    )
    assert isinstance(command, GenerateCommand)
    request = command.to_job()
    assert request.source == "coast.png"
    assert request.mask == "mask.png"
    assert request.model == "flux1-dev"
    assert request.quantization == 6
    assert request.weights == "local/weights"
    assert request.size.as_tuple == (32, 16)
    assert request.prompt == "A rocky coast at dusk"
    assert request.seed == 7
    assert request.steps == 12
    assert request.strength == 0.5
    assert request.guidance == 4.5


def test_separate_mask_flag_requests_individual_cpu_masks() -> None:
    """The documented separate flag keeps its individual-object meaning."""
    command = parse_command(
        [
            "mask",
            "coast.png",
            "--output",
            "people.png",
            "--prompt",
            "person. bicycle.",
            "--threshold",
            "0.4",
            "--device",
            "cpu",
            "--separate",
        ]
    )
    assert isinstance(command, MaskCommand)
    request = command.to_job()
    assert request.source == "coast.png"
    assert request.prompt == "person. bicycle."
    assert request.threshold == 0.4
    assert request.device == "cpu"
    assert not request.merge


def test_video_generation_options_preserve_both_guides_and_model_settings() -> (
    None
):
    """Endpoint, prompt, timing, and memory options survive typed CLI parsing."""
    command = parse_command(
        [
            "generate-video",
            "--first-frame",
            "left.png",
            "--last-frame",
            "right.png",
            "--output",
            "glide.mp4",
            "--prompt",
            "A continuous rightward camera glide",
            "--negative-prompt",
            "Dissolves, morphing buildings, extra moons",
            "--size",
            "576",
            "1024",
            "--frames",
            "49",
            "--fps",
            "24",
            "--seed",
            "7",
            "--steps",
            "20",
            "--refine-steps",
            "2",
            "--weights",
            "local/video",
            "--text-weights",
            "local/text",
            "--no-low-memory",
        ]
    )
    assert isinstance(command, GenerateVideoCommand)
    request = command.to_job()
    assert (request.first_frame, request.last_frame) == (
        "left.png",
        "right.png",
    )
    assert request.prompt == "A continuous rightward camera glide"
    assert (
        request.negative_prompt == "Dissolves, morphing buildings, extra moons"
    )
    assert request.size.as_tuple == (576, 1024)
    assert request.frames == 49
    assert request.fps == 24
    assert request.seed == 7
    assert request.steps == 20
    assert request.refine_steps == 2
    assert request.weights == "local/video"
    assert request.text_weights == "local/text"
    assert not request.low_memory


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Demo rendering requires FFmpeg and FFprobe",
)
def test_demo_options_override_defaults_and_render_the_selected_motion(
    tmp_path: Path,
) -> None:
    """A demo's zoom default can be replaced by a right-to-left pan."""
    source = tmp_path / "bands.png"
    output = tmp_path / "pan.mp4"
    with Image.new("RGB", (120, 40)) as image:
        draw = ImageDraw.Draw(image)
        for index, color in enumerate(("red", "lime", "blue")):
            draw.rectangle((index * 40, 0, index * 40 + 39, 39), fill=color)
        image.save(source)
    result = successful_job(
        run_cli(
            [
                str(source),
                "--output",
                str(output),
                "--motion",
                "panning-rl",
                "--duration",
                "0.4",
                "--fps",
                "5",
                "--size",
                "40",
                "40",
                "--crf",
                "16",
                "--preset",
                "slow",
            ],
            module="pic2panning.demos.zoom",
        )
    )
    assert result.frames == 2
    with closing(decode_frames(str(output), probe(str(output)), 5)) as frames:
        colors = []
        for frame in frames:
            with frame:
                colors.append(
                    TypeAdapter(tuple[int, int, int]).validate_python(
                        frame.getpixel((20, 20))
                    )
                )
    assert colors[0][2] > 240
    assert colors[0][0] < 10
    assert colors[-1][0] > 240
    assert colors[-1][2] < 10
