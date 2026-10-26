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
    }


def test_schema_discovery_does_not_import_optional_model_frameworks() -> None:
    """Schema discovery stays usable without loading optional native code."""
    program = """
import json
import sys
from pic2panning.cli import main
status = main(["schema"])
frameworks = {"torch", "transformers", "mflux", "mlx", "diffusers", "sam2"}
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
    assert "invalid choice" in error.message


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


@pytest.mark.parametrize("flag", ["--reverse", "--focus-center"])
def test_image_only_motion_options_reject_video_inputs(
    tmp_path: Path,
    flag: str,
) -> None:
    """Video jobs reject incompatible flags before attempting media I/O."""
    error = failed_job(
        run_cli(
            [
                "render",
                str(tmp_path / "missing.mp4"),
                "--video",
                flag,
                "--output",
                str(tmp_path / "out.mp4"),
            ]
        )
    )
    assert error.error == "ValueError"
    assert "require image inputs" in error.message
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
