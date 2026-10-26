"""Exercise camera terminal contracts and calibration without model imports."""

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from pydantic import JsonValue, TypeAdapter, ValidationError

from pic2panning.cli import main
from pic2panning.command.camera import CameraCommand
from pic2panning.command.parse import parse_command
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.camera import (
    CameraPose,
    CameraRequest,
    SceneCalibration,
    SceneColorSpace,
    Vector3,
)
from pic2panning.schemas.jobs import JOB_ADAPTER, JobResult
from pic2panning.schemas.media import EncodingPreset
from pic2panning.tools import camera as camera_tools
from pic2panning.tools.media import decode_frames, probe


class CliError(PydanticBaseModel):
    """The final machine-readable error emitted by a failed camera command."""

    error: str
    message: str


class CameraBoundary:
    """Replace only the GPU scene provider while real tools encode a movie."""

    closed: bool
    loaded_scene: str | None
    loaded_calibration: SceneCalibration | None

    def __init__(self) -> None:
        """Retain observable resource state at the external model boundary."""
        self.closed = False
        self.loaded_scene = None
        self.loaded_calibration = None

    def load(
        self, scene: str, calibration: SceneCalibration
    ) -> CameraBoundary:
        """Open the supplied calibrated scene without a GPU dependency."""
        self.loaded_scene = scene
        self.loaded_calibration = calibration
        return self

    @staticmethod
    def pose_from_offset(
        base: CameraPose, offset: Vector3, focus_distance: float
    ) -> CameraPose:
        """Return an observable displaced camera for the synthetic scene."""
        if focus_distance <= 0:
            raise ValueError("The camera requires a positive focus distance")
        return base.model_copy(update={"translation": offset})

    @staticmethod
    def render(camera: CameraPose) -> Image.Image:
        """Return blue RGB pixels at the requested output dimensions."""
        return Image.new("RGB", (camera.width, camera.height), "blue")

    def close(self) -> None:
        """Release the fake provider when encoding has consumed its frames."""
        self.closed = True


def run_cli(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Exercise the installed entry point in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-m", "pic2panning", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def failed_job(result: subprocess.CompletedProcess[str]) -> CliError:
    """Require empty stdout and a structured final error after diagnostics."""
    assert result.returncode != 0, result.stdout + result.stderr
    assert result.stdout == ""
    return CliError.model_validate_json(result.stderr.splitlines()[-1])


def calibration_document() -> dict[str, JsonValue]:
    """Describe a small valid source camera in serialized sidecar form."""
    return {
        "width": 64,
        "height": 32,
        "intrinsics": [[40, 0, 32], [0, 41, 16], [0, 0, 1]],
        "world_to_camera": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "sh_degree": 0,
        "sh_color_space": "sRGB",
        "z_percentiles_metres": {"10": 12.5},
        "source_sha256": "a" * 64,
    }


def test_camera_terminal_controls_describe_the_complete_timeline() -> None:
    """Tyro preserves descending pans, metre offsets, timing, and quality."""
    command = parse_command(
        [
            "camera",
            "https://example.com/original.png",
            "--scene",
            "scene.ply",
            "--metadata",
            "scene.json",
            "--output",
            "camera.mp4",
            "--pan-start",
            "0.9",
            "--pause-at",
            "0.6",
            "--pan-end",
            "0.1",
            "--before",
            "2",
            "--duration",
            "3",
            "--after",
            "1.5",
            "--hold",
            "0",
            "--position",
            "0.2",
            "-0.1",
            "0.05",
            "--size",
            "320",
            "576",
            "--fps",
            "30",
            "--shutter-angle",
            "90",
            "--crf",
            "14",
            "--preset",
            "slow",
            "--overwrite",
        ]
    )
    assert isinstance(command, CameraCommand)
    request = command.to_job()
    assert request.source == "https://example.com/original.png"
    assert (request.scene, request.metadata) == ("scene.ply", "scene.json")
    assert request.output == "camera.mp4"
    assert (request.pan_start, request.pause_at, request.pan_end) == (
        0.9,
        0.6,
        0.1,
    )
    assert (request.before, request.duration, request.after, request.hold) == (
        2,
        3,
        1.5,
        0,
    )
    assert request.position == (0.2, -0.1, 0.05)
    assert request.size.as_tuple == (320, 576)
    assert request.fps == 30
    assert request.shutter_angle == 90
    assert request.encoding.crf == 14
    assert request.encoding.preset is EncodingPreset.SLOW
    assert request.overwrite
    assert JOB_ADAPTER.validate_json(request.model_dump_json()) == request


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Camera rendering requires FFmpeg and FFprobe",
)
def test_camera_command_encodes_the_pan_scene_and_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real CLI dispatch writes a silent movie and closes its provider."""
    source = tmp_path / "photo.png"
    scene = tmp_path / "scene.ply"
    metadata = tmp_path / "scene.json"
    output = tmp_path / "camera.mp4"
    Image.new("RGB", (64, 32), "red").save(source)
    scene.write_bytes(b"Synthetic scene consumed only by the GPU boundary")
    document = calibration_document()
    document["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    document["scene_sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(document))
    boundary = CameraBoundary()
    monkeypatch.setattr(camera_tools, "NativeCameraBackend", lambda: boundary)
    status = main(
        [
            "camera",
            str(source),
            "--scene",
            str(scene),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
            "--before",
            "0.2",
            "--duration",
            "0.3",
            "--after",
            "0.2",
            "--hold",
            "0",
            "--fps",
            "10",
            "--size",
            "32",
            "32",
            "--shutter-angle",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    result = JobResult.model_validate_json(captured.out)
    assert result.outputs == (str(output),)
    assert result.frames == 7
    assert result.duration == pytest.approx(0.7)
    info = probe(str(output))
    assert not info.has_audio
    colors: list[tuple[int, int, int]] = []
    with closing(decode_frames(str(output), info, 10)) as frames:
        for frame in frames:
            with frame:
                colors.append(
                    cast("tuple[int, int, int]", frame.getpixel((16, 16)))
                )
    assert colors[0][0] > 220
    assert colors[3][2] > 220
    assert colors[3][0] < 30
    assert colors[-1][0] > 220
    assert boundary.closed
    assert boundary.loaded_scene == str(scene.resolve())
    assert boundary.loaded_calibration == SceneCalibration.model_validate(
        document
    )
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == document["source_sha256"]
    )


def test_camera_validation_reads_no_scene_or_source(tmp_path: Path) -> None:
    """A saved camera job validates with unavailable local and remote inputs."""
    request = CameraRequest(
        source="https://127.0.0.1:1/unavailable.png",
        scene=str(tmp_path / "missing.ply"),
        metadata=str(tmp_path / "missing.json"),
        output=str(tmp_path / "absent" / "camera.mp4"),
    )
    job = tmp_path / "camera.json"
    job.write_text(request.model_dump_json())
    result = run_cli(["validate", str(job)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert CameraRequest.model_validate_json(result.stdout) == request
    assert {path.name for path in tmp_path.iterdir()} == {"camera.json"}


def test_camera_schema_and_help_keep_native_frameworks_unloaded() -> None:
    """Camera discovery works in a fresh process without optional packages."""
    program = """
import json
import sys
from pic2panning.cli import main
status = main(["schema"])
frameworks = {"torch", "moge", "mlx", "mlx3d", "mflux"}
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
    discriminator = schema["discriminator"]
    assert isinstance(discriminator, dict)
    mapping = discriminator["mapping"]
    assert isinstance(mapping, dict)
    assert "camera" in mapping
    assert TypeAdapter(list[str]).validate_json(result.stderr) == []
    help_result = run_cli(["camera", "--help"])
    assert help_result.returncode == 0
    for option in (
        "--scene",
        "--metadata",
        "--pause-at",
        "--position",
        "--shutter-angle",
        "--crf",
        "--preset",
    ):
        assert option in help_result.stdout


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--before", "0.001"], "before"),
        (["--duration", "0.02"], "duration"),
        (["--after", "0.001"], "after"),
        (["--hold", "-1"], "hold"),
        (["--pause-at", "1.1"], "pause_at"),
        (["--pan-start", "0.8"], "advance"),
        (["--size", "33", "64"], "even"),
        (["--shutter-angle", "361"], "shutter_angle"),
        (["--position", "nan", "0", "0"], "position"),
        (["--fps", "241"], "fps"),
        (["--crf", "52"], "crf"),
    ],
)
def test_invalid_camera_options_fail_before_input_access(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    """Invalid options produce JSON errors without reading missing inputs."""
    error = failed_job(
        run_cli(
            [
                "camera",
                str(tmp_path / "missing.png"),
                "--scene",
                str(tmp_path / "missing.ply"),
                "--metadata",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "missing" / "out.mp4"),
                *arguments,
            ]
        )
    )
    assert error.error in {"ValidationError", "ValueError"}
    assert message in error.message
    assert list(tmp_path.iterdir()) == []


def test_camera_requires_its_scene_and_metadata_options() -> None:
    """A missing imported scene fails parsing with one structured error."""
    error: CliError = failed_job(
        run_cli(["camera", "source.png", "--output", "out.mp4"])
    )
    assert error.error == "ValueError"
    assert "--scene" in error.message
    assert "--metadata" in error.message


@pytest.mark.parametrize("remote", ["scene", "metadata", "output"])
def test_camera_artifact_paths_require_local_files(
    tmp_path: Path, remote: str
) -> None:
    """Remote source images remain distinct from local scene and output paths."""
    paths = {
        "scene": str(tmp_path / "scene.ply"),
        "metadata": str(tmp_path / "scene.json"),
        "output": str(tmp_path / "out.mp4"),
    }
    paths[remote] = "https://example.com/" + Path(paths[remote]).name
    arguments = ["camera", "https://example.com/source.jpg"]
    for name, path in paths.items():
        arguments.extend(("--" + name, path))
    error = failed_job(run_cli(arguments))
    assert "local" in error.message
    assert list(tmp_path.iterdir()) == []


def test_sidecar_selects_calibration_and_retains_optional_scene_identity(
    tmp_path: Path,
) -> None:
    """Additional provenance does not prevent importing valid calibration."""
    document = calibration_document()
    document.update(
        {
            "scene_sha256": "b" * 64,
            "source": "source.png",
            "prediction_seconds": 1.25,
            "generator": "external-scene-exporter",
        }
    )
    sidecar = tmp_path / "scene.json"
    sidecar.write_text(json.dumps(document))
    calibration = SceneCalibration.from_sidecar(str(sidecar))
    assert calibration.source_sha256 == "a" * 64
    assert calibration.scene_sha256 == "b" * 64
    assert calibration.intrinsics[1][1] == 41
    assert calibration.z_percentiles_metres["10"] == 12.5
    assert calibration.sh_color_space is SceneColorSpace.SRGB
    assert "generator" not in calibration.model_dump()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"width": True}, "width"),
        ({"source_sha256": "A" * 64}, "source_sha256"),
        ({"scene_sha256": "not-a-hash"}, "scene_sha256"),
        ({"sh_degree": 1}, "sh_degree"),
        ({"z_percentiles_metres": {"50": 10}}, "foreground"),
        ({"z_percentiles_metres": {"10": 0}}, "z_percentiles_metres"),
        ({"intrinsics": [[-1, 0, 32], [0, 40, 16], [0, 0, 1]]}, "focal"),
        ({"intrinsics": [[40, 1, 32], [0, 40, 16], [0, 0, 1]]}, "skew"),
        ({"intrinsics": [[40, 0, 32], [0, 40, 16], [0, 1, 1]]}, "skew"),
        (
            {
                "world_to_camera": [
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            },
            "determinant",
        ),
        (
            {
                "world_to_camera": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 1],
                ]
            },
            "last row",
        ),
        (
            {
                "world_to_camera": [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            },
            "orthogonal",
        ),
    ],
)
def test_sidecar_rejects_mismatched_camera_geometry(
    changes: dict[str, JsonValue], message: str
) -> None:
    """Invalid hashes, depths, and camera matrices cannot reach a renderer."""
    with pytest.raises(ValidationError, match=message):
        SceneCalibration.model_validate(calibration_document() | changes)


def test_camera_pose_rejects_an_inverted_clip_range() -> None:
    """Near and far planes must describe a nonempty camera viewing range."""
    with pytest.raises(ValidationError, match="far plane"):
        CameraPose(
            width=64,
            height=32,
            fx=40,
            fy=40,
            cx=32,
            cy=16,
            rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            translation=(0, 0, 0),
            znear=1,
            zfar=0.5,
        )
