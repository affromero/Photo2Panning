"""Exercise reconstruction commands and scene discovery without real weights."""

import hashlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue, TypeAdapter, ValidationError

from pic2panning.cli import main
from pic2panning.command.camera import SceneCommand
from pic2panning.command.parse import parse_command
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.camera import SceneCalibration, SceneColorSpace
from pic2panning.schemas.jobs import JOB_ADAPTER
from pic2panning.schemas.scene import SceneRequest, SceneResult
from pic2panning.tools import camera as camera_tools


class CliError(PydanticBaseModel):
    """The final JSON error written by an invalid reconstruction command."""

    error: str
    message: str


class SceneBoundary:
    """Replace the heavy predictor while exercising real scene publication."""

    request: SceneRequest | None
    image_size: tuple[int, int] | None

    def __init__(self) -> None:
        """Retain the predictor's observable input options and source size."""
        self.request = None
        self.image_size = None

    def predict(
        self,
        request: SceneRequest,
        image: Image.Image,
        output: str,
        source_sha256: str,
    ) -> tuple[SceneCalibration, int]:
        """Write a tiny valid Gaussian PLY with known camera calibration."""
        self.request = request
        self.image_size = image.size
        names = (
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        )
        header = "ply\nformat ascii 1.0\nelement vertex 4\n"
        header += "".join(f"property float {name}\n" for name in names)
        header += "end_header\n"
        rows = "".join(
            f"{x} {y} 2 0 0 0 0 0 0 2 -3 -3 -3 1 0 0 0\n"
            for x, y in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))
        )
        Path(output).write_text(header + rows)
        return (
            SceneCalibration(
                width=image.width,
                height=image.height,
                intrinsics=(
                    (40, 0, image.width / 2),
                    (0, 40, image.height / 2),
                    (0, 0, 1),
                ),
                world_to_camera=(
                    (1, 0, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                ),
                sh_degree=0,
                sh_color_space=SceneColorSpace.SRGB,
                z_percentiles_metres={"10": 2},
                source_sha256=source_sha256,
            ),
            4,
        )


def run_cli(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run the actual module entry point with independent imported state."""
    return subprocess.run(
        [sys.executable, "-m", "pic2panning", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_scene_terminal_controls_preserve_prediction_options() -> None:
    """Tyro forwards explicit devices and geometry budgets to the JSON job."""
    command = parse_command(
        [
            "scene",
            "photo.jpg",
            "--output",
            "photo.ply",
            "--metadata",
            "photo-scene.json",
            "--device",
            "cpu",
            "--num-tokens",
            "1200",
            "--max-edge",
            "512",
            "--overwrite",
        ]
    )
    assert isinstance(command, SceneCommand)
    request = command.to_job()
    assert request.source == "photo.jpg"
    assert (request.output, request.metadata) == (
        "photo.ply",
        "photo-scene.json",
    )
    assert request.device == "cpu"
    assert request.num_tokens == 1200
    assert request.max_edge == 512
    assert request.overwrite
    assert JOB_ADAPTER.validate_json(request.model_dump_json()) == request


def test_scene_command_publishes_bound_calibration_and_json_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI dispatch produces reusable scene artifacts through the predictor."""
    source = tmp_path / "photo.png"
    output = tmp_path / "photo.ply"
    metadata = tmp_path / "photo-scene.json"
    Image.new("RGB", (64, 32), "orange").save(source)
    original = source.read_bytes()
    boundary = SceneBoundary()
    monkeypatch.setattr(camera_tools, "NativeSceneBackend", lambda: boundary)
    status = main(
        [
            "scene",
            str(source),
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            "--device",
            "cpu",
            "--num-tokens",
            "1200",
            "--max-edge",
            "512",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    result = SceneResult.model_validate_json(captured.out)
    assert (result.output, result.metadata) == (str(output), str(metadata))
    assert result.points == 4
    assert result.predictor == "moge-2-vitl-normal"
    assert result.source_sha256 == hashlib.sha256(original).hexdigest()
    assert "element vertex 4" in output.read_text()
    calibration = SceneCalibration.from_sidecar(str(metadata))
    assert (calibration.width, calibration.height) == (64, 32)
    assert calibration.source_sha256 == result.source_sha256
    assert (
        calibration.scene_sha256
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert boundary.request == SceneRequest(
        source=str(source),
        output=str(output),
        metadata=str(metadata),
        device="cpu",
        num_tokens=1200,
        max_edge=512,
    )
    assert boundary.image_size == (64, 32)
    assert source.read_bytes() == original


def test_scene_validation_requires_no_source_or_output_directory(
    tmp_path: Path,
) -> None:
    """Saved reconstruction jobs validate without media or model access."""
    request = SceneRequest(
        source="https://127.0.0.1:1/missing.png",
        output=str(tmp_path / "absent" / "scene.ply"),
        metadata=str(tmp_path / "absent" / "scene.json"),
    )
    job = tmp_path / "request.json"
    job.write_text(request.model_dump_json())
    result = run_cli(["validate", str(job)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert SceneRequest.model_validate_json(result.stdout) == request
    assert {path.name for path in tmp_path.iterdir()} == {"request.json"}


def test_scene_schema_and_help_keep_predictor_frameworks_unloaded() -> None:
    """A core installation can discover scene fields and supported devices."""
    program = """
import json
import sys
from pic2panning.cli import main
status = main(["schema"])
frameworks = {"torch", "moge", "mlx", "mlx3d", "mflux"}
sys.stderr.write(json.dumps(sorted(frameworks.intersection(sys.modules))) + "\\n")
raise SystemExit(status)
"""
    process = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    schema = TypeAdapter(dict[str, JsonValue]).validate_json(process.stdout)
    discriminator = schema["discriminator"]
    assert isinstance(discriminator, dict)
    mapping = discriminator["mapping"]
    assert isinstance(mapping, dict)
    assert "scene" in mapping
    assert TypeAdapter(list[str]).validate_json(process.stderr) == []
    help_result = run_cli(["scene", "--help"])
    assert help_result.returncode == 0
    for option in (
        "--output",
        "--metadata",
        "--num-tokens",
        "--max-edge",
        "mps",
        "cpu",
    ):
        assert option in help_result.stdout


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        (["--num-tokens", "1199"], "num_tokens"),
        (["--num-tokens", "3601"], "num_tokens"),
        (["--max-edge", "255"], "max_edge"),
        (["--max-edge", "4097"], "max_edge"),
        (["--device", "cuda"], "cuda"),
    ],
)
def test_invalid_scene_controls_fail_before_accessing_inputs(
    tmp_path: Path, arguments: list[str], fragment: str
) -> None:
    """Budget and device errors leave missing source and output paths alone."""
    result = run_cli(
        [
            "scene",
            str(tmp_path / "missing.png"),
            "--output",
            str(tmp_path / "absent" / "scene.ply"),
            "--metadata",
            str(tmp_path / "absent" / "scene.json"),
            *arguments,
        ]
    )
    assert result.returncode != 0
    assert result.stdout == ""
    error = CliError.model_validate_json(result.stderr.splitlines()[-1])
    assert error.error in {"ValueError", "ValidationError"}
    assert fragment in error.message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("remote", ["output", "metadata"])
def test_scene_command_requires_local_artifact_destinations(
    tmp_path: Path, remote: str
) -> None:
    """Remote scene artifacts fail validation before an HTTP source is read."""
    paths = {
        "output": str(tmp_path / "scene.ply"),
        "metadata": str(tmp_path / "scene.json"),
    }
    paths[remote] = "https://example.com/" + Path(paths[remote]).name
    result = run_cli(
        [
            "scene",
            "https://127.0.0.1:1/source.png",
            "--output",
            paths["output"],
            "--metadata",
            paths["metadata"],
        ]
    )
    assert result.returncode != 0
    assert result.stdout == ""
    error = CliError.model_validate_json(result.stderr.splitlines()[-1])
    assert "local" in error.message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"output": "scene.png"}, "ply"),
        ({"metadata": "scene.txt"}, "json"),
        ({"num_tokens": True}, "num_tokens"),
        ({"max_edge": 512.5}, "max_edge"),
        ({"device": "cuda"}, "device"),
        ({"output": "https://example.com/scene.ply"}, "local"),
        ({"metadata": "http://example.com/scene.json"}, "local"),
    ],
)
def test_scene_requests_reject_invalid_formats_and_noninteger_budgets(
    changes: dict[str, JsonValue], fragment: str
) -> None:
    """The typed JSON contract rejects unsupported scene artifact options."""
    document: dict[str, JsonValue] = {
        "source": "source.jpg",
        "output": "scene.ply",
        "metadata": "scene.json",
    }
    with pytest.raises(ValidationError, match=fragment):
        SceneRequest.model_validate(document | changes)


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"points": 0}, "points"),
        ({"points": True}, "points"),
        ({"source_sha256": "not-a-hash"}, "source_sha256"),
        ({"predictor": "unknown-predictor"}, "predictor"),
    ],
)
def test_scene_results_reject_unusable_artifact_identity(
    changes: dict[str, JsonValue], fragment: str
) -> None:
    """Result envelopes require positive geometry and a known source hash."""
    document: dict[str, JsonValue] = {
        "output": "scene.ply",
        "metadata": "scene.json",
        "points": 4,
        "source_sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match=fragment):
        SceneResult.model_validate(document | changes)
