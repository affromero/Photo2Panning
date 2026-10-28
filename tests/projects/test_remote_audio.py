"""Remote project sound is downloaded as media before hashing and processing."""

import hashlib
import math
import struct
import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from pic2panning.api import run_project
from pic2panning.errors import DependencyError
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.audio import (
    BeatAnalysis,
    BeatRequest,
    MixTrack,
)
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    ImageLayer,
    LayerKeyframe,
)
from pic2panning.schemas.creative.operations import Operation
from pic2panning.schemas.creative.project import (
    ProjectDocument,
    ProjectNode,
    ProjectRequest,
)
from pic2panning.schemas.media import AudioTrack, ImageClip, RenderRequest
from tests.creative_comp.test_audio import tone_level

YOUTUBE_SOURCE = "https://youtu.be/offline-audio-fixture"


class DownloadedAudio:
    """Mutable observations and media returned by the external downloader."""

    payload: bytes
    error: str | None
    sources: list[str]

    def __init__(self, payload: bytes) -> None:
        """Keep the subprocess boundary independent of project processing."""
        self.payload = payload
        self.error = None
        self.sources = []


def audio_bytes(path: Path, *, period: float | None = None) -> bytes:
    """Write a real PCM tone, optionally gated into known periodic beats."""
    rate = 48000
    count = round(rate * (2.2 if period is not None else 1))
    samples = (
        round(
            math.sin(2 * math.pi * 400 * index / rate)
            * 16000
            * (period is None or (index / rate) % period < 0.03)
        )
        for index in range(count)
    )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(
            b"".join(struct.pack("<h", sample) for sample in samples)
        )
    return path.read_bytes()


def install_downloader(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> DownloadedAudio:
    """Replace only yt-dlp work and run every media subprocess normally."""
    state = DownloadedAudio(payload)
    real_run = subprocess.run

    def download(
        command: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if not text:
            raise ValueError("Media subprocess diagnostics must use text")
        if tuple(command[1:3]) != ("-m", "yt_dlp"):
            return real_run(
                command,
                capture_output=capture_output,
                text=True,
                check=check,
                timeout=timeout,
            )
        state.sources.append(command[-1])
        if state.error is not None:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr=state.error
            )
        template = command[command.index("-o") + 1]
        Path(template.replace("%(ext)s", "wav")).write_bytes(state.payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", download)
    return state


def project(workspace: Path, operation: Operation) -> ProjectRequest:
    """Run an actual public operation inside an ordinary saved graph."""
    return ProjectRequest(
        project=ProjectDocument(
            name="Remote sound",
            nodes=(ProjectNode(id="audio", job=operation),),
        ),
        workspace=str(workspace),
        output="report.json",
        overwrite=True,
    )


def test_youtube_beats_hash_decoded_audio_and_invalidate_changed_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloaded waveform bytes drive beat results and persistent cache identity."""
    payload = audio_bytes(tmp_path / "download.wav", period=0.5)
    downloader = install_downloader(monkeypatch, payload)
    request = project(
        tmp_path,
        BeatRequest(source=YOUTUBE_SOURCE, output="beats.json"),
    )
    first = run_project(request)
    analysis = BeatAnalysis.model_validate_json(
        (tmp_path / "beats.json").read_text()
    )
    assert not first.nodes[0].cache_hit
    assert set(first.nodes[0].input_hashes.values()) == {
        hashlib.sha256(payload).hexdigest()
    }
    assert all(path.endswith(".wav") for path in first.nodes[0].input_hashes)
    for moment in (0, 0.5, 1, 1.5, 2):
        assert any(abs(marker - moment) < 0.06 for marker in analysis.markers)
    assert analysis.bpm is not None
    assert 110 < analysis.bpm < 130
    cached = run_project(request)
    assert cached.nodes[0].cache_hit
    assert cached.nodes[0].fingerprint == first.nodes[0].fingerprint
    changed = audio_bytes(tmp_path / "download.wav", period=0.25)
    downloader.payload = changed
    regenerated = run_project(request)
    changed_analysis = BeatAnalysis.model_validate_json(
        (tmp_path / "beats.json").read_text()
    )
    assert not regenerated.nodes[0].cache_hit
    assert regenerated.nodes[0].fingerprint != first.nodes[0].fingerprint
    assert set(regenerated.nodes[0].input_hashes.values()) == {
        hashlib.sha256(changed).hexdigest()
    }
    assert changed_analysis.bpm is not None
    assert 220 < changed_analysis.bpm < 260
    assert any(
        abs(marker - 0.25) < 0.06 for marker in changed_analysis.markers
    )
    assert set(downloader.sources) == {YOUTUBE_SOURCE}


@pytest.mark.parametrize("mode", ["compose", "render", "clip-audio"])
def test_remote_soundtrack_bindings_produce_audible_project_movies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["compose", "render", "clip-audio"],
) -> None:
    """Global, nested, and mixed soundtracks load media and retain audible content."""
    payload = audio_bytes(tmp_path / "download.wav")
    install_downloader(monkeypatch, payload)
    photo = tmp_path / "photo.png"
    with Image.new("RGB", (16, 16), "blue") as image:
        image.save(photo)
    operation: Operation
    if mode == "compose":
        operation = ComposeRequest(
            canvas=Size(width=16, height=16),
            duration=1,
            camera=(CameraKeyframe(x=8, y=8),),
            layers=(
                ImageLayer(
                    id="photo",
                    source="photo.png",
                    duration=1,
                    size=Size(width=16, height=16),
                    keyframes=(LayerKeyframe(x=8, y=8),),
                ),
            ),
            audio=(MixTrack(source=YOUTUBE_SOURCE, duration=1),),
            output="movie.mp4",
            size=Size(width=16, height=16),
            fps=6,
        )
    else:
        track = AudioTrack(source=YOUTUBE_SOURCE)
        operation = RenderRequest(
            clips=(
                ImageClip(
                    source="photo.png",
                    duration=1,
                    audio=track if mode == "clip-audio" else None,
                ),
            ),
            audio=track if mode == "render" else None,
            output="movie.mp4",
            size=Size(width=16, height=16),
            fps=6,
        )
    result = run_project(project(tmp_path, operation))
    assert (
        hashlib.sha256(payload).hexdigest()
        in result.nodes[0].input_hashes.values()
    )
    assert (
        tone_level(
            str(tmp_path / "movie.mp4"), tmp_path / "heard.wav", 0.2, 400
        )
        > 0.3
    )


def test_youtube_download_failure_surfaces_without_project_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing optional downloader fails before cache or output publication."""
    downloader = install_downloader(monkeypatch, b"unused failed download")
    downloader.error = "No module named yt_dlp"
    request = project(
        tmp_path,
        BeatRequest(source=YOUTUBE_SOURCE, output="beats.json"),
    )
    with pytest.raises(DependencyError, match="youtube"):
        run_project(request)
    assert not (tmp_path / "beats.json").exists()
    assert not (tmp_path / "report.json").exists()
