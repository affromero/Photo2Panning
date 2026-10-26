"""Verify input preservation and publication safety at external boundaries."""

import io
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import override

import pytest

from pic2panning.errors import MediaError
from pic2panning.tools import io as media_io


class DownloadResponse(io.BytesIO):
    """A network boundary exposing response status, headers, and bytes."""

    status: int
    headers: dict[str, str]

    def __init__(self, body: bytes, status: int = 200) -> None:
        """Describe a complete bounded HTTP response."""
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}


class InterruptedResponse(DownloadResponse):
    """A connection that delivers a prefix before the network times out."""

    @override
    def read(self, size: int | None = -1) -> bytes:
        """Return available bytes once, then surface the network failure."""
        if self.tell():
            raise TimeoutError("Connection timed out after partial download")
        return super().read(size)


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_overwrite_never_replaces_an_input_alias(
    tmp_path: Path, alias: str
) -> None:
    """An explicit overwrite still protects every spelling of an input."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"unique source")
    output = source if alias == "direct" else tmp_path / "alias.bin"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, output)
    with (
        pytest.raises(ValueError, match=r"aliases|symbolic link"),
        media_io.atomic_output(
            str(output), (str(source),), overwrite=True
        ) as temporary,
    ):
        Path(temporary).write_bytes(b"replacement")
    assert source.read_bytes() == b"unique source"
    assert output.read_bytes() == b"unique source"
    if alias == "symlink":
        assert output.is_symlink()
    assert all(
        not path.name.startswith(".pic2panning-")
        for path in tmp_path.iterdir()
    )


def test_successful_output_becomes_visible_only_after_publication(
    tmp_path: Path,
) -> None:
    """Consumers see the complete result after the write context succeeds."""
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    source.write_bytes(b"source bytes")
    with media_io.atomic_output(str(output), (str(source),)) as temporary:
        Path(temporary).write_bytes(b"finished result")
        assert not output.exists()
    assert output.read_bytes() == b"finished result"
    assert source.read_bytes() == b"source bytes"
    assert {path.name for path in tmp_path.iterdir()} == {
        "source.bin",
        "output.bin",
    }


def test_failed_processing_preserves_existing_output_and_cleans_temporary(
    tmp_path: Path,
) -> None:
    """A failed producer cannot destroy a previously published artifact."""
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    source.write_bytes(b"source bytes")
    output.write_bytes(b"previous result")

    def failed_producer() -> None:
        with media_io.atomic_output(
            str(output), (str(source),), overwrite=True
        ) as temporary:
            Path(temporary).write_bytes(b"incomplete result")
            raise OSError("Encoder failed")

    with pytest.raises(OSError, match="Encoder failed"):
        failed_producer()
    assert output.read_bytes() == b"previous result"
    assert source.read_bytes() == b"source bytes"
    assert {path.name for path in tmp_path.iterdir()} == {
        "source.bin",
        "output.bin",
    }


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    """The default policy rejects an existing destination before writing."""
    output = tmp_path / "output.bin"
    output.write_bytes(b"previous result")
    with (
        pytest.raises(FileExistsError, match="already exists"),
        media_io.atomic_output(str(output), ()) as temporary,
    ):
        Path(temporary).write_bytes(b"replacement")
    assert output.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [output]


def test_concurrent_destination_creation_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    """A destination created during rendering retains the other writer's data."""
    output = tmp_path / "output.bin"

    def competing_producer() -> None:
        with media_io.atomic_output(str(output), ()) as temporary:
            Path(temporary).write_bytes(b"our prepared result")
            output.write_bytes(b"another writer finished first")

    with pytest.raises(FileExistsError):
        competing_producer()
    assert output.read_bytes() == b"another writer finished first"
    assert list(tmp_path.iterdir()) == [output]


def test_destination_changed_to_input_symlink_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    """Late alias creation cannot bypass protection when overwrite is enabled."""
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    source.write_bytes(b"protected source")

    def destination_changed_during_processing() -> None:
        with media_io.atomic_output(
            str(output), (str(source),), overwrite=True
        ) as temporary:
            Path(temporary).write_bytes(b"replacement")
            output.symlink_to(source)

    with pytest.raises(ValueError, match="symbolic link"):
        destination_changed_during_processing()
    assert source.read_bytes() == b"protected source"
    assert output.is_symlink()
    assert {path.name for path in tmp_path.iterdir()} == {
        "source.bin",
        "output.bin",
    }


def test_http_materialization_retains_response_bytes_and_separates_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct remote resources sharing a basename remain distinct files."""

    def open_response(source: str, *, timeout: float) -> DownloadResponse:
        if timeout <= 0:
            raise ValueError("A positive network timeout is required")
        return DownloadResponse(source.encode())

    monkeypatch.setattr(media_io, "urlopen", open_response)
    first_url = "https://example.test/photo.png?version=1"
    second_url = "https://example.test/photo.png?version=2"
    first = Path(media_io.materialize(first_url, str(tmp_path)))
    second = Path(media_io.materialize(second_url, str(tmp_path)))
    assert first != second
    assert first.read_bytes() == first_url.encode()
    assert second.read_bytes() == second_url.encode()


def test_http_error_status_is_reported_without_leaving_a_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTTP failure cannot become a cached media file."""

    def unavailable(source: str, *, timeout: float) -> DownloadResponse:
        if not source or timeout <= 0:
            raise ValueError("Expected a bounded media request")
        return DownloadResponse(b"Service unavailable", status=503)

    monkeypatch.setattr(media_io, "urlopen", unavailable)
    with pytest.raises(MediaError, match="HTTP 503"):
        media_io.materialize("https://example.test/photo.png", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_partial_network_timeout_removes_incomplete_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout after receiving data propagates and clears partial content."""

    def interrupted(source: str, *, timeout: float) -> InterruptedResponse:
        if not source or timeout <= 0:
            raise ValueError("Expected a bounded media request")
        return InterruptedResponse(b"partial media bytes")

    monkeypatch.setattr(media_io, "urlopen", interrupted)
    with pytest.raises(TimeoutError, match="partial download"):
        media_io.materialize("https://example.test/photo.png", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_different_youtube_sources_keep_their_own_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two clips cannot silently share a downloader's first completed file."""

    def download(
        command: Sequence[str],
        **options: object,
    ) -> subprocess.CompletedProcess[str]:
        if options.get("text") is not True:
            raise ValueError(
                "The downloader boundary must use text diagnostics"
            )
        template = command[command.index("-o") + 1]
        destination = Path(template.replace("%(ext)s", "wav"))
        if not destination.exists():
            destination.write_bytes(command[-1].encode())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", download)
    first_url = "https://www.youtube.com/watch?v=first"
    second_url = "https://www.youtube.com/watch?v=second"
    first = Path(media_io.load_audio(first_url, str(tmp_path)))
    second = Path(media_io.load_audio(second_url, str(tmp_path)))
    assert first != second
    assert first.read_bytes() == first_url.encode()
    assert second.read_bytes() == second_url.encode()
