"""Verify input preservation and publication safety at external boundaries."""

import io
import os
import subprocess
from collections.abc import Sequence
from gzip import compress
from pathlib import Path
from typing import override

import pytest
from PIL import Image

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


@pytest.mark.parametrize("declared_length", [1, 100])
def test_download_size_mismatch_removes_incomplete_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_length: int,
) -> None:
    """A successful HTTP status cannot hide an incomplete response body."""

    def mismatched(source: str, *, timeout: float) -> DownloadResponse:
        if not source or timeout <= 0:
            raise ValueError("Expected a bounded media request")
        response = DownloadResponse(b"received media bytes")
        response.headers["Content-Length"] = str(declared_length)
        return response

    monkeypatch.setattr(media_io, "urlopen", mismatched)
    with pytest.raises(MediaError, match="Content-Length"):
        media_io.materialize("https://example.test/photo.png", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("chunked", [False, True])
def test_download_without_a_reliable_length_retains_all_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunked: bool,
) -> None:
    """Unknown lengths and dechunked bodies need no declared byte count."""
    body = b"complete media bytes"

    def unbounded(source: str, *, timeout: float) -> DownloadResponse:
        if not source or timeout <= 0:
            raise ValueError("Expected a bounded media request")
        response = DownloadResponse(body)
        response.headers = (
            {"Transfer-Encoding": "chunked", "Content-Length": "100"}
            if chunked
            else {}
        )
        return response

    monkeypatch.setattr(media_io, "urlopen", unbounded)
    output = media_io.materialize(
        "https://example.test/photo.png", str(tmp_path)
    )
    assert Path(output).read_bytes() == body


def test_download_compares_compressed_body_bytes_with_content_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP content encoding does not change urllib's raw response bytes."""
    body = compress(b"media bytes" * 100)

    def compressed(source: str, *, timeout: float) -> DownloadResponse:
        if not source or timeout <= 0:
            raise ValueError("Expected a bounded media request")
        response = DownloadResponse(body)
        response.headers["Content-Encoding"] = "gzip"
        return response

    monkeypatch.setattr(media_io, "urlopen", compressed)
    output = media_io.materialize(
        "https://example.test/photo.png", str(tmp_path)
    )
    assert Path(output).read_bytes() == body


def test_download_rejects_a_declared_body_over_the_input_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized media is rejected before a partial download can remain."""

    def oversized(source: str, *, timeout: float) -> DownloadResponse:
        if not source or timeout <= 0:
            raise ValueError("Expected a bounded media request")
        response = DownloadResponse(b"media prefix")
        response.headers["Content-Length"] = str(2 * 1024**3 + 1)
        return response

    monkeypatch.setattr(media_io, "urlopen", oversized)
    with pytest.raises(MediaError, match="input limit"):
        media_io.materialize("https://example.test/photo.png", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_image_loading_keeps_rgb_default_and_optional_oriented_alpha(
    tmp_path: Path,
) -> None:
    """Editing can retain alpha without changing the RGB media contract."""
    source = tmp_path / "oriented.png"
    with Image.new("RGBA", (2, 1), (10, 20, 30, 0)) as image:
        image.putpixel((1, 0), (40, 50, 60, 128))
        exif = Image.Exif()
        exif[274] = 6
        image.save(source, exif=exif)
    with (
        media_io.read_image(str(source)) as rgb,
        media_io.read_image(str(source), preserve_alpha=True) as rgba,
    ):
        assert rgb.mode == "RGB"
        assert rgb.size == (1, 2)
        assert rgb.getpixel((0, 1)) == (40, 50, 60)
        assert rgba.mode == "RGBA"
        assert rgba.size == (1, 2)
        assert rgba.getpixel((0, 0)) == (10, 20, 30, 0)
        assert rgba.getpixel((0, 1)) == (40, 50, 60, 128)


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
