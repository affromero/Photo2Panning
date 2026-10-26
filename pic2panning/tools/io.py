"""Checked input loading and atomic publication of local artifacts."""

import hashlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import BinaryIO, cast
from urllib.parse import urlparse
from urllib.request import urlopen

from klogr.path import (
    get_suffix,
    path_dirname,
    path_exists,
    path_expanduser,
    path_is_file,
    path_islink,
    path_join,
    path_mkdir,
    path_open,
    path_remove,
    path_rename,
    path_resolve,
    path_stat,
)
from PIL import Image, ImageOps

from pic2panning.errors import DependencyError, MediaError

DOWNLOAD_CHUNK = 1024 * 1024
DOWNLOAD_LIMIT = 2 * 1024**3


def is_remote(source: str) -> bool:
    """Recognize HTTP sources without treating local names as URLs."""
    return urlparse(source).scheme in {"http", "https"}


def require_file(source: str) -> str:
    """Resolve a local file and fail with its original input name."""
    resolved = path_resolve(path_expanduser(source))
    if not path_is_file(resolved):
        raise FileNotFoundError(f"Input file does not exist: {source}")
    return resolved


def materialize(source: str, directory: str) -> str:
    """Stream an HTTP resource into scoped storage, or resolve a local file."""
    if not is_remote(source):
        return require_file(source)
    parsed = urlparse(source)
    suffix = get_suffix(parsed.path)
    name = hashlib.sha256(source.encode()).hexdigest() + suffix
    target = path_join(directory, name)
    if path_is_file(target):
        return target
    path_mkdir(directory, parents=True, exist_ok=True)
    try:
        with urlopen(source, timeout=30) as response:
            if response.status != 200:
                raise MediaError(f"HTTP {response.status} for {source}")
            length = response.headers.get("Content-Length")
            chunked = "chunked" in {
                coding.strip().lower()
                for coding in response.headers.get(
                    "Transfer-Encoding", ""
                ).split(",")
            }
            expected = (
                int(length) if length is not None and not chunked else None
            )
            if expected is not None and expected < 0:
                raise MediaError("HTTP Content-Length must be nonnegative")
            if expected is not None and expected > DOWNLOAD_LIMIT:
                raise MediaError("Download exceeds the 2 GiB input limit")
            with cast("BinaryIO", path_open(target, "wb")) as destination:
                total = 0
                while chunk := response.read(DOWNLOAD_CHUNK):
                    total += len(chunk)
                    if total > DOWNLOAD_LIMIT:
                        raise MediaError(
                            "Download exceeds the 2 GiB input limit"
                        )
                    destination.write(chunk)
                if expected is not None and total != expected:
                    raise MediaError(
                        "HTTP Content-Length mismatch: "
                        f"expected {expected} bytes, received {total}"
                    )
    except BaseException:
        path_remove(target, non_exist_ok=True)
        raise
    return target


def read_image(source: str, *, preserve_alpha: bool = False) -> Image.Image:
    """Read an oriented RGB image, retaining transparency when requested."""
    with tempfile.TemporaryDirectory(prefix="pic2panning-image-") as directory:
        local = materialize(source, directory)
        with (
            cast("BinaryIO", path_open(local, "rb")) as handle,
            Image.open(handle) as image,
            ImageOps.exif_transpose(image) as oriented,
        ):
            mode = (
                "RGBA"
                if preserve_alpha and oriented.has_transparency_data
                else "RGB"
            )
            return oriented.convert(mode)


def validate_destination(
    output: str, inputs: Sequence[str], *, overwrite: bool
) -> str:
    """Reject aliases of inputs, symbolic links, and accidental overwrites."""
    expanded = path_expanduser(output)
    if path_islink(expanded):
        raise ValueError(f"Output cannot be a symbolic link: {output}")
    target = path_resolve(expanded)
    for source in inputs:
        if is_remote(source):
            continue
        resolved = path_resolve(path_expanduser(source))
        same_inode = False
        if path_exists(target) and path_exists(resolved):
            target_stat = path_stat(target)
            source_stat = path_stat(resolved)
            same_inode = (
                target_stat.st_dev == source_stat.st_dev
                and target_stat.st_ino == source_stat.st_ino
            )
        if target == resolved or same_inode:
            raise ValueError(f"Output aliases an input file: {source}")
    if path_exists(target) and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    return target


@contextmanager
def atomic_output(
    output: str, inputs: Sequence[str], *, overwrite: bool = False
) -> Generator[str]:
    """Publish a completed artifact atomically; reserve existing names safely."""
    target = validate_destination(output, inputs, overwrite=overwrite)
    parent = path_dirname(target)
    path_mkdir(parent, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".pic2panning-",
        suffix=get_suffix(target),
        dir=parent,
        delete=False,
    ) as handle:
        temporary = handle.name
    try:
        yield temporary
        validate_destination(output, inputs, overwrite=overwrite)
        if overwrite:
            path_rename(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        path_remove(temporary, non_exist_ok=True)


def load_audio(source: str, directory: str) -> str:
    """Resolve audio; YouTube downloads require the explicit youtube extra."""
    host = (urlparse(source).hostname or "").lower()
    if host not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }:
        return materialize(source, directory)
    name = "youtube-" + hashlib.sha256(source.encode()).hexdigest()
    target = path_join(directory, name + ".%(ext)s")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-f",
            "bestaudio",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            target,
            "--",
            source,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if process.returncode:
        if "No module named yt_dlp" in process.stderr:
            raise DependencyError(
                "Install audio support: uv sync --extra youtube"
            )
        raise MediaError(f"Audio download failed: {process.stderr.strip()}")
    return require_file(path_join(directory, name + ".wav"))
