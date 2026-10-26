"""Render validated timelines through the shared image and media tools."""

import tempfile
from collections.abc import Generator, Iterator
from contextlib import closing

from klogr import get_logger
from klogr.path import get_suffix, path_copy, path_join
from PIL import Image

from pic2panning.schemas.base import Size
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.media import (
    AudioTrack,
    CoverPosition,
    ImageClip,
    RenderRequest,
    VideoClip,
)
from pic2panning.tools.frames import frame_count, image_frames, video_frame
from pic2panning.tools.io import (
    atomic_output,
    is_remote,
    load_audio,
    materialize,
    read_image,
    require_file,
)
from pic2panning.tools.media import (
    MediaInfo,
    concatenate,
    decode_frames,
    encode_frames,
    normalize_audio,
    probe,
    remove_audio,
    to_gif,
)
from pic2panning.tools.text import title_card

logger = get_logger()


def transformed_video_frames(
    source: str, info: MediaInfo, clip: VideoClip, size: Size, fps: int
) -> Generator[Image.Image]:
    """Decode one source frame at a time and crop it to the output viewport."""
    count = frame_count(info.duration / clip.speed, fps)
    with closing(decode_frames(source, info, fps, clip.speed)) as decoded:
        for index, frame in enumerate(decoded):
            with frame:
                yield video_frame(frame, clip, size, index, count)


def repeated_frames(image: Image.Image, count: int) -> Iterator[Image.Image]:
    """Yield independent title frames so encoding can close each promptly."""
    for _ in range(count):
        yield image.copy()


def request_inputs(request: RenderRequest) -> tuple[str, ...]:
    """Collect every source protected against accidental output replacement."""
    sources = [clip.source for clip in request.clips]
    sources.extend(clip.audio.source for clip in request.clips if clip.audio)
    if request.audio:
        sources.append(request.audio.source)
    if request.cover and request.cover.font:
        sources.append(request.cover.font)
    return tuple(sources)


def attach_track(
    source: str,
    output: str,
    duration: float,
    track: AudioTrack | None,
    directory: str,
) -> None:
    """Load an external track or give a silent clip compatible audio."""
    local_track = (
        AudioTrack(
            source=load_audio(track.source, directory),
            start=track.start,
            loop=track.loop,
        )
        if track
        else None
    )
    normalize_audio(
        source,
        output,
        duration,
        track=local_track,
    )


def render_clip(
    clip: ImageClip | VideoClip,
    request: RenderRequest,
    directory: str,
    index: int,
) -> tuple[str, int, bool]:
    """Encode one clip and preserve or replace its audio as requested."""
    raw_path = path_join(directory, f"pixels-{index}.mp4")
    final_path = path_join(directory, f"clip-{index}.mp4")
    original: str | None = None
    info: MediaInfo | None = None
    if isinstance(clip, ImageClip):
        with read_image(clip.source) as image:
            count = encode_frames(
                image_frames(image, clip, request.size, request.fps),
                raw_path,
                request.size,
                request.fps,
            )
    else:
        original = materialize(clip.source, directory)
        info = probe(original)
        with closing(
            transformed_video_frames(
                original, info, clip, request.size, request.fps
            )
        ) as frames:
            count = encode_frames(frames, raw_path, request.size, request.fps)
    duration = count / request.fps
    preserve = (
        isinstance(clip, VideoClip)
        and clip.keep_audio
        and clip.audio is None
        and request.audio is None
        and info is not None
        and info.has_audio
    )
    if preserve and isinstance(clip, VideoClip) and original is not None:
        normalize_audio(
            raw_path,
            final_path,
            duration,
            track=AudioTrack(source=original),
            speed=clip.speed,
        )
    else:
        attach_track(raw_path, final_path, duration, clip.audio, directory)
    return final_path, count, preserve or clip.audio is not None


def render_cover(request: RenderRequest, directory: str) -> tuple[str, int]:
    """Encode a silent title card for its independently specified duration."""
    cover = request.cover
    if cover is None:
        message = "A title card requires cover options."
        raise ValueError(message)
    raw_path = path_join(directory, "cover-pixels.mp4")
    final_path = path_join(directory, "cover.mp4")
    count = frame_count(cover.duration, request.fps)
    with title_card(cover, request.size) as image:
        encode_frames(
            repeated_frames(image, count), raw_path, request.size, request.fps
        )
    normalize_audio(raw_path, final_path, count / request.fps)
    return final_path, count


def render(request: RenderRequest) -> JobResult:
    """Render an image/video timeline atomically while retaining all inputs."""
    for source in request_inputs(request):
        if not is_remote(source):
            require_file(source)
    with (
        atomic_output(
            request.output,
            request_inputs(request),
            overwrite=request.overwrite,
        ) as destination,
        tempfile.TemporaryDirectory(prefix="pic2panning-render-") as directory,
    ):
        clips: list[str] = []
        total = 0
        has_audio = False
        for index, clip in enumerate(request.clips):
            logger.info(f"Rendering clip {index + 1}/{len(request.clips)}")
            path, count, clip_has_audio = render_clip(
                clip, request, directory, index
            )
            clips.append(path)
            total += count
            has_audio = has_audio or clip_has_audio
        if request.cover:
            path, count = render_cover(request, directory)
            if request.cover.position == CoverPosition.START:
                clips.insert(0, path)
            else:
                clips.append(path)
            total += count
        duration = total / request.fps
        combined = path_join(directory, "combined.mp4")
        if len(clips) == 1:
            path_copy(clips[0], combined)
        else:
            concatenate(
                clips,
                path_join(directory, "timeline.txt"),
                combined,
                duration,
            )
        if get_suffix(request.output).lower() == ".gif":
            to_gif(combined, destination, request.fps)
        elif request.audio:
            attach_track(
                combined, destination, duration, request.audio, directory
            )
        elif has_audio:
            path_copy(combined, destination)
        else:
            remove_audio(combined, destination)
    return JobResult(
        outputs=(request.output,),
        width=request.size.width,
        height=request.size.height,
        frames=total,
        duration=duration,
    )
