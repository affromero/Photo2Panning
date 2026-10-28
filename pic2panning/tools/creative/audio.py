"""Mix explicit soundtracks and detect spectral onsets with bounded memory."""

import statistics
import tempfile
import wave
from collections import deque
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from klogr.path import path_join, path_open

from pic2panning.errors import MediaError
from pic2panning.schemas.creative.audio import (
    BeatAnalysis,
    BeatRequest,
    MixTrack,
)
from pic2panning.schemas.jobs import JobResult
from pic2panning.schemas.media import AudioTrack
from pic2panning.tools.io import atomic_output, load_audio
from pic2panning.tools.media import (
    ffmpeg,
    normalize_audio,
    probe_metadata,
    selected_audio,
)

SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 512


def snap_to_beat(
    seconds: float, markers: tuple[float, ...], tolerance: float = 0.15
) -> float:
    """Snap a timeline position to its nearest marker within tolerance."""
    if tolerance < 0:
        raise ValueError("Beat snap tolerance must be nonnegative")
    if not markers:
        return seconds
    nearest = min(markers, key=lambda marker: abs(marker - seconds))
    return nearest if abs(nearest - seconds) <= tolerance else seconds


def track_envelope(track: MixTrack, duration: float) -> str:
    """Build one source-relative envelope followed by its timeline delay."""
    filters = [
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo",
        "asetpts=N/SR/TB",
        f"apad=whole_len={round(track.duration * 48000)}",
        f"atrim=end_sample={round(track.duration * 48000)}",
        f"volume={track.gain_db:.12g}dB",
    ]
    if track.fade_in:
        filters.append(f"afade=t=in:st=0:d={track.fade_in:.12g}")
    if track.fade_out:
        filters.append(
            f"afade=t=out:st={track.duration - track.fade_out:.12g}:d={track.fade_out:.12g}"
        )
    filters.extend(
        [
            f"adelay={round(track.start * 48000)}S:all=1",
            f"apad=whole_len={round(duration * 48000)}",
            f"atrim=end_sample={round(duration * 48000)}",
            "asetpts=N/SR/TB",
        ]
    )
    return ",".join(filters)


def encode_pcm(arguments: Sequence[str], output: str, duration: float) -> None:
    """Stage floating-point sound without clipping gain or losing the tail."""
    ffmpeg(
        [
            *arguments,
            "-c:a",
            "pcm_f32le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{duration:.12g}",
            output,
        ]
    )
    metadata = probe_metadata(output)
    audio = next(
        (
            stream
            for stream in metadata.streams
            if stream.codec_type == "audio"
        ),
        None,
    )
    if audio is None or audio.duration in {None, "N/A"}:
        raise MediaError("Audio PCM output has no measured duration")
    if abs(float(audio.duration) - duration) > 1 / 48000:
        raise MediaError(
            f"Audio PCM duration {audio.duration} does not match {duration:g} seconds"
        )


def stage_track(
    track: MixTrack, duration: float, directory: str, index: int
) -> str:
    """Apply source selection and the full envelope before shared mixing."""
    local = load_audio(track.source, directory)
    output = path_join(directory, f"track-{index}.wav")
    with selected_audio(
        AudioTrack(source=local, start=track.source_start, loop=track.loop),
        track.duration,
    ) as selected:
        if selected is None:
            raise ValueError("A soundtrack source is required")
        arguments = ["-nostdin", "-y"]
        if selected.loop:
            arguments.extend(["-stream_loop", "-1"])
        arguments.extend(
            [
                "-i",
                selected.source,
                "-map",
                "0:a:0",
                "-vn",
                "-af",
                track_envelope(track, duration),
            ]
        )
        encode_pcm(arguments, output, duration)
    return output


def voice_bus(
    tracks: Sequence[MixTrack],
    paths: Sequence[str],
    directory: str,
    duration: float,
) -> str | None:
    """Mix full-length voice stems once for independent audible/detector reads."""
    voices = [
        path
        for track, path in zip(tracks, paths, strict=True)
        if track.role == "voice"
    ]
    if not voices:
        return None
    if len(voices) == 1:
        return voices[0]
    arguments = ["-nostdin", "-y"]
    for path in voices:
        arguments.extend(["-i", path])
    inputs = "".join(f"[{index}:a:0]" for index in range(len(voices)))
    graph = f"{inputs}amix=inputs={len(voices)}:normalize=0:dropout_transition=0,atrim=end_sample={round(duration * 48000)},asetpts=N/SR/TB[voices]"
    output = path_join(directory, "voices.wav")
    encode_pcm(
        [*arguments, "-filter_complex", graph, "-map", "[voices]"],
        output,
        duration,
    )
    return output


def stage_ducked(
    track: MixTrack, source: str, voices: str, duration: float, output: str
) -> str:
    """Retain the verified music tail before compression and exact sample trim.

    Keep EOF outside both compressor inputs until its internal queues drain.
    Padding begins after each complete staged stem; the output stops at the
    declared timeline sample count.
    """
    options = track.ducking
    if options is None:
        raise ValueError("Ducked audio requires compressor options")
    threshold = 10 ** (options.threshold_db / 20)
    graph = (
        "[0:a:0]apad[main];[1:a:0]apad[detector];"
        f"[main][detector]sidechaincompress=threshold={threshold:.12g}:"
        f"ratio={options.ratio:.12g}:attack={options.attack_ms:.12g}:"
        f"release={options.release_ms:.12g},"
        f"atrim=end_sample={round(duration * 48000)},asetpts=N/SR/TB[ducked]"
    )
    encode_pcm(
        [
            "-nostdin",
            "-y",
            "-i",
            source,
            "-i",
            voices,
            "-filter_complex",
            graph,
            "-map",
            "[ducked]",
        ],
        output,
        duration,
    )
    return output


def stage_mix(
    tracks: Sequence[MixTrack],
    paths: Sequence[str],
    voices: str | None,
    duration: float,
    directory: str,
) -> str:
    """Mix completed stems after independent voice-controlled compression."""
    audible: list[str] = []
    for index, (track, path) in enumerate(zip(tracks, paths, strict=True)):
        if track.role == "voice":
            continue
        selected = path
        if track.ducking is not None:
            if voices is None:
                raise ValueError("Ducking requires a voice track")
            selected = stage_ducked(
                track,
                path,
                voices,
                duration,
                path_join(directory, f"ducked-{index}.wav"),
            )
        audible.append(selected)
    if voices is not None:
        audible.append(voices)
    arguments = ["-nostdin", "-y"]
    for path in audible:
        arguments.extend(["-i", path])
    inputs = "".join(f"[{index}:a:0]" for index in range(len(audible)))
    graph = f"{inputs}amix=inputs={len(audible)}:normalize=0:dropout_transition=0,atrim=end_sample={round(duration * 48000)},asetpts=N/SR/TB[mixed]"
    output = path_join(directory, "mixed.wav")
    encode_pcm(
        [*arguments, "-filter_complex", graph, "-map", "[mixed]"],
        output,
        duration,
    )
    return output


def mix_audio(
    video: str, output: str, tracks: tuple[MixTrack, ...], duration: float
) -> None:
    """Attach gain envelopes and ducking through the canonical FFmpeg tool."""
    if not tracks:
        raise ValueError("Audio mixing requires at least one track")
    with tempfile.TemporaryDirectory(prefix="pic2panning-mix-") as directory:
        paths = tuple(
            stage_track(track, duration, directory, index)
            for index, track in enumerate(tracks)
        )
        voices = voice_bus(tracks, paths, directory, duration)
        mixed = stage_mix(tracks, paths, voices, duration, directory)
        normalize_audio(video, output, duration, AudioTrack(source=mixed))


def analyze_pcm(path: str, request: BeatRequest) -> BeatAnalysis:
    """Track positive spectral flux over a small rolling onset window."""
    history: deque[float] = deque(maxlen=43)
    previous = np.zeros(WINDOW_SIZE // 2 + 1)
    window = np.hanning(WINDOW_SIZE)
    markers: list[float] = []
    processed = 0
    with wave.open(path, "rb") as audio:
        duration = audio.getnframes() / SAMPLE_RATE
        overlap = np.zeros(HOP_SIZE)
        while raw := audio.readframes(HOP_SIZE):
            samples = (
                np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
            )
            padded = np.zeros(HOP_SIZE)
            padded[: len(samples)] = samples
            block = np.concatenate((overlap, padded))
            spectrum = np.abs(np.fft.rfft(block * window))
            flux = float(np.maximum(spectrum - previous, 0).sum())
            baseline = statistics.fmean(history) if history else 0
            moment = processed / SAMPLE_RATE
            loud = float(np.square(samples).mean()) > 1e-6
            separated = (
                not markers or moment - markers[-1] >= request.minimum_gap
            )
            if (
                loud
                and flux > max(0.05, baseline * request.sensitivity)
                and separated
            ):
                markers.append(moment)
            history.append(flux)
            previous = spectrum
            overlap = padded
            processed += len(samples)
    intervals = [right - left for left, right in pairwise(markers)]
    bpm = 60 / statistics.median(intervals) if intervals else None
    return BeatAnalysis(markers=tuple(markers), duration=duration, bpm=bpm)


def detect_beats(request: BeatRequest) -> JobResult:
    """Decode a mono PCM stream and atomically publish detected beat markers."""
    with (
        atomic_output(
            request.output, (request.source,), overwrite=request.overwrite
        ) as target,
        tempfile.TemporaryDirectory(prefix="pic2panning-beats-") as directory,
    ):
        source = load_audio(request.source, directory)
        pcm = path_join(directory, "analysis.wav")
        arguments = [
            "-nostdin",
            "-y",
            "-ss",
            f"{request.start:.12g}",
            "-i",
            source,
            "-map",
            "0:a:0",
            "-vn",
        ]
        if request.duration is not None:
            arguments.extend(["-t", f"{request.duration:.12g}"])
        arguments.extend(
            ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", pcm]
        )
        ffmpeg(arguments)
        analysis = analyze_pcm(pcm, request)
        if not analysis.duration:
            raise MediaError("Audio selection contains no samples")
        with path_open(target, "w") as handle:
            handle.write(analysis.model_dump_json(indent=2) + "\n")
    return JobResult(
        outputs=(request.output,),
        duration=analysis.duration,
        details=f"Detected {len(analysis.markers)} beat markers",
    )
