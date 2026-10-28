"""Audible mix envelopes, timing, sidechain ducking, and CPU beat markers."""

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from pic2panning.errors import MediaError
from pic2panning.schemas.creative.audio import (
    BeatAnalysis,
    BeatRequest,
    Ducking,
    MixTrack,
)
from pic2panning.schemas.creative.composition import ComposeRequest
from pic2panning.tools.creative.audio import detect_beats, snap_to_beat
from pic2panning.tools.creative.composition import compose
from pic2panning.tools.media import ffmpeg, probe, probe_metadata


def write_tone(
    path: Path,
    frequency: float,
    *,
    pulses: bool = False,
    silence_prefix: float = 0,
) -> None:
    """Write a deterministic mono sound at a known gain or pulsed envelope."""
    rate = 48000
    duration = 2.2 if pulses else 1
    time = np.arange(round(duration * rate)) / rate
    envelope = (time % 0.5 < 0.03) if pulses else np.ones(len(time))
    envelope = envelope * (time >= silence_prefix)
    samples = np.rint(
        np.sin(2 * math.pi * frequency * time) * envelope * 16000
    ).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


def tone_level(
    video: str, output: Path, start: float, frequency: float
) -> float:
    """Measure one frequency after decoding an actual mixed audio selection."""
    ffmpeg(
        [
            "-nostdin",
            "-y",
            "-ss",
            str(start),
            "-i",
            video,
            "-t",
            "0.1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    with wave.open(str(output), "rb") as audio:
        samples = (
            np.frombuffer(
                audio.readframes(audio.getnframes()), dtype="<i2"
            ).astype(np.float64)
            / 32768
        )
    time = np.arange(len(samples)) / 48000
    real = float(np.sum(samples * np.cos(2 * math.pi * frequency * time)))
    imaginary = float(np.sum(samples * np.sin(2 * math.pi * frequency * time)))
    return 2 * math.hypot(real, imaginary) / len(samples)


def test_mix_gain_delay_and_fade_have_audible_outcomes(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """An explicit soundtrack begins at its delay and fades before the video ends."""
    tone = tmp_path / "tone.wav"
    write_tone(tone, 400)
    track = MixTrack(
        source=str(tone), start=0.25, duration=0.75, gain_db=-6, fade_out=0.25
    )
    request = composition.model_copy(update={"audio": (track,)})
    result = compose(request)
    assert result.duration == 1
    assert probe(request.output).has_audio
    early = tone_level(request.output, tmp_path / "early.wav", 0.05, 400)
    middle = tone_level(request.output, tmp_path / "middle.wav", 0.4, 400)
    ending = tone_level(request.output, tmp_path / "ending.wav", 0.88, 400)
    assert early < 0.002
    assert 0.16 < middle < 0.19
    assert ending < middle * 0.5


def test_voice_ducking_reduces_music_during_voice(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """Music energy falls during audible voice and recovers afterward."""
    music, voice = tmp_path / "music.wav", tmp_path / "voice.wav"
    write_tone(music, 400)
    write_tone(voice, 2000)
    request = composition.model_copy(
        update={
            "audio": (
                MixTrack(
                    source=str(music),
                    duration=1,
                    gain_db=-12,
                    ducking=Ducking(threshold_db=-35, ratio=10, release_ms=50),
                ),
                MixTrack(
                    source=str(voice), role="voice", start=0.3, duration=0.35
                ),
            )
        }
    )
    compose(request)
    audio_stream = next(
        stream
        for stream in probe_metadata(request.output).streams
        if stream.codec_type == "audio"
    )
    assert audio_stream.duration is not None
    assert abs(float(audio_stream.duration) - 1) < 0.03
    before = tone_level(request.output, tmp_path / "before.wav", 0.1, 400)
    during = tone_level(request.output, tmp_path / "during.wav", 0.45, 400)
    after = tone_level(request.output, tmp_path / "after.wav", 0.85, 400)
    assert during < before * 0.4
    assert after > during * 2
    assert after > before * 0.8


def test_audio_loop_excludes_trimmed_source_prefix(
    composition: ComposeRequest, tmp_path: Path
) -> None:
    """A selected final quarter repeats without bringing back skipped silence."""
    source = tmp_path / "prefix.wav"
    write_tone(source, 400, silence_prefix=0.75)
    track = MixTrack(
        source=str(source), source_start=0.75, duration=1, loop=True
    )
    request = composition.model_copy(update={"audio": (track,)})
    compose(request)
    for index, moment in enumerate((0.05, 0.45, 0.85)):
        assert (
            tone_level(
                request.output,
                tmp_path / f"selection-{index}.wav",
                moment,
                400,
            )
            > 0.3
        )


def test_cpu_beat_detection_finds_pulse_times_and_snaps(
    tmp_path: Path,
) -> None:
    """Spectral onset markers follow known audible pulses without model packages."""
    source, output = tmp_path / "pulses.wav", tmp_path / "beats.json"
    write_tone(source, 400, pulses=True)
    detect_beats(BeatRequest(source=str(source), output=str(output)))
    analysis = BeatAnalysis.model_validate_json(output.read_text())
    for expected in (0, 0.5, 1, 1.5, 2):
        assert any(
            abs(marker - expected) < 0.06 for marker in analysis.markers
        )
    assert analysis.bpm is not None
    assert 110 < analysis.bpm < 130
    assert snap_to_beat(0.47, analysis.markers) == min(
        analysis.markers, key=lambda marker: abs(marker - 0.47)
    )
    assert snap_to_beat(0.25, analysis.markers, tolerance=0.01) == 0.25


def test_beat_analysis_protects_input_aliases(tmp_path: Path) -> None:
    """An audio file cannot be replaced through a differently named hardlink."""
    source = tmp_path / "source.wav"
    write_tone(source, 400)
    alias = tmp_path / "beats.json"
    alias.hardlink_to(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="aliases an input"):
        detect_beats(
            BeatRequest(source=str(source), output=str(alias), overwrite=True)
        )
    assert source.read_bytes() == original


def test_empty_beat_selection_fails_without_publishing(tmp_path: Path) -> None:
    """An offset beyond the audio raises an error instead of a valid empty result."""
    source, output = tmp_path / "tone.wav", tmp_path / "beats.json"
    write_tone(source, 400)
    with pytest.raises(MediaError, match="no samples"):
        detect_beats(
            BeatRequest(source=str(source), output=str(output), start=2)
        )
    assert not output.exists()
