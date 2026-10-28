"""Create reproducible social examples with an original synthesized track."""

import math
import sys
import wave

import numpy as np
import tyro
from klogr.path import path_join, path_open

from pic2panning import (
    CameraKeyframe,
    ComposeRequest,
    Easing,
    ImageLayer,
    LayerKeyframe,
    MixTrack,
    PreviewRequest,
    ProjectDocument,
    ProjectNode,
    ProjectRequest,
    SafeInsets,
    Size,
    SocialPackRequest,
    TextLayer,
    run_job,
)
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.audio import BeatRequest
from pic2panning.schemas.editing import Adjust, EditRequest
from pic2panning.schemas.jobs import Job
from pic2panning.schemas.media import Encoding, EncodingPreset
from pic2panning.tools.io import atomic_output, read_image

SAMPLE_RATE = 48000
DURATION = 8.0
MUSIC = "examples/media/demo-music.wav"
PHOTO = "examples/media/waterfall.jpg"
SECOND_PHOTO = "examples/media/mountains.jpg"


class Options(PydanticBaseModel):
    """Build example jobs, optionally rendering their final social package."""

    output_directory: str = "outputs/social-demo"
    draft: bool = False
    render: bool = False
    overwrite: bool = False
    jobs_directory: str = "examples"


def make_music(output: str, *, overwrite: bool) -> None:
    """Synthesize an original, quiet 120 BPM instrumental for the examples."""
    count = round(DURATION * SAMPLE_RATE)
    seconds = np.arange(count) / SAMPLE_RATE
    samples = np.zeros(count, dtype=np.float64)
    chords = ((220, 261.63, 329.63, 440), (174.61, 220, 261.63, 349.23))
    for beat in range(16):
        start = beat / 2
        elapsed = np.maximum(seconds - start, 0)
        active = (seconds >= start) & (seconds < start + 1.2)
        envelope = (1 - np.exp(-elapsed * 80)) * np.exp(-elapsed * 4)
        for note in chords[(beat // 8) % 2]:
            samples += (
                np.sin(2 * math.pi * note * elapsed)
                * envelope
                * active
                * 0.045
            )
        kick = np.sin(
            2
            * math.pi
            * (45 * elapsed + 30 * (1 - np.exp(-elapsed * 25)) / 25)
        )
        samples += kick * np.exp(-elapsed * 35) * active * 0.2
    fade = np.minimum(seconds / 0.02, 1) * np.minimum(
        (DURATION - seconds) / 0.3, 1
    )
    pcm = np.rint(np.clip(samples * fade, -1, 1) * 32767).astype("<i2")
    with (
        atomic_output(output, (), overwrite=overwrite) as destination,
        wave.open(destination, "wb") as audio,
    ):
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())


def story(output: str, *, draft: bool) -> ComposeRequest:
    """Create one continuous camera path across two independent public photos."""
    height = 1280
    with read_image(PHOTO) as first, read_image(SECOND_PHOTO) as second:
        first_width = round(height * first.width / first.height)
        second_width = round(height * second.width / second.height)
    width = first_width + second_width
    size = (
        Size(width=360, height=640) if draft else Size(width=1080, height=1920)
    )
    text_scale = size.width / 1080
    return ComposeRequest(
        canvas=Size(width=width, height=height),
        duration=DURATION,
        camera=(
            CameraKeyframe(
                at=0,
                x=first_width * 0.34,
                y=height / 2,
                ease=Easing.EASE_IN_OUT,
            ),
            CameraKeyframe(
                at=2, x=first_width * 0.65, y=height / 2, zoom=1.12
            ),
            CameraKeyframe(
                at=3,
                x=first_width * 0.65,
                y=height / 2,
                zoom=1.12,
                ease=Easing.EASE_IN_OUT,
            ),
            CameraKeyframe(
                at=6, x=first_width + second_width / 2, y=height / 2
            ),
            CameraKeyframe(
                at=8, x=first_width + second_width / 2, y=height / 2
            ),
        ),
        layers=(
            ImageLayer(
                id="coast",
                source="outputs/social-demo/prepared-waterfall.png",
                size=Size(width=first_width, height=height),
                duration=DURATION,
                keyframes=(LayerKeyframe(x=first_width / 2, y=height / 2),),
            ),
            ImageLayer(
                id="aurora",
                source=SECOND_PHOTO,
                size=Size(width=second_width, height=height),
                duration=DURATION,
                keyframes=(
                    LayerKeyframe(
                        x=first_width + second_width / 2, y=height / 2
                    ),
                ),
            ),
            TextLayer(
                id="title",
                text="FROM THE COAST\nTO THE NORTHERN LIGHTS",
                space="viewport",
                size=Size(
                    width=round(960 * text_scale),
                    height=round(240 * text_scale),
                ),
                font_size=round(60 * text_scale),
                stroke_width=max(1, round(2 * text_scale)),
                shadow_offset=(round(2 * text_scale), round(3 * text_scale)),
                duration=DURATION,
                keyframes=(
                    LayerKeyframe(
                        at=0, x=size.width / 2, y=size.height * 0.17, opacity=0
                    ),
                    LayerKeyframe(
                        at=0.5, x=size.width / 2, y=size.height * 0.17
                    ),
                    LayerKeyframe(
                        at=6.5, x=size.width / 2, y=size.height * 0.17
                    ),
                    LayerKeyframe(
                        at=7.5,
                        x=size.width / 2,
                        y=size.height * 0.17,
                        opacity=0,
                    ),
                ),
            ),
        ),
        audio=(
            MixTrack(
                source=MUSIC, duration=DURATION, gain_db=-2, fade_out=0.5
            ),
        ),
        beats=tuple(beat / 2 for beat in range(16)),
        output=output,
        size=size,
        fps=12 if draft else 30,
        shutter_angle=0 if draft else 180,
        shutter_samples=8,
        encoding=Encoding(
            crf=22 if draft else 16,
            preset=EncodingPreset.FAST if draft else EncodingPreset.SLOW,
        ),
    )


def write_job(path: str, request: Job, *, overwrite: bool) -> None:
    """Publish example JSON through the same atomic output protection."""
    with (
        atomic_output(
            path, (PHOTO, SECOND_PHOTO, MUSIC), overwrite=overwrite
        ) as target,
        path_open(target, "w") as handle,
    ):
        handle.write(request.model_dump_json(indent=2) + "\n")


def main(options: Options) -> None:
    """Write reproducible jobs and optionally render one selected project."""
    make_music(MUSIC, overwrite=options.overwrite)
    prepared = path_join(options.output_directory, "prepared-waterfall.png")
    preparation = EditRequest(
        source=PHOTO, output=prepared, operations=(Adjust(brightness=1.04),)
    )
    composition = story(
        path_join(options.output_directory, "story.mp4"), draft=options.draft
    )
    first = composition.layers[0]
    if not isinstance(first, ImageLayer):
        raise TypeError("The story must start with its prepared photo")
    composition = composition.model_copy(
        update={
            "layers": (
                first.model_copy(update={"source": prepared}),
                *composition.layers[1:],
            )
        }
    )
    pack = SocialPackRequest(
        composition=composition,
        output_directory=path_join(options.output_directory, "social"),
        cover_at=2,
        carousel_at=(0, 1, 2, 3, 4, 5, 6, 7),
        safe_insets=SafeInsets(),
    )
    project = ProjectRequest(
        project=ProjectDocument(
            name="Coast to northern lights",
            nodes=(
                ProjectNode(id="prepare", job=preparation),
                ProjectNode(
                    id="film", job=composition, depends_on=("prepare",)
                ),
                ProjectNode(id="social", job=pack, depends_on=("prepare",)),
            ),
        ),
        workspace=".",
        output=path_join(options.output_directory, "project-report.json"),
        targets=("social",),
        overwrite=options.overwrite,
    )
    standalone = composition.model_copy(
        update={
            "layers": (
                first.model_copy(update={"source": PHOTO}),
                *composition.layers[1:],
            )
        }
    )
    requests: dict[str, Job] = {
        "compose.json": standalone,
        "social-pack.json": pack.model_copy(
            update={"composition": standalone}
        ),
        "project.json": project,
        "preview.json": PreviewRequest(
            composition=standalone,
            at=2,
            output=path_join(options.output_directory, "preview.png"),
        ),
        "beats.json": BeatRequest(
            source=MUSIC,
            output=path_join(options.output_directory, "beats.json"),
        ),
    }
    for filename, request in requests.items():
        write_job(
            path_join(options.jobs_directory, filename),
            request,
            overwrite=options.overwrite,
        )
    if options.render:
        sys.stdout.write(run_job(project).model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Options))
