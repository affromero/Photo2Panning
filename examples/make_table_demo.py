"""Build the public depth-fitted letter-drop composition and its preview job."""

import sys

import tyro
from klogr.path import path_join, path_open

from pic2panning import (
    CameraKeyframe,
    ComposeRequest,
    DepthStyle,
    ImageLayer,
    LayerKeyframe,
    MixTrack,
    PreviewRequest,
    Size,
    TextLayer,
    run_job,
)
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.surface import (
    LetterDrop,
    SolidText,
    SurfaceAttachment,
)
from pic2panning.schemas.creative.typography import TextCue
from pic2panning.schemas.media import Encoding, EncodingPreset
from pic2panning.tools.io import atomic_output

PHOTO = "examples/media/table.jpg"
DEPTH = "examples/media/table-depth/table.depth.json"
FONT = "examples/media/fonts/Anton-Regular.ttf"
MUSIC = "examples/media/demo-music.wav"
DURATION = 7.0


class Options(PydanticBaseModel):
    """Write portable table jobs and optionally render their movie."""

    output_directory: str = "outputs/table-demo"
    jobs_directory: str = "examples"
    draft: bool = False
    render: bool = False
    overwrite: bool = False


def table_story(output: str, *, draft: bool) -> ComposeRequest:
    """Fall onto a depth-fitted tabletop, hold, and morph the resting sentence."""
    return ComposeRequest(
        canvas=Size(width=1080, height=1920),
        duration=DURATION,
        camera=(CameraKeyframe(x=540, y=960),),
        background="#ede5d6",
        layers=(
            ImageLayer(
                id="table",
                source=PHOTO,
                duration=DURATION,
                size=Size(width=1068, height=1600),
                keyframes=(LayerKeyframe(x=540, y=820),),
                depth=DepthStyle(metadata=DEPTH),
            ),
            TextLayer(
                id="falling-letters",
                text="MAKE ROOM",
                text_cues=(
                    TextCue(
                        at=4,
                        text="MAKE TIME",
                        transition=1.1,
                        stagger=0.3,
                        blur=0,
                        travel=0.12,
                    ),
                ),
                font=FONT,
                font_size=240,
                color="#eee1c3",
                spacing=10,
                duration=DURATION,
                size=Size(width=1200, height=300),
                keyframes=(LayerKeyframe(x=640, y=1740, scale=0.9),),
                surface=SurfaceAttachment(
                    layer_id="table",
                    region=(220, 1660, 1120, 1890),
                    roll=-6,
                    depth_tolerance_metres=0.005,
                    feather_metres=0.003,
                ),
                letter_drop=LetterDrop(
                    duration=2,
                    stagger=0.6,
                    height=12,
                    rotation=(65, -25, -28),
                    bounce=0.12,
                ),
                solid=SolidText(),
            ),
        ),
        audio=(MixTrack(source=MUSIC, duration=DURATION, fade_out=0.6),),
        beats=tuple(index / 2 for index in range(14)),
        output=output,
        size=Size(width=360, height=640)
        if draft
        else Size(width=1080, height=1920),
        fps=12 if draft else 30,
        shutter_angle=0,
        encoding=Encoding(
            crf=22 if draft else 16,
            preset=EncodingPreset.FAST if draft else EncodingPreset.SLOW,
        ),
    )


def main(options: Options) -> None:
    """Write the same validated requests used by Python, Studio, and the CLI."""
    composition = table_story(
        path_join(options.output_directory, "table-letters.mp4"),
        draft=options.draft,
    )
    preview = PreviewRequest(
        composition=composition,
        output=path_join(options.output_directory, "table-cover.png"),
        at=3,
    )
    for name, request in (
        ("compose-table.json", composition),
        ("preview-table.json", preview),
    ):
        with (
            atomic_output(
                path_join(options.jobs_directory, name),
                (PHOTO, DEPTH, FONT, MUSIC),
                overwrite=options.overwrite,
            ) as temporary,
            path_open(temporary, "w") as handle,
        ):
            handle.write(request.model_dump_json(indent=2) + "\n")
    if options.render:
        request = composition.model_copy(
            update={"overwrite": options.overwrite}
        )
        sys.stdout.write(run_job(request).model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Options))
