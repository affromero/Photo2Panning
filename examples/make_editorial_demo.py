"""Build the single-photo editorial story through the public composition API."""

import sys

import tyro
from klogr.path import path_join, path_open

from pic2panning import (
    CameraKeyframe,
    ComposeRequest,
    Easing,
    ImageLayer,
    LayerKeyframe,
    PreviewRequest,
    Size,
    TextLayer,
    run_job,
)
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.media import Alignment, Encoding, EncodingPreset
from pic2panning.tools.creative.composition import (
    composition_inputs,
    reframe_composition,
)
from pic2panning.tools.io import atomic_output

PHOTO = "examples/media/waterfall.jpg"
KICKER = "examples/media/faroe-islands-label.png"
SERIF = "examples/media/fonts/InstrumentSerif-Regular.ttf"
SANS = "examples/media/fonts/Manrope-Medium.ttf"
CREAM = "#f6f0e4"
DURATION = 5.5


class Options(PydanticBaseModel):
    """Write portable editorial jobs and optionally render their movie."""

    output_directory: str = "outputs/editorial-demo"
    jobs_directory: str = "examples"
    draft: bool = False
    render: bool = False
    overwrite: bool = False


def location_label(
    identifier: str,
    text: str,
    *,
    start: float,
    duration: float,
    keyframes: tuple[LayerKeyframe, ...],
) -> TextLayer:
    """Keep secondary location changes separate from the main title."""
    return TextLayer(
        id=identifier,
        text=text,
        font=SANS,
        font_size=24,
        color=CREAM,
        align=Alignment.LEFT,
        start=start,
        duration=duration,
        size=Size(width=900, height=60),
        space="viewport",
        keyframes=keyframes,
    )


def editorial_story(output: str, *, draft: bool) -> ComposeRequest:
    """Hold a location title while the photograph and small labels move."""
    request = ComposeRequest(
        canvas=Size(width=1920, height=1280),
        size=Size(width=1080, height=1920),
        duration=DURATION,
        camera=(
            CameraKeyframe(x=1030, y=640, ease=Easing.EASE_IN_OUT),
            CameraKeyframe(at=DURATION, x=1090, y=640),
        ),
        layers=(
            ImageLayer(
                id="waterfall",
                source=PHOTO,
                duration=DURATION,
                size=Size(width=1920, height=1280),
                keyframes=(LayerKeyframe(x=960, y=640),),
            ),
            ImageLayer(
                id="geography",
                source=KICKER,
                duration=DURATION,
                size=Size(width=900, height=60),
                space="viewport",
                keyframes=(LayerKeyframe(x=538, y=282),),
            ),
            TextLayer(
                id="place",
                text="Múlafossur",
                font=SERIF,
                font_size=205,
                color=CREAM,
                align=Alignment.LEFT,
                duration=DURATION,
                size=Size(width=900, height=260),
                space="viewport",
                keyframes=(LayerKeyframe(x=530, y=440),),
            ),
            location_label(
                "island",
                "VÁGAR",
                start=0,
                duration=2.4,
                keyframes=(
                    LayerKeyframe(x=538, y=550),
                    LayerKeyframe(at=2, x=538, y=550, ease=Easing.EASE_IN_OUT),
                    LayerKeyframe(at=2.35, x=538, y=542, opacity=0),
                ),
            ),
            location_label(
                "village",
                "GÁSADALUR",
                start=2.45,
                duration=3.05,
                keyframes=(
                    LayerKeyframe(
                        x=538,
                        y=558,
                        opacity=0,
                        ease=Easing.EASE_OUT,
                    ),
                    LayerKeyframe(at=0.5, x=538, y=550),
                ),
            ),
        ),
        output=output,
        fps=24 if draft else 30,
        shutter_angle=180,
        shutter_samples=4 if draft else 8,
        encoding=Encoding(
            crf=18 if draft else 16,
            preset=EncodingPreset.FAST if draft else EncodingPreset.SLOW,
        ),
    )
    if draft:
        return reframe_composition(request, Size(width=540, height=960))
    return request


def main(options: Options) -> None:
    """Publish the same validated requests consumed by Studio and the CLI."""
    composition = editorial_story(
        path_join(options.output_directory, "waterfall-editorial.mp4"),
        draft=options.draft,
    )
    preview = PreviewRequest(
        composition=composition,
        output=path_join(options.output_directory, "waterfall-cover.png"),
        at=1,
    )
    for name, request in (
        ("compose-editorial.json", composition),
        ("preview-editorial.json", preview),
    ):
        with (
            atomic_output(
                path_join(options.jobs_directory, name),
                composition_inputs(composition),
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
