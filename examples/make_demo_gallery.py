"""Lay out actual social covers and carousel stills using the compositor."""

from tempfile import TemporaryDirectory

import tyro
from klogr.path import path_expanduser, path_join

from pic2panning import (
    Adjust,
    CameraKeyframe,
    ComposeRequest,
    EditRequest,
    ImageLayer,
    LayerKeyframe,
    PreviewRequest,
    Size,
    TextLayer,
    edit,
    preview,
)
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.tools.creative.composition import composition_inputs
from pic2panning.tools.io import validate_destination

FONT = "examples/media/fonts/Anton-Regular.ttf"


class Options(PydanticBaseModel):
    """Choose an exported social pack and a publication image destination."""

    source_directory: str = "outputs/social-demo/social"
    output: str = "examples/media/social-formats.jpg"
    overwrite: bool = False


def label(
    identifier: str,
    text: str,
    center: tuple[float, float],
    *,
    width: int = 324,
    font_size: int = 28,
) -> TextLayer:
    """Keep publication labels in an ordinary reusable text layer."""
    return TextLayer(
        id=identifier,
        text=text,
        font=FONT,
        font_size=font_size,
        color="#241d14",
        duration=1,
        size=Size(width=width, height=40),
        keyframes=(LayerKeyframe(x=center[0], y=center[1]),),
    )


def gallery(directory: str, output: str) -> ComposeRequest:
    """Use the exported pixels without recreating their source composition."""
    layers: list[ImageLayer | TextLayer] = []
    formats = (
        ("portrait", "9:16 PORTRAIT", Size(width=324, height=576), 180),
        ("feed", "4:5 FEED", Size(width=324, height=405), 576),
        ("square", "1:1 SQUARE", Size(width=324, height=324), 972),
    )
    for name, title, size, x in formats:
        layers.extend(
            (
                ImageLayer(
                    id=name,
                    source=path_join(directory, name + "-cover.png"),
                    duration=1,
                    size=size,
                    keyframes=(LayerKeyframe(x=x, y=72 + size.height / 2),),
                ),
                label(name + "-label", title, (x, 30)),
            )
        )
    layers.extend(
        (
            label("cover", "COVERS", (180, 680)),
            label("carousel", "CAROUSEL FRAMES", (714, 530), width=720),
        )
    )
    for index, moment, x in ((1, 0, 462), (6, 5, 714), (8, 7, 966)):
        layers.extend(
            (
                ImageLayer(
                    id=f"slide-{index}",
                    source=path_join(
                        directory, f"square-carousel-{index:03d}.png"
                    ),
                    duration=1,
                    size=Size(width=216, height=216),
                    keyframes=(LayerKeyframe(x=x, y=672),),
                ),
                label(
                    f"slide-time-{index}",
                    f"{moment} SECONDS",
                    (x, 808),
                    width=216,
                    font_size=22,
                ),
            )
        )
    return ComposeRequest(
        canvas=Size(width=1152, height=840),
        size=Size(width=1152, height=840),
        duration=1,
        camera=(CameraKeyframe(x=576, y=420),),
        layers=tuple(layers),
        background="#ede5d6",
        output=output,
    )


def main(options: Options) -> None:
    """Render the contact sheet, then publish JPEG through the image API."""
    directory = path_expanduser(options.source_directory)
    with TemporaryDirectory(prefix="pic2panning-demo-gallery-") as temporary:
        image = path_join(temporary, "gallery.png")
        request = gallery(directory, path_join(temporary, "gallery.mp4"))
        validate_destination(
            options.output,
            composition_inputs(request),
            overwrite=options.overwrite,
        )
        preview(PreviewRequest(composition=request, output=image))
        edit(
            EditRequest(
                source=image,
                output=options.output,
                operations=(Adjust(),),
                overwrite=options.overwrite,
            )
        )


if __name__ == "__main__":
    main(tyro.cli(Options))
