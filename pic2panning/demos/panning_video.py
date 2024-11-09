"""Demo for panning video."""

import tempfile
from typing import cast

import tyro

from pic2panning.demos.add_audio_to_video import demo_add_audio_to_video
from pic2panning.main import main_videos as main
from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import VALID_VIDEO_PROCESS, CoverImageOpts, Opts
from pic2panning.utils.process import read_video

logger = get_logger()


def demo_panning_video(
    videos: list[str] = [
        "https://videos.pexels.com/video-files/4133023/4133023-uhd_3840_2160_30fps.mp4"
    ],
    /,
    speed: int = 5,
    output_file: str = "assets/demo_panning_video.mp4",
    with_cover: bool = False,
    output_size: tuple[int, int] = (1080, 1920),
    song: str = "https://www.youtube.com/watch?v=v2AC41dglnM",
) -> None:
    """Demo function for zoom-in."""
    if song:
        _tempfile = tempfile.NamedTemporaryFile(
            delete=True, suffix=".mp4"
        ).name
    else:
        _tempfile = output_file
    opts = Opts(
        data=videos,
        speed=speed,
        output_file=_tempfile,
        output_size=output_size,
        process=[cast(VALID_VIDEO_PROCESS, "panning-lr")],
        add_image=(
            CoverImageOpts(
                text="Este video no estaba en 9:16\nY para no cortarlo, queria crearle un efecto bouncing\n ----- \n... Así que escribí un código que lo hiciera por mi",
                size=tuple(read_video(videos[0]).size),
                font_color="white",
                justification="center",
                time=1,
                insert_at="start",
                font_path="/Library/Fonts/Arial Unicode.ttf",
            )
            if with_cover
            else None
        ),
    )
    main(opts)
    if song:
        demo_add_audio_to_video(
            [opts.output_file],
            song=song,
            start_at=83,
            output_file=output_file,
            output_size=output_size,
        )


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_panning_video)
