"""Demo for speed up with video."""

import tempfile
from typing import cast

import tyro

from pic2panning.demos.add_audio_to_video import demo_add_audio_to_video
from pic2panning.main import main_videos as main
from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import VALID_VIDEO_PROCESS, CoverImageOpts, Opts
from pic2panning.utils.process import read_video

logger = get_logger()


def demo_speed_up(
    videos: list[str] = [
        "https://videos.pexels.com/video-files/4133023/4133023-uhd_3840_2160_30fps.mp4"
    ],
    /,
    speed: float = 25.0,
    output_file: str = "assets/demo_speed.mp4",
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
        output_file=_tempfile,
        output_size=output_size,
        process=[cast(VALID_VIDEO_PROCESS, "speed")],
        speed=speed,
        add_image=(
            CoverImageOpts(
                text="I wanted to add audio to this video but I was lazy to download an app\nand IG somehow destroys the quality when there is an audio?\n ----- \n...So I wrote an algorithm to add it from a YouTube link.",
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
    tyro.cli(demo_speed_up)
