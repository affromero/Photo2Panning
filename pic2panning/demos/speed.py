"""Demo for speed up with video."""

from typing import cast

import moviepy.editor as mpe
import tyro

from pic2panning.demos.add_audio_to_video import demo_add_audio_to_video
from pic2panning.main import main_videos as main
from pic2panning.utils.logger import get_logger
from pic2panning.utils.options import VALID_VIDEO_PROCESS, CoverImageOpts, Opts

logger = get_logger()


def demo_speed_up(
    videos: list[str] = [
        "https://videos.pexels.com/video-files/4133023/4133023-uhd_3840_2160_30fps.mp4"
    ],
    /,
    speed: int = 25,
) -> None:
    """Demo function for zoom-in."""
    opts = Opts(
        data=videos,
        output_file="assets/demo_speed_cover.mp4",
        output_size=None,
        process=[cast(VALID_VIDEO_PROCESS, "speed-up")],
        speed=speed,
        add_image=CoverImageOpts(
            text="I wanted to add audio to this video but I was lazy to download an app\nand IG somehow destroys the quality when there is an audio?\n ----- \n...So I wrote an algorithm to add it from a YouTube link.",
            size=tuple(mpe.VideoFileClip(videos[0]).size),
            font_color="white",
            justification="center",
            time=1,
            insert_at="start",
            font_path="/Library/Fonts/Arial Unicode.ttf",
        ),
    )
    main(opts)
    demo_add_audio_to_video(
        videos=[opts.output_file],
        song="https://www.youtube.com/watch?v=8ui9umU0C2g",
        start_at=83,
        output_file=opts.output_file,
    )


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_speed_up)
