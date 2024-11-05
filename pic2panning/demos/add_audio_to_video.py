"""Demo for audio with video."""

from typing import cast

import tyro

from pic2panning.main import main_videos as main
from pic2panning.utils.options import VALID_VIDEO_PROCESS, AudioOpts, Opts


def demo_add_audio_to_video(
    videos: list[str] = [
        "https://videos.pexels.com/video-files/1437396/1437396-uhd_3840_2160_24fps.mp4"
    ],
    song: str = "https://www.youtube.com/watch?v=v2AC41dglnM",
    output_file: str = "assets/demo_audio.mp4",
    start_at: int = 0,
) -> None:
    """Demo function for zoom-in."""
    opts = Opts(
        data=videos,
        output_file=output_file,
        time=[2],
        ratio="16:9",
        audio=[AudioOpts(audio_file=song, start_at=start_at)],
        output_size=(1080, 1920),
        fps=[240],
        process=[cast(VALID_VIDEO_PROCESS, "add-audio-to-video")],
    )
    main(opts)


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_add_audio_to_video)
