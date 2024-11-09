"""Demo for audio with video."""

from typing import cast

import moviepy.editor as mpe
import tyro

from pic2panning.main import get_temp_video
from pic2panning.main import main_videos as main
from pic2panning.utils.options import (
    VALID_VIDEO_PROCESS,
    AudioOpts,
    CoverImageOpts,
    Opts,
)
from pic2panning.utils.process import read_video


def demo_add_audio_to_video(
    videos: list[str] = [
        "https://videos.pexels.com/video-files/1437396/1437396-uhd_3840_2160_24fps.mp4"
    ],
    /,
    song: str = "https://www.youtube.com/watch?v=v2AC41dglnM",
    output_file: str = "assets/demo_audio.mp4",
    start_at: int = 0,
    output_size: tuple[int, int] = (1080, 1920),
    add_cover: str = "",
) -> None:
    """Demo function for zoom-in."""
    if len(videos) > 1:
        # concatenate videos
        _videos = []
        for video in videos:
            _videos.append(read_video(video))
        videos_cat = mpe.concatenate_videoclips(_videos, method="compose")
        video_temp = get_temp_video()
        videos_cat.write_videofile(video_temp)
        videos = [video_temp]
    opts = Opts(
        data=videos,
        output_file=output_file,
        audio=[AudioOpts(audio_file=song, start_at=start_at)],
        output_size=output_size,
        process=[cast(VALID_VIDEO_PROCESS, "add-audio-to-video")],
        add_image=(
            CoverImageOpts(
                text=add_cover,
                size=output_size,
                font_color="white",
                justification="center",
                time=1,
                insert_at="start",
                font_path="/Library/Fonts/Arial Unicode.ttf",
            )
            if add_cover
            else None
        ),
    )
    main(opts)


if __name__ == "__main__":
    """Run the demo."""
    tyro.cli(demo_add_audio_to_video)
