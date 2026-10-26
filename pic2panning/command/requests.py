"""Translate terminal arguments into the same requests Python callers use."""

from pic2panning.command.parser import Arguments
from pic2panning.schemas.base import Size
from pic2panning.schemas.editing import (
    Adjust,
    Crop,
    EditOperation,
    EditRequest,
    Flip,
    Resize,
    Rotate,
)
from pic2panning.schemas.jobs import Job
from pic2panning.schemas.media import (
    AudioTrack,
    Clip,
    Cover,
    ImageClip,
    Motion,
    RenderRequest,
    VideoClip,
)
from pic2panning.schemas.models import (
    GenerateRequest,
    MaskRequest,
    Quantization,
)


def render_request(arguments: Arguments) -> RenderRequest:
    """Build a uniform image or video timeline."""
    clips: list[Clip] = []
    for source in arguments.sources:
        if arguments.video:
            if arguments.duration is not None:
                raise ValueError("--duration requires image inputs")
            if arguments.reverse or arguments.focus_center:
                raise ValueError(
                    "--reverse and --focus-center require image inputs"
                )
            clips.append(
                VideoClip(
                    source=source,
                    speed=arguments.speed,
                    motion=arguments.motion,
                    pan_speed=arguments.pan_speed
                    if arguments.pan_speed is not None
                    else 5,
                    zoom=arguments.zoom,
                )
            )
        else:
            if arguments.speed != 1:
                raise ValueError("--speed requires --video")
            if arguments.pan_speed is not None:
                raise ValueError("--pan-speed requires --video")
            clips.append(
                ImageClip(
                    source=source,
                    duration=arguments.duration
                    if arguments.duration is not None
                    else 5,
                    motion=arguments.motion or Motion.PAN_LR,
                    reverse=arguments.reverse,
                    focus_center=arguments.focus_center,
                    zoom=arguments.zoom,
                )
            )
    if not arguments.audio and (arguments.audio_start or arguments.loop_audio):
        raise ValueError("Audio offset and loop options require --audio")
    return RenderRequest(
        clips=tuple(clips),
        output=arguments.output,
        size=Size(width=arguments.size[0], height=arguments.size[1]),
        fps=arguments.fps,
        audio=AudioTrack(
            source=arguments.audio,
            start=arguments.audio_start,
            loop=arguments.loop_audio,
        )
        if arguments.audio
        else None,
        cover=Cover(text=arguments.cover) if arguments.cover else None,
        overwrite=arguments.overwrite,
    )


def edit_request(arguments: Arguments) -> EditRequest:
    """Apply crop, resize, rotate, flip, then color controls from the CLI."""
    operations: list[EditOperation] = []
    if arguments.source is None:
        raise ValueError("Image editing requires a source")
    if arguments.crop:
        left, top, right, bottom = arguments.crop
        operations.append(Crop(left=left, top=top, right=right, bottom=bottom))
    if arguments.resize:
        operations.append(
            Resize(
                size=Size(
                    width=arguments.resize[0], height=arguments.resize[1]
                )
            )
        )
    if arguments.rotate is not None:
        operations.append(Rotate(degrees=arguments.rotate))
    if arguments.flip:
        operations.append(Flip(axis=arguments.flip))
    if (arguments.brightness, arguments.contrast, arguments.saturation) != (
        1,
        1,
        1,
    ):
        operations.append(
            Adjust(
                brightness=arguments.brightness,
                contrast=arguments.contrast,
                saturation=arguments.saturation,
            )
        )
    return EditRequest(
        source=arguments.source,
        output=arguments.output,
        operations=tuple(operations),
        overwrite=arguments.overwrite,
    )


def to_job(arguments: Arguments) -> Job:
    """Build and validate a request before processing begins."""
    match arguments.command:
        case "render":
            return render_request(arguments)
        case "edit":
            return edit_request(arguments)
        case "generate":
            return GenerateRequest(
                source=arguments.source,
                output=arguments.output,
                prompt=arguments.prompt,
                model=arguments.model,
                weights=arguments.weights,
                quantization=Quantization(arguments.quantization),
                seed=arguments.seed,
                steps=arguments.steps,
                size=Size(width=arguments.size[0], height=arguments.size[1]),
                strength=arguments.strength,
                guidance=arguments.guidance,
                mask=arguments.mask,
                overwrite=arguments.overwrite,
            )
        case "mask":
            if arguments.source is None:
                raise ValueError("Masking requires a source")
            return MaskRequest(
                source=arguments.source,
                output=arguments.output,
                prompt=arguments.prompt,
                threshold=arguments.threshold,
                device=arguments.device,
                merge=not arguments.separate,
                overwrite=arguments.overwrite,
            )
        case _:
            raise ValueError(
                f"Command does not create a job: {arguments.command}"
            )
