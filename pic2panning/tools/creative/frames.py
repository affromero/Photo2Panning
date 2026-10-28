"""Render one world-canvas frame through the shared composition pipeline."""

import math
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from itertools import pairwise
from types import TracebackType
from typing import Self

from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageOps,
)

from pic2panning.errors import MediaError
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import (
    CameraKeyframe,
    ComposeRequest,
    Easing,
    ImageLayer,
    Layer,
    LayerKeyframe,
    TextLayer,
    VideoLayer,
)
from pic2panning.tools.creative.depth import DepthAsset, load_depth
from pic2panning.tools.creative.effects import (
    apply_depth_parallax,
    apply_depth_style,
    depth_mask,
    rescale_depth_style,
)
from pic2panning.tools.creative.sampling import VideoSampler, source_time
from pic2panning.tools.frames import frame_count, weighted_linear_blend
from pic2panning.tools.io import materialize, read_image, require_file
from pic2panning.tools.media import probe
from pic2panning.tools.text import wrap_text


def eased(progress: float, easing: Easing) -> float:
    """Map a bounded segment fraction to its requested easing curve."""
    if easing == Easing.HOLD:
        return 0
    if easing == Easing.EASE_IN:
        return progress * progress
    if easing == Easing.EASE_OUT:
        return 1 - (1 - progress) ** 2
    if easing == Easing.EASE_IN_OUT:
        return progress * progress * (3 - 2 * progress)
    return progress


def camera_at(request: ComposeRequest, seconds: float) -> CameraKeyframe:
    """Evaluate the camera, retaining exact keyframe poses and final holds."""
    for left, right in pairwise(request.camera):
        if seconds < right.at:
            fraction = eased(
                (seconds - left.at) / (right.at - left.at), left.ease
            )
            return CameraKeyframe(
                at=seconds,
                x=left.x + (right.x - left.x) * fraction,
                y=left.y + (right.y - left.y) * fraction,
                zoom=left.zoom + (right.zoom - left.zoom) * fraction,
            )
    return request.camera[-1]


def transform_at(layer: Layer, seconds: float) -> LayerKeyframe:
    """Evaluate a layer transform in its own local timeline."""
    for left, right in pairwise(layer.keyframes):
        if seconds < right.at:
            fraction = eased(
                (seconds - left.at) / (right.at - left.at), left.ease
            )
            return LayerKeyframe(
                at=seconds,
                x=left.x + (right.x - left.x) * fraction,
                y=left.y + (right.y - left.y) * fraction,
                scale=left.scale + (right.scale - left.scale) * fraction,
                rotation=left.rotation
                + (right.rotation - left.rotation) * fraction,
                opacity=left.opacity
                + (right.opacity - left.opacity) * fraction,
            )
    return layer.keyframes[-1]


def text_image(layer: TextLayer) -> Image.Image:
    """Draw wrapped text onto a transparent rectangle without a title card."""
    image = Image.new("RGBA", layer.size.as_tuple)
    draw = ImageDraw.Draw(image)
    font = (
        ImageFont.truetype(require_file(layer.font), size=layer.font_size)
        if layer.font
        else ImageFont.load_default(size=layer.font_size)
    )
    text = wrap_text(
        layer.text,
        draw,
        font,
        max(1, layer.size.width - 2 * layer.stroke_width),
    )
    bounds = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        align=layer.align.value,
        spacing=layer.spacing,
        stroke_width=layer.stroke_width,
    )
    width = bounds[2] - bounds[0]
    x = 0.0
    if layer.align.value == "center":
        x = (layer.size.width - width) / 2
    elif layer.align.value == "right":
        x = layer.size.width - width
    position = (x - bounds[0], -bounds[1])
    if layer.shadow_offset != (0, 0):
        draw.multiline_text(
            (
                position[0] + layer.shadow_offset[0],
                position[1] + layer.shadow_offset[1],
            ),
            text,
            font=font,
            fill=ImageColor.getcolor(layer.shadow_color, "RGBA"),
            align=layer.align.value,
            spacing=layer.spacing,
        )
    draw.multiline_text(
        position,
        text,
        font=font,
        fill=ImageColor.getcolor(layer.color, "RGBA"),
        align=layer.align.value,
        spacing=layer.spacing,
        stroke_width=layer.stroke_width,
        stroke_fill=ImageColor.getcolor(layer.stroke_color, "RGBA"),
    )
    return image


def fit_layer(
    image: Image.Image, layer: Layer, bitmap: Size | None = None
) -> Image.Image:
    """Preserve source proportions inside the declared layer rectangle."""
    target = bitmap or layer.size
    if isinstance(layer, TextLayer):
        return image.resize(target.as_tuple, Image.Resampling.LANCZOS)
    with image.convert("RGBA") as rgba:
        if layer.fit == "cover":
            return ImageOps.fit(
                rgba, target.as_tuple, Image.Resampling.LANCZOS
            )
        with ImageOps.contain(
            rgba, target.as_tuple, Image.Resampling.LANCZOS
        ) as contained:
            fitted = Image.new("RGBA", target.as_tuple)
            fitted.paste(
                contained,
                (
                    (fitted.width - contained.width) // 2,
                    (fitted.height - contained.height) // 2,
                ),
            )
            return fitted


def place_layer(
    image: Image.Image,
    layer: Layer,
    transform: LayerKeyframe,
    request: ComposeRequest,
    camera: CameraKeyframe,
) -> Image.Image:
    """Apply the same affine mapping to visual pixels and occlusion masks."""
    factor = request.size.height / request.canvas.height * camera.zoom
    center_x = (transform.x - camera.x) * factor + request.size.width / 2
    center_y = (transform.y - camera.y) * factor + request.size.height / 2
    if layer.space == "viewport":
        factor = 1
        center_x, center_y = transform.x, transform.y
    radians = math.radians(transform.rotation)
    divisor = transform.scale * factor
    a = math.cos(radians) / divisor * image.width / layer.size.width
    b = math.sin(radians) / divisor * image.width / layer.size.width
    d = -math.sin(radians) / divisor * image.height / layer.size.height
    e = math.cos(radians) / divisor * image.height / layer.size.height
    return image.transform(
        request.size.as_tuple,
        Image.Transform.AFFINE,
        (
            a,
            b,
            image.width / 2 - a * center_x - b * center_y,
            d,
            e,
            image.height / 2 - d * center_x - e * center_y,
        ),
        resample=Image.Resampling.BICUBIC,
    )


def projected_box(
    layer: Layer,
    transform: LayerKeyframe,
    request: ComposeRequest,
    camera: CameraKeyframe,
) -> Size:
    """Keep intermediate layer rectangles at their required output density."""
    factor = (
        1
        if layer.space == "viewport"
        else request.size.height / request.canvas.height * camera.zoom
    )
    density = min(1.0, factor * transform.scale)
    return Size(
        width=max(1, math.ceil(layer.size.width * density)),
        height=max(1, math.ceil(layer.size.height * density)),
    )


class CompositionSession:
    """Own source media and subprocesses while keeping frame memory bounded."""

    request: ComposeRequest
    preview: bool
    stack: ExitStack
    directory: str
    images: dict[str, Image.Image]
    masks: dict[str, Image.Image]
    videos: dict[str, VideoSampler]
    plates: dict[str, Image.Image]
    depths: dict[str, DepthAsset]

    def __init__(
        self, request: ComposeRequest, *, preview: bool = False
    ) -> None:
        """Prepare ownership; input loading starts inside the context."""
        self.request = request
        self.preview = preview
        self.stack = ExitStack()
        self.directory = ""
        self.images = {}
        self.masks = {}
        self.videos = {}
        self.plates = {}
        self.depths = {}

    def __enter__(self) -> Self:
        """Validate and load sources, closing partial state on any failure."""
        try:
            self.directory = self.stack.enter_context(
                tempfile.TemporaryDirectory(prefix="pic2panning-compose-")
            )
            for layer in self.request.layers:
                self.load_layer(layer)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close all images and decoder processes, including interrupted runs."""
        self.stack.close()

    def own_image(self, image: Image.Image) -> Image.Image:
        """Register an image for deterministic cleanup at session exit."""
        self.stack.callback(image.close)
        return image

    def load_layer(self, layer: Layer) -> None:
        """Load static assets and validate video and aligned-mask dimensions."""
        if isinstance(layer, TextLayer):
            self.images[layer.id] = self.own_image(text_image(layer))
        elif isinstance(layer, ImageLayer):
            self.images[layer.id] = self.own_image(
                read_image(layer.source, preserve_alpha=True)
            )
        else:
            local = materialize(layer.source, self.directory)
            info = probe(local)
            if source_time(layer, layer.duration) > info.duration + 1e-8:
                raise MediaError(
                    f"Video layer {layer.id} exceeds its source duration"
                )
            sampler = VideoSampler(local, info, preview=self.preview)
            self.videos[layer.id] = sampler
            self.stack.callback(sampler.close)
            self.load_cinemagraph(layer, sampler)
        if layer.mask:
            with read_image(layer.mask) as mask:
                selected = mask.convert("L")
            if selected.size != self.source_size(layer):
                selected.close()
                raise MediaError(
                    "Layer mask dimensions must match oriented source"
                )
            self.masks[layer.id] = self.own_image(selected)
        if layer.depth is not None and isinstance(layer, ImageLayer):
            self.depths[layer.id] = self.stack.enter_context(
                load_depth(layer.depth.metadata, layer.source)
            )

    def source_size(self, layer: Layer) -> tuple[int, int]:
        """Return oriented source dimensions without decoding video pixels."""
        if isinstance(layer, VideoLayer):
            info = self.videos[layer.id].info
            return info.width, info.height
        return self.images[layer.id].size

    def load_cinemagraph(
        self, layer: VideoLayer, sampler: VideoSampler
    ) -> None:
        """Load a declared clean plate and strictly aligned motion mask."""
        options = layer.cinemagraph
        if options is None:
            return
        plate = (
            read_image(options.reference)
            if options.reference is not None
            else sampler.seek(options.reference_time or 0)
        )
        self.plates[layer.id] = self.own_image(plate)
        with read_image(options.mask) as image:
            mask = image.convert("L")
        if options.feather:
            previous = mask
            mask = previous.filter(ImageFilter.GaussianBlur(options.feather))
            previous.close()
        self.masks[f"cinemagraph:{layer.id}"] = self.own_image(mask)
        size = sampler.info.width, sampler.info.height
        if mask.size != size or plate.size != size:
            raise MediaError(
                "Cinemagraph mask and clean plate must match oriented video dimensions"
            )

    def working_size(
        self, layer: Layer, transform: LayerKeyframe, camera: CameraKeyframe
    ) -> tuple[int, int]:
        """Retain enough native pixels for the current camera and layer zoom."""
        source_width, source_height = self.source_size(layer)
        if layer.depth is None or isinstance(layer, TextLayer):
            return source_width, source_height
        fit = min if layer.fit == "contain" else max
        fit_factor = fit(
            layer.size.width / source_width, layer.size.height / source_height
        )
        factor = (
            1
            if layer.space == "viewport"
            else self.request.size.height
            / self.request.canvas.height
            * camera.zoom
        )
        density = min(1.0, fit_factor * factor * transform.scale)
        return max(1, math.ceil(source_width * density)), max(
            1, math.ceil(source_height * density)
        )

    def source_image(
        self,
        layer: Layer,
        local_time: float,
        camera: CameraKeyframe,
        transform: LayerKeyframe,
    ) -> Image.Image:
        """Return an owned layer source with its explicit selective motion."""
        if isinstance(layer, VideoLayer):
            image = self.videos[layer.id].frame(source_time(layer, local_time))
            if layer.cinemagraph is not None:
                previous = image
                image = Image.composite(
                    previous,
                    self.plates[layer.id],
                    self.masks[f"cinemagraph:{layer.id}"],
                )
                previous.close()
            return image
        working = self.working_size(layer, transform, camera)
        image = self.images[layer.id].resize(working, Image.Resampling.LANCZOS)
        if layer.depth is not None:
            style = rescale_depth_style(
                layer.depth, self.source_size(layer), working
            )
            with image:
                styled = apply_depth_style(
                    image,
                    self.depths[layer.id],
                    style,
                    local_time / layer.duration,
                )
                if image.mode == "RGBA":
                    with (
                        image.getchannel("A") as alpha,
                        apply_depth_parallax(
                            alpha,
                            self.depths[layer.id],
                            style,
                            local_time / layer.duration,
                        ) as warped,
                    ):
                        styled.putalpha(warped)
                return styled
        return image

    def layer_mask(
        self, layer: Layer, progress: float, working: tuple[int, int]
    ) -> Image.Image:
        """Move an explicit source mask with its depth-driven pixel warp."""
        mask = self.masks[layer.id].resize(working, Image.Resampling.LANCZOS)
        if layer.depth is not None:
            with mask:
                return apply_depth_parallax(
                    mask,
                    self.depths[layer.id],
                    rescale_depth_style(
                        layer.depth, self.source_size(layer), working
                    ),
                    progress,
                )
        return mask


def occlusion_mask(
    layer: Layer,
    seconds: float,
    request: ComposeRequest,
    session: CompositionSession,
    camera: CameraKeyframe,
) -> Image.Image | None:
    """Transport a nearer-source depth mask through its exact layer mapping."""
    options = layer.occlusion
    if options is None:
        return None
    source = next(
        item for item in request.layers if item.id == options.layer_id
    )
    if not source.start <= seconds < source.start + source.duration:
        return None
    if source.depth is None:
        raise ValueError("Occlusion source requires depth effects")
    asset = session.depths[source.id]
    transform = transform_at(source, seconds - source.start)
    working = session.working_size(source, transform, camera)
    style = rescale_depth_style(
        source.depth, session.source_size(source), working
    )
    with (
        depth_mask(
            asset.depth, options.plane_metres, options.feather_metres
        ) as small,
        ImageChops.multiply(small, asset.mask) as foreground,
        foreground.resize(working, Image.Resampling.BILINEAR) as expanded,
        session.images[source.id].resize(
            working, Image.Resampling.LANCZOS
        ) as resized,
        resized.convert("RGBA") as source_pixels,
        source_pixels.getchannel("A") as alpha,
        ImageChops.multiply(expanded, alpha) as opaque,
    ):
        if source.id in session.masks:
            with (
                session.masks[source.id].resize(
                    working, Image.Resampling.LANCZOS
                ) as source_mask,
                ImageChops.multiply(opaque, source_mask) as masked,
            ):
                opaque.paste(masked)
        with (
            apply_depth_parallax(
                opaque,
                asset,
                style,
                (seconds - source.start) / source.duration,
            ) as warped,
            warped.convert("RGBA") as rgba,
            fit_layer(
                rgba, source, projected_box(source, transform, request, camera)
            ) as fitted,
            place_layer(
                fitted,
                source,
                transform,
                request,
                camera,
            ) as placed,
            placed.convert("L") as selected,
            placed.getchannel("A") as coverage,
            ImageChops.multiply(selected, coverage) as covered,
        ):
            return covered.point(
                [round(value * transform.opacity) for value in range(256)]
            )


def composition_at_time(
    request: ComposeRequest, seconds: float, session: CompositionSession
) -> Image.Image:
    """Render one instantaneous RGB sample at an explicit timeline time."""
    if not 0 <= seconds < request.duration:
        raise ValueError("Composition sample time is outside its timeline")
    camera = camera_at(request, seconds)
    frame = Image.new(
        "RGBA", request.size.as_tuple, ImageColor.getrgb(request.background)
    )
    try:
        for layer in request.layers:
            if not layer.start <= seconds < layer.start + layer.duration:
                continue
            local = seconds - layer.start
            transform = transform_at(layer, local)
            with (
                session.source_image(layer, local, camera, transform) as raw,
                raw.convert("RGBA") as source,
            ):
                if layer.id in session.masks:
                    with (
                        source.getchannel("A") as alpha,
                        session.layer_mask(
                            layer, local / layer.duration, source.size
                        ) as layer_mask,
                        ImageChops.multiply(alpha, layer_mask) as masked,
                    ):
                        source.putalpha(masked)
                with (
                    fit_layer(
                        source,
                        layer,
                        projected_box(layer, transform, request, camera),
                    ) as fitted,
                    place_layer(
                        fitted, layer, transform, request, camera
                    ) as placed,
                ):
                    if transform.opacity < 1:
                        with (
                            placed.getchannel("A") as alpha,
                            alpha.point(
                                [
                                    round(value * transform.opacity)
                                    for value in range(256)
                                ]
                            ) as faded,
                        ):
                            placed.putalpha(faded)
                    mask = occlusion_mask(
                        layer, seconds, request, session, camera
                    )
                    if mask is not None:
                        with (
                            mask,
                            ImageOps.invert(mask) as visible,
                            placed.getchannel("A") as alpha,
                            ImageChops.multiply(alpha, visible) as masked,
                        ):
                            placed.putalpha(masked)
                    frame.alpha_composite(placed)
        return frame.convert("RGB")
    finally:
        frame.close()


def composition_frame(
    request: ComposeRequest, index: int, session: CompositionSession
) -> Image.Image:
    """Render the canonical frame, integrating requested interior exposures."""
    count = frame_count(request.duration, request.fps)
    if not 0 <= index < count:
        raise ValueError("Composition frame index is outside its timeline")
    seconds = index / request.fps
    if not request.shutter_angle or index in {0, count - 1}:
        return composition_at_time(request, seconds, session)

    def samples() -> Iterator[tuple[Image.Image, float]]:
        interval = request.shutter_angle / (360 * request.fps)
        for sample in range(request.shutter_samples):
            offset = (
                (sample + 0.5) / request.shutter_samples - 0.5
            ) * interval
            yield composition_at_time(request, seconds + offset, session), 1.0

    return weighted_linear_blend(samples())
