"""Atomic metric-depth extraction and source-bound cached asset loading."""

from collections.abc import Generator
from contextlib import ExitStack, closing, contextmanager
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast

import numpy as np
from klogr.path import (
    path_basename,
    path_dirname,
    path_join,
    path_open,
    path_read_text,
    path_resolve,
    path_write_text,
)
from PIL import Image
from pydantic import ConfigDict

from pic2panning.backends.creative.depth import (
    DepthBackend,
    DepthPrediction,
    NativeDepthBackend,
)
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.creative.depth import (
    DepthArtifact,
    DepthMetadata,
    DepthRequest,
    DepthResult,
)
from pic2panning.tools.creative.effects import normal_image, validate_depth
from pic2panning.tools.io import (
    atomic_output,
    file_sha256,
    materialize,
    read_image,
    require_file,
    validate_destination,
)


class DepthAsset(PydanticBaseModel):
    """Owned cached pixels and their verified source calibration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    depth: Image.Image
    mask: Image.Image
    normals: Image.Image
    metadata: DepthMetadata

    def close(self) -> None:
        """Release all pixel buffers owned by the loaded asset."""
        self.depth.close()
        self.mask.close()
        self.normals.close()


def depth_artifacts(request: DepthRequest | DepthResult) -> tuple[str, ...]:
    """Enumerate every depth output for job graphs and alias protection."""
    return request.artifacts


def _prediction_size(image: Image.Image, max_edge: int) -> tuple[int, int]:
    """Resolve the exact Pillow thumbnail grid shared with native inference."""
    with image.copy() as geometry:
        geometry.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return geometry.size


def _validate_prediction(
    prediction: DepthPrediction, expected: tuple[int, int]
) -> None:
    """Validate pixel semantics before any artifact can be published."""
    validate_depth(prediction.depth)
    if prediction.depth.size != expected:
        raise ValueError(
            "Predicted depth dimensions differ from bounded source"
        )
    if prediction.mask.mode != "L" or prediction.mask.size != expected:
        raise ValueError("Predicted depth mask dimensions or mode differ")
    values = np.asarray(prediction.mask)
    if not bool(((values == 0) | (values == 255)).all()) or not bool(
        values.any()
    ):
        raise ValueError("Predicted depth mask must select binary foreground")


def _preview(prediction: DepthPrediction) -> tuple[Image.Image, float, float]:
    """Normalize only the visualization using foreground depth percentiles."""
    values = np.asarray(prediction.depth)
    selected = values[np.asarray(prediction.mask) > 0]
    near, far = (float(value) for value in np.percentile(selected, (1, 99)))
    pixels = (
        np.clip((far - values) / (far - near), 0, 1)
        if far > near
        else (values <= near).astype(np.float32)
    )
    preview = Image.fromarray(np.rint(pixels * 65535).astype(np.uint16))
    return preview, near, far


def _artifact(path: str, temporary: str) -> DepthArtifact:
    """Bind an output sibling name to exact completed file bytes."""
    return DepthArtifact(
        path=path_basename(path), sha256=file_sha256(temporary)
    )


def _save_prediction(
    request: DepthRequest,
    prediction: DepthPrediction,
    outputs: dict[str, str],
    source_info: tuple[str, tuple[int, int]],
) -> DepthMetadata:
    """Save metric and visualization files, then build their typed manifest."""
    source_sha256, size = source_info
    prediction.depth.save(
        outputs[request.output], format="TIFF", compression="raw"
    )
    prediction.mask.save(outputs[request.mask], format="PNG")
    with normal_image(prediction.depth, prediction.intrinsics) as normals:
        normals.save(outputs[request.normals], format="PNG")
    preview, near, far = _preview(prediction)
    with preview:
        preview.save(outputs[request.preview], format="PNG")
    return DepthMetadata(
        source_sha256=source_sha256,
        width=size[0],
        height=size[1],
        grid_width=prediction.depth.width,
        grid_height=prediction.depth.height,
        intrinsics=prediction.intrinsics,
        depth=_artifact(request.output, outputs[request.output]),
        preview=_artifact(request.preview, outputs[request.preview]),
        normals=_artifact(request.normals, outputs[request.normals]),
        mask=_artifact(request.mask, outputs[request.mask]),
        preview_near_metres=near,
        preview_far_metres=far,
        device=request.device,
        num_tokens=request.num_tokens,
        max_edge=request.max_edge,
    )


def extract_depth(
    request: DepthRequest, backend: DepthBackend | None = None
) -> DepthResult:
    """Publish exact metric depth and a manifest after all outputs validate."""
    for destination in request.artifacts:
        inputs = (
            request.source,
            *(
                artifact
                for artifact in request.artifacts
                if artifact != destination
            ),
        )
        validate_destination(destination, inputs, overwrite=request.overwrite)
    with (
        ExitStack() as stack,
        TemporaryDirectory(prefix="pic2panning-depth-") as directory,
    ):
        outputs = {}
        # The manifest is entered first, so it is published last.
        for destination in (request.metadata, *request.artifacts[:-1]):
            inputs = (
                request.source,
                *(
                    artifact
                    for artifact in request.artifacts
                    if artifact != destination
                ),
            )
            outputs[destination] = stack.enter_context(
                atomic_output(destination, inputs, overwrite=request.overwrite)
            )
        local = materialize(request.source, directory)
        source_sha256 = file_sha256(local)
        selected = backend if backend is not None else NativeDepthBackend()
        with (
            read_image(local) as image,
            closing(selected.predict(request, image)) as prediction,
        ):
            _validate_prediction(
                prediction, _prediction_size(image, request.max_edge)
            )
            document = _save_prediction(
                request, prediction, outputs, (source_sha256, image.size)
            )
        path_write_text(
            outputs[request.metadata],
            document.model_dump_json(indent=2) + "\n",
        )
        return DepthResult(
            output=request.output,
            metadata=request.metadata,
            preview=request.preview,
            normals=request.normals,
            mask=request.mask,
            source_sha256=source_sha256,
            width=document.width,
            height=document.height,
            grid_width=document.grid_width,
            grid_height=document.grid_height,
        )


def _read_artifact(
    artifact: DepthArtifact, directory: str, mode: str
) -> Image.Image:
    """Verify bytes and image format before retaining an independent copy."""
    local = require_file(path_join(directory, artifact.path))
    if file_sha256(local) != artifact.sha256:
        raise ValueError("Depth artifact checksum differs from metadata")
    with (
        cast("BinaryIO", path_open(local, "rb")) as handle,
        Image.open(handle) as image,
    ):
        image.load()
        expected_format = "TIFF" if mode == "F" else "PNG"
        if image.mode != mode or image.format != expected_format:
            raise ValueError("Depth artifact image mode or format differs")
        return image.copy()


def _validate_source(document: DepthMetadata, source: str) -> None:
    """Bind cached geometry to original file bytes and oriented dimensions."""
    with TemporaryDirectory(prefix="pic2panning-depth-source-") as directory:
        local = materialize(source, directory)
        if file_sha256(local) != document.source_sha256:
            raise ValueError("Depth source checksum differs from metadata")
        with read_image(local) as image:
            if image.size != (document.width, document.height):
                raise ValueError(
                    "Depth source dimensions differ from metadata"
                )
            if _prediction_size(image, document.max_edge) != (
                document.grid_width,
                document.grid_height,
            ):
                raise ValueError("Depth grid dimensions differ from metadata")


@contextmanager
def load_depth(metadata: str, source: str) -> Generator[DepthAsset]:
    """Own source-verified cached depth without importing model frameworks."""
    local = require_file(metadata)
    document = DepthMetadata.model_validate_json(path_read_text(local))
    _validate_source(document, source)
    directory = path_dirname(path_resolve(local))
    with ExitStack() as stack:
        depth = stack.enter_context(
            _read_artifact(document.depth, directory, "F")
        )
        mask = stack.enter_context(
            _read_artifact(document.mask, directory, "L")
        )
        normals = stack.enter_context(
            _read_artifact(document.normals, directory, "RGB")
        )
        preview = stack.enter_context(
            _read_artifact(document.preview, directory, "I;16")
        )
        size = (document.grid_width, document.grid_height)
        if any(
            image.size != size for image in (depth, mask, normals, preview)
        ):
            raise ValueError("Depth artifact dimensions differ from metadata")
        validate_depth(depth)
        values = np.asarray(mask)
        if not bool(((values == 0) | (values == 255)).all()) or not bool(
            values.any()
        ):
            raise ValueError("Cached depth mask must select binary foreground")
        yield DepthAsset(
            depth=depth, mask=mask, normals=normals, metadata=document
        )
