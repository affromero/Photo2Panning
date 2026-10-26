"""Small Grounding DINO and SAM 2 models with verified CPU execution."""

from collections.abc import ItemsView, Iterator
from contextlib import AbstractContextManager
from typing import Protocol, cast

from klogr import get_logger
from PIL import Image
from pydantic import TypeAdapter

from pic2panning.backends.loading import load_dependency
from pic2panning.schemas.base import PydanticBaseModel
from pic2panning.schemas.models import Device, MaskRequest

logger = get_logger()


class BoundingBox(PydanticBaseModel):
    """Pixel coordinates produced by the grounding detector."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Detection(PydanticBaseModel):
    """A validated label, score, and box from Grounding DINO."""

    score: float
    label: str
    box: BoundingBox


class TensorValue(Protocol):
    """Tensor methods used at the third-party inference boundary."""

    def cpu(self) -> "TensorValue":
        """Copy tensor data to the CPU."""
        ...

    def squeeze(self) -> "TensorValue":
        """Remove singleton object and prediction dimensions."""
        ...

    def tolist(self) -> object:
        """Return primitive Python values for validated conversion."""
        ...


class SamInputs(Protocol):
    """Processor output containing tensor values and original sizes."""

    def to(self, device: str) -> "SamInputs":
        """Move inference inputs onto the selected device."""
        ...

    def __getitem__(self, key: str) -> object:
        """Read an input field."""
        ...

    def items(self) -> ItemsView[str, object]:
        """Expose named model inputs."""
        ...


class SamOutput(Protocol):
    """The prediction fields needed from SAM 2."""

    pred_masks: TensorValue


class SamModel(Protocol):
    """The inference portion of Transformers' SAM 2 model."""

    def to(self, device: str) -> "SamModel":
        """Move model parameters to the selected device."""
        ...

    def eval(self) -> "SamModel":
        """Disable training-only behavior."""
        ...

    def __call__(self, **inputs: object) -> SamOutput:
        """Predict one mask for each object box."""
        ...


class SamProcessor(Protocol):
    """Image and bounding-box conversion for SAM 2."""

    def __call__(self, **inputs: object) -> SamInputs:
        """Prepare image and bounding boxes for inference."""
        ...

    def post_process_masks(
        self, masks: TensorValue, original_sizes: object
    ) -> list[TensorValue]:
        """Restore masks to the original input image resolution."""
        ...


class ProcessorFactory(Protocol):
    """Transformers' processor loader."""

    def from_pretrained(self, model: str) -> SamProcessor:
        """Load the preprocessing configuration."""
        ...


class SamFactory(Protocol):
    """Transformers' model loader."""

    def from_pretrained(self, model: str) -> SamModel:
        """Load the selected small model weights."""
        ...


class Detector(Protocol):
    """Grounding DINO pipeline inference interface."""

    def __call__(self, image: Image.Image, **options: object) -> object:
        """Find requested text labels in an image."""
        ...


class TransformersModule(Protocol):
    """Only the public Transformers exports used by this adapter."""

    Sam2Processor: ProcessorFactory
    Sam2Model: SamFactory

    def pipeline(self, **options: object) -> Detector:
        """Build a text-conditioned object detector."""
        ...


class TorchModule(Protocol):
    """Inference-only PyTorch context manager."""

    def inference_mode(self) -> AbstractContextManager[object]:
        """Disable gradient allocation during inference."""
        ...


def image_from_mask(values: object, size: tuple[int, int]) -> Image.Image:
    """Validate binary model output and convert it to an editable mask."""
    rows = TypeAdapter(list[list[bool]]).validate_python(values)
    width, height = size
    if len(rows) != height or any(len(row) != width for row in rows):
        message = "The segmentation backend returned the wrong mask size."
        raise ValueError(message)
    result = Image.new("L", size)
    result.putdata(
        [255 if selected else 0 for row in rows for selected in row]
    )
    return result


class GroundedSamBackend:
    """Detect text labels and segment them using small local models."""

    def masks(
        self, request: MaskRequest, image: Image.Image
    ) -> tuple[Image.Image, ...]:
        """Return a full-resolution mask for each matching object."""
        transformers = cast(
            "TransformersModule", load_dependency("transformers")
        )
        torch = cast("TorchModule", load_dependency("torch"))
        device = request.device.value
        logger.info(
            f"Loading small grounding and segmentation models on {device}"
        )
        detector = transformers.pipeline(
            task="zero-shot-object-detection",
            model=request.detector,
            device=device,
        )
        labels = [
            f"{label.strip()}."
            for label in request.prompt.split(".")
            if label.strip()
        ]
        with torch.inference_mode():
            detections = TypeAdapter(tuple[Detection, ...]).validate_python(
                detector(
                    image,
                    candidate_labels=labels,
                    threshold=request.threshold,
                )
            )
        if not detections:
            return ()
        processor = transformers.Sam2Processor.from_pretrained(
            request.segmenter
        )
        model = transformers.Sam2Model.from_pretrained(request.segmenter)
        model = model.to(device).eval()
        session = SamSession(processor, model, torch, request.device)
        return tuple(session.object_masks(detections, image))


class SamSession:
    """Keep one loaded segmentation model for all objects in an image."""

    processor: SamProcessor
    model: SamModel
    torch: TorchModule
    device: Device

    def __init__(
        self,
        processor: SamProcessor,
        model: SamModel,
        torch: TorchModule,
        device: Device,
    ) -> None:
        """Declare the reusable inference resources for an image."""
        self.processor = processor
        self.model = model
        self.torch = torch
        self.device = device

    def object_masks(
        self, detections: tuple[Detection, ...], image: Image.Image
    ) -> Iterator[Image.Image]:
        """Process one object at a time to limit inference memory."""
        for detection in detections:
            box = detection.box
            inputs = self.processor(
                images=image,
                input_boxes=[[[box.xmin, box.ymin, box.xmax, box.ymax]]],
                return_tensors="pt",
            ).to(self.device.value)
            with self.torch.inference_mode():
                prediction = self.model(
                    **dict(inputs.items()), multimask_output=False
                )
            masks = self.processor.post_process_masks(
                prediction.pred_masks.cpu(), inputs["original_sizes"]
            )
            yield image_from_mask(masks[0].squeeze().tolist(), image.size)
