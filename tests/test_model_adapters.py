"""Exercise production adapters at their external framework boundaries."""

import sys
from collections.abc import ItemsView
from contextlib import nullcontext
from types import ModuleType
from typing import Self

import pytest
from PIL import Image
from pydantic import TypeAdapter

from pic2panning.backends.mflux import (
    MfluxBackend,
    canonical_default_checkpoint,
)
from pic2panning.backends.segmentation import GroundedSamBackend
from pic2panning.schemas.base import PydanticBaseModel, Size
from pic2panning.schemas.models import (
    GenerateRequest,
    GenerationModel,
    Lora,
    MaskRequest,
    Quantization,
)


class FluxProvider:
    """Capture model loading and inference at the MFLUX boundary."""

    variant: str
    initialization: dict[str, object]
    generation: dict[str, object]
    reference_pixels: bytes | None
    references_pixels: tuple[bytes, ...]
    mask_pixels: bytes | None
    image: Image.Image

    def __init__(self) -> None:
        """Keep provider observations separate from application helpers."""
        self.variant = ""
        self.initialization = {}
        self.generation = {}
        self.reference_pixels = None
        self.references_pixels = ()
        self.mask_pixels = None
        self.image = Image.new("RGBA", (1, 1))

    def generate_image(self, **parameters: object) -> Self:
        """Read temporary conditioning files while inference can access them."""
        self.generation = parameters
        references = parameters.get("image_paths")
        if references is not None:
            pixels = []
            for path in TypeAdapter(list[str]).validate_python(references):
                with Image.open(path) as image:
                    pixels.append(image.tobytes())
            self.references_pixels = tuple(pixels)
        reference = (
            TypeAdapter(list[str]).validate_python(references)[0]
            if references is not None
            else TypeAdapter(str | None).validate_python(
                parameters.get("image_path")
            )
        )
        if reference is not None:
            with Image.open(reference) as image:
                self.reference_pixels = image.tobytes()
        selection = parameters.get("masked_image_path")
        if selection is not None:
            with Image.open(
                TypeAdapter(str).validate_python(selection)
            ) as mask:
                self.mask_pixels = mask.tobytes()
        self.image.close()
        self.image = Image.new(
            "RGBA",
            (
                TypeAdapter(int).validate_python(parameters["width"]),
                TypeAdapter(int).validate_python(parameters["height"]),
            ),
            (16, 32, 64, 128),
        )
        return self


class FluxFactory:
    """Represent one independently selected external model class."""

    provider: FluxProvider
    variant: str

    def __init__(self, provider: FluxProvider, variant: str) -> None:
        """Associate constructor arguments with the selected model family."""
        self.provider = provider
        self.variant = variant

    def __call__(self, **parameters: object) -> FluxProvider:
        """Record the options the external model receives."""
        self.provider.variant = self.variant
        self.provider.initialization = parameters
        return self.provider


class ModelRegistry:
    """Make external configuration selection visible without loading weights."""

    @staticmethod
    def from_name(model_name: str) -> str:
        """Return an identifiable configuration supplied by the registry."""
        return f"configured:{model_name}"


class Flux3Module(ModuleType):
    """Expose the external FLUX 3 generation and reference-edit classes."""

    Flux3Klein: FluxFactory
    Flux3KleinEdit: FluxFactory

    def __init__(self, provider: FluxProvider) -> None:
        """Supply only exports available in the actual FLUX 3 module."""
        super().__init__("mflux.models.flux3.variants")
        self.Flux3Klein = FluxFactory(provider, "Flux3Klein")
        self.Flux3KleinEdit = FluxFactory(provider, "Flux3KleinEdit")


class Flux1Module(ModuleType):
    """Expose the FLUX 1 text-to-image and reference-edit class."""

    Flux1: FluxFactory

    def __init__(self, provider: FluxProvider) -> None:
        """Supply only the export available in the FLUX 1 module."""
        super().__init__("mflux.models.flux.variants.txt2img.flux")
        self.Flux1 = FluxFactory(provider, "Flux1")


class FluxFillModule(ModuleType):
    """Expose the masked-fill class independently of other model variants."""

    Flux1Fill: FluxFactory

    def __init__(self, provider: FluxProvider) -> None:
        """Supply only the export available in the masked-fill module."""
        super().__init__("mflux.models.flux.variants.fill.flux_fill")
        self.Flux1Fill = FluxFactory(provider, "Flux1Fill")


class ConfigModule(ModuleType):
    """Expose the external configuration registry."""

    ModelConfig: type[ModelRegistry]

    def __init__(self) -> None:
        """Keep model configuration imports distinct from model classes."""
        super().__init__("mflux.models.common.config.model_config")
        self.ModelConfig = ModelRegistry


@pytest.fixture
def flux_provider(monkeypatch: pytest.MonkeyPatch) -> FluxProvider:
    """Replace optional external modules without replacing adapter logic."""
    provider = FluxProvider()
    for module in (
        Flux3Module(provider),
        Flux1Module(provider),
        FluxFillModule(provider),
        ConfigModule(),
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return provider


class GenerationCase(PydanticBaseModel):
    """Expected external contracts for each model and conditioning mode."""

    model: GenerationModel
    reference: bool
    variant: str
    config: str
    checkpoint: str
    steps: int
    guidance: float | None = None


@pytest.mark.parametrize(
    "case",
    [
        GenerationCase(
            model=GenerationModel.FLUX3,
            reference=False,
            variant="Flux3Klein",
            config="flux3-image-klein",
            checkpoint="flux-3-image-klein",
            steps=16,
        ),
        GenerationCase(
            model=GenerationModel.FLUX3,
            reference=True,
            variant="Flux3KleinEdit",
            config="flux3-image-klein",
            checkpoint="flux-3-image-klein",
            steps=16,
        ),
        GenerationCase(
            model=GenerationModel.FLUX1_SCHNELL,
            reference=True,
            variant="Flux1",
            config="schnell",
            checkpoint="flux-1-schnell",
            steps=4,
        ),
        GenerationCase(
            model=GenerationModel.FLUX1_DEV,
            reference=True,
            variant="Flux1",
            config="dev",
            checkpoint="flux-1-dev",
            steps=28,
            guidance=3.5,
        ),
        GenerationCase(
            model=GenerationModel.FLUX1_FILL,
            reference=True,
            variant="Flux1Fill",
            config="dev-fill",
            checkpoint="flux-1-dev-fill",
            steps=28,
            guidance=30,
        ),
    ],
    ids=["flux3-text", "flux3-reference", "schnell", "dev", "fill"],
)
def test_generation_preserves_model_selection_and_conditioning_defaults(
    flux_provider: FluxProvider, case: GenerationCase
) -> None:
    """Each family receives its own class, weights, controls, and image files."""
    fill = case.model == GenerationModel.FLUX1_FILL
    request = GenerateRequest(
        source="reference.png" if case.reference else None,
        mask="selection.png" if fill else None,
        output="result.png",
        prompt="A blue coast at dusk",
        seed=19,
        model=case.model,
        size=Size(width=32, height=16),
    )
    with (
        Image.new("RGB", (32, 16), "blue") as reference,
        Image.new("L", (32, 16), 255) as selection,
        MfluxBackend().generate(
            request,
            reference if case.reference else None,
            selection if fill else None,
        ) as result,
    ):
        assert result.mode == "RGB"
        assert result.size == (32, 16)
        assert result.getpixel((0, 0)) == (16, 32, 64)
        assert flux_provider.reference_pixels == (
            reference.tobytes() if case.reference else None
        )
        assert flux_provider.mask_pixels == (
            selection.tobytes() if fill else None
        )
    assert flux_provider.variant == case.variant
    assert flux_provider.initialization == {
        "quantize": 4,
        "model_path": f"mflux-community/{case.checkpoint}-mflux-q4",
        "model_config": f"configured:{case.config}",
        "lora_paths": None,
        "lora_scales": None,
        "bake_lora": False,
    }
    parameters = flux_provider.generation
    assert parameters["prompt"] == "A blue coast at dusk"
    assert parameters["seed"] == 19
    assert (parameters["width"], parameters["height"]) == (32, 16)
    assert parameters["num_inference_steps"] == case.steps
    assert parameters.get("guidance") == case.guidance
    if case.model in {
        GenerationModel.FLUX1_SCHNELL,
        GenerationModel.FLUX1_DEV,
    }:
        assert parameters["image_strength"] == 0.7
    else:
        assert "image_strength" not in parameters
    if case.model == GenerationModel.FLUX3 and case.reference:
        assert "image_paths" in parameters
        assert "image_path" not in parameters


def test_flux3_uses_edit_variant_and_every_ordered_reference_without_source(
    flux_provider: FluxProvider,
) -> None:
    """References-only editing passes all images to the MFLUX edit class."""
    request = GenerateRequest(
        references=("first.png", "second.png"),
        output="result.png",
        prompt="Combine both references",
        size=Size(width=32, height=16),
    )
    with (
        Image.new("RGB", (19, 13), "red") as first,
        Image.new("RGB", (23, 17), "blue") as second,
        MfluxBackend().generate_references(
            request, (first, second), None
        ) as result,
    ):
        assert result.size == (32, 16)
        assert flux_provider.references_pixels == (
            first.tobytes(),
            second.tobytes(),
        )
    assert flux_provider.variant == "Flux3KleinEdit"
    assert "image_path" not in flux_provider.generation


def test_checkpoint_identity_matches_model_selection_and_explicit_weights(
    flux_provider: FluxProvider,
) -> None:
    """Cache identity resolves the exact checkpoint used for inference."""
    request = GenerateRequest(
        output="result.png",
        prompt="An image",
        quantization=Quantization.Q8,
        size=Size(width=16, height=16),
    )
    assert canonical_default_checkpoint(request).endswith("-q8")
    with MfluxBackend().generate(request, None, None):
        assert flux_provider.initialization["model_path"] == (
            canonical_default_checkpoint(request)
        )
    explicit = request.model_copy(update={"weights": "local/custom"})
    assert canonical_default_checkpoint(explicit) == "local/custom"


@pytest.mark.parametrize(
    "model", [GenerationModel.FLUX1_DEV, GenerationModel.FLUX1_FILL]
)
def test_explicit_generation_controls_and_adapters_reach_the_provider(
    flux_provider: FluxProvider, model: GenerationModel
) -> None:
    """User inference and LoRA settings survive model-specific translation."""
    fill = model == GenerationModel.FLUX1_FILL
    request = GenerateRequest(
        source="reference.png",
        mask="selection.png" if fill else None,
        output="result.png",
        prompt="A bright red coast",
        seed=23,
        model=model,
        weights="local/checkpoint",
        quantization=Quantization.Q6,
        steps=7,
        size=Size(width=64, height=32),
        strength=None if fill else 0.25,
        guidance=8.5,
        loras=(
            Lora(path="style.safetensors", scale=0.6),
            Lora(path="detail.safetensors", scale=1.2),
        ),
    )
    with (
        Image.new("RGB", (64, 32), "blue") as reference,
        Image.new("L", (64, 32), 255) as selection,
        MfluxBackend().generate(
            request, reference, selection if fill else None
        ) as result,
    ):
        assert result.size == (64, 32)
        assert result.getpixel((63, 31)) == (16, 32, 64)
    initialization = flux_provider.initialization
    assert initialization["model_path"] == "local/checkpoint"
    assert initialization["quantize"] == 6
    assert initialization["lora_paths"] == [
        "style.safetensors",
        "detail.safetensors",
    ]
    assert initialization["lora_scales"] == [0.6, 1.2]
    assert initialization["bake_lora"] is False
    parameters = flux_provider.generation
    assert parameters["prompt"] == "A bright red coast"
    assert parameters["seed"] == 23
    assert (parameters["width"], parameters["height"]) == (64, 32)
    assert parameters["num_inference_steps"] == 7
    assert parameters["guidance"] == 8.5
    assert parameters.get("image_strength") == (None if fill else 0.25)


class MaskTensor:
    """Provide the tensor conversion methods used at the SAM boundary."""

    values: list[list[bool]]

    def __init__(self, values: list[list[bool]]) -> None:
        """Hold a small full-resolution binary prediction."""
        self.values = values

    def cpu(self) -> Self:
        """Represent an already CPU-resident prediction."""
        return self

    def squeeze(self) -> Self:
        """Expose the two image dimensions expected after singleton removal."""
        return self

    def tolist(self) -> list[list[bool]]:
        """Return native values without importing a tensor framework."""
        return self.values


class SamInput:
    """Carry named processor results and an observable inference device."""

    values: dict[str, object]
    device: str | None

    def __init__(self, values: dict[str, object]) -> None:
        """Store prepared pixels, original dimensions, and object boxes."""
        self.values = values
        self.device = None

    def to(self, device: str) -> Self:
        """Record the requested device at the external tensor boundary."""
        self.device = device
        return self

    def __getitem__(self, key: str) -> object:
        """Expose original dimensions as Transformers' processor does."""
        return self.values[key]

    def items(self) -> ItemsView[str, object]:
        """Pass the prepared values through the model keyword boundary."""
        return self.values.items()


class SamProcessor:
    """Observe object boxes and restore a known full-resolution prediction."""

    requests: list[dict[str, object]]
    prepared: list[SamInput]
    restoration_sizes: object
    prediction: MaskTensor

    def __init__(self) -> None:
        """Use an asymmetric prediction to expose transposed mask axes."""
        self.requests = []
        self.prepared = []
        self.restoration_sizes = None
        self.prediction = MaskTensor(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, False, False, True],
            ]
        )

    def __call__(self, **inputs: object) -> SamInput:
        """Prepare named values that can be traced through actual inference."""
        self.requests.append(inputs)
        image = inputs["images"]
        assert isinstance(image, Image.Image)
        prepared = SamInput(
            {
                "pixel_values": image.tobytes(),
                "input_boxes": inputs["input_boxes"],
                "original_sizes": [[image.height, image.width]],
            }
        )
        self.prepared.append(prepared)
        return prepared

    def post_process_masks(
        self, masks: MaskTensor, original_sizes: object
    ) -> list[MaskTensor]:
        """Return the model prediction at the requested original dimensions."""
        self.restoration_sizes = original_sizes
        return [masks]


class SamModel:
    """Capture SAM inference inputs and execution mode."""

    pred_masks: MaskTensor
    device: str | None
    evaluated: bool
    inputs: dict[str, object]

    def __init__(self, prediction: MaskTensor) -> None:
        """Provide a prediction without loading a model checkpoint."""
        self.pred_masks = prediction
        self.device = None
        self.evaluated = False
        self.inputs = {}

    def to(self, device: str) -> Self:
        """Observe model placement at the external framework boundary."""
        self.device = device
        return self

    def eval(self) -> Self:
        """Record inference mode rather than training mode."""
        self.evaluated = True
        return self

    def __call__(self, **inputs: object) -> Self:
        """Return the known mask for the actual prepared model inputs."""
        self.inputs = inputs
        return self


class CheckpointLoader[Component]:
    """Observe which external checkpoint is requested for each component."""

    component: Component
    checkpoints: list[str]

    def __init__(self, component: Component) -> None:
        """Retain the component provided by the fake external framework."""
        self.component = component
        self.checkpoints = []

    def from_pretrained(self, model: str) -> Component:
        """Return the component and retain its externally selected weights."""
        self.checkpoints.append(model)
        return self.component


class Detector:
    """Return a fractional object box and observe the grounding request."""

    detections: list[dict[str, object]]
    options: dict[str, object]
    image_pixels: bytes | None

    def __init__(self) -> None:
        """Describe one object without invoking a detector model."""
        self.detections = [
            {
                "score": 0.9,
                "label": "truck",
                "box": {
                    "xmin": 1.5,
                    "ymin": 0.25,
                    "xmax": 4.5,
                    "ymax": 2.75,
                },
            }
        ]
        self.options = {}
        self.image_pixels = None

    def __call__(
        self, image: Image.Image, **options: object
    ) -> list[dict[str, object]]:
        """Record the actual image, labels, and confidence passed to grounding."""
        self.options = options
        self.image_pixels = image.tobytes()
        return self.detections


class TransformersModule(ModuleType):
    """Supply external detector, processor, and model exports."""

    detector: Detector
    processor: SamProcessor
    segmenter: SamModel
    Sam2Processor: CheckpointLoader[SamProcessor]
    Sam2Model: CheckpointLoader[SamModel]
    pipeline_options: dict[str, object]

    def __init__(self) -> None:
        """Keep external observations available after production adaptation."""
        super().__init__("transformers")
        self.detector = Detector()
        self.processor = SamProcessor()
        self.segmenter = SamModel(self.processor.prediction)
        self.Sam2Processor = CheckpointLoader(self.processor)
        self.Sam2Model = CheckpointLoader(self.segmenter)
        self.pipeline_options = {}

    def pipeline(self, **options: object) -> Detector:
        """Record explicit detection task, checkpoint, and execution device."""
        self.pipeline_options = options
        return self.detector


class TorchModule(ModuleType):
    """Provide the external inference context without importing PyTorch."""

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        """Execute a small fake prediction without allocating gradients."""
        return nullcontext()


@pytest.fixture
def transformers_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> TransformersModule:
    """Install fake external exports while preserving the real loader."""
    provider = TransformersModule()
    monkeypatch.setitem(sys.modules, "transformers", provider)
    monkeypatch.setitem(sys.modules, "torch", TorchModule("torch"))
    return provider


def test_text_grounding_keeps_cpu_controls_and_full_resolution_object_mask(
    transformers_provider: TransformersModule,
) -> None:
    """Text labels and boxes produce a binary mask with the original axes."""
    request = MaskRequest(
        source="reference.png",
        output="mask.png",
        prompt="person . truck..",
        detector="local/grounding-tiny",
        segmenter="local/sam-tiny",
        threshold=0.65,
    )
    with Image.new("RGB", (5, 3), "blue") as reference:
        masks = GroundedSamBackend().masks(request, reference)
        assert (
            transformers_provider.detector.image_pixels == reference.tobytes()
        )
        assert len(masks) == 1
        with masks[0] as selection:
            assert selection.mode == "L"
            assert selection.size == (5, 3)
            assert selection.tobytes() == bytes(
                [255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255]
            )
    assert transformers_provider.pipeline_options == {
        "task": "zero-shot-object-detection",
        "model": "local/grounding-tiny",
        "device": "cpu",
    }
    assert transformers_provider.detector.options == {
        "candidate_labels": ["person.", "truck."],
        "threshold": 0.65,
    }
    assert transformers_provider.Sam2Processor.checkpoints == [
        "local/sam-tiny"
    ]
    assert transformers_provider.Sam2Model.checkpoints == ["local/sam-tiny"]
    processor = transformers_provider.processor
    assert processor.requests[0]["input_boxes"] == [[[1.5, 0.25, 4.5, 2.75]]]
    assert processor.requests[0]["return_tensors"] == "pt"
    assert processor.prepared[0].device == "cpu"
    assert processor.restoration_sizes == [[3, 5]]
    segmenter = transformers_provider.segmenter
    assert segmenter.device == "cpu"
    assert segmenter.evaluated
    assert segmenter.inputs == {
        **processor.prepared[0].values,
        "multimask_output": False,
    }


def test_unmatched_text_does_not_load_segmentation_weights(
    transformers_provider: TransformersModule,
) -> None:
    """An empty detection result produces no masks or unnecessary SAM load."""
    transformers_provider.detector.detections = []
    request = MaskRequest(
        source="reference.png", output="mask.png", prompt="person"
    )
    with Image.new("RGB", (5, 3)) as reference:
        assert GroundedSamBackend().masks(request, reference) == ()
    assert transformers_provider.detector.options["candidate_labels"] == [
        "person."
    ]
    assert transformers_provider.Sam2Processor.checkpoints == []
    assert transformers_provider.Sam2Model.checkpoints == []
