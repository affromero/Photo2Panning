# Local models on this Mac

The model tools target Apple Silicon. Install their dependencies with:

```bash
uv sync --frozen --extra models
```

Python 3.14.9 is the supported version. Python 3.15 triggers a reproduced
native PyTorch crash when worker threads copy tensors to the CPU.
The project pins 3.14 while that inference path is affected.

Model packages load only when a generation or masking request needs them.
The first use downloads weights into the normal Hugging Face cache.
Choose quantized weights and a small image for an initial run. A successful
package import verifies installation; generating an image also verifies
that the selected weights and inference path work on the machine.

## FLUX generation and editing

| Model value | Operation | Initial settings |
| --- | --- | --- |
| `flux3` | Text-to-image or reference-image editing | 4-bit, 512 by 512, 16 steps |
| `flux1-schnell` | Image-to-image or text-to-image | 4-bit, 512 by 512, 4 steps |
| `flux1-dev` | Image-to-image or text-to-image | 4-bit, 512 by 512, 28 steps |
| `flux1-fill` | Fill a white region in a supplied mask | 4-bit, 512 by 512, 28 steps |

Omit `steps` to use the model's default: 16 for FLUX 3, 4 for Schnell, and
28 for Dev or Fill. Default guidance is 3.5 for Dev and 30 for Fill. The
distilled FLUX 3 and Schnell models reject guidance controls.

FLUX 3 Image Klein is the default. Its distilled model combines a 5.6B
transformer with a Qwen3-VL-4B text encoder. MFLUX runs it with MLX and
supports 4, 6, or 8-bit quantization. Use `--quantization 4` to keep the
weight footprint small. The default FLUX 3 4-bit checkpoint is
`mflux-community/flux-3-image-klein-mflux-q4`, which stores MFLUX-format
quantized components. Every model has an explicit prequantized community
checkpoint for each supported bit width. Higher resolution increases
activation memory even with quantized weights.

Source: [MFLUX FLUX 3 implementation](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux3/README.md).

Run a small text-to-image job:

```bash
uv run --extra models pic2panning generate \
  --output model-check.png \
  --model flux3 --quantization 4 --size 256 256 --steps 16 --seed 1 \
  --prompt "A red ceramic mug on a wooden table, soft daylight, photograph"
```

Add `--source photo.jpg` for reference-image editing. Describe the complete
desired result, including the content to retain. The output size must use
multiples of 16. `--weights` selects a compatible local or Hugging Face
MFLUX checkpoint; check the checkpoint's architecture and quantization
before using it.

The FLUX 3 distilled model does not use classifier-free guidance. Use
guidance only with a model that supports it. Image strength belongs to
FLUX 1 image-to-image processing. Model requests reject incompatible
controls for the selected model.

FLUX 1 Fill takes `--source` and `--mask`. A white mask pixel selects an
area for editing. Black pixels retain the source. Create a mask with the
`mask` command, or provide your own PNG. Source, mask, and output dimensions
must match. FLUX 3 applies a supplied mask by compositing the edited image
over the original; FLUX 1 Fill conditions its generation on the mask.
Both paths retain unmasked source pixels exactly.
Masks and masked edits require lossless PNG. Other generated images
support PNG, JPEG, and WebP.

LoRA adapters are explicit request options in the Python API and JSON
jobs. Each adapter supplies a `path` and a numeric `scale`. Use adapters
trained for the selected model architecture.

Some checkpoints require accepting their terms on Hugging Face and
authenticating before download:

```bash
uv run --extra models hf auth login
```

Model loading errors propagate to the caller. A failed request does not
switch to a different model or device.

On this 36 GB Mac with Python 3.14.9 and cached weights, FLUX 3 4-bit
generated a 256 by 256 image at 16 steps in 15.51 seconds, including model
loading. Peak memory footprint was 10.98 GB. A reference edit at that size
and 8 steps completed in 16.90 seconds with an 11.34 GB peak. Use 512 by
512 and 16 steps for normal FLUX 3 edits; the smaller runs check execution.

FLUX 1 Fill 4-bit also completed a 256 by 256 masked edit on this Mac.
That four-step execution check used guidance 30 and peaked at 17.42 GB.
The edited region changed and unselected pixels stayed identical, but
four steps produced a rough fill. Use the default 28 steps for ordinary
Fill requests, or tune within 25 to 50 steps. Other model variants and
larger images have different memory requirements.

## Text-guided segmentation

Mask requests use Grounding DINO Tiny to find text-described objects and
SAM 2.1 Tiny to segment them. The default CPU path uses the small models;
`--device cpu` makes that execution choice explicit. Generation uses
Metal through MLX; segmentation runs on the CPU.

```bash
uv run --extra models pic2panning mask photo.jpg \
  --prompt "person" --threshold 0.3 --device cpu --output people.png
```

The default merges detections into one mask. `--separate` requests an
individual output for each detected object, named with `_000`, `_001`,
and subsequent numeric suffixes. Read the result's `outputs` field for the
resulting paths. If no objects match, the request fails with a
`No objects matched` error and creates no output.

A CPU smoke run with the default Tiny models and threshold `0.3` found and
segmented a truck in a 512 by 341 image in 6.70 seconds on this Mac,
including cached model loading. Peak process memory was 2.50 GB.

Model inference checks belong outside the ordinary test suite because
they download weights and depend on the machine's memory and available
model access. The normal suite checks the request and output behavior at
the model boundary without requiring those downloads.
