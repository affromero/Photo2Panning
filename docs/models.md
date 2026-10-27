# Local models on this Mac

The model tools target Apple Silicon. Install their dependencies with:

```bash
uv sync --frozen --extra models
```

Python 3.14.9 is the supported version. Python 3.15 triggers a reproduced
native PyTorch crash when worker threads copy tensors to the CPU.
The project pins 3.14 while that inference path is affected.

Model packages load only when their image, mask, or video request needs
them.
The first use downloads weights into the normal Hugging Face cache.
Choose quantized weights and a small image for an initial run. A successful
package import verifies installation; generating an image also verifies
that the selected weights and inference path work on the machine.

## Camera scenes

```bash
uv sync --frozen --extra camera
uv run --extra camera pic2panning scene examples/media/waterfall.jpg \
  --output waterfall.ply --metadata waterfall-scene.json \
  --num-tokens 1800 --max-edge 1920 --device mps
```

The camera extra installs MoGe-2 for local geometry prediction and MLX3D
for native Metal Gaussian rendering. Model code and renderer revisions are
pinned. The official `Ruicheng/moge-2-vitl-normal` checkpoint is pinned to
`cb0e8bbd6b1e243589717c78e750b1ba4c093acf`. Its `model.pt` file is
1,323,815,904 bytes. Downloads enter the normal Hugging Face cache, and
the adapter verifies the checkpoint SHA256 before loading it.

The [official weight card](https://huggingface.co/Ruicheng/moge-2-vitl-normal/blob/cb0e8bbd6b1e243589717c78e750b1ba4c093acf/README.md)
declares the MIT license. The [MoGe source license](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/LICENSE)
contains Microsoft MIT terms and Apache 2.0 terms for the DINOv2 components.
The [MLX3D renderer](https://github.com/amirhossein-razlighi/mlx3D/tree/29ba415cc8a9004750b44fe9083e1ce6715d5f0d)
runs the exported Gaussian scene locally.

MoGe predicts metric points, depth, normals, and intrinsics. The adapter
creates one source-colored, degree-zero Gaussian per predicted pixel, with
an analytic footprint based on pixel size and depth. It retains finite
raw sky predictions and uses the foreground mask to choose conservative
default camera offsets. No hidden surfaces are reconstructed behind
foreground objects, so keep camera movements modest.

`--num-tokens` accepts 1200 to 3600 and defaults to 1800. More tokens
increase inference detail and cost. `--max-edge` bounds the geometry's
longest image dimension and defaults to 1920; it accepts 256 to 4096.
Calibration scales back to the original photo dimensions. The original
photo stays available for sharp pans and pause endpoints. Explicit `mps`
and `cpu` devices are supported for prediction; a failed MPS request does
not switch devices. Camera rendering requires MLX Metal.

The PLY scene and JSON calibration can be reused:

```bash
uv run --extra camera pic2panning camera examples/media/waterfall.jpg \
  --scene waterfall.ply --metadata waterfall-scene.json \
  --pause-at 0.6 --output waterfall-camera.mp4 --crf 16 --preset slow
```

The default timeline is 4 seconds of panning, a 0.25-second hold,
a 5-second excursion and return, another hold, and 3.2 seconds of resumed
panning. It produces a silent 12.7-second MP4 at 1080 by 1920 and 60 fps.
Each phase rounds up to whole frames. `--before`, `--duration`, `--after`,
and `--hold` control these phases.

`--pan-start`, `--pause-at`, and `--pan-end` select horizontal crop
positions between 0 and 1. They must advance through the pause in either
direction. `--position X Y Z` sets the far displacement in estimated
metres: right, down, forward. Negative Y raises the camera. Omitting it
uses a small offset based on foreground depth. The camera eases out and
back to the same pose. Short linear-light handoffs use the original paused
photo; H.264 encoding changes the final pixels slightly.

Source and optional scene checksums protect the calibration relationship.
The PLY scene and JSON metadata must be local files. Rendering streams frames
and does not retain the completed video in memory. Inferred geometry may
soften fine details or reveal gaps during wider camera moves.

On a 36 GB M3 Max, a cached offline CLI run reconstructed the public
1920 by 1280 waterfall photo into 2,457,600 Gaussians in 7.42 seconds.
It used MPS, 1800 tokens, and a 1920-pixel maximum edge. Peak physical
memory was 4.46 GB. The production camera API rendered the complete
12.7-second, 762-frame timeline at 1080 by 1920 and 60 fps. All frames
decoded successfully, and inspected pause and return boundaries retained
the original framing. This verifies one public scene; other photos and
wider camera paths need their own visual check.

## First and last frame video

Install the separate video dependencies:

```bash
uv sync --frozen --extra video
```

This backend uses the native MLX LTX two-stage pipeline on Apple Silicon.
The pipeline packages are pinned to upstream Git revision
`f14dbf930626eafd86b9f29030ac7a23b4e914b8`. Its 0.18 implementation provides
the joint first/last image conditioning used here; that revision is
installed directly from Git because the required release is unpublished
on PyPI.

The default checkpoints are pinned too:

| Component | Checkpoint | Revision |
| --- | --- | --- |
| LTX-2.3 q4 | `dgrauet/ltx-2.3-mlx-q4` | `56a5866d638ecfe37c54d348e88938235185c2d4` |
| Gemma 3 12B 4-bit | `mlx-community/gemma-3-12b-it-4bit` | `86cc6a8dedbc456dd0e4af01a9d09f396f77e558` |

The first default run downloads about 40 GB into the Hugging Face cache.
Downloads select the LTX components needed by this pipeline and the Gemma
text encoder. Local input images stay on the Mac. `--weights` and
`--text-weights` accept compatible local checkpoint directories to use
instead of downloading the defaults.

```bash
uv run --extra video pic2panning generate-video \
  --first-frame examples/media/waterfall.jpg \
  --last-frame examples/media/mountains.jpg \
  --output generated.mp4 \
  --prompt "A continuous camera move across the landscape toward the final view." \
  --size 576 1024 --frames 65 --fps 24 --steps 20 --refine-steps 3 --seed 0
```

Both stages receive guides at frame 0 and the last frame. These guides
influence appearance and motion; generated endpoints can differ from the
original pixels. Choose overlapping views of the same scene when the goal
is a continuous camera move. The bundled public photos illustrate the API.

Inputs receive EXIF orientation and a centered crop matching the requested
aspect ratio. Use `edit` to crop each input before generation when the
centered framing removes a landmark you need to retain. Both output
dimensions must be multiples of 64. `frames` must be at least 9 and equal
`8k + 1`, such as 49, 65, or 97. Frame rate ranges from 1 to 60. Duration
is the frame count divided by frame rate; 65 frames at 24 fps take about
2.71 seconds of playback.

Defaults are 576 by 1024, 65 frames, 24 fps, seed 0, 20 first-stage steps,
and 3 refinement steps. Refinement accepts 1 to 3 steps. An optional
`--negative-prompt` names unwanted content or motion. `low_memory` defaults
to true and releases stages promptly. Transformer block streaming stays
enabled even when `low_memory` is false.

Generation writes a silent MP4. To add sound afterward, use the ordinary
`render` command with `--video` and `--audio`. Output dimensions, frame
count, frame rate, duration, and absence of audio are checked before the
file is published. Inference errors propagate to the caller.

On a 36 GB M3 Max Mac, one local smoke run generated a silent 576 by 1024
clip with 49 frames at 24 fps (2.042 seconds of playback). It used 20
first-stage steps and 3 refinement steps and completed in 16 minutes,
57 seconds. Peak Metal memory was 12.95 GiB; peak physical memory was
15.6 GiB. Inspection of all 49 frames found a coherent scene with no
obvious jumps. This single run checks execution and that clip's continuity;
output quality depends on the input images and prompt.

Sources: [pinned LTX pipeline](https://github.com/dgrauet/ltx-2-mlx/tree/f14dbf930626eafd86b9f29030ac7a23b4e914b8),
[LTX checkpoint](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4/tree/56a5866d638ecfe37c54d348e88938235185c2d4),
[Gemma checkpoint](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit/tree/86cc6a8dedbc456dd0e4af01a9d09f396f77e558).

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

The [README examples](../readme.md#change-the-season-with-flux) show two
actual 768 by 512 edits using 4-bit FLUX 3, 16 steps, and seed 7.
The [winter job](../examples/generate.json) edits the whole reference.
For a local sky change, run the [SAM job](../examples/mask.json) followed
by the [masked edit](../examples/generate-masked.json). The cyan overlay
shows the selected sky on the photo. The job uses its binary PNG mask.

On this 36 GB M3 Max with cached weights, the winter edit took 85.79
seconds and peaked at 23.25 GB physical memory. The sky edit took 73.04
seconds and peaked at 23.65 GB. CPU sky segmentation took 23.86 seconds.
All 277,164 pixels outside the sky selection stayed byte-identical in the
final PNG. The JPEG comparison panels are for display; use the lossless
PNGs when checking preservation.

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
