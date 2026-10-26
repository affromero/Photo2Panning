# Photo2Panning

[![CI](https://github.com/affromero/Photo2Panning/actions/workflows/quality.yml/badge.svg)](https://github.com/affromero/Photo2Panning/actions/workflows/quality.yml)
[![Release](https://img.shields.io/github/v/release/affromero/Photo2Panning)](https://github.com/affromero/Photo2Panning/releases/latest)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](https://docs.python.org/3.14/)
[![MIT license](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)](docs/models.md)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv)](https://docs.astral.sh/uv/)

Turn photos into pans, zooms, and 3D camera moves. Reframe footage, change its
speed, and add music or title cards. Photo2Panning also provides image
editing and local model tools through a CLI, Python library, and validated
JSON jobs.

| Photo pan | Zoom and return | Pan, 3D camera, resume |
| :---: | :---: | :---: |
| [<img src="examples/media/waterfall-pan.gif" alt="Portrait pan across the waterfall and coastal cliffs" width="180"/>](https://github.com/affromero/Photo2Panning/releases/download/v0.3.0/waterfall-pan.mp4) | [<img src="examples/media/mountain-zoom.gif" alt="Zoom into the snowy mountains and return to the original view" width="180"/>](https://github.com/affromero/Photo2Panning/releases/download/v0.3.0/mountain-zoom.mp4) | [<img src="examples/media/waterfall-camera.gif" alt="Complete waterfall pan, pause, 3D camera excursion, return, and resumed pan" width="180"/>](https://github.com/affromero/Photo2Panning/releases/download/v0.3.0/waterfall-camera.mp4) |

Pan and zoom previews are short excerpts. The camera preview shows the
complete sequence at 8 fps. Click a result for its full 1080 by 1920,
60 fps MP4.

[Quick start](#quick-start) · [Demos](#demos) · [Features](#features) ·
[Python](#python-library) · [LLM tools](#llm-tools-and-json-jobs) ·
[Local models](#local-models)

## Quick start

Requires an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and [Homebrew](https://brew.sh/).

```bash
brew install ffmpeg
git clone https://github.com/affromero/Photo2Panning.git
cd Photo2Panning
uv sync --frozen
uv run pic2panning --help
```

`uv` installs the pinned Python 3.14.9 interpreter and creates `.venv`.
Photo motion, video processing, and ordinary image edits work with this
core installation. Try either public photo below.

## Demos

### Pan across Múlafossur waterfall

<img src="examples/media/waterfall.jpg" alt="Original Múlafossur waterfall photo" width="320"/>

[Original photo on Pexels](https://images.pexels.com/photos/29188556/pexels-photo-29188556/free-photo-of-stunning-sunset-over-mulafossur-waterfall-faroe-islands.jpeg).

```bash
uv run pic2panning render examples/media/waterfall.jpg \
  --output waterfall-pan.mp4 --motion panning-lr \
  --duration 7.5 --size 1080 1920 --fps 60 \
  --shutter-angle 180 --crf 16 --preset slow
```

[Watch the full pan](https://github.com/affromero/Photo2Panning/releases/download/v0.3.0/waterfall-pan.mp4).

### Zoom into the mountains and return

<img src="examples/media/mountains.jpg" alt="Original northern lights and snowy mountains photo" width="320"/>

[Original photo on Pexels](https://images.pexels.com/photos/2113566/pexels-photo-2113566.jpeg).

```bash
uv run pic2panning render examples/media/mountains.jpg \
  --output mountain-zoom.mp4 --motion zoom-in --zoom 1.5 --reverse \
  --duration 4 --size 1080 1920 --fps 60 \
  --shutter-angle 180 --crf 16 --preset slow
```

[Watch the full zoom](https://github.com/affromero/Photo2Panning/releases/download/v0.3.0/mountain-zoom.mp4).

The bundled photos are resized copies of the original public demos and
follow the [Pexels license](https://www.pexels.com/license/).

### Pan, pause, move a 3D camera, and resume

Install the camera tools and reconstruct a reusable scene from the photo:

```bash
uv sync --frozen --extra camera
uv run --extra camera pic2panning scene examples/media/waterfall.jpg \
  --output waterfall.ply --metadata waterfall-scene.json --max-edge 1920

uv run --extra camera pic2panning camera examples/media/waterfall.jpg \
  --scene waterfall.ply --metadata waterfall-scene.json \
  --pause-at 0.6 --before 4 --duration 5 --after 3.2 --hold 0.25 \
  --output waterfall-camera.mp4 --crf 16 --preset slow
```

[Watch the full camera sequence](https://github.com/affromero/Photo2Panning/releases/download/v0.3.0/waterfall-camera.mp4).

MoGe-2 predicts metric depth and camera intrinsics locally on MPS. The
first scene downloads a pinned 1.32 GB checkpoint. The model card declares
MIT licensing; DINOv2 components use Apache 2.0. See
[camera models and measured runs](docs/models.md#camera-scenes).

The camera moves through source-colored Gaussian geometry, returns to its
original pose, and resumes the photo pan. Pans and pause endpoints use the
original photo; short handoffs blend in linear light. Video compression
changes output pixels slightly. Inferred depth supports modest moves;
larger excursions can reveal missing surfaces or soften details.

`--max-edge` limits geometry resolution while retaining the original photo
for the pans. Reuse the PLY and JSON for later camera renders. The metadata
binds them to the source photo's checksum.

## Features

| Tool | Controls |
| --- | --- |
| Photo motion | Four pan directions, zoom, return path, slower center motion |
| 3D camera | Local scene reconstruction, calibrated excursion, return and resumed pan |
| Video | Crop to a new aspect ratio, bouncing pan, zoom, playback speed |
| Generated video | Local LTX shot guided by a first and last photo |
| Audio and titles | Retain source sound, replace or loop music, add title cards |
| Image editing | Crop, resize, rotate, flip, brightness, contrast, saturation |
| Local models | Prompted generation, reference-image edits, masked fill, segmentation |

Sizes use **width, height**. `1080 1920` produces a 9:16 portrait video;
`1920 1080` produces 16:9 landscape. Images retain their proportions and
crop to fill the frame. Video reframing respects display rotation and
pixel aspect ratio. MP4 dimensions must be even.

### Motion and export quality

| `--motion` | Movement |
| --- | --- |
| `panning-lr`, `panning-rl` | Left to right, right to left |
| `panning-ud`, `panning-du` | Top to bottom, bottom to top |
| `zoom-in`, `zoom-out` | Toward the center, away from the center |

For photos, `--duration` sets the forward motion. `--reverse` appends the
return path and `--focus-center` slows movement near the center. `--zoom`
sets the closest magnification and defaults to `4/3`.

`--shutter-angle 180` samples half a frame interval to soften motion along
its direction and reduce strobing during fast pans. Angles range from 0 to 360;
the default `0` keeps instantaneous frames. An enabled shutter renders
eight samples per interior photo frame, increasing render time. The first
and last frames stay sharp. This option applies only to photos.

Choose timing explicitly: `--duration 7.5 --fps 60 --shutter-angle 180`
combines a slower photo pan with more output frames and motion exposure.
Results depend on the scene, movement speed, and playback display.

MP4 exports use H.264. `--crf` controls detail: lower values produce larger
files. The default is `18`. `--preset` controls compression effort and
accepts `veryfast`, `fast`, `medium`, `slow`, or `veryslow`. The default is
`veryfast`; the demos use `slow` for the final export.

GIF output requires an explicit frame rate that divides 100 exactly:
`1`, `2`, `4`, `5`, `10`, `20`, `25`, `50`, or `100`.

```bash
uv run pic2panning render photo.jpg \
  --output pan.gif --motion panning-lr --duration 3 --fps 20
```

### Video, music, and title cards

Use `--video` for footage. Speed changes preserve the pitch of retained
audio. A video pan bounces at the source edges; `--pan-speed` sets movement
in output pixels per frame and defaults to `5`.

```bash
uv run pic2panning render walk.mp4 \
  --video --speed 2 --output walk-fast.mp4

uv run pic2panning render wide.mp4 \
  --video --motion panning-lr --output wide-portrait.mp4

uv run pic2panning render coast.jpg forest.jpg \
  --output trip.mp4 --motion zoom-in \
  --audio music.m4a --audio-start 12 --loop-audio \
  --cover "A weekend away"

uv run pic2panning inspect trip.mp4
```

Multiple sources appear in input order. The CLI applies the same options
to each source; Python requests and JSON jobs support separate clip
settings. Original video sound stays unless you replace it or set
`keep_audio=False` in a request.

A global soundtrack spans the complete timeline, including title cards.
Per-clip soundtracks apply to their own clips. Short tracks end in silence
unless looping is enabled. An audio offset excludes the skipped prefix
from every repeat. Title cards keep their own duration.

For YouTube audio, install its optional dependency with
`uv sync --frozen --extra youtube`, then pass the URL to `--audio`.

### Soundtrack a finished video

```bash
uv run python examples/add_music.py --help
uv run python examples/add_music.py story.mp4 music.m4a \
  --output story-scored.mp4 --start 12 --loop
```

The example copies the video stream without recompression and replaces any
existing soundtrack. The audio input can also be an HTTP URL. The selected
track starts at `--start` and is trimmed or padded with silence to match
the video duration. `--loop` repeats it from that offset. Existing outputs
require `--overwrite`. You can also upload a silent video and add music in
Instagram.

### Edit a picture

<img src="examples/media/waterfall-edit.jpg" alt="Original waterfall photo beside a portrait crop with brighter tones" width="704"/>

```bash
uv run pic2panning edit examples/media/waterfall.jpg \
  --crop 680 0 1640 1280 --resize 450 600 \
  --brightness 1.12 --contrast 1.08 --saturation 1.1 \
  --output waterfall-edited.jpg
```

Crop coordinates are `left top right bottom` in source pixels. The CLI
applies crop, resize, rotation, flip, then adjustments. Rotation uses
counterclockwise degrees and expands the canvas. Flip accepts `horizontal`
or `vertical`. Adjustment factors use `1` for no change. JSON jobs let you
choose the operation order.

## Python library

```python
from pic2panning import render
from pic2panning.schemas.base import Size
from pic2panning.schemas.media import ImageClip, Motion, RenderRequest

result = render(
    RenderRequest(
        clips=(
            ImageClip(
                source="examples/media/waterfall.jpg",
                duration=5,
                motion=Motion.PAN_LR,
            ),
            ImageClip(
                source="examples/media/mountains.jpg",
                duration=3,
                motion=Motion.ZOOM_IN,
            ),
        ),
        output="trip.mp4",
        size=Size(width=1080, height=1920),
        fps=30,
    )
)
print(result.model_dump_json(indent=2))
```

The public functions are `render`, `reconstruct_scene`, `render_camera`,
`edit`, `generate`, `generate_video`, `mask`, and `run_job`.
Requests validate before processing. Results include
output paths, dimensions, and timing when applicable. Existing outputs
require `overwrite=True` or `--overwrite`. Input aliases stay protected.

## LLM tools and JSON jobs

Start with [llms.txt](llms.txt) and the [tool guide](docs/agent-tools.md).
The CLI generates its JSON Schema from the same request models the Python
library uses:

```bash
uv run pic2panning schema > job.schema.json
uv run pic2panning validate examples/render.json
uv run pic2panning run examples/render.json
```

A job selects `render`, `scene`, `camera`, `edit`, `generate`,
`generate-video`, or `mask`
with its `kind` field.
Unknown fields fail validation. `validate` checks the request without
reading media or loading models. Commands write JSON results to stdout
and return a nonzero exit status on failure. Relative paths resolve from
the process working directory.

Examples: [render](examples/render.json), [image edits](examples/edit.json),
[scene reconstruction](examples/scene.json),
[camera timeline](examples/camera.json),
[image generation](examples/generate.json),
[video generation](examples/generate-video.json),
[segmentation](examples/mask.json).

## Local models

Install model dependencies separately. Weights download on first use.

```bash
uv sync --frozen --extra models

uv run --extra models pic2panning generate \
  --source examples/media/waterfall.jpg --output waterfall-evening.png \
  --prompt "The same waterfall at dusk, warm light on the cliffs, calm sea, photograph" \
  --model flux3 --quantization 4 --size 512 512 --seed 7
```

FLUX 3 Image Klein uses quantized MLX weights on Metal. It supports
4, 6, and 8-bit weights; the default is 4-bit. Source images are optional
for generation. Width and height must be multiples of 16.

Grounding DINO Tiny and SAM 2.1 Tiny create masks on the CPU. This request
selects the falling water in the original public photo:

```bash
uv run --extra models pic2panning mask examples/media/waterfall.jpg \
  --prompt waterfall --device cpu --threshold 0.3 --output waterfall-mask.png
```

<img src="examples/media/waterfall-mask.png" alt="Actual waterfall mask: falling water is white, the sky and land are black" width="480"/>

Masks use white for the region to edit and black for the region to retain.
Source, mask, and output dimensions must match for masked edits. Masks and
masked edits require PNG. Other image outputs support PNG, JPEG, and WebP.
Use a mask with `flux1-fill` or FLUX 3; see
[masked editing controls](docs/models.md#flux-generation-and-editing).
Some weights require Hugging Face authentication and acceptance of the
model terms. See [local model setup and measured runs](docs/models.md).

### Generate a shot between two photos

The `video` extra adds local LTX-2.3 generation on MLX. Its two stages use
the first and last photos as image guides. Generated endpoints can differ
from the input pixels. For a continuous camera move, choose adjacent views
of the same scene. The bundled public photos illustrate the request format.

```bash
uv sync --frozen --extra video
uv run --extra video pic2panning generate-video \
  --first-frame examples/media/waterfall.jpg \
  --last-frame examples/media/mountains.jpg \
  --output generated.mp4 \
  --prompt "A continuous camera move across the landscape toward the final view." \
  --size 576 1024 --frames 65 --steps 20
```

The output is a silent MP4 at 24 fps by default. Inputs are center-cropped
to fit; use `edit` first to choose another framing. Width and height must
be multiples of 64, and frame counts must be `8k + 1` with at least 9
frames. The first default run downloads about 40 GB of model weights.
Local input images stay on this Mac. See
[video models and controls](docs/models.md#first-and-last-frame-video).

## Development

```bash
uv sync --frozen --extra camera
uv run --extra camera pre-commit install
uv run --extra camera pre-commit run --all-files
uv run --extra camera mypy
uv run --extra camera pytest
uv build
```

Ruff checks style and formatting. `ty` checks types at commit time; strict
mypy also runs in CI. The camera dependencies allow both tools to check
the optional adapters; installing them does not download model weights.
Tests inspect image content and real FFmpeg output. Model execution checks
and memory measurements are documented in [local models](docs/models.md).

Demo entry points in [`pic2panning/demos`](pic2panning/demos) use the same
CLI and processing functions. See [AGENTS.md](AGENTS.md) for contributor
rules and [LICENSE](LICENSE) for the MIT license.
