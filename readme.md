# Photo2Panning

Turn photos into panning and zooming videos. Edit pictures, reframe video,
change playback speed, and add music or title cards. Use the Python library,
the CLI, or validated JSON jobs from an internal LLM.

This project targets Apple Silicon Macs with Python 3.14.9. Pillow handles
image operations and FFmpeg streams video frames. Local model tools use
quantized MLX models and small CPU segmentation models.

## Setup

Install [uv 0.11.17 or newer](https://docs.astral.sh/uv/getting-started/installation/) and
[Homebrew](https://brew.sh/), then run:

```bash
brew install ffmpeg
git clone https://github.com/affromero/Photo2Panning.git
cd Photo2Panning
uv sync --frozen
uv run pic2panning --help
```

`uv` creates `.venv` and installs the pinned Python interpreter when needed.
Photo, video, and ordinary image editing work with the core installation.
Model weights download only when a model operation first needs them.

For local generation, reference-image editing, and segmentation:

```bash
uv sync --frozen --extra models
```

For YouTube soundtracks:

```bash
uv sync --frozen --extra youtube
```

Extras can be combined with `uv sync --frozen --all-extras`.

## Photos to video

Pan across a photo and produce a five-second portrait video:

```bash
uv run pic2panning render coast.jpg \
  --output coast.mp4 \
  --motion panning-lr \
  --duration 5 \
  --size 1080 1920 \
  --fps 30
```

The size is always **width, height**. Use `1920 1080` for landscape output.
Source images retain their proportions and crop to fill the output frame.
MP4 dimensions must be even.

| Motion | Camera movement |
| --- | --- |
| `panning-lr` | Left to right |
| `panning-rl` | Right to left |
| `panning-ud` | Top to bottom |
| `panning-du` | Bottom to top |
| `zoom-in` | Toward the center |
| `zoom-out` | Away from the center |

Add `--reverse` to return along the same path. `--focus-center` slows the
motion around the center. `--zoom` sets the zoom factor and defaults to
about 1.33. A `.gif` output produces an animated GIF.

Multiple inputs form one video in input order. The CLI applies its options
to every clip; JSON jobs and the Python API allow different options for
each clip.

```bash
uv run pic2panning render coast.jpg forest.jpg \
  --output trip.mp4 --motion zoom-in \
  --audio music.m4a --audio-start 12 --loop-audio \
  --cover "A weekend away"
```

## Video and sound

Use `--video` for video inputs. Speed changes affect the source video and
its retained audio. A title card keeps its own duration. Original sound
stays unless a replacement soundtrack is supplied.

Video panning bounces at the source edges. `--pan-speed` sets the movement
in pixels per output frame and defaults to `5`.

```bash
uv run pic2panning render walk.mp4 \
  --video --speed 2 --output walk-fast.mp4

uv run pic2panning render wide.mp4 \
  --video --motion panning-lr --output wide-portrait.mp4

uv run pic2panning render walk.mp4 \
  --video --motion zoom-in --zoom 2 --output walk-zoom.mp4

uv run pic2panning render silent.mp4 \
  --video --audio music.m4a --output scored.mp4

uv run pic2panning inspect scored.mp4
```

A global soundtrack spans the complete timeline, including title cards.
Per-clip audio belongs to that clip. Choose either global or per-clip
replacement audio for a job. Short tracks end in silence
unless looping is enabled.

## Edit a picture

Ordinary image edits need no model:

```bash
uv run pic2panning edit coast.jpg \
  --output coast-edited.png \
  --crop 80 0 1600 1000 \
  --resize 1280 720 \
  --brightness 1.1 --contrast 1.05 --saturation 0.9
```

Crop coordinates are `left top right bottom` in source pixels. JSON jobs
make the operation order explicit. The CLI applies crop, resize, rotation,
flip, then adjustments. Rotation uses counterclockwise degrees and expands
the canvas; flip accepts `horizontal` or `vertical`. Adjustment factors use
`1` for no change.

For a model edit, describe the desired complete image and provide a
reference:

```bash
uv run --extra models pic2panning generate \
  --source coast.jpg \
  --output coast-evening.png \
  --prompt "The same rocky coast at dusk, warm light on the cliffs, calm sea, photograph" \
  --model flux3 --quantization 4 --size 512 512 --seed 7
```

FLUX 3 Image Klein uses a 5.6B transformer and a Qwen3-VL-4B text encoder.
Its MLX implementation supports 4, 6, and 8-bit weights. The default is
4-bit. Width and height must be multiples of 16. Weights may require
Hugging Face authentication and acceptance on the model page. See
[local models](docs/models.md) for model selection and small test jobs.

Make a text-guided mask, then fill the selected region:

```bash
uv run --extra models pic2panning mask coast-edited.png \
  --prompt person --output person-mask.png

uv run --extra models pic2panning generate \
  --model flux1-fill --source coast-edited.png --mask person-mask.png \
  --output coast-clear.png \
  --prompt "An empty rocky coast, matching the surrounding cliffs and sea" \
  --quantization 4 --size 1280 720 --steps 28
```

Masks use white for the region to edit and black for the region to retain.
For a masked edit, source, mask, and output dimensions must match. The
`coast-edited.png` above comes from the earlier 1280 by 720 image edit.
Masks and masked edits save as lossless PNG to retain exact pixel values.
Other image outputs support PNG, JPEG, and WebP.

## Python library

```python
from pic2panning import render
from pic2panning.schemas.base import Size
from pic2panning.schemas.media import (
    ImageClip,
    Motion,
    RenderRequest,
)

result = render(
    RenderRequest(
        clips=(
            ImageClip(
                source="coast.jpg",
                duration=5,
                motion=Motion.PAN_LR,
            ),
            ImageClip(
                source="forest.jpg",
                duration=3,
                motion=Motion.ZOOM_IN,
            ),
        ),
        output="trip.mp4",
        size=Size(width=1080, height=1920),
        fps=30,
    )
)
```

The public functions are `render`, `edit`, `generate`, `mask`, and
`run_job`. Requests validate before processing. Results name the output
files and include dimensions and timing when applicable. Input files are
protected, and existing outputs require `overwrite=True` or `--overwrite`.

## Internal LLMs and JSON jobs

Point an agent at [llms.txt](llms.txt) and the
[tool guide](docs/agent-tools.md). It can inspect the generated schema,
validate a job, and execute that same job:

```bash
uv run pic2panning schema > job.schema.json
uv run pic2panning validate examples/render.json
uv run pic2panning run examples/render.json
```

The schema comes from the runtime request models. A job selects an
operation with `kind`: `render`, `edit`, `generate`, or `mask`. Unknown
fields fail validation. `validate` checks the request without downloading
assets, loading models, or producing media. CLI results are JSON on stdout;
failures return a nonzero status.

Examples: [render](examples/render.json), [image edits](examples/edit.json),
[generation](examples/generate.json), and [segmentation](examples/mask.json).
Paths in a job resolve from the process working directory.

## Demos

Every demo uses the same CLI and processing functions as the library.
Pass your own inputs; demos have no personal file paths or default music.

```bash
uv run python -m pic2panning.demos.panning coast.jpg --output pan.mp4
uv run python -m pic2panning.demos.zoom coast.jpg --output zoom.mp4
uv run python -m pic2panning.demos.panning_video wide.mp4 --output pan-video.mp4
uv run python -m pic2panning.demos.speed walk.mp4 --output fast.mp4
uv run python -m pic2panning.demos.add_audio_to_video walk.mp4 --audio music.m4a --output music.mp4
uv run --extra models python -m pic2panning.demos.img2img --source coast.jpg --prompt "A rocky coast at dusk, photograph" --output dusk.png
uv run --extra models python -m pic2panning.demos.inpaint --source coast-edited.png --mask person-mask.png --prompt "An empty rocky coast" --size 1280 720 --steps 28 --output clear.png
uv run --extra models python -m pic2panning.demos.sam coast.jpg --prompt person --output mask.png
```

## Development

```bash
uv sync --frozen
uv run pre-commit install
uv run pre-commit run --all-files
uv run mypy
uv run pytest
uv build
```

Ruff checks style and correctness with 79-character formatting. `ty` checks
types at commit time; strict mypy also runs in CI. Repository hooks enforce
the shared Pydantic base, declared instance attributes, shaped array
annotations, root-level tests, and source size limits. Tests exercise image
content and real FFmpeg outputs. Model tests use explicit model boundaries
so core checks run without downloading weights.

MIT licensed. See [LICENSE](LICENSE).
