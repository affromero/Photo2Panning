# Tool guide

Photo2Panning accepts typed Python requests and equivalent JSON jobs. Both
routes call the same processing functions. Prefer JSON jobs for LLM tool
calls because field names, allowed values, and validation errors come from
the runtime schemas.

## Select the operation

| Request `kind` | Use it for | Python function |
| --- | --- | --- |
| `render` | Photo pan or zoom, video reframing, speed, music, title cards | `render` |
| `edit` | Crop, resize, rotate, flip, brightness, contrast, saturation | `edit` |
| `generate` | Prompted generation, reference-image editing, masked fill | `generate` |
| `mask` | Find objects from text and produce segmentation masks | `mask` |

Inspect the current contract before constructing a job:

```bash
uv run pic2panning schema
```

Each request rejects unknown fields. Numeric values must be finite. Use
enum values exactly as the schema lists them. Models and media encoders
load during execution, so schema discovery works with the core install.

## Construct a job

Render requests contain `clips` and an `output`. Each clip declares its
`kind` as `image` or `video`. Images take a duration and motion. Videos take
a playback speed and optional panning motion. Width comes before height in
every size. Choose portrait output with `{"width": 1080, "height": 1920}`.

```json
{
  "kind": "render",
  "clips": [
    {
      "kind": "image",
      "source": "coast.jpg",
      "duration": 5,
      "motion": "panning-lr"
    }
  ],
  "output": "coast.mp4",
  "size": {"width": 1080, "height": 1920},
  "fps": 30
}
```

For different motion, timing, or audio per source, put those options on
each clip. `reverse` adds a return trip after the forward motion.
`focus_center` changes the motion curve. Image and video `zoom` set the
zoom factor, defaulting to `4/3`. A video clip's `pan_speed` is pixels per
output frame; the camera bounces when it reaches an edge. CLI `--duration`
applies to images and `--pan-speed` applies to videos.

Edit requests apply `operations` in order. Every operation selects itself
with `op`. Use a crop before resize when crop coordinates refer to source
pixels. `adjust` contains brightness, contrast, and saturation factors.
Unspecified adjustment factors remain at `1`.

```json
{
  "kind": "edit",
  "source": "coast.jpg",
  "output": "coast-edited.png",
  "operations": [
    {"op": "crop", "left": 80, "top": 0, "right": 1600, "bottom": 1000},
    {"op": "resize", "size": {"width": 1280, "height": 720}},
    {"op": "adjust", "brightness": 1.1, "saturation": 0.9}
  ]
}
```

Generation and masking need the `models` extra. See
[local models](models.md) before selecting a model or changing model
controls. An edit prompt should describe the complete target scene. Use
`edit` for geometric changes that need exact pixel behavior.

Generation uses quantized MLX models on Metal. Mask requests use the small
CPU models; the only supported `device` value is `cpu`.

For generation, omit `steps` to choose 16 for FLUX 3, 4 for Schnell, or 28
for Dev and Fill. Default guidance is 3.5 for Dev and 30 for Fill. FLUX 3
and Schnell reject guidance controls. Explicit steps override the default.

Image outputs support PNG, JPEG, and WebP. Masks and masked edits require
PNG so their pixel values remain lossless. Unsupported output suffixes
fail request validation before model weights load.

## Validate, then execute

```bash
uv run pic2panning validate job.json
uv run pic2panning run job.json
```

Validation checks the structure and option relationships without reading
source media or downloading weights. Execution checks that sources can be
read and that output paths are safe. A valid request can still fail during
execution if a file is missing, a model is unavailable, or FFmpeg fails.

Inputs may be local files or supported HTTP URLs. Relative paths resolve
from the current working directory. Use a new output path. Existing output
files require explicit `overwrite: true`; input files stay protected even
when overwrite is enabled.

CLI success writes a JSON result to stdout. Read `outputs` for the actual
files produced, including separate segmentation masks. `width`, `height`,
`frames`, and `duration` describe the produced media when applicable.
Use the exit status to distinguish failures; do not infer success from the
existence of a previously created file.

## Embed the library

```python
from pic2panning import run_job
from pic2panning.schemas.jobs import JOB_ADAPTER

request = JOB_ADAPTER.validate_json(job_json)
result = run_job(request)
```

Keep the validated request with your tool invocation record. The schema
and Python request classes are the source of truth for available options.
Call public APIs directly; avoid spawning the CLI from an application.

## Audio and timing

A global `audio` track replaces sound across the complete output timeline.
It includes title cards. A clip's own `audio` replaces sound for that clip.
Use one replacement policy per job. A video retains its original sound by
default; `keep_audio: false` silences that clip.

`start` selects an offset in seconds. `loop: true` repeats the soundtrack
to cover its timeline. Without looping, unused time is silent. Video speed
changes preserve audio pitch. Cover duration stays independent of speed.
Frame counts determine the encoded duration at the requested frame rate.
