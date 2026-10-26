# Tool guide

Photo2Panning accepts typed Python requests and equivalent JSON jobs. Both
routes call the same processing functions. Prefer JSON jobs for LLM tool
calls because field names, allowed values, and validation errors come from
the runtime schemas.

## Select the operation

| Request `kind` | Use it for | Python function |
| --- | --- | --- |
| `render` | Photo pan or zoom, video reframing, speed, music, title cards | `render` |
| `scene` | Reconstruct a photo into a reusable calibrated Gaussian scene | `reconstruct_scene` |
| `camera` | Pan, pause for a 3D camera excursion, return, and resume | `render_camera` |
| `edit` | Crop, resize, rotate, flip, brightness, contrast, saturation | `edit` |
| `generate` | Prompted generation, reference-image editing, masked fill | `generate` |
| `generate-video` | One local shot guided by first and last image inputs | `generate_video` |
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
  "fps": 30,
  "encoding": {"crf": 16, "preset": "slow"}
}
```

For different motion, timing, or audio per source, put those options on
each clip. `reverse` adds a return trip after the forward motion.
`focus_center` changes the motion curve. Image and video `zoom` set the
zoom factor, defaulting to `4/3`. A video clip's `pan_speed` is pixels per
output frame; the camera bounces when it reaches an edge. CLI `--duration`
applies to images and `--pan-speed` applies to videos.

Image clips accept `shutter_angle` from 0 to 360 degrees. The default `0`
samples one instant per frame. A value of `180` integrates eight subframe
samples across half a frame interval, in linear light. This softens detail
along the motion direction to reduce strobing during fast pans, with extra render
cost. The first and last frames remain sharp at their exact endpoints.
Exposure samples follow the clip's `focus_center` curve when enabled.

The CLI exposes this as `--shutter-angle` and rejects nonzero values with
`--video`. Frame rate and duration remain explicit controls: increasing
`fps` or slowing the pan with a longer `duration` reduces movement between
frames. Appearance depends on scene detail, pan speed, and playback display.
For a slower photo pan, choose `duration: 7.5`, `fps: 60`, and
`shutter_angle: 180` explicitly.

`encoding` selects H.264 quality. `crf` ranges from 0 to 51; lower values
retain more detail and produce larger files. Its default is 18. `preset`
accepts `veryfast`, `fast`, `medium`, `slow`, or `veryslow` and defaults to
`veryfast`. The CLI exposes these fields as `--crf` and `--preset`.

MP4 dimensions must be even. GIF output requires an explicit `fps` from
`1`, `2`, `4`, `5`, `10`, `20`, `25`, `50`, or `100`, so frame delays fit
GIF's centisecond clock. GIFs cannot contain replacement audio. Source
video rotation and pixel aspect ratio are applied before reframing.

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

Image generation and masking need the `models` extra. Video generation
needs the separate `video` extra. See
[local models](models.md) before selecting a model or changing model
controls. An edit prompt should describe the complete target scene. Use
`edit` for geometric changes that need exact pixel behavior.

Generation uses quantized MLX models on Metal. Mask requests use the small
CPU models; the only supported `device` value is `cpu`.

For image generation, omit `steps` to choose 16 for FLUX 3, 4 for Schnell,
or 28 for Dev and Fill. Default guidance is 3.5 for Dev and 30 for Fill. FLUX 3
and Schnell reject guidance controls. Explicit steps override the default.

Image outputs support PNG, JPEG, and WebP. Masks and masked edits require
PNG so their pixel values remain lossless. Unsupported output suffixes
fail request validation before model weights load.

### Reconstruct a scene, then move the camera

The `camera` extra adds local geometry prediction and Gaussian rendering.
Create a scene once and reuse its PLY and calibration for camera variants:

```bash
uv sync --frozen --extra camera
uv run --extra camera pic2panning run examples/scene.json
uv run --extra camera pic2panning run examples/camera.json
```

The equivalent reconstruction command is:

```bash
uv run --extra camera pic2panning scene examples/media/waterfall.jpg \
  --output waterfall.ply --metadata waterfall-scene.json \
  --device mps --num-tokens 1800
```

A `scene` request requires `source`, a PLY `output`, and a JSON `metadata`
destination. It uses the pinned `moge-2-vitl-normal` geometry predictor.
`device` defaults to `mps`; `cpu` is an explicit alternative. A device
failure stops execution. `num_tokens` ranges from 1200 to 3600 and defaults
to 1800. Larger budgets increase inference detail, memory use, and time.
`max_edge` bounds the geometry image's longest edge, from 256 to 4096
pixels, defaulting to 1920. Larger source images receive an aspect-preserving
resize for reconstruction. The original photo remains unchanged for panning,
and calibration intrinsics still refer to its original oriented dimensions.
The first run downloads the checkpoint into the normal model cache.

`SceneResult` reports `output`, `metadata`, the positive Gaussian `points`
count, `source_sha256`, and `predictor`. The source hash identifies the
original file bytes. Keep the original image with its generated artifacts.
The metadata records the oriented source dimensions, pixel intrinsics,
OpenCV world-to-camera transform, color encoding, and positive foreground
depth percentile `10`. It also binds the scene file with `scene_sha256`.

A `camera` request requires the same `source`, the PLY `scene`, its JSON
`metadata`, and an MP4 `output`. Execution verifies the source hash and
dimensions before loading scene geometry. It verifies the scene hash when
the sidecar supplies one. Changes to the source require reconstruction.
You can also supply an externally generated degree-zero Gaussian scene
with a matching `SceneCalibration` sidecar.

`pan_start`, `pause_at`, and `pan_end` are horizontal crop positions from
0 at the left edge to 1 at the right. They must strictly increase or
strictly decrease through the pause. The defaults are `0`, `0.5`, and `1`.
`before` defaults to 4 seconds, `duration` to 5 seconds for the whole 3D
excursion and return, and `after` to 3.2 seconds. `hold` adds 0.25 seconds
on each side of the excursion; `0` omits both holds.

Each phase rounds its duration up to a whole output frame. Both pan phases
need at least two frames, and the excursion needs at least three. Total
frames equal `ceil(before * fps) + ceil(duration * fps)` plus
`ceil(after * fps) + 2 * ceil(hold * fps)`. Default timing at 60 fps produces 762 frames,
or 12.7 seconds. The camera returns to the paused source composition before
the pan resumes. Original pixels anchor the handoff frames before encoding;
lossy MP4 decoding can change individual pixel values.

`position` is the far camera offset in model-estimated metres, measured in
the original camera's axes: positive X moves right, positive Y moves down,
and positive Z moves forward. Negative Y raises the camera. Omit it for
the conservative default `(0.02, -0.005, 0.01)` times foreground depth.
The path eases outward and back while retaining its optical aim. A short
linear-light blend handles the source-photo handoff near each endpoint.
Choose small offsets: single-image geometry can expose holes or stretched
surfaces at unseen viewpoints.

Camera output defaults to 1080 by 1920 at 60 fps. Dimensions must be even;
`fps` ranges from 1 to 240. `shutter_angle` defaults to 180 for its photo
pans and accepts 0 to 360. `encoding` has the same CRF and preset controls
as `render`. The complete timeline streams through one encoder and writes
a silent MP4. Add music afterward with the soundtrack example.

See the paired [scene job](../examples/scene.json) and
[camera job](../examples/camera.json), and [local models](models.md) for
the predictor and renderer setup.

### Generate video from first and last frames

`generate-video` uses the local MLX LTX pipeline. Both stages receive a
guide at frame 0 and another at `frames - 1`. Generated endpoints can
differ from the input pixels. Prefer overlapping views of the same scene
for continuous camera movement.

```json
{
  "kind": "generate-video",
  "first_frame": "examples/media/waterfall.jpg",
  "last_frame": "examples/media/mountains.jpg",
  "output": "generated.mp4",
  "prompt": "A continuous camera move across the landscape toward the final view.",
  "size": {"width": 576, "height": 1024},
  "frames": 65,
  "fps": 24,
  "steps": 20,
  "refine_steps": 3,
  "seed": 0,
  "low_memory": true
}
```

Input images receive EXIF orientation and a centered crop to fit. Use
`EditRequest` first for another crop. Width and height must be multiples
of 64. Frame count must be at least 9 and equal `8k + 1`; frame rate must
be between 1 and 60. Duration equals `frames / fps`.

Defaults are 576 by 1024, 65 frames, 24 fps, 20 first-stage steps, 3
refinement steps, seed 0, and `low_memory: true`. Refinement accepts 1 to
3 steps. Transformer block streaming stays enabled independently of
`low_memory`. `negative_prompt` is optional. `weights` and `text_weights`
accept compatible local checkpoint directories; omit them to use the
pinned public defaults. The first default download is about 40 GB.

The output is a silent MP4. Add a soundtrack in a separate `render` job
using that output as a video clip. Generated media must match the
requested dimensions, frame count, frame rate, and duration before it is
published. See [local video setup](models.md#first-and-last-frame-video)
and the [complete job example](../examples/generate-video.json).

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

CLI success writes a JSON result to stdout. Scene reconstruction returns
`SceneResult` with singular `output` and `metadata` paths plus `points` and
source identity. Other jobs return `JobResult`: read `outputs` for the
actual files produced, including separate segmentation masks. `width`,
`height`, `frames`, and `duration` describe produced media when applicable.
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

`start` selects an offset in seconds. `loop: true` repeats the selected
audio from that offset to its end, excluding the skipped prefix from every
cycle. An offset at or beyond the source duration fails during execution.
Without looping, unused time is silent. Video speed changes preserve audio
pitch. Cover duration stays independent of speed. Frame counts determine
the encoded duration at the requested frame rate.
