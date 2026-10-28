# Perspective text and letter animation

Text can follow a photographed surface, fall into place one letter at a
time, and change into another sentence. The same composition renders in
Studio, Python, and the CLI. Cached depth and typography run with the core
installation.

## Letters on a photographed table

[![Letters fall onto a table, hold, then morph into another sentence](../examples/media/table-letters.gif)](../examples/media/table-letters.mp4)

```bash
uv run pic2panning run examples/compose-table.json
uv run pic2panning run examples/preview-table.json
```

The seven-second example drops **MAKE ROOM** onto a table, holds the
sentence, then reshapes it into **MAKE TIME**. Letters begin with different
start times and rotate around X, Y, and Z before settling. The text uses
the table's inferred perspective. The cup and notebook can hide lettering
when they are nearer to the camera.

The photograph sits in a warm poster frame with a separate bottom safe
area. Its aspect ratio stays unchanged, and the landed caption remains
above the Story reply controls.

The [composition](../examples/compose-table.json) references the bundled
[photo](../examples/media/table.jpg) and its real
[depth manifest](../examples/media/table-depth/table.depth.json). Repeating
the render needs no model download. To extract a replacement asset:

```bash
uv run --extra camera pic2panning run examples/table-depth.json
```

This extraction uses the explicit MPS device, 1,800 tokens, and a
768-pixel maximum edge. On a 36 GiB M3 Max, it produced a 513×768 grid from
the 1280×1917 photo in 30.77 seconds, with a 2.87 GiB peak process RSS.

## Attach text to depth

Set `TextLayer.surface` to a `SurfaceAttachment`. Its `layer_id` identifies
an earlier image layer with matching cached depth. `region` selects a
planar patch in the oriented source image, using exclusive right and
bottom edges. Choose the table itself and leave cups, books, and edges
outside the fitting patch.

The renderer unprojects the sampled depth with its calibrated intrinsics,
fits a plane, and projects the text back into the photograph. A rejected
fit raises an error. It does not substitute a flat screen overlay.

Surface-text keyframe `x` and `y` use the host image's source pixels.
Their camera rays intersect the fitted plane. `scale` changes physical
text width, and `rotation` rotates it within that plane. The host image
supplies its placement, camera movement, mask, opacity, and depth warp.

| Surface control | Meaning |
| --- | --- |
| `tilt`, `yaw`, `roll` | Rotation in degrees around the local surface axes |
| `offset_metres` | Additional displacement along the plane's right and down axes |
| `lift_metres` | Move toward the camera along the surface normal |
| `depth_tolerance_metres` | Depth separation tolerated before nearer pixels hide the text |
| `feather_metres` | Depth interval used to soften an occlusion edge |
| `max_relative_error` | Largest accepted plane-fit RMS relative to sampled depth |

Surface text uses world space and automatic depth visibility. Its `surface`
field replaces a separate `occlusion` declaration. A single photo provides
estimated geometry; a curved or cluttered region may not fit one plane.

## Drop letters through 3D

Add `letter_drop` to a text layer. The first sentence falls during the
first `duration` seconds of that layer's local timeline. `stagger` controls
the fraction of that interval used to start the glyphs in reading order.
The remaining interval lets every glyph finish its fall.

`height` uses font heights above the resting plane. `rotation` supplies
initial X/Y/Z angles that settle to the layer's declared pose. `bounce`
adds a bounded rebound. A surface attachment places this movement in the
depth camera's metric coordinates. Free text can use the effect in its
own perspective view.

Leave time after landing so the sentence can be read. Sentence transitions
must begin after the complete letter drop; overlapping intervals fail
validation.

## Morph between sentences

[![Perspective headlines morph from the coast to the light and the long way home](../examples/media/social-story.gif)](../examples/media/social-story.mp4)

`TextLayer.text` contains the first sentence. Ordered `text_cues` describe
later sentences, with `at` measured from the layer's start:

```json
{
  "text": "CHASE THE\nCOAST",
  "text_cues": [
    {
      "at": 3,
      "text": "CHASE THE\nLIGHT",
      "transition": 1,
      "stagger": 0.25,
      "blur": 1.5,
      "travel": 0.55
    }
  ]
}
```

Matching glyphs move to their new positions. Changed glyphs reshape their
ink contours while traveling along curved paths. `stagger` spreads their
starts within the declared transition interval. `blur` controls temporary
softening in text pixels; `travel` uses font sizes. At the transition's
end, the target sentence is rendered exactly and holds until another cue.

The [coast-to-aurora composition](../examples/compose.json) changes its
headline at 3 and 5.5 seconds, both on its supplied 120 BPM markers. It
holds each completed sentence and uses an original instrumental track.
Its MP4 retains the soundtrack; GIF previews are silent.

## Free perspective text

Layer keyframes expose `rotation_x` and `rotation_y`, with `rotation` for
clockwise Z rotation. A positive `perspective` distance controls the
projection. Use a surface attachment when the angle must come from the
photographed table or another measured patch.

## Edit the animation in Studio

Select a text layer and open **Sentence morphs** to set each target
sentence, start, and transition. **Letter entrance** enables the 3D fall
and sets its initial X/Y/Z angles. **Attach to image surface**
selects the depth-enabled host and a flat patch in its source pixels.
The layer's **3D orientation** controls free perspective rotation.

Use the playhead to check a landing or transition, then export a draft.
Preview frames use the same renderer as the final movie. Reusing the
bundled depth does not start model inference.

## Source credits

The tabletop photo is by
[Pavel Danilyuk on Pexels](https://www.pexels.com/photo/coffee-cup-on-wooden-table-6763887/),
used under the [Pexels license](https://www.pexels.com/license/).
The examples bundle [Anton](https://github.com/googlefonts/AntonFont),
with its [SIL Open Font License](../examples/media/fonts/OFL.txt).
The font is selected explicitly in these jobs; the library's default font
is unchanged.
