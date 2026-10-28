"""Physical glyph geometry, native boundaries, and bounded worker ownership."""

import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest
from PIL import ImageChops

from pic2panning.backends.solid.session import SolidSession
from pic2panning.errors import MediaError
from pic2panning.schemas.base import Size
from pic2panning.schemas.creative.composition import LayerKeyframe, TextLayer
from pic2panning.schemas.creative.surface import (
    LetterDrop,
    SolidText,
    SurfaceAttachment,
)
from pic2panning.schemas.creative.typography import TextCue
from pic2panning.tools.creative.depth import DepthAsset
from pic2panning.tools.typography.contours import contour_area
from pic2panning.tools.typography.drop import SurfaceContext
from pic2panning.tools.typography.morph import TextAnimator
from pic2panning.tools.typography.solid import (
    foreground_mesh,
    solid_frame,
    solid_scene,
)
from pic2panning.tools.typography.surface import dot, fit_surface


def layer(*, color: str = "white") -> TextLayer:
    """Place a small real solid on the calibrated fixture surface."""
    return TextLayer(
        id="solid",
        text="ROOM",
        color=color,
        font_size=20,
        size=Size(width=80, height=24),
        duration=7,
        keyframes=(LayerKeyframe(x=80, y=60),),
        surface=SurfaceAttachment(layer_id="table", region=(40, 40, 120, 80)),
        solid=SolidText(samples=1),
    )


def context_for(text: TextLayer, table: DepthAsset) -> SurfaceContext:
    """Own no model resources while sharing a once-fitted analytic plane."""
    if text.surface is None:
        raise ValueError("Test solid has no source attachment")
    return SurfaceContext(
        asset=table,
        attachment=text.surface,
        plane=fit_surface(table, text.surface),
    )


def fake_blender(directory: Path, behavior: str = "valid") -> str:
    """Model only the external executable/socket/PNG boundary."""
    executable = directory / "native-boundary"
    program = dedent(
        f"""\
        #!{sys.executable}
        import json
        import os
        import socket
        import sys
        from pathlib import Path
        from PIL import Image

        if '--version' in sys.argv:
            print('Blender 4.2.1 boundary fixture')
            sys.exit(0)
        scene_path, endpoint = sys.argv[sys.argv.index('--')+1:]
        Path({str(directory / "pid")!r}).write_text(str(os.getpid()))
        config = json.loads(Path(scene_path).read_text())
        behavior = {behavior!r}
        if behavior == 'startup_exit':
            sys.exit(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(endpoint)
            with connection.makefile('rwb') as stream:
                stream.write(b'{{"status":"ready"}}'+bytes([10]))
                stream.flush()
                while payload := stream.readline():
                    command = json.loads(payload)
                    if command.get('close'):
                        sys.exit(0)
                    if behavior == 'render_exit':
                        sys.exit(1)
                    if behavior == 'bad_reply':
                        stream.write(b'not-json'+bytes([10]))
                        stream.flush()
                        continue
                    size = config['size']['width'],config['size']['height']
                    if behavior == 'wrong_size':
                        size = size[0]+1,size[1]
                    mode = 'RGB' if behavior == 'wrong_mode' else 'RGBA'
                    red = config['color'][0] > config['color'][2]
                    color = (255,0,0) if red else (0,0,255)
                    if mode == 'RGBA':
                        color += (255,)
                    with Image.new(mode,size,color) as image:
                        image.save(Path(scene_path).parent/command['output'])
                    stream.write(b'{{"status":"rendered"}}'+bytes([10]))
                    stream.flush()
        """
    )
    executable.write_text(program)
    executable.chmod(0o755)
    return str(executable)


def test_solid_scene_preserves_calibration_at_requested_render_density(
    table: DepthAsset,
) -> None:
    """A lower density still projects the same source rays and real thickness."""
    text = layer()
    context = context_for(text, table)
    scene = solid_scene(text, table, context.plane, Size(width=80, height=60))
    assert scene.intrinsics[0] == pytest.approx((80, 0, 40))
    assert scene.intrinsics[1] == pytest.approx((0, 80, 30))
    assert scene.intrinsics[0][0] * 0.5 / 5 + scene.intrinsics[0][2] == 48
    assert scene.intrinsics[1][1] * 0.25 / 5 + scene.intrinsics[1][2] == 34
    assert scene.font_size * scene.thickness == 4
    assert scene.lights[0].position == pytest.approx((0.22, -0.18, 4.7))


def test_every_morph_phase_has_closed_ink_and_positive_extrusion(
    table: DepthAsset,
) -> None:
    """Holds and R/T,O/I,O/M,M/E morphs all retain actual metric volume."""
    text = layer().model_copy(
        update={
            "text_cues": (TextCue(at=3, text="TIME", transition=1, blur=0),)
        }
    )
    context = context_for(text, table)
    animator = TextAnimator(text)
    try:
        for at in (0, 2, 3.1, 3.25, 3.5, 3.75, 3.9, 4, 6):
            frame = solid_frame(text, context, animator, at, text.keyframes[0])
            assert frame.glyphs
            for glyph in frame.glyphs:
                assert sum(contour_area(loop) for loop in glyph.contours) < 0
                depth_axis = tuple(glyph.transform[row][2] for row in range(3))
                physical_depth = (
                    abs(
                        dot(
                            context.plane.normal,
                            (depth_axis[0], depth_axis[1], depth_axis[2]),
                        )
                    )
                    * 4
                )
                assert physical_depth == pytest.approx(0.125)
    finally:
        animator.close()


def test_letter_drop_moves_complete_solids_before_exact_rest(
    table: DepthAsset,
) -> None:
    """Per-glyph orientation and camera-Z displacement settle to the same body."""
    text = layer().model_copy(update={"letter_drop": LetterDrop(height=1)})
    context = context_for(text, table)
    animator = TextAnimator(text)
    try:
        empty = solid_frame(text, context, animator, 0, text.keyframes[0])
        moving = solid_frame(text, context, animator, 1.5, text.keyframes[0])
        settled = solid_frame(text, context, animator, 2, text.keyframes[0])
        hold = solid_frame(text, context, animator, 5, text.keyframes[0])
        assert not empty.glyphs
        assert (
            moving.glyphs[-1].transform[2][3]
            < settled.glyphs[-1].transform[2][3] - 0.1
        )
        assert (
            moving.glyphs[-1].transform[0][:3]
            != settled.glyphs[-1].transform[0][:3]
        )
        assert settled == hold
    finally:
        animator.close()


def test_floating_text_keeps_its_shadow_receiver_on_the_original_table(
    table: DepthAsset,
) -> None:
    """A text lift raises real geometry while preserving the photographed floor."""
    text = layer().model_copy(
        update={
            "surface": SurfaceAttachment(
                layer_id="table", region=(40, 40, 120, 80), lift_metres=0.3
            )
        }
    )
    context = context_for(text, table)
    animator = TextAnimator(text)
    try:
        frame = solid_frame(text, context, animator, 3, text.keyframes[0])
        for point in frame.catcher:
            assert dot(context.plane.normal, point) == pytest.approx(
                dot(context.plane.normal, context.plane.center)
            )
        assert frame.glyphs[0].transform[2][3] < 4.71
    finally:
        animator.close()


def test_depth_holdouts_retain_foreground_at_the_last_source_pixel(
    table: DepthAsset,
) -> None:
    """Mesh generation preserves corner objects and excludes the fitted floor."""
    text = layer()
    plane = context_for(text, table).plane
    table.depth.paste(3, (150, 110, 160, 120))
    vertices, triangles = foreground_mesh(
        table, plane, Size(width=160, height=120)
    )
    used = {index for face in triangles for index in face}
    assert vertices[-1] == pytest.approx(
        ((159.5 - 80) / 160 * 3, (119.5 - 60) / 160 * 3, 3)
    )
    assert len(vertices) - 1 in used
    assert all(vertices[index][2] == 3 for index in used)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_whole_sentence_rotation_preserves_rigid_distances_between_glyphs(
    table: DepthAsset,
    axis: str,
) -> None:
    """One floor clearance preserves the sentence's full rotated 3D geometry."""
    text = layer().model_copy(
        update={"text": "ROOM\nTIME", "size": Size(width=80, height=60)}
    )
    context = context_for(text, table)
    animator = TextAnimator(text)
    try:
        flat = solid_frame(text, context, animator, 3, text.keyframes[0])
        rotated = solid_frame(
            text,
            context,
            animator,
            3,
            LayerKeyframe(
                x=80,
                y=60,
                rotation_x=35 if axis == "x" else 0,
                rotation_y=35 if axis == "y" else 0,
            ),
        )
        for first, second in zip(flat.glyphs, rotated.glyphs, strict=True):
            flat_delta = tuple(
                first.transform[row][3] - flat.glyphs[0].transform[row][3]
                for row in range(3)
            )
            rotated_delta = tuple(
                second.transform[row][3] - rotated.glyphs[0].transform[row][3]
                for row in range(3)
            )
            assert math.dist(flat_delta, (0, 0, 0)) == pytest.approx(
                math.dist(rotated_delta, (0, 0, 0))
            )
        heights = tuple(
            dot(
                context.plane.normal,
                (
                    glyph.transform[0][3],
                    glyph.transform[1][3],
                    glyph.transform[2][3],
                ),
            )
            for glyph in rotated.glyphs
        )
        assert max(heights) - min(heights) > 0.1
    finally:
        animator.close()


def test_settled_letters_stay_fixed_as_later_letters_arrive_and_after_seeking(
    table: DepthAsset,
) -> None:
    """The full resting sentence determines contact before every glyph arrives."""
    text = layer().model_copy(update={"letter_drop": LetterDrop(height=1)})
    context = context_for(text, table)
    animator = TextAnimator(text)
    pose = LayerKeyframe(x=80, y=60, rotation_y=35)
    try:
        early = solid_frame(text, context, animator, 0.9, pose)
        final = solid_frame(text, context, animator, 2, pose)
        first_rest = tuple(
            value for row in early.glyphs[0].transform for value in row
        )
        final_rest = tuple(
            value for row in final.glyphs[0].transform for value in row
        )
        assert first_rest == pytest.approx(final_rest)
        solid_frame(text, context, animator, 1.8, pose)
        assert solid_frame(text, context, animator, 0.9, pose) == early
        assert solid_frame(text, context, animator, 4, pose) == final
    finally:
        animator.close()


def test_two_workers_share_no_frames_and_return_independent_cached_pixels(
    table: DepthAsset,
    tmp_path: Path,
) -> None:
    """Worker directories, calibrated sizes, colors, and one-frame caches isolate."""
    executable = fake_blender(tmp_path)
    directory = tmp_path / "workers"
    red, blue = layer(color="red"), layer(color="blue")
    context = context_for(red, table)
    animator = TextAnimator(red)
    try:
        descriptor = solid_frame(red, context, animator, 3, red.keyframes[0])
        with ExitStack() as stack:
            first = SolidSession(
                solid_scene(
                    red, table, context.plane, Size(width=80, height=60)
                ),
                str(directory),
                executable,
            )
            stack.callback(first.close)
            second = SolidSession(
                solid_scene(
                    blue, table, context.plane, Size(width=64, height=48)
                ),
                str(directory),
                executable,
            )
            stack.callback(second.close)
            with first.frame(descriptor) as image:
                assert image.size == (80, 60)
                assert image.getpixel((20, 20)) == (255, 0, 0, 255)
                image.paste("green", (0, 0, *image.size))
            with second.frame(descriptor) as image:
                assert image.size == (64, 48)
                assert image.getpixel((20, 20)) == (0, 0, 255, 255)
            with first.frame(descriptor) as image:
                assert image.getpixel((20, 20)) == (255, 0, 0, 255)
        assert not tuple(directory.iterdir())
    finally:
        animator.close()


@pytest.mark.parametrize(
    "behavior", ["bad_reply", "wrong_size", "wrong_mode", "render_exit"]
)
def test_failed_native_frames_reap_workers_and_remove_scoped_artifacts(
    table: DepthAsset,
    tmp_path: Path,
    behavior: str,
) -> None:
    """Broken native replies and images never become accepted flat output."""
    executable = fake_blender(tmp_path, behavior)
    text = layer()
    context = context_for(text, table)
    directory = tmp_path / "workers"
    session = SolidSession(
        solid_scene(text, table, context.plane, Size(width=80, height=60)),
        str(directory),
        executable,
    )
    animator = TextAnimator(text)
    try:
        with pytest.raises(MediaError):
            session.frame(
                solid_frame(text, context, animator, 3, text.keyframes[0])
            )
        assert not tuple(directory.iterdir())
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)
    finally:
        session.close()
        animator.close()


def test_failed_native_startup_reaps_process_and_cleans_both_directories(
    table: DepthAsset,
    tmp_path: Path,
) -> None:
    """A worker exiting before readiness leaves no process or frame artifacts."""
    executable = fake_blender(tmp_path, "startup_exit")
    text = layer()
    context = context_for(text, table)
    directory = tmp_path / "workers"
    with pytest.raises(MediaError, match="startup failed"):
        SolidSession(
            solid_scene(text, table, context.plane, Size(width=80, height=60)),
            str(directory),
            executable,
        )
    assert not tuple(directory.iterdir())
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


@pytest.mark.skipif(
    sys.platform != "darwin"
    or not Path("/Applications/Blender.app/Contents/MacOS/Blender").is_file(),
    reason="Native Cycles Metal integration requires Blender on macOS",
)
def test_native_metal_workers_project_real_solids_and_isolate_render_grids(
    table: DepthAsset,
    tmp_path: Path,
) -> None:
    """Real Blender validates row-major placement, actual shading, and two workers."""
    red, blue = layer(color="red"), layer(color="blue")
    context = context_for(red, table)
    animator = TextAnimator(red)
    try:
        descriptor = solid_frame(red, context, animator, 3, red.keyframes[0])
        with ExitStack() as stack:
            first = SolidSession(
                solid_scene(
                    red, table, context.plane, Size(width=80, height=60)
                ),
                str(tmp_path),
                None,
            )
            stack.callback(first.close)
            second = SolidSession(
                solid_scene(
                    blue, table, context.plane, Size(width=64, height=48)
                ),
                str(tmp_path),
                None,
            )
            stack.callback(second.close)
            with (
                first.frame(descriptor) as first_image,
                second.frame(descriptor) as second_image,
            ):
                assert first_image.size == (80, 60)
                assert second_image.size == (64, 48)
                first_pixels = [
                    cast(
                        "tuple[int, int, int, int]",
                        first_image.getpixel((x, y)),
                    )
                    for y in range(60)
                    for x in range(80)
                ]
                second_pixels = [
                    cast(
                        "tuple[int, int, int, int]",
                        second_image.getpixel((x, y)),
                    )
                    for y in range(48)
                    for x in range(64)
                ]
                opaque_ink = tuple(
                    index % 80
                    for index, pixel in enumerate(first_pixels)
                    if pixel[3] >= 240 and pixel[0] > pixel[2] + 20
                )
                assert opaque_ink
                assert 0 < min(opaque_ink) < max(opaque_ink) < 79
                assert (
                    sum(
                        pixel[0] - pixel[2]
                        for pixel in first_pixels
                        if pixel[3]
                    )
                    > 0
                )
                assert (
                    sum(
                        pixel[2] - pixel[0]
                        for pixel in second_pixels
                        if pixel[3]
                    )
                    > 0
                )
                with (
                    first.frame(descriptor) as cached,
                    ImageChops.difference(first_image, cached) as difference,
                ):
                    assert difference.getbbox(alpha_only=False) is None
        assert not tuple(tmp_path.iterdir())
    finally:
        animator.close()
