"""Standalone Blender worker using only its standard library and native API."""

import hashlib
import importlib
import json
import math
import socket
import sys
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import BinaryIO, Protocol, TypedDict, cast


class SizeData(TypedDict):
    """Validated host image dimensions."""

    width: int
    height: int


class PlaneData(TypedDict):
    """Metric fitted axes needed for fixed source lighting."""

    center: tuple[float, float, float]
    normal: tuple[float, float, float]


class LightData(TypedDict):
    """World-space area emitter sent by the validated host."""

    position: tuple[float, float, float]
    energy: float
    size: float
    color: tuple[float, float, float]


class SceneData(TypedDict):
    """JSON boundary for static source geometry and physical material."""

    version: int
    size: SizeData
    intrinsics: tuple[tuple[float, float, float], ...]
    plane: PlaneData
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    lights: tuple[LightData, ...]
    color: tuple[float, float, float]
    font_size: float
    thickness: float
    bevel: float
    roughness: float
    metallic: float
    ambient: float
    samples: int


class GlyphData(TypedDict):
    """One closed glyph body and its metric transform."""

    contours: tuple[tuple[tuple[float, float], ...], ...]
    transform: tuple[tuple[float, float, float, float], ...]


class FrameData(TypedDict):
    """Only the current solids and finite shadow-catching patch."""

    glyphs: tuple[GlyphData, ...]
    catcher: tuple[tuple[float, float, float], ...]


class CommandData(TypedDict, total=False):
    """A render request or orderly worker shutdown."""

    close: bool
    frame: FrameData
    output: str


class InputSocket(Protocol):
    """One explicitly assigned native shader input."""

    default_value: float | tuple[float, ...]


class InputCollection(Protocol):
    """Shader socket lookup at the native API boundary."""

    def __getitem__(self, name: str) -> InputSocket:
        """Find the named native input."""
        ...


class ShaderNode(Protocol):
    """The standard material/background shader controls we use."""

    inputs: InputCollection


class NodeCollection(Protocol):
    """Named shader lookup independent of dynamic Blender classes."""

    def get(self, name: str) -> ShaderNode | None:
        """Find an existing standard shader node."""
        ...


class NodeTree(Protocol):
    """Minimal native node tree boundary."""

    nodes: NodeCollection


class Material(Protocol):
    """One physical surface material."""

    use_nodes: bool
    node_tree: NodeTree


class MaterialCollection(Protocol):
    """Material factory supplied by Blender."""

    def new(self, name: str) -> Material:
        """Allocate an owned physical material."""
        ...


class MaterialSlots(Protocol):
    """Geometry's explicit material assignment."""

    def append(self, material: Material) -> None:
        """Assign the shared material to geometry."""
        ...


class CurvePoint(Protocol):
    """One homogeneous spline control point."""

    co: tuple[float, float, float, float]


class CurvePoints(Protocol):
    """Closed polygon point storage."""

    def add(self, count: int) -> None:
        """Allocate the remaining contour points."""
        ...

    def __getitem__(self, index: int) -> CurvePoint:
        """Access a preallocated contour point."""
        ...


class Spline(Protocol):
    """One simple closed exterior or hole contour."""

    points: CurvePoints
    use_cyclic_u: bool


class SplineCollection(Protocol):
    """Native polygon spline factory."""

    def new(self, kind: str) -> Spline:
        """Create a polygonal closed contour."""
        ...


class Curve(Protocol):
    """Native filled extrusion, converted to a complete mesh before beveling."""

    dimensions: str
    fill_mode: str
    extrude: float
    resolution_u: int
    splines: SplineCollection
    materials: MaterialSlots


class CurveCollection(Protocol):
    """Curve allocation and explicit orphan cleanup."""

    def new(self, name: str, kind: str) -> Curve:
        """Allocate one owned contour body."""
        ...

    def remove(self, curve: Curve) -> None:
        """Release contour storage after mesh conversion."""
        ...


class Polygon(Protocol):
    """Surface shading flag for a mesh face."""

    use_smooth: bool
    vertices: Sequence[int]


class MeshVertex(Protocol):
    """Local geometric coordinates for volume validation."""

    co: tuple[float, float, float]


class Mesh(Protocol):
    """Native indexed mesh for the source holdouts and catcher."""

    materials: MaterialSlots
    polygons: Iterable[Polygon]
    vertices: Sequence[MeshVertex]

    def from_pydata(
        self,
        vertices: Sequence[tuple[float, float, float]],
        edges: Sequence[tuple[int, int]],
        faces: Sequence[tuple[int, ...]],
    ) -> None:
        """Assign calibrated vertices and explicit polygon indices."""
        ...

    def clear_geometry(self) -> None:
        """Release the previous receiver's vertices."""
        ...

    def update(self) -> None:
        """Publish updated mesh geometry."""
        ...


class MeshCollection(Protocol):
    """Mesh allocation and explicit orphan cleanup."""

    def new(self, name: str) -> Mesh:
        """Allocate one owned mesh."""
        ...

    def remove(self, mesh: Mesh) -> None:
        """Release replaced mesh storage."""
        ...


class Modifier(Protocol):
    """Only bevel and weighted-normal settings at the native API boundary."""

    width: float
    segments: int
    use_clamp_overlap: bool
    harden_normals: bool
    limit_method: str
    keep_sharp: bool
    mode: str


class ModifierCollection(Protocol):
    """Native modifier factory."""

    def new(self, name: str, kind: str) -> Modifier:
        """Create one explicitly configured native modifier."""
        ...


class MatrixValue(Protocol):
    """A native row-oriented matrix, rather than an RNA column array."""

    def __getitem__(self, index: int) -> Sequence[float]:
        """Read a native matrix row."""
        ...


class MatrixFactory(Protocol):
    """Explicit row-major construction avoids RNA's transposed tuple coercion."""

    def __call__(self, rows: Sequence[Sequence[float]]) -> MatrixValue:
        """Construct a native transform from validated host matrix rows."""
        ...


class Mathutils(Protocol):
    """Typed native matrix constructor without project dependencies."""

    Matrix: MatrixFactory


class BlenderObject(Protocol):
    """Mesh, light, or camera instance without an untyped API escape."""

    data: object
    location: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]
    matrix_world: MatrixValue
    is_shadow_catcher: bool
    is_holdout: bool
    visible_shadow: bool
    modifiers: ModifierCollection

    def select_set(self, *, state: bool) -> None:
        """Set the conversion operator's exact selected object."""
        ...


class ObjectCollection(Protocol):
    """Explicit object ownership, including removal from its collections."""

    def new(self, name: str, data: object) -> BlenderObject:
        """Create an instance of explicit native data."""
        ...

    def remove(self, obj: BlenderObject, *, do_unlink: bool) -> None:
        """Remove a replaced object from all scene collections."""
        ...

    def link(self, obj: BlenderObject) -> None:
        """Make an owned object visible in the current scene."""
        ...


class Camera(Protocol):
    """Horizontal sensor-fit camera calibrated from source intrinsics."""

    sensor_fit: str
    sensor_width: float
    lens: float
    shift_x: float
    shift_y: float
    clip_start: float
    clip_end: float


class CameraCollection(Protocol):
    """Native camera factory."""

    def new(self, name: str) -> Camera:
        """Allocate the source-calibrated native camera."""
        ...


class Light(Protocol):
    """World-oriented circular area emitter."""

    energy: float
    color: tuple[float, float, float]
    shape: str
    size: float


class LightCollection(Protocol):
    """Native light factory."""

    def new(self, name: str, kind: str) -> Light:
        """Allocate one fixed source area light."""
        ...


class Data(Protocol):
    """The small subset of Blender data factories used by this worker."""

    objects: ObjectCollection
    materials: MaterialCollection
    curves: CurveCollection
    meshes: MeshCollection
    cameras: CameraCollection
    lights: LightCollection


class Device(Protocol):
    """Explicit Cycles backend selection."""

    type: str
    use: bool


class CyclesPreferences(Protocol):
    """Native Metal enumeration with no CPU render fallback."""

    compute_device_type: str
    devices: Iterable[Device]

    def get_devices(self) -> None:
        """Enumerate the explicitly requested Metal devices."""
        ...


class Addon(Protocol):
    """Cycles exposes its native preferences at this package boundary."""

    preferences: object


class AddonCollection(Protocol):
    """Installed Blender add-on lookup."""

    def __getitem__(self, name: str) -> Addon:
        """Access the installed Cycles add-on."""
        ...


class Preferences(Protocol):
    """Blender's loaded add-on preferences."""

    addons: AddonCollection


class ImageSettings(Protocol):
    """Unambiguous transparent PNG serialization."""

    file_format: str
    color_mode: str
    color_depth: str


class RenderSettings(Protocol):
    """Calibrated source-grid output settings."""

    engine: str
    resolution_x: int
    resolution_y: int
    resolution_percentage: int
    film_transparent: bool
    pixel_aspect_y: float
    image_settings: ImageSettings
    filepath: str


class CyclesSettings(Protocol):
    """Explicit samples, Metal device, and bounded light transport."""

    device: str
    samples: int
    use_adaptive_sampling: bool
    use_denoising: bool
    max_bounces: int
    transparent_max_bounces: int


class ViewSettings(Protocol):
    """View transform applies only to the rendered geometry, never the photo."""

    view_transform: str
    look: str


class World(Protocol):
    """Indirect environment illumination."""

    use_nodes: bool
    node_tree: NodeTree


class Scene(Protocol):
    """Native current scene boundary."""

    render: RenderSettings
    cycles: CyclesSettings
    view_settings: ViewSettings
    world: World
    camera: BlenderObject


class ActiveObjects(Protocol):
    """Object conversion has an explicit active mesh context."""

    active: BlenderObject


class ViewLayer(Protocol):
    """The active native object context."""

    objects: ActiveObjects


class Collection(Protocol):
    """Scene object linking."""

    objects: ObjectCollection


class Context(Protocol):
    """Static native context access without importing project dependencies."""

    scene: Scene
    preferences: Preferences
    collection: Collection
    view_layer: ViewLayer


class ObjectOperators(Protocol):
    """Mesh conversion and deterministic selection state."""

    def select_all(self, *, action: str) -> None:
        """Set deterministic scene selection."""
        ...

    def delete(self, *, use_global: bool) -> None:
        """Remove the factory scene objects."""
        ...

    def convert(self, *, target: str) -> None:
        """Convert one whole closed glyph to a mesh."""
        ...

    def mode_set(self, *, mode: str) -> None:
        """Enter the mesh editing context explicitly."""
        ...


class MeshOperators(Protocol):
    """Weld duplicated cap seams before applying a solid bevel."""

    def select_all(self, *, action: str) -> None:
        """Include every cap and side seam in the weld."""
        ...

    def remove_doubles(self, *, threshold: float) -> None:
        """Weld duplicated extrusion seams without changing its outline."""
        ...

    def normals_make_consistent(self, *, inside: bool) -> None:
        """Orient the welded closed surface consistently outward."""
        ...


class TransformOperators(Protocol):
    """Position the symmetric extrusion's bottom exactly on its local plane."""

    def translate(self, *, value: tuple[float, float, float]) -> None:
        """Move the whole local mesh above its receiving plane."""
        ...


class RenderOperators(Protocol):
    """One synchronous native render per requested time sample."""

    def render(self, *, write_still: bool) -> None:
        """Write one explicitly requested transparent frame."""
        ...


class Operators(Protocol):
    """Minimal operator package boundary."""

    object: ObjectOperators
    mesh: MeshOperators
    transform: TransformOperators
    render: RenderOperators


class Bpy(Protocol):
    """Typed entry point for the separately installed Blender application."""

    data: Data
    context: Context
    ops: Operators


def material(
    bpy: Bpy, name: str, config: SceneData, *, catcher: bool = False
) -> Material:
    """Use a real rough ceramic BSDF and warm receiving-surface bounce."""
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        raise RuntimeError("Blender has no Principled BSDF material node")
    color = (0.55, 0.38, 0.22) if catcher else config["color"]
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = (
        0.7 if catcher else config["roughness"]
    )
    shader.inputs["Metallic"].default_value = (
        0 if catcher else config["metallic"]
    )
    shader.inputs["Specular IOR Level"].default_value = 0.36
    return result


def camera(bpy: Bpy, config: SceneData) -> None:
    """Use exact source intrinsics, including an off-center principal point."""
    data = bpy.data.cameras.new("Source calibrated camera")
    obj = bpy.data.objects.new("Source calibrated camera", data)
    bpy.context.collection.objects.link(obj)
    obj.rotation_euler = (math.pi, 0, 0)
    matrix, size = config["intrinsics"], config["size"]
    fx, fy = matrix[0][0], matrix[1][1]
    aspect = fx / fy
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = fx / size["width"] * 36
    data.shift_x = 0.5 - matrix[0][2] / size["width"]
    data.shift_y = (matrix[1][2] - size["height"] / 2) * aspect / size["width"]
    data.clip_start, data.clip_end = 0.00001, 10000
    bpy.context.scene.render.pixel_aspect_y = aspect
    bpy.context.scene.camera = obj


def area_light(
    bpy: Bpy, light: LightData, center: tuple[float, float, float]
) -> None:
    """Aim a circular area light at the fixed fitted source patch."""
    data = bpy.data.lights.new("Fixed source area light", "AREA")
    data.energy, data.size, data.color = (
        light["energy"],
        light["size"],
        light["color"],
    )
    data.shape = "DISK"
    obj = bpy.data.objects.new("Fixed source area light", data)
    bpy.context.collection.objects.link(obj)
    obj.location = light["position"]
    direction = tuple(
        center[index] - obj.location[index] for index in range(3)
    )
    horizontal = math.hypot(direction[0], direction[1])
    obj.rotation_euler = (
        math.atan2(horizontal, -direction[2]),
        0,
        math.atan2(-direction[0], direction[1]),
    )


def configure_render(scene: Scene, config: SceneData) -> None:
    """Keep explicit sample count and photographic background transparency."""
    scene.cycles.device = "GPU"
    scene.cycles.samples = config["samples"]
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.transparent_max_bounces = 8
    scene.render.resolution_x = config["size"]["width"]
    scene.render.resolution_y = config["size"]["height"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("Blender has no world background node")
    background.inputs["Color"].default_value = (0.72, 0.79, 0.9, 1)
    background.inputs["Strength"].default_value = config["ambient"]


def validate_solid(mesh: Mesh) -> None:
    """Require a closed, consistently oriented volume before physical beveling."""
    uses: dict[tuple[int, int], int] = {}
    volume = 0.0
    for polygon in mesh.polygons:
        indices = tuple(polygon.vertices)
        for first_index, second_index in zip(
            indices, (*indices[1:], indices[0]), strict=True
        ):
            edge = (
                min(first_index, second_index),
                max(first_index, second_index),
            )
            uses[edge] = uses.get(edge, 0) + 1
        first = mesh.vertices[indices[0]].co
        for index in range(1, len(indices) - 1):
            second, third = (
                mesh.vertices[indices[index]].co,
                mesh.vertices[indices[index + 1]].co,
            )
            volume += (
                first[0] * (second[1] * third[2] - second[2] * third[1])
                + first[1] * (second[2] * third[0] - second[0] * third[2])
                + first[2] * (second[0] * third[1] - second[1] * third[0])
            ) / 6
    if not uses or any(count != 2 for count in uses.values()) or volume <= 0:
        raise ValueError(
            "Glyph contours did not produce a closed positive-volume solid: "
            f"{sum(count != 2 for count in uses.values())} nonmanifold edges, "
            f"signed volume {volume}"
        )


class Renderer:
    """Reuse static geometry and only the bounded current glyph mesh set."""

    bpy: Bpy
    mathutils: Mathutils
    config: SceneData
    material: Material
    catcher: Mesh
    glyphs: list[BlenderObject]
    fingerprints: list[str]

    def __init__(self, config: SceneData) -> None:
        """Select Metal explicitly before acknowledging worker readiness."""
        self.bpy = cast("Bpy", importlib.import_module("bpy"))
        self.mathutils = cast(
            "Mathutils", importlib.import_module("mathutils")
        )
        self.config = config
        self.glyphs, self.fingerprints = [], []
        bpy = self.bpy
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        preferences = cast(
            "CyclesPreferences",
            bpy.context.preferences.addons["cycles"].preferences,
        )
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        metal = False
        for device in preferences.devices:
            device.use = device.type == "METAL"
            metal = metal or device.use
        if not metal:
            raise RuntimeError(
                "Solid text requires an available Cycles Metal device"
            )
        configure_render(scene, config)
        camera(bpy, config)
        self.material = material(bpy, "Solid type surface", config)
        self.catcher = bpy.data.meshes.new("Fitted shadow receiver")
        catcher = bpy.data.objects.new("Fitted shadow receiver", self.catcher)
        bpy.context.collection.objects.link(catcher)
        catcher.is_shadow_catcher = True
        self.catcher.materials.append(
            material(bpy, "Source wood bounce", config, catcher=True)
        )
        mesh = bpy.data.meshes.new("Source depth foreground")
        mesh.from_pydata(config["vertices"], (), config["triangles"])
        mesh.update()
        holdout = bpy.data.objects.new("Source depth foreground", mesh)
        bpy.context.collection.objects.link(holdout)
        holdout.is_holdout, holdout.visible_shadow = True, False
        for light in config["lights"]:
            area_light(bpy, light, config["plane"]["center"])

    def remove_glyph(self, index: int) -> None:
        """Release replaced mesh storage before the next geometry is allocated."""
        obj = self.glyphs.pop(index)
        mesh = cast("Mesh", obj.data)
        self.bpy.data.objects.remove(obj, do_unlink=True)
        self.bpy.data.meshes.remove(mesh)
        self.fingerprints.pop(index)

    def glyph(
        self, contours: tuple[tuple[tuple[float, float], ...], ...]
    ) -> BlenderObject:
        """Extrude closed hole-aware contours, weld seams, and bevel actual edges."""
        bpy = self.bpy
        curve = bpy.data.curves.new("Closed solid glyph", "CURVE")
        curve.dimensions, curve.fill_mode = "2D", "BOTH"
        curve.resolution_u = 1
        depth = self.config["font_size"] * self.config["thickness"]
        curve.extrude = depth / 2
        curve.materials.append(self.material)
        for contour in contours:
            spline = curve.splines.new("POLY")
            spline.points.add(len(contour) - 1)
            for index, point in enumerate(contour):
                spline.points[index].co = (point[0], point[1], 0, 1)
            spline.use_cyclic_u = True
        obj = bpy.data.objects.new("Complete solid glyph", curve)
        bpy.context.collection.objects.link(obj)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(state=True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")
        bpy.data.curves.remove(curve)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(
            threshold=self.config["font_size"] * 0.000001
        )
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.transform.translate(value=(0, 0, depth / 2))
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh = cast("Mesh", obj.data)
        validate_solid(mesh)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        if self.config["bevel"]:
            bevel = obj.modifiers.new("Clamped physical bevel", "BEVEL")
            bevel.width = self.config["font_size"] * self.config["bevel"]
            bevel.segments = 4
            bevel.use_clamp_overlap, bevel.harden_normals = True, True
            bevel.limit_method = "ANGLE"
        normals = obj.modifiers.new(
            "Weighted solid normals", "WEIGHTED_NORMAL"
        )
        normals.mode, normals.keep_sharp = "FACE_AREA_WITH_ANGLE", True
        obj.select_set(state=False)
        return obj

    def frame(self, descriptor: FrameData, output: str) -> None:
        """Render complete volumes at each exact requested animation sample."""
        incoming = descriptor["glyphs"]
        while len(self.glyphs) > len(incoming):
            self.remove_glyph(-1)
        for index, glyph in enumerate(incoming):
            fingerprint = hashlib.sha256(
                json.dumps(glyph["contours"], separators=(",", ":")).encode()
            ).hexdigest()
            if (
                index < len(self.glyphs)
                and self.fingerprints[index] != fingerprint
            ):
                self.remove_glyph(index)
                self.glyphs.insert(index, self.glyph(glyph["contours"]))
                self.fingerprints.insert(index, fingerprint)
            elif index == len(self.glyphs):
                self.glyphs.append(self.glyph(glyph["contours"]))
                self.fingerprints.append(fingerprint)
            self.glyphs[index].matrix_world = self.mathutils.Matrix(
                glyph["transform"]
            )
        self.catcher.clear_geometry()
        self.catcher.from_pydata(descriptor["catcher"], (), ((0, 1, 2, 3),))
        self.catcher.update()
        self.bpy.context.scene.render.filepath = output
        self.bpy.ops.render.render(write_still=True)


def reply(stream: BinaryIO, status: str, error: str = "") -> None:
    """Keep the protocol separate from Blender's verbose native diagnostics."""
    stream.write(
        (json.dumps({"status": status, "error": error}) + "\n").encode()
    )
    stream.flush()


def main() -> None:
    """Serve one host-owned local connection until explicit close or failure."""
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    scene_path, endpoint = arguments
    with Path(scene_path).open(encoding="utf-8") as handle:
        config = cast("SceneData", json.load(handle))
    if config["version"] != 1:
        raise ValueError("Unsupported solid scene protocol")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(endpoint)
        with cast("BinaryIO", connection.makefile("rwb")) as stream:
            try:
                renderer = Renderer(config)
                reply(stream, "ready")
                while payload := stream.readline(64 * 1024 * 1024 + 1):
                    if len(payload) > 64 * 1024 * 1024:
                        raise ValueError(
                            "Solid frame exceeds the message limit"
                        )
                    command = cast("CommandData", json.loads(payload))
                    if command.get("close"):
                        return
                    output = str(Path(scene_path).parent / command["output"])
                    renderer.frame(command["frame"], output)
                    reply(stream, "rendered")
            except BaseException as error:
                reply(stream, "error", f"{type(error).__name__}: {error}")
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
