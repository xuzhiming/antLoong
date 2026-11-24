"""Main orchestrator to build, light, inspect, and export the Cyberpunk Dragon in Blender 5.2."""

import os
import sys
import math
import bpy
from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Reload submodules so edits take effect cleanly inside live Blender
import importlib
import dragon_spine
import dragon_materials
import dragon_body
import dragon_head
import dragon_limbs

for mod in (dragon_spine, dragon_materials, dragon_body, dragon_head, dragon_limbs):
    importlib.reload(mod)


def _setup_clean_scene(scene_name: str = "Cyberpunk Dragon"):
    """Create or cleanly replace the target scene while preserving other scenes in the .blend session."""
    old_scene = bpy.data.scenes.get(scene_name)
    new_scene = bpy.data.scenes.new(scene_name + "_TMP")
    bpy.context.window.scene = new_scene

    if old_scene is not None:
        # Remove objects belonging exclusively to old_scene
        old_objs = list(old_scene.objects)
        old_cols = list(old_scene.collection.children)
        bpy.data.scenes.remove(old_scene)
        for obj in old_objs:
            if obj.users == 0 or (len(obj.users_scene) == 0):
                mesh_data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh_data is not None and mesh_data.users == 0:
                    if isinstance(mesh_data, bpy.types.Mesh):
                        bpy.data.meshes.remove(mesh_data)
                    elif isinstance(mesh_data, bpy.types.Light):
                        bpy.data.lights.remove(mesh_data)
                    elif isinstance(mesh_data, bpy.types.Camera):
                        bpy.data.cameras.remove(mesh_data)
        for col in old_cols:
            if col.users == 0:
                bpy.data.collections.remove(col)

    new_scene.name = scene_name
    return new_scene


def _make_collection(name: str, parent_col):
    col = bpy.data.collections.new(name)
    parent_col.children.link(col)
    return col


def _aim_object_at(obj, target_pt: Vector):
    direction = target_pt - obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def _setup_world_and_studio(scene, studio_col):
    """Configure studio lighting, camera, and Cycles GPU render settings."""
    try:
        scene.render.engine = "CYCLES"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    if scene.render.engine == "CYCLES":
        scene.cycles.device = "GPU"
        scene.cycles.samples = 48
        scene.cycles.use_denoising = True

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720

    # World background
    world = bpy.data.worlds.get("Cyberpunk_Studio_World")
    if world is None:
        world = bpy.data.worlds.new("Cyberpunk_Studio_World")
    world.use_nodes = True
    w_nt = world.node_tree
    w_nt.nodes.clear()
    w_out = w_nt.nodes.new("ShaderNodeOutputWorld")
    w_bg = w_nt.nodes.new("ShaderNodeBackground")
    w_bg.inputs["Color"].default_value = (0.048, 0.055, 0.072, 1.0)
    w_bg.inputs["Strength"].default_value = 0.50
    w_nt.links.new(w_bg.outputs["Background"], w_out.inputs["Surface"])
    scene.world = world

    # Studio Area Lights
    light_specs = [
        # (name, location, target, energy, size, color)
        ("Key_Light",          (-5.5, -7.5,  6.5), (0.0, -0.5, 1.4), 1600.0, 6.0, (1.00, 0.96, 0.92)),
        ("Fill_Light",         ( 6.5, -5.5,  3.8), (0.0,  0.0, 1.4),  850.0, 7.0, (0.78, 0.92, 1.00)),
        ("Rim_Light",          ( 1.5,  6.5,  6.8), (0.0, -0.8, 1.6), 1900.0, 6.5, (0.88, 0.94, 1.00)),
        ("Under_Bounce_Light", ( 0.0, -0.5, -4.5), (0.0,  0.0, 1.0),  520.0, 9.0, (0.92, 0.95, 1.00)),
    ]
    for l_name, loc, tgt, energy, size, col in light_specs:
        ldata = bpy.data.lights.new(l_name + "_Data", type="AREA")
        ldata.energy = energy
        ldata.size = size
        ldata.color = col
        lobj = bpy.data.objects.new(l_name, ldata)
        lobj.location = Vector(loc)
        _aim_object_at(lobj, Vector(tgt))
        studio_col.objects.link(lobj)

    # Hero 3/4 Camera framed cleanly over entire 10m dragon
    cam_data = bpy.data.cameras.new("Camera_Hero_Data")
    cam_data.lens = 38.0
    cam_obj = bpy.data.objects.new("Camera_Hero", cam_data)
    cam_obj.location = Vector((-8.2, -9.8, 3.9))
    _aim_object_at(cam_obj, Vector((0.0, -0.05, 1.45)))
    studio_col.objects.link(cam_obj)
    scene.camera = cam_obj


def _world_bbox(objs):
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in objs:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for corner in obj.bound_box:
            w_co = mw @ Vector(corner)
            mins.x = min(mins.x, w_co.x)
            mins.y = min(mins.y, w_co.y)
            mins.z = min(mins.z, w_co.z)
            maxs.x = max(maxs.x, w_co.x)
            maxs.y = max(maxs.y, w_co.y)
            maxs.z = max(maxs.z, w_co.z)
    return mins, maxs


def build_all():
    scene = _setup_clean_scene("Cyberpunk Dragon")
    dragon_col = _make_collection("Cyberpunk_Dragon", scene.collection)
    studio_col = _make_collection("Studio_Setup", scene.collection)

    materials = dragon_materials.create_all_materials()

    root = bpy.data.objects.new("Cyberpunk_Dragon_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.5
    dragon_col.objects.link(root)

    dragon_objs = []
    dragon_objs.extend(dragon_body.build_body_objects(materials, dragon_col, root))
    dragon_objs.extend(dragon_head.build_head_objects(materials, dragon_col, root))
    dragon_objs.extend(dragon_limbs.build_limb_objects(materials, dragon_col, root))

    _setup_world_and_studio(scene, studio_col)

    # Force depsgraph update before measuring world bounding boxes
    bpy.context.view_layer.update()

    # Check ground contact and shift root if needed so claws rest strictly on Z = 0.000
    mins, maxs = _world_bbox(dragon_objs)
    if abs(mins.z) > 1e-4:
        root.location.z -= mins.z
        bpy.context.view_layer.update()
        mins, maxs = _world_bbox(dragon_objs)

    dims = maxs - mins
    print("=== Cyberpunk Dragon Build Summary ===")
    total_faces = 0
    for obj in dragon_objs:
        f_cnt = len(obj.data.polygons) if obj.type == "MESH" else 0
        total_faces += f_cnt
        print(f"  {obj.name:24s} | faces={f_cnt:6d} | mats={[m.name for m in obj.data.materials]}")
    print(f"Total faces: {total_faces}")
    print(f"World BBox min=({mins.x:.3f}, {mins.y:.3f}, {mins.z:.3f}) max=({maxs.x:.3f}, {maxs.y:.3f}, {maxs.z:.3f})")
    print(f"Dimensions: X={dims.x:.3f}m, Y={dims.y:.3f}m, Z={dims.z:.3f}m (Aspect L/H={dims.y/max(dims.z, 1e-4):.2f})")

    # Export clean standalone .blend containing ONLY Cyberpunk Dragon scene
    blend_path = os.path.join(SCRIPT_DIR, "cyberpunk_dragon.blend")
    bpy.data.libraries.write(blend_path, {scene}, compress=True)
    print(f"Saved .blend -> {blend_path}")

    # Export .glb containing strictly dragon objects
    for o in scene.objects:
        o.select_set(o in dragon_objs or o == root)
    bpy.context.view_layer.objects.active = root
    glb_path = os.path.join(SCRIPT_DIR, "cyberpunk_dragon.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        use_selection=True,
    )
    print(f"Exported .glb -> {glb_path}")
    return dragon_objs


if __name__ == "__main__":
    build_all()
