"""Main orchestrator to build, light, inspect, and export the Crystal Baby Dragon in Blender 5.2."""

import os
import sys
import importlib
import bpy
from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import baby_math
import baby_materials
import baby_body
import baby_head
import baby_limbs
import baby_wings
import baby_saddle
import baby_rig

for mod in (baby_math, baby_materials, baby_body, baby_head, baby_limbs, baby_wings, baby_saddle, baby_rig):
    importlib.reload(mod)


def _setup_clean_scene(scene_name: str = "Crystal Baby Dragon"):
    old_scene = bpy.data.scenes.get(scene_name)
    new_scene = bpy.data.scenes.new(scene_name + "_TMP")
    if bpy.context.window is not None:
        bpy.context.window.scene = new_scene

    if old_scene is not None:
        old_objs = list(old_scene.objects)
        old_cols = list(old_scene.collection.children)
        bpy.data.scenes.remove(old_scene)
        for obj in old_objs:
            if obj.users == 0 or len(obj.users_scene) == 0:
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if data is not None and data.users == 0:
                    if isinstance(data, bpy.types.Mesh):
                        bpy.data.meshes.remove(data)
                    elif isinstance(data, bpy.types.Armature):
                        bpy.data.armatures.remove(data)
                    elif isinstance(data, bpy.types.Light):
                        bpy.data.lights.remove(data)
                    elif isinstance(data, bpy.types.Camera):
                        bpy.data.cameras.remove(data)
        for col in old_cols:
            if col.users == 0:
                bpy.data.collections.remove(col)

    for act in list(bpy.data.actions):
        if act.users == 0 or act.name.startswith(("Takeoff_And_Fly", "Hover_Fly_Loop", "Walk_Loop")):
            bpy.data.actions.remove(act, do_unlink=True)

    new_scene.name = scene_name
    return new_scene


def _make_collection(name: str, parent_col):
    col = bpy.data.collections.new(name)
    parent_col.children.link(col)
    return col


def _aim_object_at(obj, target_pt: Vector):
    direction = target_pt - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _setup_world_and_studio(scene, studio_col):
    try:
        scene.render.engine = "CYCLES"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    if scene.render.engine == "CYCLES":
        scene.cycles.device = "GPU"
        scene.cycles.samples = 48
        scene.cycles.use_denoising = True

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800

    world = bpy.data.worlds.get("Crystal_Studio_World")
    if world is None:
        world = bpy.data.worlds.new("Crystal_Studio_World")
    world.use_nodes = True
    w_nt = world.node_tree
    w_nt.nodes.clear()
    w_out = w_nt.nodes.new("ShaderNodeOutputWorld")
    w_bg = w_nt.nodes.new("ShaderNodeBackground")
    w_bg.inputs["Color"].default_value = (0.015, 0.015, 0.024, 1.0)
    w_bg.inputs["Strength"].default_value = 0.55
    w_nt.links.new(w_bg.outputs["Background"], w_out.inputs["Surface"])
    scene.world = world

    light_specs = [
        ("Crystal_Key_Light",    (-3.8, -5.2,  4.8), (0.0,  0.0, 1.1),  900.0, 4.5, (1.00, 0.94, 0.92)),
        ("Crystal_Fill_Light",   ( 4.6, -4.0,  2.8), (0.0,  0.0, 1.1),  520.0, 5.5, (0.86, 0.90, 1.00)),
        ("Crystal_Rim_Light",    ( 0.8,  5.2,  4.6), (0.0, -0.2, 1.2), 1100.0, 5.0, (0.94, 0.92, 1.00)),
        ("Crystal_Under_Bounce", ( 0.0,  0.0, -3.5), (0.0,  0.0, 0.8),  320.0, 6.5, (0.96, 0.94, 1.00)),
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

    cam_data = bpy.data.cameras.new("Camera_Crystal_Hero_Data")
    cam_data.lens = 34.0
    cam_obj = bpy.data.objects.new("Camera_Hero", cam_data)
    cam_obj.location = Vector((-5.8, -7.2, 3.0))
    _aim_object_at(cam_obj, Vector((0.0, 0.10, 1.60)))
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
    scene = _setup_clean_scene("Crystal Baby Dragon")
    dragon_col = _make_collection("Crystal_Dragon", scene.collection)
    studio_col = _make_collection("Crystal_Studio", scene.collection)

    materials = baby_materials.create_all_materials()

    root = bpy.data.objects.new("Crystal_Dragon_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.4
    dragon_col.objects.link(root)

    dragon_objs = []
    dragon_objs.extend(baby_body.build_body_objects(materials, dragon_col, root))
    dragon_objs.extend(baby_head.build_head_objects(materials, dragon_col, root))
    dragon_objs.extend(baby_limbs.build_limb_objects(materials, dragon_col, root))
    dragon_objs.append(baby_wings.build_wing_objects(materials, dragon_col, root))
    dragon_objs.append(baby_saddle.build_saddle_object(materials, dragon_col, root))

    _setup_world_and_studio(scene, studio_col)
    bpy.context.view_layer.update()

    mins, maxs = _world_bbox(dragon_objs)
    if abs(mins.z) > 1e-4:
        dz = -mins.z
        for obj in dragon_objs:
            if obj.type == "MESH":
                for v in obj.data.vertices:
                    v.co.z += dz
                obj.data.update()
        bpy.context.view_layer.update()
        mins, maxs = _world_bbox(dragon_objs)

    rig_obj = baby_rig.build_and_bind_rig(dragon_objs, dragon_col, root)
    baby_rig.create_flight_animations(rig_obj, scene)

    dims = maxs - mins
    print("=== Crystal Baby Dragon Build Summary ===")
    total_faces = 0
    for obj in dragon_objs:
        f_cnt = len(obj.data.polygons) if obj.type == "MESH" else 0
        total_faces += f_cnt
        print(f"  {obj.name:20s} | faces={f_cnt:6d} | mats={[m.name for m in obj.data.materials]}")
    print(f"Total faces: {total_faces}")
    print(f"Rig: {rig_obj.name} | bones={len(rig_obj.data.bones)} | fps={scene.render.fps} | frames={scene.frame_start}..{scene.frame_end}")
    print(f"World BBox min=({mins.x:.3f}, {mins.y:.3f}, {mins.z:.3f}) max=({maxs.x:.3f}, {maxs.y:.3f}, {maxs.z:.3f})")
    print(f"Dimensions: X={dims.x:.3f}m, Y={dims.y:.3f}m, Z={dims.z:.3f}m")

    blend_path = os.path.join(ROOT_DIR, "crystal_dragon.blend")
    bpy.data.libraries.write(blend_path, {scene}, compress=True)
    print(f"Saved .blend -> {blend_path}")

    for s in bpy.data.scenes:
        for o in s.objects:
            o.select_set(False)
    for o in scene.objects:
        o.select_set(o in dragon_objs or o in (root, rig_obj))
    bpy.context.view_layer.objects.active = rig_obj
    glb_path = os.path.join(ROOT_DIR, "crystal_dragon.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        use_selection=True,
        use_active_scene=True,
        export_animations=True,
        export_skins=True,
        export_def_bones=False,
        export_animation_mode="ACTIONS",
        export_frame_range=False,
        export_force_sampling=True,
        export_anim_slide_to_zero=True,
        export_rest_position_armature=True,
    )
    print(f"Exported .glb -> {glb_path}")
    return dragon_objs, rig_obj


if __name__ == "__main__":
    build_all()
