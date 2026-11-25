"""Automatic rigging, deterministic multi-part skinning, and wing-flap flight animation for Crystal Baby Dragon."""

import math
import bpy
from mathutils import Vector, Matrix

from baby_math import smoothstep


def _pt_seg_dist(p: Vector, a: Vector, b: Vector) -> float:
    ab = b - a
    len_sq = ab.length_squared
    if len_sq < 1e-10:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / len_sq))
    return (p - (a + ab * t)).length


def _segment_weights(p: Vector, named_segments, power: float = 3.4, eps: float = 0.008, max_influences: int = 4):
    accum = {}
    for bone_name, a, b, mul in named_segments:
        d = _pt_seg_dist(p, a, b)
        w = mul / (math.pow(d, power) + eps)
        accum[bone_name] = accum.get(bone_name, 0.0) + w
    ranked = sorted(accum.items(), key=lambda kv: kv[1], reverse=True)[:max_influences]
    total = sum(w for _, w in ranked)
    if total <= 1e-12:
        return [(named_segments[0][0], 1.0)]
    return [(name, w / total) for name, w in ranked if w > 1e-6]


def _normalize_pairs(pairs, max_influences: int = 4):
    merged = {}
    for name, w in pairs:
        if w > 1e-6:
            merged[name] = merged.get(name, 0.0) + w
    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:max_influences]
    total = sum(w for _, w in ranked)
    if total <= 1e-12:
        return []
    return [(name, w / total) for name, w in ranked]


def _get_or_create_vg(obj, name: str):
    vg = obj.vertex_groups.get(name)
    if vg is None:
        vg = obj.vertex_groups.new(name=name)
    return vg


def _assign_vertex_weights(obj, weights_by_vert):
    obj.vertex_groups.clear()
    grouped = {}
    for v_idx, pairs in enumerate(weights_by_vert):
        normed = _normalize_pairs(pairs, max_influences=4)
        for bone_name, w in normed:
            grouped.setdefault(bone_name, []).append((v_idx, w))

    for bone_name, entries in grouped.items():
        vg = _get_or_create_vg(obj, bone_name)
        for v_idx, w in entries:
            vg.add((v_idx,), w, "REPLACE")


def _create_armature(collection, root):
    arm_data = bpy.data.armatures.new("Crystal_Dragon_Armature")
    arm_data.display_type = "OCTAHEDRAL"

    rig_obj = bpy.data.objects.new("Crystal_Dragon_Rig", arm_data)
    rig_obj.show_in_front = True
    collection.objects.link(rig_obj)
    rig_obj.parent = root
    rig_obj.matrix_parent_inverse = root.matrix_world.inverted()

    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_data.edit_bones

    def _add_bone(name, head, tail, parent_bone=None, use_connect=False, use_deform=True):
        eb = ebones.new(name)
        eb.head = Vector(head)
        eb.tail = Vector(tail)
        eb.parent = parent_bone
        eb.use_connect = use_connect and (parent_bone is not None)
        eb.use_deform = use_deform
        vec = (eb.tail - eb.head)
        if vec.length > 1e-6:
            vec.normalize()
            up_ref = Vector((0.0, -1.0, 0.0)) if abs(vec.dot(Vector((0.0, 0.0, 1.0)))) > 0.92 else Vector((0.0, 0.0, 1.0))
            eb.align_roll(up_ref)
        return eb

    # Root & Spine-Head-Tail (10 bones)
    b_root = _add_bone("Root", (0.0, 0.0, 0.0), (0.0, -0.35, 0.0), None, False, use_deform=False)
    b_pelvis = _add_bone("Pelvis", (0.0, 0.36, 0.80), (0.0, -0.14, 0.85), b_root, False, True)
    b_spine = _add_bone("Spine", (0.0, -0.14, 0.85), (0.0, -0.66, 0.94), b_pelvis, True, True)
    b_chest = _add_bone("Chest", (0.0, -0.66, 0.94), (0.0, -0.86, 1.24), b_spine, True, True)
    b_neck = _add_bone("Neck", (0.0, -0.86, 1.24), (0.0, -0.98, 1.56), b_chest, True, True)
    _add_bone("Head", (0.0, -0.98, 1.56), (0.0, -1.28, 1.92), b_neck, True, True)

    b_tail1 = _add_bone("Tail_01", (0.0, 0.42, 0.78), (0.0, 0.85, 0.58), b_pelvis, False, True)
    b_tail2 = _add_bone("Tail_02", (0.0, 0.85, 0.58), (0.0, 1.45, 0.52), b_tail1, True, True)
    b_tail3 = _add_bone("Tail_03", (0.0, 1.45, 0.52), (0.0, 1.94, 0.68), b_tail2, True, True)
    _add_bone("Tail_04", (0.0, 1.94, 0.68), (0.0, 2.48, 0.84), b_tail3, True, True)

    # Symmetrical Wings (12 bones) & Limbs (12 bones)
    for s, sfx in ((1.0, "L"), (-1.0, "R")):
        b_wsh = _add_bone(
            f"Wing_Shoulder.{sfx}",
            (s * 0.24, -0.58, 1.18),
            (s * 0.72, -0.48, 1.58),
            b_chest,
            False,
            True,
        )
        b_wel = _add_bone(
            f"Wing_Elbow.{sfx}",
            (s * 0.72, -0.48, 1.58),
            (s * 1.26, -0.42, 2.24),
            b_wsh,
            True,
            True,
        )
        _add_bone(f"Wing_Finger1.{sfx}", (s * 1.26, -0.42, 2.24), (s * 2.16, -0.14, 1.12), b_wel, True, True)
        _add_bone(f"Wing_Finger2.{sfx}", (s * 1.26, -0.42, 2.24), (s * 1.82,  0.28, 0.74), b_wel, False, True)
        _add_bone(f"Wing_Finger3.{sfx}", (s * 1.26, -0.42, 2.24), (s * 1.32,  0.58, 0.78), b_wel, False, True)
        _add_bone(f"Wing_Finger4.{sfx}", (s * 1.26, -0.42, 2.24), (s * 0.80,  0.52, 0.90), b_wel, False, True)

        b_fu = _add_bone(f"Fore_Upper.{sfx}", (s * 0.26, -0.70, 1.02), (s * 0.34, -0.66, 0.58), b_chest, False, True)
        b_fl = _add_bone(f"Fore_Lower.{sfx}", (s * 0.34, -0.66, 0.58), (s * 0.34, -0.82, 0.24), b_fu, True, True)
        _add_bone(f"Fore_Paw.{sfx}", (s * 0.34, -0.82, 0.24), (s * 0.35, -1.04, 0.08), b_fl, True, True)

        b_hu = _add_bone(f"Hind_Upper.{sfx}", (s * 0.38, 0.20, 0.78), (s * 0.48, 0.34, 0.38), b_pelvis, False, True)
        b_hl = _add_bone(f"Hind_Lower.{sfx}", (s * 0.48, 0.34, 0.38), (s * 0.44, 0.19, 0.16), b_hu, True, True)
        _add_bone(f"Hind_Paw.{sfx}", (s * 0.44, 0.19, 0.16), (s * 0.46, -0.02, 0.07), b_hl, True, True)

    bpy.ops.object.mode_set(mode="OBJECT")
    if prev_active is not None:
        bpy.context.view_layer.objects.active = prev_active

    for pb in rig_obj.pose.bones:
        pb.rotation_mode = "XYZ"

    return rig_obj


def _bind_all_meshes(dragon_objs, rig_obj):
    # Spine-to-tail segments shared by torso/scutes/spines
    spine_segments = [
        ("Head",    Vector((0.0, -0.98, 1.56)), Vector((0.0, -1.28, 1.92)), 1.0),
        ("Neck",    Vector((0.0, -0.86, 1.24)), Vector((0.0, -0.98, 1.56)), 1.0),
        ("Chest",   Vector((0.0, -0.66, 0.94)), Vector((0.0, -0.86, 1.24)), 1.1),
        ("Spine",   Vector((0.0, -0.14, 0.85)), Vector((0.0, -0.66, 0.94)), 1.0),
        ("Pelvis",  Vector((0.0,  0.40, 0.79)), Vector((0.0, -0.14, 0.85)), 1.0),
        ("Tail_01", Vector((0.0,  0.42, 0.78)), Vector((0.0,  0.85, 0.58)), 1.0),
        ("Tail_02", Vector((0.0,  0.85, 0.58)), Vector((0.0,  1.45, 0.52)), 1.0),
        ("Tail_03", Vector((0.0,  1.45, 0.52)), Vector((0.0,  1.94, 0.68)), 1.0),
        ("Tail_04", Vector((0.0,  1.94, 0.68)), Vector((0.0,  2.55, 0.86)), 1.0),
    ]
    tail_plume_segments = [
        ("Tail_02", Vector((0.0, 0.85, 0.58)), Vector((0.0, 1.45, 0.52)), 0.7),
        ("Tail_03", Vector((0.0, 1.45, 0.52)), Vector((0.0, 1.94, 0.68)), 1.0),
        ("Tail_04", Vector((0.0, 1.94, 0.68)), Vector((0.0, 2.55, 0.86)), 1.25),
    ]

    side_limb_segments = {}
    side_wing_segments = {}
    for s, sfx in ((1.0, "L"), (-1.0, "R")):
        side_limb_segments[(sfx, "fore")] = [
            ("Chest",            Vector((s * 0.20, -0.70, 1.08)), Vector((s * 0.18, -0.66, 0.90)), 1.15),
            (f"Fore_Upper.{sfx}", Vector((s * 0.26, -0.70, 1.02)), Vector((s * 0.34, -0.66, 0.58)), 1.0),
            (f"Fore_Lower.{sfx}", Vector((s * 0.34, -0.66, 0.58)), Vector((s * 0.34, -0.82, 0.24)), 1.0),
            (f"Fore_Paw.{sfx}",   Vector((s * 0.34, -0.82, 0.24)), Vector((s * 0.35, -1.04, 0.08)), 1.1),
        ]
        side_limb_segments[(sfx, "hind")] = [
            ("Pelvis",           Vector((s * 0.26,  0.22, 0.80)), Vector((s * 0.28,  0.34, 0.72)), 1.15),
            (f"Hind_Upper.{sfx}", Vector((s * 0.38,  0.20, 0.78)), Vector((s * 0.48,  0.34, 0.38)), 1.0),
            (f"Hind_Lower.{sfx}", Vector((s * 0.48,  0.34, 0.38)), Vector((s * 0.44,  0.19, 0.16)), 1.0),
            (f"Hind_Paw.{sfx}",   Vector((s * 0.44,  0.19, 0.16)), Vector((s * 0.46, -0.02, 0.07)), 1.1),
        ]

        sh_pt = Vector((s * 0.24, -0.58, 1.18))
        elbow_pt = Vector((s * 0.72, -0.48, 1.58))
        wrist_pt = Vector((s * 1.26, -0.42, 2.24))
        tip1 = Vector((s * 2.16, -0.14, 1.12))
        tip2 = Vector((s * 1.82,  0.28, 0.74))
        tip3 = Vector((s * 1.32,  0.58, 0.78))
        tip4 = Vector((s * 0.80,  0.52, 0.90))
        flank_mid = Vector((s * 0.34, -0.22, 1.04))
        flank_rear = Vector((s * 0.42,  0.12, 0.96))

        side_wing_segments[sfx] = [
            ("Chest",               sh_pt,                            flank_mid,                    1.05),
            ("Spine",               flank_mid,                        flank_rear,                   1.10),
            (f"Wing_Shoulder.{sfx}", sh_pt.lerp(elbow_pt, 0.05),      sh_pt.lerp(elbow_pt, 0.98),   1.0),
            (f"Wing_Elbow.{sfx}",    elbow_pt.lerp(wrist_pt, 0.02),   wrist_pt,                     1.25),
            (f"Wing_Finger1.{sfx}",  wrist_pt.lerp(tip1, 0.07),       tip1,                         1.0),
            (f"Wing_Finger2.{sfx}",  wrist_pt.lerp(tip2, 0.07),       tip2,                         1.0),
            (f"Wing_Finger3.{sfx}",  wrist_pt.lerp(tip3, 0.07),       tip3,                         1.0),
            (f"Wing_Finger4.{sfx}",  wrist_pt.lerp(tip4, 0.07),       tip4,                         1.0),
        ]

    for obj in dragon_objs:
        if obj.type != "MESH":
            continue
        obj.parent = rig_obj
        obj.matrix_parent_inverse = rig_obj.matrix_world.inverted()

        for m in list(obj.modifiers):
            if m.type == "ARMATURE":
                obj.modifiers.remove(m)
        arm_mod = obj.modifiers.new("Crystal_Rig_Deform", type="ARMATURE")
        arm_mod.object = rig_obj
        arm_mod.use_vertex_groups = True
        arm_mod.use_bone_envelopes = False

        name = obj.name
        verts = obj.data.vertices
        weights_by_vert = []

        if name in ("Baby_Eyes", "Baby_Horns", "Baby_Head_Frills"):
            weights_by_vert = [[("Head", 1.0)] for _ in verts]

        elif name == "Baby_Head":
            for v in verts:
                p = v.co
                if p.z >= 1.56 or p.y <= -1.16:
                    weights_by_vert.append([("Head", 1.0)])
                else:
                    t_head = smoothstep(1.34, 1.56, p.z)
                    weights_by_vert.append([("Head", t_head), ("Neck", 1.0 - t_head)])

        elif name == "Baby_Pauldrons":
            for v in verts:
                p = v.co
                sfx = "L" if p.x >= 0.0 else "R"
                if p.z > 1.06:
                    weights_by_vert.append([("Chest", 0.85), (f"Fore_Upper.{sfx}", 0.15)])
                elif p.z > 0.93:
                    weights_by_vert.append([("Chest", 0.65), (f"Fore_Upper.{sfx}", 0.35)])
                else:
                    weights_by_vert.append([("Chest", 0.45), (f"Fore_Upper.{sfx}", 0.55)])

        elif name == "Baby_Claws":
            for v in verts:
                p = v.co
                sfx = "L" if p.x >= 0.0 else "R"
                kind = "Fore_Paw" if p.y < -0.25 else "Hind_Paw"
                weights_by_vert.append([(f"{kind}.{sfx}", 1.0)])

        elif name == "Baby_Tail_Plume":
            for v in verts:
                weights_by_vert.append(_segment_weights(v.co, tail_plume_segments, power=3.0, eps=0.008, max_influences=3))

        elif name in ("Baby_Body", "Baby_Dorsal_Spines"):
            for v in verts:
                weights_by_vert.append(_segment_weights(v.co, spine_segments, power=3.4, eps=0.012, max_influences=3))

        elif name == "Baby_Saddle":
            saddle_segs = spine_segments[2:5]  # Chest, Spine, Pelvis
            for v in verts:
                weights_by_vert.append(_segment_weights(v.co, saddle_segs, power=3.6, eps=0.015, max_influences=3))

        elif name == "Baby_Belly_Scutes":
            chin_segs = spine_segments[:2]
            for v in verts:
                p = v.co
                if p.y < -0.96 and p.z > 1.38:
                    weights_by_vert.append(_segment_weights(p, chin_segs, power=3.2, eps=0.010, max_influences=2))
                else:
                    weights_by_vert.append(_segment_weights(p, spine_segments, power=3.4, eps=0.012, max_influences=3))

        elif name == "Baby_Limbs":
            for v in verts:
                p = v.co
                sfx = "L" if p.x >= 0.0 else "R"
                kind = "fore" if p.y < -0.25 else "hind"
                weights_by_vert.append(_segment_weights(p, side_limb_segments[(sfx, kind)], power=3.0, eps=0.018, max_influences=3))

        elif name == "Baby_Wings":
            for v in verts:
                p = v.co
                sfx = "L" if p.x >= 0.0 else "R"
                weights_by_vert.append(_segment_weights(p, side_wing_segments[sfx], power=3.2, eps=0.004, max_influences=4))

        else:
            for v in verts:
                weights_by_vert.append(_segment_weights(v.co, spine_segments, power=3.0, eps=0.012, max_influences=3))

        _assign_vertex_weights(obj, weights_by_vert)


def _audit_rig_binding(dragon_objs, rig_obj):
    deform_bones = {b.name for b in rig_obj.data.bones if b.use_deform}
    used_bones = set()
    total_verts = 0
    unweighted_total = 0

    for obj in dragon_objs:
        if obj.type != "MESH":
            continue
        vg_names = {vg.index: vg.name for vg in obj.vertex_groups}
        v_cnt = len(obj.data.vertices)
        total_verts += v_cnt
        unweighted = 0
        for v in obj.data.vertices:
            w_sum = 0.0
            for g in v.groups:
                if g.weight > 1e-6:
                    w_sum += g.weight
                    b_name = vg_names.get(g.group)
                    if b_name in deform_bones:
                        used_bones.add(b_name)
            if w_sum < 1e-4:
                unweighted += 1
        unweighted_total += unweighted

    unused = sorted(deform_bones - used_bones)
    print(
        f"Rig Skin Audit: bones={len(rig_obj.data.bones)} (deform={len(deform_bones)}), "
        f"used={len(used_bones)}, total_verts={total_verts}, unweighted={unweighted_total}, unused_deform={unused}"
    )
    if unweighted_total > 0 or unused:
        raise RuntimeError(f"Skin audit failed: unweighted={unweighted_total}, unused_deform={unused}")


def _set_world_delta_rot(arm_obj, pb, world_rot_mat3: Matrix):
    rest_m3 = arm_obj.data.bones[pb.name].matrix_local.to_3x3()
    local_m3 = rest_m3.transposed() @ world_rot_mat3 @ rest_m3
    pb.rotation_euler = local_m3.to_euler("XYZ")


def _set_world_delta_loc(arm_obj, pb, world_offset: Vector):
    rest_m3 = arm_obj.data.bones[pb.name].matrix_local.to_3x3()
    pb.location = rest_m3.transposed() @ world_offset


def _reset_pose(rig_obj):
    for pb in rig_obj.pose.bones:
        pb.location = Vector((0.0, 0.0, 0.0))
        pb.rotation_euler = (0.0, 0.0, 0.0)


def _bind_action(rig_obj, action):
    if rig_obj.animation_data is None:
        rig_obj.animation_data_create()
    ad = rig_obj.animation_data
    ad.action = action
    if hasattr(action, "slots"):
        slot = next((s for s in action.slots if getattr(s, "target_id_type", "OBJECT") == "OBJECT"), None)
        if slot is None:
            slot = action.slots.new(id_type="OBJECT", name=rig_obj.name)
        ad.action_slot = slot


def _insert_pose_keys(rig_obj, frame: int):
    for pb in rig_obj.pose.bones:
        if pb.name == "Root":
            pb.keyframe_insert(data_path="location", frame=frame)
        pb.keyframe_insert(data_path="rotation_euler", frame=frame)


def _apply_flight_pose(
    rig_obj,
    phase: float,
    wing_amp: float,
    air_blend: float,
    root_z: float,
    spread_blend: float = 0.0,
    crouch_amt: float = 0.0,
    prelift_deg: float = 0.0,
    landing_flare: float = 0.0,
):
    pbones = rig_obj.pose.bones
    _reset_pose(rig_obj)

    axis_pitch_up = Vector((-1.0, 0.0, 0.0))
    _set_world_delta_loc(rig_obj, pbones["Root"], Vector((0.0, 0.0, root_z)))

    # Wingbeat stroke waveform: +1 = peak upstroke, -1 = peak power downstroke
    # Asymmetric timing (slightly brisker downstroke, smooth recovery upstroke)
    flap_raw = math.sin(phase) + 0.16 * math.sin(2.0 * phase)
    flap_norm = max(-1.12, min(1.12, flap_raw))
    sweep_raw = -math.cos(phase)  # +1 = sweeping forward on downstroke, -1 = sweeping back on upstroke

    # Elbow and fingers lag sequentially for fluid avian/dragon wing wave
    elbow_wave = math.sin(phase - 0.72)
    f_wave_1 = math.sin(phase - 1.25)
    f_wave_2 = math.sin(phase - 1.45)
    f_wave_3 = math.sin(phase - 1.65)
    f_wave_4 = math.sin(phase - 1.85)

    # Torso & head flight posture + coupled lift bob
    pitch_chest_deg = air_blend * (9.0 - 3.5 * flap_norm + 6.0 * landing_flare) - crouch_amt * 9.0
    pitch_spine_deg = air_blend * (5.0 - 2.0 * flap_norm) - crouch_amt * 5.0
    pitch_pelvis_deg = air_blend * (-3.0 - 1.5 * flap_norm) + crouch_amt * 7.0
    pitch_neck_deg = -pitch_chest_deg * 0.45 + air_blend * 2.0
    pitch_head_deg = -(pitch_chest_deg + pitch_spine_deg + pitch_neck_deg) * 0.72 + crouch_amt * 6.0

    _set_world_delta_rot(rig_obj, pbones["Pelvis"], Matrix.Rotation(math.radians(pitch_pelvis_deg), 3, axis_pitch_up))
    _set_world_delta_rot(rig_obj, pbones["Spine"], Matrix.Rotation(math.radians(pitch_spine_deg), 3, axis_pitch_up))
    _set_world_delta_rot(rig_obj, pbones["Chest"], Matrix.Rotation(math.radians(pitch_chest_deg), 3, axis_pitch_up))
    _set_world_delta_rot(rig_obj, pbones["Neck"], Matrix.Rotation(math.radians(pitch_neck_deg), 3, axis_pitch_up))
    _set_world_delta_rot(rig_obj, pbones["Head"], Matrix.Rotation(math.radians(pitch_head_deg), 3, axis_pitch_up))

    # Tail traveling wave
    for idx, t_name in enumerate(("Tail_01", "Tail_02", "Tail_03", "Tail_04")):
        t_phase = phase - 0.85 - idx * 0.72
        t_pitch = air_blend * ((-5.0 - idx * 1.5) + (5.5 + idx * 1.8) * math.sin(t_phase)) - crouch_amt * 4.0
        t_yaw = air_blend * (4.0 * math.sin(0.5 * phase - idx * 0.65))
        r_tail = Matrix.Rotation(math.radians(t_yaw), 3, Vector((0.0, 0.0, 1.0))) @ Matrix.Rotation(
            math.radians(-t_pitch), 3, axis_pitch_up
        )
        _set_world_delta_rot(rig_obj, pbones[t_name], r_tail)

    # Wings (L/R symmetric): rest pose forms a folded tent (shoulder->wrist +46 deg,
    # wrist->tips -51..-109 deg). spread_blend unfurls the entire wing into a wide horizontal
    # airfoil first, then applies power downstroke/upstroke wave and finger fan follow-through.
    down_open = max(0.0, -flap_norm)
    up_fold = max(0.0, flap_norm)

    sh_flap_deg = -26.0 * spread_blend + prelift_deg + wing_amp * (34.0 * flap_norm + 6.0 * landing_flare)
    sh_sweep_deg = spread_blend * 6.0 + wing_amp * (-6.0 * flap_norm + 10.0 * sweep_raw + 12.0 * landing_flare)
    sh_twist_deg = wing_amp * (-8.0 * math.cos(phase))

    el_flap_deg = -14.0 * spread_blend + wing_amp * (-10.0 * elbow_wave + 5.0 * up_fold)
    el_sweep_deg = spread_blend * 10.0 + wing_amp * (5.0 * sweep_raw - 4.0 * up_fold)

    finger_waves = (f_wave_1, f_wave_2, f_wave_3, f_wave_4)
    finger_unfold = (86.0, 78.0, 70.0, 54.0)
    finger_fan = (4.0, -4.0, -10.0, -12.0)
    finger_wave_amp = (10.0, 12.0, 14.0, 14.0)

    for s, sfx in ((1.0, "L"), (-1.0, "R")):
        axis_flap = Vector((0.0, -s, 0.0))
        axis_sweep = Vector((0.0, 0.0, -s))

        r_sh = (
            Matrix.Rotation(math.radians(sh_sweep_deg), 3, axis_sweep)
            @ Matrix.Rotation(math.radians(sh_flap_deg), 3, axis_flap)
            @ Matrix.Rotation(math.radians(sh_twist_deg), 3, axis_pitch_up)
        )
        _set_world_delta_rot(rig_obj, pbones[f"Wing_Shoulder.{sfx}"], r_sh)

        r_el = (
            Matrix.Rotation(math.radians(el_sweep_deg), 3, axis_sweep)
            @ Matrix.Rotation(math.radians(el_flap_deg), 3, axis_flap)
        )
        _set_world_delta_rot(rig_obj, pbones[f"Wing_Elbow.{sfx}"], r_el)

        for fi, (fw, f_unfold, f_fan, fwa) in enumerate(
            zip(finger_waves, finger_unfold, finger_fan, finger_wave_amp), start=1
        ):
            f_flap_deg = spread_blend * f_unfold + wing_amp * (10.0 * down_open - 14.0 * up_fold - fwa * fw)
            f_sweep_deg = spread_blend * f_fan + wing_amp * (4.0 * down_open - 4.0 * fw + 5.0 * landing_flare)
            r_fi = (
                Matrix.Rotation(math.radians(f_sweep_deg), 3, axis_sweep)
                @ Matrix.Rotation(math.radians(f_flap_deg), 3, axis_flap)
            )
            _set_world_delta_rot(rig_obj, pbones[f"Wing_Finger{fi}.{sfx}"], r_fi)

    # Limbs: negative rotation around (-1,0,0) tucks legs BACK (+Y)!
    # Keep crouch mild (stay on ground without floor clipping), then tuck smoothly in air
    leg_tuck = max(0.0, air_blend * (1.0 - 0.72 * landing_flare))
    fore_up_deg = -40.0 * leg_tuck + crouch_amt * 8.0
    fore_lo_deg = 22.0 * leg_tuck - crouch_amt * 14.0
    fore_pw_deg = -32.0 * leg_tuck + crouch_amt * 6.0

    hind_up_deg = -46.0 * leg_tuck + crouch_amt * 10.0
    hind_lo_deg = 26.0 * leg_tuck - crouch_amt * 14.0
    hind_pw_deg = -38.0 * leg_tuck + crouch_amt * 4.0

    for s, sfx in ((1.0, "L"), (-1.0, "R")):
        _set_world_delta_rot(rig_obj, pbones[f"Fore_Upper.{sfx}"], Matrix.Rotation(math.radians(fore_up_deg), 3, axis_pitch_up))
        _set_world_delta_rot(rig_obj, pbones[f"Fore_Lower.{sfx}"], Matrix.Rotation(math.radians(fore_lo_deg), 3, axis_pitch_up))
        _set_world_delta_rot(rig_obj, pbones[f"Fore_Paw.{sfx}"],   Matrix.Rotation(math.radians(fore_pw_deg), 3, axis_pitch_up))

        _set_world_delta_rot(rig_obj, pbones[f"Hind_Upper.{sfx}"], Matrix.Rotation(math.radians(hind_up_deg), 3, axis_pitch_up))
        _set_world_delta_rot(rig_obj, pbones[f"Hind_Lower.{sfx}"], Matrix.Rotation(math.radians(hind_lo_deg), 3, axis_pitch_up))
        _set_world_delta_rot(rig_obj, pbones[f"Hind_Paw.{sfx}"],   Matrix.Rotation(math.radians(hind_pw_deg), 3, axis_pitch_up))


def _apply_folded_wings(rig_obj, fold_blend: float, walk_phase: float = 0.0, walk_amp: float = 0.0):
    """Tuck wings sleekly back along the flanks/back above the haunches and well clear of the floor."""
    pbones = rig_obj.pose.bones
    axis_pitch_up = Vector((-1.0, 0.0, 0.0))
    bounce = walk_amp * math.sin(2.0 * walk_phase)

    sh_flap_deg = -8.0 * fold_blend + 3.0 * bounce
    sh_sweep_deg = -44.0 * fold_blend - 2.0 * bounce
    sh_twist_deg = 12.0 * fold_blend

    el_flap_deg = 8.0 * fold_blend + 2.0 * bounce
    el_sweep_deg = -14.0 * fold_blend

    # Positive flap lifts tips well above the ground and flanks; inner fingers sweep outward (+deg) to clear haunches
    f_flap_fold = (26.0, 32.0, 38.0, 46.0)
    f_sweep_fold = (-16.0, 0.0, 16.0, 32.0)

    for s, sfx in ((1.0, "L"), (-1.0, "R")):
        axis_flap = Vector((0.0, -s, 0.0))
        axis_sweep = Vector((0.0, 0.0, -s))

        r_sh = (
            Matrix.Rotation(math.radians(sh_sweep_deg), 3, axis_sweep)
            @ Matrix.Rotation(math.radians(sh_flap_deg), 3, axis_flap)
            @ Matrix.Rotation(math.radians(sh_twist_deg), 3, axis_pitch_up)
        )
        _set_world_delta_rot(rig_obj, pbones[f"Wing_Shoulder.{sfx}"], r_sh)

        r_el = (
            Matrix.Rotation(math.radians(el_sweep_deg), 3, axis_sweep)
            @ Matrix.Rotation(math.radians(el_flap_deg), 3, axis_flap)
        )
        _set_world_delta_rot(rig_obj, pbones[f"Wing_Elbow.{sfx}"], r_el)

        for fi, (ff, fs) in enumerate(zip(f_flap_fold, f_sweep_fold), start=1):
            r_fi = (
                Matrix.Rotation(math.radians(fs * fold_blend), 3, axis_sweep)
                @ Matrix.Rotation(math.radians(ff * fold_blend + 2.0 * bounce), 3, axis_flap)
            )
            _set_world_delta_rot(rig_obj, pbones[f"Wing_Finger{fi}.{sfx}"], r_fi)


def _apply_walk_pose(rig_obj, walk_phase: float, walk_amp: float, fold_blend: float):
    """Apply grounded quadruped walking gait with wings folded smoothly along the back."""
    pbones = rig_obj.pose.bones
    _reset_pose(rig_obj)

    axis_pitch_up = Vector((-1.0, 0.0, 0.0))
    axis_roll = Vector((0.0, -1.0, 0.0))
    axis_yaw = Vector((0.0, 0.0, 1.0))

    root_z = walk_amp * (0.032 + 0.022 * abs(math.sin(walk_phase)))
    _set_world_delta_loc(rig_obj, pbones["Root"], Vector((0.0, 0.0, root_z)))

    # Torso, neck, and head lively walk sway & step bounce
    s1 = math.sin(walk_phase)
    c1 = math.cos(walk_phase)
    c2 = math.cos(2.0 * walk_phase)

    def _rot_ryp(roll_d, yaw_d, pitch_d):
        return (
            Matrix.Rotation(math.radians(yaw_d), 3, axis_yaw)
            @ Matrix.Rotation(math.radians(roll_d), 3, axis_roll)
            @ Matrix.Rotation(math.radians(pitch_d), 3, axis_pitch_up)
        )

    _set_world_delta_rot(rig_obj, pbones["Pelvis"], _rot_ryp(2.5 * walk_amp * s1, 4.0 * walk_amp * c1, 1.5 * walk_amp * c2))
    _set_world_delta_rot(rig_obj, pbones["Spine"], _rot_ryp(-1.5 * walk_amp * s1, -3.0 * walk_amp * c1, 0.0))
    _set_world_delta_rot(rig_obj, pbones["Chest"], _rot_ryp(-2.5 * walk_amp * s1, -4.0 * walk_amp * c1, 2.5 * walk_amp * abs(s1)))

    head_bob = 4.0 * walk_amp * math.sin(2.0 * walk_phase - 0.5)
    head_yaw = 4.5 * walk_amp * c1
    _set_world_delta_rot(rig_obj, pbones["Neck"], _rot_ryp(1.2 * walk_amp * s1, head_yaw * 0.45, head_bob * 0.45))
    _set_world_delta_rot(rig_obj, pbones["Head"], _rot_ryp(0.0, head_yaw * 0.55, head_bob * 0.55))

    # Cheerful puppy-dragon tail wag
    for idx, t_name in enumerate(("Tail_01", "Tail_02", "Tail_03", "Tail_04")):
        t_yaw = walk_amp * (7.0 + idx * 2.8) * math.sin(walk_phase - idx * 0.65)
        t_pitch = walk_amp * 3.5 * math.cos(2.0 * walk_phase - idx * 0.5)
        r_tail = Matrix.Rotation(math.radians(t_yaw), 3, axis_yaw) @ Matrix.Rotation(math.radians(-t_pitch), 3, axis_pitch_up)
        _set_world_delta_rot(rig_obj, pbones[t_name], r_tail)

    # Diagonal 4-beat walk cycle: swinging legs lift forward (+u_deg), keeping lower joints flexed forward
    # (never un-bending through vertical into the floor!) and paws parallel during stance.
    leg_phases = (
        ("L", "Fore", walk_phase),
        ("R", "Hind", walk_phase + 0.35),
        ("R", "Fore", walk_phase + math.pi),
        ("L", "Hind", walk_phase + math.pi + 0.35),
    )
    for sfx, kind, p in leg_phases:
        swing = max(0.0, math.sin(p))
        stride = -math.cos(p)
        if kind == "Fore":
            u_deg = walk_amp * (11.0 * stride + 26.0 * swing)
            l_deg = walk_amp * (4.0 * stride - 10.0 * swing)
            p_deg = -(u_deg + l_deg) * 0.92 - walk_amp * 10.0 * swing
        else:
            u_deg = walk_amp * (12.0 * stride + 28.0 * swing)
            l_deg = walk_amp * (5.0 * stride - 8.0 * swing)
            p_deg = -(u_deg + l_deg) * 0.92 - walk_amp * 10.0 * swing

        _set_world_delta_rot(rig_obj, pbones[f"{kind}_Upper.{sfx}"], Matrix.Rotation(math.radians(u_deg), 3, axis_pitch_up))
        _set_world_delta_rot(rig_obj, pbones[f"{kind}_Lower.{sfx}"], Matrix.Rotation(math.radians(l_deg), 3, axis_pitch_up))
        _set_world_delta_rot(rig_obj, pbones[f"{kind}_Paw.{sfx}"],   Matrix.Rotation(math.radians(p_deg), 3, axis_pitch_up))

    _apply_folded_wings(rig_obj, fold_blend=fold_blend, walk_phase=walk_phase, walk_amp=walk_amp)


def _bake_takeoff_and_fly(rig_obj):
    act = bpy.data.actions.new("Takeoff_And_Fly")
    act.use_fake_user = True
    _bind_action(rig_obj, act)

    period = 24.0
    key_markers = [1, 12, 14, 24, 32, 96, 112, 126, 144, 160, 170, 180]
    frames = sorted(set(list(range(1, 181, 2)) + key_markers))
    for f in frames:
        if f == 1 or f == 180:
            _apply_flight_pose(rig_obj, phase=0.0, wing_amp=0.0, air_blend=0.0, root_z=0.0)
        elif f <= 112:
            # Phase A (1..112): Unfurl -> takeoff -> full-span soaring flight -> landing touchdown at f=112
            air_up = smoothstep(13.0, 30.0, float(f))
            air_dn = 1.0 - smoothstep(94.0, 111.0, float(f))
            air_blend = air_up * air_dn

            spread_blend = smoothstep(2.0, 14.0, float(f)) * (1.0 - smoothstep(98.0, 112.0, float(f)))
            crouch = smoothstep(2.0, 10.0, float(f)) * (1.0 - smoothstep(10.0, 18.0, float(f)))
            prelift = 24.0 * smoothstep(2.0, 13.0, float(f)) * (1.0 - smoothstep(13.0, 21.0, float(f)))
            landing_flare = smoothstep(92.0, 102.0, float(f)) * (1.0 - smoothstep(102.0, 112.0, float(f)))

            wing_env = smoothstep(11.0, 20.0, float(f)) * (1.0 - smoothstep(96.0, 111.0, float(f)))
            phase = ((float(f) - 8.0) / period) * math.tau
            bob = 0.16 * math.sin(phase - 1.25) * wing_env
            root_z = max(0.0, air_blend * 1.12 + bob)

            _apply_flight_pose(
                rig_obj,
                phase=phase,
                wing_amp=wing_env,
                air_blend=air_blend,
                root_z=root_z,
                spread_blend=spread_blend,
                crouch_amt=crouch,
                prelift_deg=prelift,
                landing_flare=landing_flare,
            )
        else:
            # Phase B (113..180): Fold wings smoothly -> cheerful quadruped walk -> unfold back to rest
            fold_in = smoothstep(112.0, 126.0, float(f))
            fold_out = 1.0 - smoothstep(170.0, 180.0, float(f))
            fold_blend = fold_in * fold_out

            walk_amp = smoothstep(124.0, 132.0, float(f)) * (1.0 - smoothstep(162.0, 171.0, float(f)))
            walk_phase = ((float(f) - 126.0) / 24.0) * math.tau
            _apply_walk_pose(rig_obj, walk_phase=walk_phase, walk_amp=walk_amp, fold_blend=fold_blend)

        _insert_pose_keys(rig_obj, f)
    return act


def _bake_hover_loop(rig_obj):
    act = bpy.data.actions.new("Hover_Fly_Loop")
    act.use_fake_user = True
    _bind_action(rig_obj, act)

    # Exact 2 wingbeats over frames 1..49 so frame 49 == frame 1 seamlessly
    frames = sorted(set(list(range(1, 50, 2)) + [1, 49]))
    for f in frames:
        phase = ((float(f) - 1.0) / 48.0) * (2.0 * math.tau)
        root_z = 1.12 + 0.16 * math.sin(phase - 1.25)
        _apply_flight_pose(
            rig_obj,
            phase=phase,
            wing_amp=1.0,
            air_blend=1.0,
            root_z=root_z,
            spread_blend=1.0,
        )
        _insert_pose_keys(rig_obj, f)
    return act


def _bake_walk_loop(rig_obj):
    act = bpy.data.actions.new("Walk_Loop")
    act.use_fake_user = True
    _bind_action(rig_obj, act)

    # Exact 2 quadruped strides over frames 1..49 so frame 49 == frame 1 seamlessly
    frames = sorted(set(list(range(1, 50, 2)) + [1, 49]))
    for f in frames:
        walk_phase = ((float(f) - 1.0) / 48.0) * (2.0 * math.tau)
        _apply_walk_pose(rig_obj, walk_phase=walk_phase, walk_amp=1.0, fold_blend=1.0)
        _insert_pose_keys(rig_obj, f)
    return act


def build_and_bind_rig(dragon_objs, collection, root):
    """Create Crystal_Dragon_Rig armature and deterministically skin all 12 dragon meshes."""
    rig_obj = _create_armature(collection, root)
    _bind_all_meshes(dragon_objs, rig_obj)
    _audit_rig_binding(dragon_objs, rig_obj)
    return rig_obj


def create_flight_animations(rig_obj, scene):
    """Bake Takeoff_And_Fly (active), Hover_Fly_Loop, and Walk_Loop onto Crystal_Dragon_Rig."""
    _reset_pose(rig_obj)
    act_hover = _bake_hover_loop(rig_obj)
    _reset_pose(rig_obj)
    act_walk = _bake_walk_loop(rig_obj)
    _reset_pose(rig_obj)
    act_takeoff = _bake_takeoff_and_fly(rig_obj)

    ad = rig_obj.animation_data
    for idx, act_extra in enumerate((act_hover, act_walk)):
        track = ad.nla_tracks.new()
        track.name = f"{act_extra.name}_Track"
        track.mute = True
        strip = track.strips.new(act_extra.name, 1 + idx * 100, act_extra)
        if hasattr(strip, "action_slot") and hasattr(act_extra, "slots") and len(act_extra.slots) > 0:
            strip.action_slot = act_extra.slots[0]

    _bind_action(rig_obj, act_takeoff)
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 180
    scene.frame_set(1)
    bpy.context.view_layer.update()
    return act_takeoff, act_hover, act_walk
