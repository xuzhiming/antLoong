"""Procedural builders for the HD-optimized stocky crouched limbs, paw pads, 20 gold claws, and wavy limb plumes."""

import math
import random
import bpy
import bmesh
from mathutils import Vector

from dragon_spine import (
    spine_point,
    smoothstep,
    loft_sections,
    add_curved_tube,
    add_flame_tuft,
    finalize_mesh_object,
)


def _catmull_rom_sample(ctrl_pts, n_samples: int):
    pts = [Vector(p) for p in ctrl_pts]
    ext = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    out = []
    segs = len(pts) - 1
    for i in range(n_samples):
        tg = (i / float(max(1, n_samples - 1))) * segs
        idx = min(segs - 1, int(tg))
        t = tg - idx
        t2, t3 = t * t, t * t * t
        p0, p1, p2, p3 = ext[idx], ext[idx + 1], ext[idx + 2], ext[idx + 3]
        out.append(0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        ))
    return out


def _build_limb_tube(bm, uv_layer, ctrl_pts, rad_profile, scale_amp=0.013):
    """Build a stocky heavily scaled muscular limb shaft."""
    n_pts = 34
    pts = _catmull_rom_sample(ctrl_pts, n_pts)
    n_rad = 22

    rads = []
    for i in range(n_pts):
        t = i / float(n_pts - 1)
        idx_f = t * (len(rad_profile) - 1)
        i0 = min(len(rad_profile) - 2, int(idx_f))
        f = idx_f - i0
        rads.append(rad_profile[i0] * (1.0 - f) + rad_profile[i0 + 1] * f)

    tangents = []
    for i in range(n_pts):
        t_vec = (pts[min(n_pts - 1, i + 1)] - pts[max(0, i - 1)]).normalized()
        tangents.append(t_vec)

    ref = Vector((0.0, -1.0, 0.0))
    if abs(tangents[0].dot(ref)) > 0.9:
        ref = Vector((1.0, 0.0, 0.0))
    n_vec = (ref - tangents[0] * tangents[0].dot(ref)).normalized()
    b_vec = tangents[0].cross(n_vec).normalized()

    rings = []
    u_coords = []
    for i in range(n_pts):
        t = i / float(n_pts - 1)
        u_coords.append(t * 2.2)
        tan = tangents[i]
        if i > 0:
            n_vec = (n_vec - tan * n_vec.dot(tan))
            if n_vec.length < 1e-6:
                n_vec = Vector((0.0, 1.0, 0.0)).cross(tan)
            n_vec.normalize()
            b_vec = tan.cross(n_vec).normalized()
        r_base = rads[i]
        ring = []
        for j in range(n_rad):
            ang = (j / float(n_rad)) * math.tau
            nu = t * 18.0
            nt = (ang / math.tau) * 10.0
            sc = 0.5 + 0.5 * math.sin(nu * math.tau + math.cos(nt * math.tau))
            r_eff = r_base + scale_amp * sc * smoothstep(0.08, 0.25, t) * (1.0 - smoothstep(0.82, 1.0, t))
            off = n_vec * (math.cos(ang) * r_eff) + b_vec * (math.sin(ang) * r_eff * 0.92)
            ring.append(pts[i] + off)
        rings.append(ring)

    loft_sections(bm, rings, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0, u_coords=u_coords)
    return pts


def _add_paw_and_toes(bm_limbs, uv_limbs, bm_claws, uv_claws, side, wrist_pt, paw_center, yaw_deg=30.0, is_fore=True):
    """Build broad heavy palm, central + toe bronze paw pads, 5 thick splayed toes, and 5 bold golden claws."""
    paw_yaw = math.radians(yaw_deg) * side
    fwd = Vector((math.sin(paw_yaw), -math.cos(paw_yaw), 0.0)).normalized()
    right = Vector((math.cos(paw_yaw), math.sin(paw_yaw), 0.0)).normalized()
    up = Vector((0.0, 0.0, 1.0))

    palm_rings = []
    palm_specs = [
        (wrist_pt + Vector((0.0, 0.0, 0.02)), 0.13, 0.095),
        (wrist_pt.lerp(paw_center, 0.55) + Vector((0.0, 0.0, 0.02)), 0.18, 0.090),
        (paw_center + Vector((0.0, 0.0, 0.008)), 0.22, 0.078),
        (paw_center + fwd * 0.11, 0.17, 0.056),
    ]
    for c_pt, rw, rh in palm_specs:
        ring = []
        for j in range(14):
            ang = (j / 14.0) * math.tau
            px = math.cos(ang) * rw
            pz = math.sin(ang) * rh
            if pz < 0.0:
                pz *= 0.55
            ring.append(c_pt + right * px + up * pz)
        palm_rings.append(ring)
    loft_sections(bm_limbs, palm_rings, cap_start=True, cap_end=True, uv_layer=uv_limbs, mat_index=0)

    pad_center = Vector((paw_center.x, paw_center.y, 0.052))
    pad_pts = [
        pad_center - fwd * 0.06 + Vector((0.0, 0.0, 0.015)),
        pad_center,
        pad_center + fwd * 0.08 + Vector((0.0, 0.0, 0.005)),
    ]
    add_curved_tube(
        bm_limbs,
        pad_pts,
        [0.07, 0.12, 0.07],
        radial_segs=10,
        uv_layer=uv_limbs,
        mat_index=1,
        rx_scale=1.25,
        rz_scale=0.35,
    )

    if is_fore:
        toe_configs = [
            (-56.0, 0.21, 0.050, 0.175),
            (-22.0, 0.29, 0.058, 0.215),
            (  5.0, 0.33, 0.064, 0.235),
            ( 32.0, 0.29, 0.058, 0.215),
            ( 60.0, 0.24, 0.052, 0.185),
        ]
    else:
        toe_configs = [
            (-54.0, 0.20, 0.050, 0.175),
            (-20.0, 0.29, 0.058, 0.215),
            (  6.0, 0.33, 0.064, 0.235),
            ( 32.0, 0.29, 0.058, 0.215),
            ( 58.0, 0.23, 0.052, 0.185),
        ]

    for rel_deg, t_len, t_rad, c_len in toe_configs:
        ang_rad = paw_yaw + math.radians(rel_deg) * side
        t_dir = Vector((math.sin(ang_rad), -math.cos(ang_rad), 0.0)).normalized()
        t_root = Vector((paw_center.x, paw_center.y, 0.092)) + t_dir * 0.11

        t_pts = [
            t_root,
            t_root + t_dir * (t_len * 0.36) + Vector((0.0, 0.0, 0.022)),
            t_root + t_dir * (t_len * 0.72) + Vector((0.0, 0.0, -0.002)),
            t_root + t_dir * t_len + Vector((0.0, 0.0, -0.012)),
        ]
        t_rads = [t_rad, t_rad * 0.92, t_rad * 0.85, t_rad * 0.70]
        add_curved_tube(bm_limbs, t_pts, t_rads, radial_segs=10, uv_layer=uv_limbs, mat_index=0)

        toe_tip = t_pts[-1]
        tpad_pts = [
            Vector((toe_tip.x, toe_tip.y, 0.042)) - t_dir * 0.05,
            Vector((toe_tip.x, toe_tip.y, 0.038)),
            Vector((toe_tip.x, toe_tip.y, 0.042)) + t_dir * 0.03,
        ]
        add_curved_tube(
            bm_limbs,
            tpad_pts,
            [t_rad * 0.65, t_rad * 0.85, t_rad * 0.55],
            radial_segs=8,
            uv_layer=uv_limbs,
            mat_index=1,
            rx_scale=1.1,
            rz_scale=0.42,
        )

        claw_start = toe_tip + Vector((0.0, 0.0, 0.006))
        claw_mid1 = claw_start + t_dir * (c_len * 0.40) + Vector((0.0, 0.0, 0.022))
        claw_mid2 = claw_start + t_dir * (c_len * 0.78) + Vector((0.0, 0.0, -0.024))
        claw_tip = Vector((
            claw_start.x + t_dir.x * c_len,
            claw_start.y + t_dir.y * c_len,
            0.002,
        ))
        add_curved_tube(
            bm_claws,
            [claw_start, claw_mid1, claw_mid2, claw_tip],
            [t_rad * 0.64, t_rad * 0.46, t_rad * 0.24, 0.004],
            radial_segs=8,
            cap_start=True,
            cap_end=True,
            uv_layer=uv_claws,
            mat_index=0,
            rx_scale=0.84,
            rz_scale=1.15,
        )


def build_limb_objects(materials: dict, collection, parent):
    """Build stocky crouched Dragon_Limbs, bold Dragon_Claws, and lush Dragon_Limb_Tufts."""
    bm_limbs = bmesh.new()
    uv_limbs = bm_limbs.loops.layers.uv.new("UVMap")

    bm_claws = bmesh.new()
    uv_claws = bm_claws.loops.layers.uv.new("UVMap")

    bm_tufts = bmesh.new()
    uv_tufts = bm_tufts.loops.layers.uv.new("UVMap")
    rng = random.Random(515)

    p_shoulder_base = spine_point(0.245)
    p_hip_base = spine_point(0.695)

    for side in (-1.0, 1.0):
        # ================= FORELIMB (stocky crouched muscular foreleg) =================
        sh_anchor = p_shoulder_base + Vector((side * 0.32, 0.02, -0.04))
        elbow = Vector((side * 0.84, -2.04, 0.50))
        forearm_mid = Vector((side * 0.80, -2.38, 0.34))
        wrist = Vector((side * 0.76, -2.66, 0.19))
        fore_paw = Vector((side * 0.80, -2.84, 0.105))

        fore_ctrl = [sh_anchor, Vector((side * 0.62, -2.26, 0.76)), elbow, forearm_mid, wrist, fore_paw]
        fore_rad = [0.27, 0.22, 0.185, 0.175, 0.140, 0.115]
        _build_limb_tube(bm_limbs, uv_limbs, fore_ctrl, fore_rad, scale_amp=0.013)

        # Subtle upper-shoulder cyan cyber fleck only (no racing stripes on forearm)
        sh_cyan = _catmull_rom_sample([
            sh_anchor + Vector((side * 0.15, -0.10, 0.04)),
            sh_anchor + Vector((side * 0.28, -0.02, -0.10)),
            Vector((side * 0.72, -2.14, 0.62)),
        ], 12)
        sc_r = [0.002 + 0.0055 * math.sin((k / 11.0) * math.pi) for k in range(12)]
        add_curved_tube(bm_limbs, sh_cyan, sc_r, radial_segs=6, uv_layer=uv_limbs, mat_index=2)

        _add_paw_and_toes(bm_limbs, uv_limbs, bm_claws, uv_claws, side, wrist, fore_paw, yaw_deg=28.0, is_fore=True)

        # Lush wavy forelimb elbow flame tufts (layered outer+rear so visible in FRONT & LEFT)
        for k in range(15):
            kt = k / 14.0
            root = elbow.lerp(forearm_mid, kt * 0.45) + Vector((side * (0.10 + 0.03 * (k % 2)), 0.10, 0.06 - kt * 0.05))
            f_dir = Vector((side * (0.34 + 0.18 * kt), 0.78, 0.52 - 0.16 * kt)).normalized()
            add_flame_tuft(
                bm_tufts,
                root=root,
                forward_dir=f_dir,
                up_dir=Vector((0.0, 0.2, 1.0)),
                length=0.40 - 0.08 * kt + rng.uniform(-0.02, 0.03),
                width=0.130,
                thickness=0.066,
                curl_up=0.20,
                curl_side=side * 0.07,
                segs=7,
                uv_layer=uv_tufts,
                mat_index=0,
                s_wave=0.08,
            )

        # ================= HINDLIMB (stocky crouched muscular hindleg) =================
        hip_anchor = p_hip_base + Vector((side * 0.30, -0.02, -0.04))
        knee = Vector((side * 0.92, 2.18, 0.68))
        hock = Vector((side * 1.00, 2.34, 0.28))
        ankle = Vector((side * 1.04, 1.98, 0.17))
        hind_paw = Vector((side * 1.08, 1.74, 0.105))

        hind_ctrl = [hip_anchor, Vector((side * 0.66, 2.04, 1.04)), knee, hock, ankle, hind_paw]
        hind_rad = [0.29, 0.23, 0.195, 0.155, 0.130, 0.115]
        _build_limb_tube(bm_limbs, uv_limbs, hind_ctrl, hind_rad, scale_amp=0.013)

        # Subtle upper-hip cyan cyber fleck only (no racing stripes on shin)
        hip_cyan = _catmull_rom_sample([
            hip_anchor + Vector((side * 0.17, -0.05, 0.05)),
            hip_anchor + Vector((side * 0.32, 0.04, -0.14)),
            Vector((side * 0.82, 2.12, 0.84)),
        ], 12)
        hc_r = [0.002 + 0.006 * math.sin((k / 11.0) * math.pi) for k in range(12)]
        add_curved_tube(bm_limbs, hip_cyan, hc_r, radial_segs=6, uv_layer=uv_limbs, mat_index=2)

        _add_paw_and_toes(bm_limbs, uv_limbs, bm_claws, uv_claws, side, ankle, hind_paw, yaw_deg=36.0, is_fore=False)

        # Lush wavy hindlimb thigh & hock flame tufts
        for k in range(14):
            kt = k / 13.0
            root = knee.lerp(hock, kt * 0.85) + Vector((side * 0.09, 0.11, 0.05))
            f_dir = Vector((side * (0.28 + 0.15 * kt), 0.84, 0.44 - 0.20 * kt)).normalized()
            add_flame_tuft(
                bm_tufts,
                root=root,
                forward_dir=f_dir,
                up_dir=Vector((0.0, 0.2, 1.0)),
                length=0.40 - 0.09 * kt + rng.uniform(-0.02, 0.03),
                width=0.115,
                thickness=0.060,
                curl_up=0.20,
                curl_side=side * 0.07,
                segs=7,
                uv_layer=uv_tufts,
                mat_index=0,
                s_wave=0.08,
            )

    z_min_claw = min((v.co.z for v in bm_claws.verts), default=0.0)
    if abs(z_min_claw) > 1e-4:
        for v in bm_claws.verts:
            v.co.z -= z_min_claw

    obj_limbs = finalize_mesh_object(
        "Dragon_Limbs",
        bm_limbs,
        [materials["dark_scales"], materials["belly_scutes"], materials["cyan_glow"]],
        collection,
        parent,
    )
    obj_claws = finalize_mesh_object(
        "Dragon_Claws",
        bm_claws,
        [materials["antique_gold"]],
        collection,
        parent,
    )
    obj_tufts = finalize_mesh_object(
        "Dragon_Limb_Tufts",
        bm_tufts,
        [materials["magenta_mane"]],
        collection,
        parent,
    )
    return [obj_limbs, obj_claws, obj_tufts]
