"""Procedural builders for the Crystal Baby Dragon four chubby legs, pink toe-bean paw pads, and pearl claws."""

import math
import bpy
import bmesh
from mathutils import Vector

from baby_math import (
    smoothstep,
    catmull_rom_sample,
    loft_sections,
    add_curved_tube,
    finalize_mesh_object,
)


def _build_chubby_leg_tube(bm, uv_layer, ctrl_pts, rad_profile, shoulder_flare: bool = True):
    n_pts = 30
    pts = catmull_rom_sample(ctrl_pts, n_pts)
    n_rad = 22

    rads = []
    for i in range(n_pts):
        t = i / float(n_pts - 1)
        idx_f = t * (len(rad_profile) - 1)
        i0 = min(len(rad_profile) - 2, int(idx_f))
        f = idx_f - i0
        rads.append(rad_profile[i0] * (1.0 - f) + rad_profile[i0 + 1] * f)

    tangents = [(pts[min(n_pts - 1, i + 1)] - pts[max(0, i - 1)]).normalized() for i in range(n_pts)]
    ref = Vector((0.0, -1.0, 0.0))
    if abs(tangents[0].dot(ref)) > 0.9:
        ref = Vector((1.0, 0.0, 0.0))
    n_vec = (ref - tangents[0] * tangents[0].dot(ref)).normalized()
    b_vec = tangents[0].cross(n_vec).normalized()

    rings = []
    u_coords = []
    v_coords = [(j / float(n_rad)) * 0.5 for j in range(n_rad + 1)]
    for i in range(n_pts):
        t = i / float(n_pts - 1)
        u_coords.append(t * 0.85)
        tan = tangents[i]
        if i > 0:
            n_vec = (n_vec - tan * n_vec.dot(tan))
            if n_vec.length < 1e-6:
                n_vec = Vector((0.0, 1.0, 0.0)).cross(tan)
            n_vec.normalize()
            b_vec = tan.cross(n_vec).normalized()
        r_base = rads[i]
        sh_t = 1.0 - smoothstep(0.0, 0.36, t)
        fwd_scale = 1.0 + (0.24 if shoulder_flare else 0.14) * sh_t
        side_scale = 0.94 - 0.06 * sh_t
        ring = []
        for j in range(n_rad):
            ang = (j / float(n_rad)) * math.tau
            sc = 0.5 + 0.5 * math.sin(t * 14.0 * math.tau + math.cos((ang / math.tau) * 8.0 * math.tau))
            r_eff = r_base + 0.003 * sc * smoothstep(0.08, 0.25, t) * (1.0 - smoothstep(0.80, 1.0, t))
            ring.append(pts[i] + n_vec * (math.cos(ang) * r_eff * fwd_scale) + b_vec * (math.sin(ang) * r_eff * side_scale))
        rings.append(ring)

    loft_sections(
        bm,
        rings,
        cap_start=True,
        cap_end=True,
        uv_layer=uv_layer,
        mat_index=0,
        u_coords=u_coords,
        v_coords=v_coords,
    )


def _add_puppy_paw(bm_limbs, uv_limbs, bm_claws, uv_claws, side, wrist_pt, paw_center, yaw_deg=12.0):
    """Build plump paw, clover-pink central pad + 4 pink toe beans, 4 stubby toes, and 4 pearl baby claws."""
    paw_yaw = math.radians(yaw_deg) * side
    fwd = Vector((math.sin(paw_yaw), -math.cos(paw_yaw), 0.0)).normalized()
    right = Vector((math.cos(paw_yaw), math.sin(paw_yaw), 0.0)).normalized()
    up = Vector((0.0, 0.0, 1.0))

    # Plump rounded paw cushion
    palm_specs = [
        (wrist_pt + Vector((0.0, 0.0, 0.01)), 0.145, 0.125),
        (wrist_pt.lerp(paw_center, 0.55),     0.170, 0.110),
        (paw_center + Vector((0.0, 0.0, -0.01)), 0.185, 0.095),
        (paw_center + fwd * 0.08 + Vector((0.0, 0.0, -0.02)), 0.155, 0.075),
    ]
    palm_rings = []
    for c_pt, rw, rh in palm_specs:
        ring = []
        for j in range(14):
            ang = (j / 14.0) * math.tau
            px = math.cos(ang) * rw
            pz = math.sin(ang) * rh
            if pz < 0.0:
                pz *= 0.62
            ring.append(c_pt + right * px + up * pz)
        palm_rings.append(ring)
    loft_sections(
        bm_limbs,
        palm_rings,
        cap_start=True,
        cap_end=True,
        uv_layer=uv_limbs,
        mat_index=0,
        u_coords=[(i / 3.0) * 0.22 for i in range(4)],
        v_coords=[(j / 14.0) * 0.25 for j in range(15)],
    )

    # Big cute pink central clover palm pad (prominent in BOTTOM)
    pad_c = Vector((paw_center.x, paw_center.y, 0.048))
    add_curved_tube(
        bm_limbs,
        [pad_c - fwd * 0.045, pad_c, pad_c + fwd * 0.055],
        [0.068, 0.105, 0.075],
        radial_segs=10,
        uv_layer=uv_limbs,
        mat_index=1,
        rx_scale=1.22,
        rz_scale=0.38,
    )

    # 4 cute forward-splayed chubby toes + 4 pink toe beans + 4 stubby pearl baby claws
    toe_configs = [
        (-28.0, 0.135, 0.056, 0.054),
        ( -9.0, 0.155, 0.062, 0.060),
        ( 10.0, 0.155, 0.062, 0.060),
        ( 29.0, 0.135, 0.056, 0.054),
    ]
    for rel_deg, t_len, t_rad, c_len in toe_configs:
        ang_rad = paw_yaw + math.radians(rel_deg) * side
        t_dir = Vector((math.sin(ang_rad), -math.cos(ang_rad), 0.0)).normalized()
        t_root = Vector((paw_center.x, paw_center.y, 0.085)) + t_dir * 0.095

        t_pts = [
            t_root,
            t_root + t_dir * (t_len * 0.45) + Vector((0.0, 0.0, 0.008)),
            t_root + t_dir * t_len - Vector((0.0, 0.0, 0.015)),
        ]
        t_rads = [t_rad, t_rad * 0.94, t_rad * 0.78]
        add_curved_tube(bm_limbs, t_pts, t_rads, radial_segs=10, uv_layer=uv_limbs, mat_index=0)

        # Adorable pink oval toe-bean cushion underneath each toe
        toe_tip = t_pts[-1]
        bean_c = Vector((toe_tip.x, toe_tip.y, 0.036)) - t_dir * 0.025
        add_curved_tube(
            bm_limbs,
            [bean_c - t_dir * 0.028, bean_c, bean_c + t_dir * 0.026],
            [t_rad * 0.55, t_rad * 0.78, t_rad * 0.50],
            radial_segs=8,
            uv_layer=uv_limbs,
            mat_index=1,
            rx_scale=1.10,
            rz_scale=0.42,
        )

        # Smooth stubby rounded pearl baby claw resting cleanly at z = 0.002
        cl_start = toe_tip + Vector((0.0, 0.0, 0.004))
        cl_mid = cl_start + t_dir * (c_len * 0.55) + Vector((0.0, 0.0, 0.008))
        cl_tip = Vector((cl_start.x + t_dir.x * c_len, cl_start.y + t_dir.y * c_len, 0.012))
        add_curved_tube(
            bm_claws,
            [cl_start, cl_mid, cl_tip],
            [t_rad * 0.56, t_rad * 0.44, 0.016],
            radial_segs=8,
            uv_layer=uv_claws,
            mat_index=0,
            rx_scale=0.92,
            rz_scale=1.08,
        )


def build_limb_objects(materials: dict, collection, parent):
    """Build Baby_Limbs and Baby_Claws."""
    bm_limbs = bmesh.new()
    uv_limbs = bm_limbs.loops.layers.uv.new("UVMap")

    bm_claws = bmesh.new()
    uv_claws = bm_claws.loops.layers.uv.new("UVMap")

    for side in (-1.0, 1.0):
        # Foreleg: high chest-anchored shoulder -> deltoid -> tucked puppy elbow -> forward forearm -> paw
        fore_ctrl = [
            Vector((side * 0.24, -0.70, 1.05)),
            Vector((side * 0.33, -0.68, 0.86)),
            Vector((side * 0.34, -0.65, 0.56)),
            Vector((side * 0.34, -0.78, 0.32)),
            Vector((side * 0.34, -0.87, 0.14)),
        ]
        fore_rad = [0.225, 0.210, 0.185, 0.168, 0.150]
        _build_chubby_leg_tube(bm_limbs, uv_limbs, fore_ctrl, fore_rad, shoulder_flare=True)
        _add_puppy_paw(bm_limbs, uv_limbs, bm_claws, uv_claws, side, fore_ctrl[-2], fore_ctrl[-1], yaw_deg=8.0)

        # Hindleg: plump rounded haunch bulging warmly outward + cute compact paw
        hind_ctrl = [
            Vector((side * 0.36, 0.20, 0.78)),
            Vector((side * 0.52, 0.18, 0.64)),
            Vector((side * 0.48, 0.36, 0.36)),
            Vector((side * 0.45, 0.24, 0.22)),
            Vector((side * 0.44, 0.16, 0.13)),
        ]
        hind_rad = [0.24, 0.24, 0.17, 0.155, 0.145]
        _build_chubby_leg_tube(bm_limbs, uv_limbs, hind_ctrl, hind_rad, shoulder_flare=False)
        _add_puppy_paw(bm_limbs, uv_limbs, bm_claws, uv_claws, side, hind_ctrl[-2], hind_ctrl[-1], yaw_deg=14.0)

    # Guarantee exact ground contact (paw bottoms / claws cleanly at z=0)
    z_min = min(min((v.co.z for v in bm_limbs.verts), default=0.0), min((v.co.z for v in bm_claws.verts), default=0.0))
    if abs(z_min) > 1e-4:
        for v in bm_limbs.verts:
            v.co.z -= z_min
        for v in bm_claws.verts:
            v.co.z -= z_min

    obj_limbs = finalize_mesh_object(
        "Baby_Limbs",
        bm_limbs,
        [materials["lavender_scales"], materials["pink_pad"]],
        collection,
        parent,
    )
    obj_claws = finalize_mesh_object(
        "Baby_Claws",
        bm_claws,
        [materials["claw_pearl"]],
        collection,
        parent,
    )
    return [obj_limbs, obj_claws]
