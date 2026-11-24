"""Procedural builders for the Crystal Baby Dragon scalloped iridescent wings, finger ribs, and cyan wing veins."""

import math
import bpy
import bmesh
from mathutils import Vector

from baby_math import (
    catmull_rom_sample,
    add_curved_tube,
    add_double_walled_membrane,
    finalize_mesh_object,
)


def _eval_poly(pts, t: float) -> Vector:
    n = len(pts)
    t_c = max(0.0, min(1.0, t)) * (n - 1)
    i0 = min(n - 2, int(t_c))
    f = t_c - i0
    return pts[i0].lerp(pts[i0 + 1], f)


def build_wing_objects(materials: dict, collection, parent):
    """Build Baby_Wings (slots: 0=lavender_scales, 1=wing_membrane, 2=ice_cyan_rune, 3=claw_pearl)."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    for side in (-1.0, 1.0):
        sh_pt = Vector((side * 0.24, -0.58, 1.18))
        elbow_pt = Vector((side * 0.72, -0.48, 1.58))
        wrist_pt = Vector((side * 1.26, -0.42, 2.24))

        tip1 = Vector((side * 2.16, -0.14, 1.12))
        tip2 = Vector((side * 1.82, 0.28, 0.74))
        tip3 = Vector((side * 1.32, 0.58, 0.78))
        tip4 = Vector((side * 0.80, 0.52, 0.90))
        flank_pt = Vector((side * 0.42, 0.12, 0.96))

        # 1. Leading-edge wing arm (shoulder -> elbow -> wrist -> tip1)
        inner_arm_pts = catmull_rom_sample([
            sh_pt,
            Vector((side * 0.48, -0.54, 1.36)),
            elbow_pt,
            Vector((side * 0.98, -0.45, 1.96)),
            wrist_pt,
        ], 20)
        inner_arm_r = [0.092 - 0.022 * (k / 19.0) for k in range(20)]
        add_curved_tube(bm, inner_arm_pts, inner_arm_r, radial_segs=10, uv_layer=uv_layer, mat_index=0)

        outer_arm_pts = catmull_rom_sample([
            wrist_pt,
            Vector((side * 1.62, -0.34, 2.04)),
            Vector((side * 1.95, -0.24, 1.62)),
            tip1,
        ], 20)
        outer_arm_r = [0.062 - 0.048 * ((k / 19.0) ** 0.85) for k in range(20)]
        add_curved_tube(bm, outer_arm_pts, outer_arm_r, radial_segs=9, uv_layer=uv_layer, mat_index=0)

        # 2. Cute curved pearlescent wing-thumb claw hugging the wrist crest
        th_ctrl = [
            wrist_pt + Vector((0.0, -0.01, 0.030)),
            wrist_pt + Vector((-side * 0.025, -0.055, 0.068)),
            wrist_pt + Vector((-side * 0.055, -0.080, 0.085)),
        ]
        add_curved_tube(bm, th_ctrl, [0.032, 0.020, 0.007], radial_segs=8, uv_layer=uv_layer, mat_index=3)

        # 3. Wing-finger ribs radiating from wrist_pt -> tip2, tip3, tip4
        finger2_pts = catmull_rom_sample([
            wrist_pt,
            Vector((side * 1.46, -0.18, 1.66)),
            Vector((side * 1.64, 0.08, 1.18)),
            tip2,
        ], 18)
        finger3_pts = catmull_rom_sample([
            wrist_pt,
            Vector((side * 1.24, -0.06, 1.64)),
            Vector((side * 1.26, 0.28, 1.18)),
            tip3,
        ], 18)
        finger4_pts = catmull_rom_sample([
            wrist_pt,
            Vector((side * 1.02, -0.06, 1.66)),
            Vector((side * 0.90, 0.24, 1.24)),
            tip4,
        ], 18)

        for f_pts in (finger2_pts, finger3_pts, finger4_pts):
            f_r = [0.038 - 0.029 * ((k / 17.0) ** 0.85) for k in range(18)]
            add_curved_tube(bm, f_pts, f_r, radial_segs=8, uv_layer=uv_layer, mat_index=0)

        # 4. Exposed Glowing Ice-Cyan Wing-Bone Veins (front crest, dorsal-rear crest & finger fork)
        n_in_vein = len(inner_arm_pts) - 3
        cyan_main = [
            p + Vector((0.0, -0.052 + 0.016 * (k / max(1, n_in_vein - 1)), inner_arm_r[k + 3] * 0.96))
            for k, p in enumerate(inner_arm_pts[3:])
        ] + [
            p + Vector((0.0, -0.036 * (1.0 - 0.45 * (k / 16.0)), outer_arm_r[k] * 0.96))
            for k, p in enumerate(outer_arm_pts[:17])
        ]
        n_cm = len(cyan_main)
        cm_r = [0.005 + 0.012 * math.sin((k / float(n_cm - 1)) * math.pi) for k in range(n_cm)]
        add_curved_tube(bm, cyan_main, cm_r, radial_segs=6, uv_layer=uv_layer, mat_index=2)

        # Matching dorsal-rear cyan vein clearly visible in REAR and TOP views
        n_in_rear = len(inner_arm_pts) - 4
        cyan_rear = [
            p + Vector((0.0, 0.044 * (1.0 - 0.18 * (k / max(1, n_in_rear - 1))), inner_arm_r[k + 4] * 0.92))
            for k, p in enumerate(inner_arm_pts[4:])
        ] + [
            p + Vector((0.0, 0.035 * (1.0 - 0.45 * (k / 14.0)), outer_arm_r[k] * 0.92))
            for k, p in enumerate(outer_arm_pts[:15])
        ]
        n_cr = len(cyan_rear)
        cr_r = [0.004 + 0.011 * math.sin((k / float(n_cr - 1)) * math.pi) for k in range(n_cr)]
        add_curved_tube(bm, cyan_rear, cr_r, radial_segs=6, uv_layer=uv_layer, mat_index=2)

        cyan_fork = [
            p + Vector((0.0, -0.018 * (1.0 - 0.35 * (k / 11.0)), 0.040 - 0.024 * (k / 11.0)))
            for k, p in enumerate(finger2_pts[:12])
        ]
        cf_r = [0.014 * (1.0 - 0.72 * (k / 11.0)) for k in range(12)]
        add_curved_tube(bm, cyan_fork, cf_r, radial_segs=6, uv_layer=uv_layer, mat_index=2)

        # 5. Scalloped Iridescent Membrane Bays (Bays 1..3 between wrist-radiating ribs)
        bay_pairs = [
            (outer_arm_pts, finger2_pts, 0.22),
            (finger2_pts,   finger3_pts, 0.24),
            (finger3_pts,   finger4_pts, 0.24),
        ]
        m_segs, n_segs = 12, 12
        for fa_pts, fb_pts, scallop_amp in bay_pairs:
            grid = []
            for i in range(m_segs):
                uf = i / float(m_segs - 1)
                col = []
                for j in range(n_segs):
                    vf = j / float(n_segs - 1)
                    # Start at s=0.04 inside wrist bone to avoid degenerate zero-width apex
                    s_base = 0.04 + vf * 0.96
                    # Concave bat-wing scallop notch along trailing edge
                    notch = math.sin(uf * math.pi) * scallop_amp * (vf ** 2.2)
                    s_eff = max(0.04, s_base - notch)
                    pa = _eval_poly(fa_pts, s_eff)
                    pb = _eval_poly(fb_pts, s_eff)
                    p = pa.lerp(pb, uf)
                    # Aerodynamic camber billow + upward curl at concave notch
                    p.z += 0.045 * math.sin(uf * math.pi) * math.sin(vf * math.pi) + 0.14 * notch
                    p.y += 0.04 * math.sin(uf * math.pi) * vf
                    col.append(p)
                grid.append(col)
            add_double_walled_membrane(bm, grid, thickness=0.016, uv_layer=uv_layer, mat_index=1)

        # 6. Innermost Bay 4 (between body flank & Finger 4, hanging below inner wing arm)
        flank_edge_pts = catmull_rom_sample([
            sh_pt,
            Vector((side * 0.34, -0.22, 1.04)),
            flank_pt,
        ], 18)
        grid4 = []
        for i in range(m_segs):
            uf = i / float(m_segs - 1)
            col = []
            for j in range(n_segs):
                vf = j / float(n_segs - 1)
                # j=0 follows inner_arm_pts (shoulder -> wrist), j=n-1 sweeps scalloped trailing edge (flank -> tip4)
                p_top = _eval_poly(inner_arm_pts, uf)
                p_left = _eval_poly(flank_edge_pts, vf)
                p_right = _eval_poly(finger4_pts, vf)
                p_bot = p_left.lerp(p_right, uf)
                p = p_top.lerp(p_bot, vf)
                # Concave scalloped arch along bottom edge
                notch = math.sin(uf * math.pi) * 0.22 * (vf ** 2.0)
                p = p.lerp(p_top, notch * 0.55)
                p.z += 0.035 * math.sin(uf * math.pi) * math.sin(vf * math.pi) + 0.10 * notch
                col.append(p)
            grid4.append(col)
        add_double_walled_membrane(bm, grid4, thickness=0.016, uv_layer=uv_layer, mat_index=1)

    return finalize_mesh_object(
        "Baby_Wings",
        bm,
        [
            materials["lavender_scales"],
            materials["wing_membrane"],
            materials["ice_cyan_rune"],
            materials["claw_pearl"],
        ],
        collection,
        parent,
    )
