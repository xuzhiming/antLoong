"""Procedural builders for the HD-optimized cyberpunk dragon head, eyes, antlers, whiskers, silver collar/beard, and wavy mane."""

import math
import random
import bpy
import bmesh
from mathutils import Vector

from dragon_spine import (
    spine_point,
    spine_frame,
    body_surface_point,
    smoothstep,
    loft_sections,
    add_curved_tube,
    add_flame_tuft,
    finalize_mesh_object,
)


def _catmull_rom_sample(ctrl_pts, n_samples: int):
    pts = [Vector(p) for p in ctrl_pts]
    if len(pts) < 2:
        return pts
    ext = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    out = []
    segs = len(pts) - 1
    for i in range(n_samples):
        t_global = (i / float(max(1, n_samples - 1))) * segs
        idx = min(segs - 1, int(t_global))
        t = t_global - idx
        t2, t3 = t * t, t * t * t
        p0, p1, p2, p3 = ext[idx], ext[idx + 1], ext[idx + 2], ext[idx + 3]
        val = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(val)
    return out


def build_head_objects(materials: dict, collection, parent):
    """Build Dragon_Head, Dragon_Eyes, Dragon_Antlers, Dragon_Beard, Dragon_Whiskers, Dragon_Mane."""
    objs = []
    objs.append(_build_head_and_jaw(materials, collection, parent))
    objs.append(_build_eyes(materials, collection, parent))
    objs.append(_build_antlers(materials, collection, parent))
    objs.append(_build_beard_and_silver_collar(materials, collection, parent))
    objs.append(_build_whiskers(materials, collection, parent))
    objs.append(_build_mane(materials, collection, parent))
    return objs


def _build_head_and_jaw(materials, collection, parent):
    """Build sculpted upper cranium/snout, closed snarling mandible, nostrils, fangs, and subtle cyan veins."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    n_y = 36
    n_t = 40
    rings = []
    u_coords = []

    for i in range(n_y):
        t = i / float(n_y - 1)
        y = -5.00 + t * 1.62
        u_coords.append(t * 1.4)

        if t < 0.25:
            z_center = 2.02 + 0.03 * smoothstep(0.0, 0.25, t)
        else:
            z_center = 2.05 - 0.05 * smoothstep(0.65, 1.0, t)

        snout_w = 0.15 + 0.10 * smoothstep(0.0, 0.18, t)
        cheek_w = 0.18 * smoothstep(0.32, 0.72, t)
        neck_blend = -0.04 * smoothstep(0.82, 1.0, t)
        rx = snout_w + cheek_w + neck_blend

        rz_top = 0.105 + 0.145 * smoothstep(0.15, 0.65, t) + 0.04 * smoothstep(0.70, 1.0, t)
        rz_bot = 0.085 + 0.055 * smoothstep(0.20, 0.75, t) + 0.075 * smoothstep(0.75, 1.0, t)

        nasal_bump = 0.0
        if 0.08 < t < 0.56:
            nasal_bump = 0.018 * max(0.0, math.sin((t - 0.08) * 32.0))

        brow_w = smoothstep(0.30, 0.46, t) * (1.0 - smoothstep(0.66, 0.84, t))
        socket_w = smoothstep(0.36, 0.46, t) * (1.0 - smoothstep(0.57, 0.66, t))
        nostril_w = smoothstep(0.05, 0.11, t) * (1.0 - smoothstep(0.15, 0.23, t))

        ring = []
        for j in range(n_t):
            theta = -math.pi + (j / float(n_t)) * math.tau
            c = math.cos(theta)
            s = math.sin(theta)
            abs_c = abs(c)

            rz = rz_top if s >= 0.0 else rz_bot
            px = c * rx
            pz = s * rz

            # Snarling upper lip wave along mouth seam (t in 0.10..0.55)
            if -0.45 < s < 0.0 and 0.08 < t < 0.58:
                lip_wave = 0.014 * math.sin((t - 0.08) * 24.0)
                pz += lip_wave * (1.0 - abs_c * 0.4)

            if s < -0.25 and t < 0.75:
                pz *= 0.70

            if s > 0.65:
                pz += nasal_bump * ((s - 0.65) / 0.35)

            tip_box = (1.0 - smoothstep(0.0, 0.45, t)) * 0.14
            if s > 0.0:
                px *= (1.0 + tip_box * (1.0 - abs_c) * s)

            # Bulbous flared nostrils
            if nostril_w > 0.01 and abs_c > 0.35 and -0.1 < s < 0.75:
                n_dome = math.exp(-((s - 0.25) / 0.32) ** 2) * math.exp(-((abs_c - 0.78) / 0.25) ** 2)
                px += math.copysign(0.062 * nostril_w * n_dome, c)
                pz += 0.036 * nostril_w * n_dome

            # Deep heavy V-shaped angled supraorbital brow ridge
            if brow_w > 0.01 and s > 0.14:
                b_arch = math.exp(-((s - 0.54) / 0.26) ** 2) * math.exp(-((abs_c - 0.66) / 0.28) ** 2)
                scowl_lift = -0.055 + 0.115 * smoothstep(0.34, 0.62, t)
                px += math.copysign(0.118 * brow_w * b_arch, c)
                pz += (0.060 + scowl_lift) * brow_w * b_arch

            # Deep concave orbital socket pocket
            if socket_w > 0.01 and -0.14 < s < 0.36 and abs_c > 0.50:
                sock_d = math.exp(-((s - 0.12) / 0.18) ** 2) * math.exp(-((abs_c - 0.86) / 0.22) ** 2)
                px -= math.copysign(0.095 * socket_w * sock_d, c)

            # High cheekbone armor ridge
            if brow_w > 0.01 and -0.45 < s < 0.0 and abs_c > 0.55:
                cheek_d = math.exp(-((s + 0.20) / 0.18) ** 2)
                px += math.copysign(0.075 * brow_w * cheek_d, c)
                pz -= 0.012 * brow_w * cheek_d

            if brow_w > 0.01 and s > 0.85:
                pz += 0.040 * brow_w * ((s - 0.85) / 0.15)

            ring.append(Vector((px, y, z_center + pz)))
        rings.append(ring)

    v_coords = [j / float(n_t) for j in range(n_t + 1)]
    loft_sections(bm, rings, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0, u_coords=u_coords, v_coords=v_coords)

    # Lower Jaw (Mandible) - raised cleanly into a closed fierce snarling frown (no black open mouth gap)
    n_jy = 26
    n_jt = 28
    j_rings = []
    for i in range(n_jy):
        t = i / float(n_jy - 1)
        y = -4.92 + t * 1.44
        # Raised z_j to nest seamlessly into upper jaw lip seam
        z_j = 1.90 - 0.025 * math.sin(t * math.pi * 0.85) + 0.02 * smoothstep(0.7, 1.0, t)
        rx_j = 0.115 + 0.195 * smoothstep(0.0, 0.72, t) + 0.03 * smoothstep(0.72, 1.0, t)
        rz_j = 0.072 + 0.078 * smoothstep(0.15, 0.85, t)
        ring = []
        for j in range(n_jt):
            theta = -math.pi + (j / float(n_jt)) * math.tau
            c = math.cos(theta)
            s = math.sin(theta)
            px = c * rx_j
            pz = s * (rz_j * (0.68 if s > 0.0 else 1.05))
            if s < -0.6 and t < 0.4:
                pz -= 0.022 * (1.0 - t / 0.4) * ((-s - 0.6) / 0.4)
            ring.append(Vector((px, y, z_j + pz)))
        j_rings.append(ring)

    loft_sections(bm, j_rings, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0)

    # Sculpted Nostril Scrolls
    for side in (-1.0, 1.0):
        scroll_pts = _catmull_rom_sample([
            (side * 0.10, -4.90, 2.06),
            (side * 0.18, -4.84, 2.12),
            (side * 0.21, -4.75, 2.08),
            (side * 0.15, -4.72, 2.02),
        ], 12)
        scroll_r = [0.018, 0.024, 0.028, 0.026, 0.023, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008]
        add_curved_tube(bm, scroll_pts, scroll_r, radial_segs=8, uv_layer=uv_layer, mat_index=0)

    # Subtle Interlocking Corner Fangs peeking outside closed snarl
    fang_specs = [
        ( 0.145, -4.78, 1.94,  0.01,  0.00, -0.068, 0.017),
        (-0.145, -4.78, 1.94, -0.01,  0.00, -0.068, 0.017),
        ( 0.205, -4.52, 1.94,  0.01,  0.01, -0.056, 0.014),
        (-0.205, -4.52, 1.94, -0.01,  0.01, -0.056, 0.014),
    ]
    for fx, fy, fz, dx, dy, dz, fr in fang_specs:
        p0 = Vector((fx, fy, fz))
        p1 = p0 + Vector((dx * 0.5, dy * 0.5, dz * 0.55))
        p2 = p0 + Vector((dx, dy, dz))
        add_curved_tube(bm, [p0, p1, p2], [fr, fr * 0.55, 0.003], radial_segs=7, uv_layer=uv_layer, mat_index=3)

    # Delicate Facial Cyan Cyber Veins tucked neatly into brow & cheek grooves
    for side in (-1.0, 1.0):
        sub_eye = _catmull_rom_sample([
            (side * 0.25, -4.46, 2.00),
            (side * 0.34, -4.24, 1.97),
            (side * 0.42, -3.96, 1.97),
            (side * 0.43, -3.68, 2.00),
        ], 18)
        r_se = [0.0025 + 0.0055 * math.sin((k / 17.0) * math.pi) for k in range(18)]
        add_curved_tube(bm, sub_eye, r_se, radial_segs=6, uv_layer=uv_layer, mat_index=1)

        brow_line = _catmull_rom_sample([
            (side * 0.16, -4.46, 2.15),
            (side * 0.30, -4.24, 2.24),
            (side * 0.39, -3.96, 2.26),
        ], 14)
        r_bl = [0.0025 + 0.0045 * math.sin((k / 13.0) * math.pi) for k in range(14)]
        add_curved_tube(bm, brow_line, r_bl, radial_segs=6, uv_layer=uv_layer, mat_index=1)

    return finalize_mesh_object(
        "Dragon_Head",
        bm,
        [
            materials["dark_scales"],
            materials["cyan_glow"],
            materials["dark_mouth"],
            materials["antique_gold"],
        ],
        collection,
        parent,
    )


def _build_eyes(materials, collection, parent):
    """Build fierce compact fiery orange-amber eyes tucked under deep V-scowling brow ridges."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    for side in (-1.0, 1.0):
        eye_center = Vector((side * 0.275, -4.24, 2.065))
        gaze = Vector((side * 0.54, -0.84, 0.02)).normalized()
        up_v = Vector((side * 0.34, 0.0, 1.0)).normalized()
        right_v = gaze.cross(up_v).normalized()
        true_up = right_v.cross(gaze).normalized()

        eye_r = 0.082
        rings_n = 14
        segs_n = 20
        verts_grid = []
        for i in range(rings_n + 1):
            phi = (i / float(rings_n)) * math.pi
            r_ring = math.sin(phi) * eye_r
            d_front = math.cos(phi) * eye_r
            row = []
            for j in range(segs_n):
                ang = (j / float(segs_n)) * math.tau
                p = (
                    eye_center
                    + gaze * d_front
                    + right_v * (math.cos(ang) * r_ring * 1.15)
                    + true_up * (math.sin(ang) * r_ring * 0.50)
                )
                row.append(bm.verts.new(p))
            verts_grid.append(row)

        for i in range(rings_n):
            for j in range(segs_n):
                jn = (j + 1) % segs_n
                try:
                    f = bm.faces.new((
                        verts_grid[i][j],
                        verts_grid[i][jn],
                        verts_grid[i + 1][jn],
                        verts_grid[i + 1][j],
                    ))
                    f.material_index = 0
                    f.smooth = True
                    for l in f.loops:
                        l[uv_layer].uv = (j / float(segs_n), i / float(rings_n))
                except ValueError:
                    pass

        pupil_pts = [
            eye_center + gaze * (eye_r * 0.98) - true_up * (eye_r * 0.38),
            eye_center + gaze * (eye_r * 1.04),
            eye_center + gaze * (eye_r * 0.98) + true_up * (eye_r * 0.38),
        ]
        add_curved_tube(
            bm,
            pupil_pts,
            [0.002, 0.010, 0.002],
            radial_segs=8,
            cap_start=True,
            cap_end=True,
            uv_layer=uv_layer,
            mat_index=1,
            rx_scale=0.45,
            rz_scale=1.1,
        )

    return finalize_mesh_object(
        "Dragon_Eyes",
        bm,
        [materials["amber_eye"], materials["pupil_black"]],
        collection,
        parent,
    )


def _build_antlers(materials, collection, parent):
    """Build thick organic gnarled burnished-gold stag antlers matching HD FRONT/REAR/LEFT/TOP."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    for side in (-1.0, 1.0):
        # Sweeping main beam: thicker base (0.095m), smooth outward arc, hooking inward at crown
        main_ctrl = [
            (side * 0.20, -3.92, 2.26),
            (side * 0.40, -3.68, 2.52),
            (side * 0.64, -3.40, 2.75),
            (side * 0.72, -3.12, 2.98),
            (side * 0.56, -2.92, 3.10),
            (side * 0.44, -2.84, 3.12),
        ]
        main_pts = _catmull_rom_sample(main_ctrl, 30)
        main_r = [0.095 - 0.079 * ((k / 29.0) ** 0.80) for k in range(30)]
        add_curved_tube(
            bm,
            main_pts,
            main_r,
            radial_segs=14,
            ring_ridge_amp=0.0,
            ring_ridge_freq=0.0,
            uv_layer=uv_layer,
            mat_index=0,
        )

        # Tine 1: Curved forward-upward brow tine
        tine1_ctrl = [
            (side * 0.38, -3.72, 2.50),
            (side * 0.46, -3.92, 2.68),
            (side * 0.36, -4.08, 2.86),
            (side * 0.26, -4.04, 2.94),
        ]
        t1_pts = _catmull_rom_sample(tine1_ctrl, 16)
        t1_r = [0.056 - 0.046 * (k / 15.0) for k in range(16)]
        add_curved_tube(bm, t1_pts, t1_r, radial_segs=10, uv_layer=uv_layer, mat_index=0)

        # Tine 2: Wide outward-sweeping mid-tine hooking upward to x=±1.02m, z=2.94m
        tine2_ctrl = [
            (side * 0.62, -3.42, 2.74),
            (side * 0.84, -3.38, 2.78),
            (side * 1.02, -3.30, 2.86),
            (side * 0.98, -3.22, 2.98),
        ]
        t2_pts = _catmull_rom_sample(tine2_ctrl, 16)
        t2_r = [0.054 - 0.044 * (k / 15.0) for k in range(16)]
        add_curved_tube(bm, t2_pts, t2_r, radial_segs=10, uv_layer=uv_layer, mat_index=0)

        # Tine 3: Rear crown fork
        tine3_ctrl = [
            (side * 0.70, -3.14, 2.96),
            (side * 0.86, -2.96, 3.04),
            (side * 0.88, -2.78, 3.10),
        ]
        t3_pts = _catmull_rom_sample(tine3_ctrl, 12)
        t3_r = [0.044 - 0.036 * (k / 11.0) for k in range(12)]
        add_curved_tube(bm, t3_pts, t3_r, radial_segs=9, uv_layer=uv_layer, mat_index=0)

    return finalize_mesh_object(
        "Dragon_Antlers",
        bm,
        [materials["antique_gold"]],
        collection,
        parent,
    )


def _build_beard_and_silver_collar(materials, collection, parent):
    """Build prominent layered silver chin goatee + icy-silver brow/temple/cheek fur collar."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rng = random.Random(77)

    # 1. Prominent triangular layered silver chin goatee (visible cleanly in FRONT down to z~1.48 & swept back in LEFT)
    for tier, (n_chin, y_off, z_off, fz_drop, ln_base, w_base) in enumerate([
        (11, -4.82, 1.82, -0.78, 0.36, 0.092),
        ( 7, -4.74, 1.78, -0.66, 0.38, 0.085),
    ]):
        for i in range(n_chin):
            frac = (i / float(max(1, n_chin - 1))) * 2.0 - 1.0
            x = frac * (0.14 if tier == 0 else 0.10)
            y = y_off + abs(frac) * 0.20
            z = z_off - 0.01 * (1.0 - abs(frac))
            f_dir = Vector((-frac * 0.24, 0.62 + 0.16 * abs(frac), fz_drop)).normalized()
            up_dir = Vector((0.0, 0.86, 0.50)).normalized()
            length = (ln_base - 0.14 * (abs(frac) ** 1.1)) + rng.uniform(-0.015, 0.02)
            add_flame_tuft(
                bm,
                root=Vector((x, y, z)),
                forward_dir=f_dir,
                up_dir=up_dir,
                length=length,
                width=w_base * (1.0 - 0.22 * abs(frac)),
                thickness=0.050,
                curl_up=0.26,
                curl_side=frac * 0.04,
                segs=7,
                uv_layer=uv_layer,
                mat_index=0,
                s_wave=0.06,
            )

    # 2. Backward-swept silver mustache / jawline fringe wisps
    for side in (-1.0, 1.0):
        for k in range(5):
            kt = k / 4.0
            root = Vector((side * (0.13 + kt * 0.15), -4.74 + kt * 0.40, 1.92 - kt * 0.04))
            f_dir = Vector((side * (0.26 + kt * 0.14), 0.88, -0.36)).normalized()
            add_flame_tuft(
                bm,
                root=root,
                forward_dir=f_dir,
                up_dir=Vector((0.0, 0.85, 0.52)).normalized(),
                length=0.28 - 0.04 * kt + rng.uniform(-0.015, 0.02),
                width=0.072,
                thickness=0.038,
                curl_up=0.22,
                curl_side=side * 0.04,
                segs=6,
                uv_layer=uv_layer,
                mat_index=0,
                s_wave=0.05,
            )

        # 3. Signature icy-silver inner brow, temple & cheekbone fur collar bridging dark muzzle -> magenta mane
        n_collar = 10
        for k in range(n_collar):
            kt = k / float(n_collar - 1)  # 0 = upper brow/temple, 1 = lower cheek
            ang = math.radians(65.0) - kt * math.radians(130.0)
            root = Vector((
                side * (0.31 + 0.07 * math.cos(ang)),
                -4.12 + 0.22 * kt,
                2.04 + 0.20 * math.sin(ang),
            ))
            f_dir = Vector((
                side * (0.66 + 0.14 * math.cos(ang)),
                0.60 + 0.14 * kt,
                0.40 * math.sin(ang) - 0.08 * kt,
            )).normalized()
            up_dir = Vector((side * math.cos(ang), 0.15, math.sin(ang))).normalized()
            add_flame_tuft(
                bm,
                root=root,
                forward_dir=f_dir,
                up_dir=up_dir,
                length=0.34 + 0.08 * math.sin(kt * math.pi) + rng.uniform(-0.02, 0.02),
                width=0.105,
                thickness=0.055,
                curl_up=0.16,
                curl_side=side * 0.06,
                segs=6,
                uv_layer=uv_layer,
                mat_index=0,
                s_wave=0.06,
            )

    return finalize_mesh_object(
        "Dragon_Beard",
        bm,
        [materials["silver_beard"]],
        collection,
        parent,
    )


def _build_whiskers(materials, collection, parent):
    """Build sleek wide figure-8 / teardrop looped neon-magenta whiskers (x=±1.44m)."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    for side in (-1.0, 1.0):
        ctrl = [
            (side * 0.19, -4.74, 2.00),
            (side * 0.54, -4.85, 2.00),
            (side * 1.06, -4.76, 2.02),
            (side * 1.42, -4.54, 1.88),
            (side * 1.44, -4.34, 1.68),
            (side * 1.22, -4.20, 1.54),
            (side * 0.88, -4.12, 1.58),
            (side * 0.58, -4.04, 1.76),
            (side * 0.44, -3.96, 1.92),
        ]
        pts = _catmull_rom_sample(ctrl, 46)
        rads = [0.016 - 0.008 * ((k / 45.0) ** 0.85) for k in range(46)]
        add_curved_tube(bm, pts, rads, radial_segs=8, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0)

        ctrl2 = [
            (side * 0.22, -4.64, 1.93),
            (side * 0.46, -4.48, 1.85),
            (side * 0.66, -4.12, 1.74),
            (side * 0.70, -3.68, 1.70),
        ]
        pts2 = _catmull_rom_sample(ctrl2, 22)
        rads2 = [0.012 * (1.0 - 0.68 * (k / 21.0)) for k in range(22)]
        add_curved_tube(bm, pts2, rads2, radial_segs=7, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0)

    return finalize_mesh_object(
        "Dragon_Whiskers",
        bm,
        [materials["magenta_whisker"]],
        collection,
        parent,
    )


def _build_mane(materials, collection, parent):
    """Build flowing S-waved neon-magenta crown crest, radial cheek halo, neck mane, and lower shoulder flares."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rng = random.Random(303)

    # 1. Crown Flame Crest rising high between the antlers
    crown_specs = [
        ( 0.00, -4.18, 2.26,  0.00,  0.42,  0.91, 0.58, 0.145, 0.075),
        (-0.08, -4.12, 2.25, -0.14,  0.46,  0.88, 0.54, 0.135, 0.070),
        ( 0.08, -4.12, 2.25,  0.14,  0.46,  0.88, 0.54, 0.135, 0.070),
        ( 0.00, -3.94, 2.28,  0.00,  0.60,  0.80, 0.66, 0.165, 0.085),
        (-0.11, -3.88, 2.26, -0.16,  0.62,  0.77, 0.60, 0.145, 0.075),
        ( 0.11, -3.88, 2.26,  0.16,  0.62,  0.77, 0.60, 0.145, 0.075),
        ( 0.00, -3.68, 2.26,  0.00,  0.76,  0.65, 0.64, 0.175, 0.090),
        (-0.14, -3.62, 2.23, -0.22,  0.78,  0.58, 0.56, 0.145, 0.075),
        ( 0.14, -3.62, 2.23,  0.22,  0.78,  0.58, 0.56, 0.145, 0.075),
        ( 0.00, -3.45, 2.20,  0.00,  0.84,  0.54, 0.58, 0.165, 0.085),
        ( 0.00, -4.34, 2.22,  0.00,  0.30,  0.95, 0.40, 0.115, 0.060),
    ]
    for cx, cy, cz, fx, fy, fz, ln, wd, th in crown_specs:
        add_flame_tuft(
            bm,
            root=Vector((cx, cy, cz)),
            forward_dir=Vector((fx, fy, fz)),
            up_dir=Vector((0.0, 0.2, 1.0)),
            length=ln,
            width=wd,
            thickness=th,
            curl_up=-0.10,
            curl_side=rng.uniform(-0.05, 0.05),
            segs=8,
            uv_layer=uv_layer,
            mat_index=0,
            s_wave=0.09,
        )

    # 2. Outer Radial Cheek / Temple / Jaw Ruff Halo (framing outside the silver collar)
    for side in (-1.0, 1.0):
        for y_off, len_mul, rad_push in [(0.04, 1.04, 0.0), (0.30, 0.94, 0.02)]:
            n_arc = 15
            for k in range(n_arc):
                kt = k / float(n_arc - 1)
                ang = (math.pi * 0.46) - kt * (math.pi * 0.98)
                root = Vector((
                    side * (0.37 + 0.06 * math.cos(ang)),
                    -3.90 + y_off + 0.12 * abs(math.sin(ang)),
                    2.00 + 0.24 * math.sin(ang) + rad_push,
                ))
                out_x = side * (0.72 + 0.14 * math.cos(ang))
                out_y = 0.54 + 0.18 * kt
                out_z = 0.40 * math.sin(ang) - 0.14 * kt
                f_dir = Vector((out_x, out_y, out_z)).normalized()
                up_dir = Vector((side * math.cos(ang), 0.15, math.sin(ang))).normalized()
                ln = (0.54 + 0.16 * math.sin(kt * math.pi) + rng.uniform(-0.04, 0.05)) * len_mul
                add_flame_tuft(
                    bm,
                    root=root,
                    forward_dir=f_dir,
                    up_dir=up_dir,
                    length=ln,
                    width=0.145,
                    thickness=0.075,
                    curl_up=0.16,
                    curl_side=side * 0.08,
                    twist=side * 0.15,
                    segs=8,
                    uv_layer=uv_layer,
                    mat_index=0,
                    s_wave=0.09,
                )

    # 3. Cascading Neck Mane along u = 0.150 .. 0.265 (seamless into dorsal crest)
    n_neck_rows = 20
    for ri in range(n_neck_rows):
        rt = ri / float(n_neck_rows - 1)
        u = 0.150 + rt * (0.265 - 0.150)
        T, R, U = spine_frame(u)

        p_mid, _ = body_surface_point(u, math.pi * 0.5, offset=-0.025)
        f_mid = (T * 0.78 + U * 0.62).normalized()
        add_flame_tuft(
            bm,
            root=p_mid,
            forward_dir=f_mid,
            up_dir=U,
            length=0.50 - 0.10 * rt + rng.uniform(-0.02, 0.03),
            width=0.185,
            thickness=0.095,
            curl_up=-0.10,
            curl_side=rng.uniform(-0.04, 0.04),
            segs=8,
            uv_layer=uv_layer,
            mat_index=0,
            s_wave=0.09,
        )

        max_spread = 1.45 * (1.0 - 0.78 * smoothstep(0.25, 1.0, rt))
        s_fracs = [0.24, 0.58, 0.96] if rt < 0.65 else [0.32, 0.78]
        for s_frac in s_fracs:
            for side in (-1.0, 1.0):
                theta = math.pi * 0.5 - side * s_frac * max_spread
                p_root, out_rad = body_surface_point(u, theta, offset=-0.020)
                f_dir = (T * (0.78 + 0.08 * rt) + out_rad * 0.42 + U * 0.30).normalized()
                ln = (0.48 - 0.12 * rt) * (1.0 - 0.14 * s_frac) + rng.uniform(-0.03, 0.04)
                add_flame_tuft(
                    bm,
                    root=p_root,
                    forward_dir=f_dir,
                    up_dir=out_rad,
                    length=ln,
                    width=0.155 * (1.0 - 0.12 * rt),
                    thickness=0.080,
                    curl_up=0.15,
                    curl_side=side * 0.05,
                    segs=8,
                    uv_layer=uv_layer,
                    mat_index=0,
                    s_wave=0.08,
                )

    # 4. Upper-lateral neck ruff blend tufts sweeping cleanly back behind cheeks (no spiky throat thorns)
    for side in (-1.0, 1.0):
        for k in range(6):
            kt = k / 5.0
            u = 0.175 + kt * 0.065
            T, R, U = spine_frame(u)
            theta = -math.pi * 0.5 + side * (1.72 + 0.14 * kt)
            p_root, out_rad = body_surface_point(u, theta, offset=-0.015)
            f_dir = (T * 0.86 + out_rad * 0.44 + U * 0.10).normalized()
            add_flame_tuft(
                bm,
                root=p_root,
                forward_dir=f_dir,
                up_dir=out_rad,
                length=0.36 + 0.06 * math.sin(kt * math.pi),
                width=0.125,
                thickness=0.065,
                curl_up=0.14,
                curl_side=side * 0.05,
                segs=7,
                uv_layer=uv_layer,
                mat_index=0,
                s_wave=0.07,
            )

    return finalize_mesh_object(
        "Dragon_Mane",
        bm,
        [materials["magenta_mane"]],
        collection,
        parent,
    )
