"""Procedural builders for the dragon torso/tail, belly scutes, cyan conduits, dorsal crest, and tail plume."""

import math
import random
import bpy
import bmesh
from mathutils import Vector

from dragon_spine import (
    spine_point,
    spine_frame,
    body_radii,
    body_surface_point,
    smoothstep,
    loft_sections,
    add_curved_tube,
    add_flame_tuft,
    finalize_mesh_object,
)


def _fract_centered(x: float) -> float:
    return (x % 1.0) - 0.5


def _diamond_scale_relief(u: float, theta: float) -> float:
    """Compute smooth 3D staggered imbricated scale height in [0, 1]."""
    nu = u * 66.0
    nt = (theta / math.tau) * 22.0

    def cell(a, b):
        fa = _fract_centered(a)
        fb = _fract_centered(b)
        dist2 = (fa - 0.12) ** 2 * 2.4 + fb ** 2 * 2.8
        return max(0.0, 1.0 - dist2 * 1.9)

    h = max(cell(nu, nt), cell(nu + 0.5, nt + 0.5))
    return h * h * (3.0 - 2.0 * h)


def build_body_objects(materials: dict, collection, parent):
    """Build Dragon_Body, Dragon_Belly_Scutes, Dragon_Cyan_Conduits, Dragon_Dorsal_Crest, Dragon_Tail_Plume."""
    objs = []
    objs.append(_build_dragon_body(materials, collection, parent))
    objs.append(_build_belly_scutes(materials, collection, parent))
    objs.append(_build_cyan_conduits(materials, collection, parent))
    objs.append(_build_dorsal_crest(materials, collection, parent))
    objs.append(_build_tail_plume(materials, collection, parent))
    return objs


def _build_dragon_body(materials, collection, parent):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    n_u = 180
    n_t = 64
    u_start = 0.155
    u_end = 0.985

    rings = []
    u_coords = []
    thetas = [(-math.pi + (j / float(n_t)) * math.tau) for j in range(n_t)]

    for i in range(n_u):
        t_u = i / float(n_u - 1)
        u = u_start + t_u * (u_end - u_start)
        u_coords.append(u * 6.0)
        rx, _ = body_radii(u)
        scale_mask_u = smoothstep(0.158, 0.185, u) * (1.0 - smoothstep(0.93, 0.985, u))

        ring = []
        for theta in thetas:
            ang_from_ventral = abs(math.atan2(math.cos(theta), -math.sin(theta)))
            flank_mask = smoothstep(0.75, 1.08, ang_from_ventral)
            relief = _diamond_scale_relief(u, theta)
            amp = 0.024 * (rx / 0.45) * scale_mask_u * flank_mask
            pos, _ = body_surface_point(u, theta, offset=amp * relief)
            ring.append(pos)
        rings.append(ring)

    v_coords = [j / float(n_t) for j in range(n_t + 1)]
    _, faces = loft_sections(
        bm,
        rings,
        cap_start=True,
        cap_end=True,
        uv_layer=uv_layer,
        mat_index=0,
        u_coords=u_coords,
        v_coords=v_coords,
    )

    for f in faces:
        if len(f.loops) >= 3:
            avg_v = sum(l[uv_layer].uv[1] for l in f.loops) / len(f.loops)
            if abs(avg_v - 0.25) < 0.13:
                f.material_index = 1

    return finalize_mesh_object(
        "Dragon_Body",
        bm,
        [materials["dark_scales"], materials["belly_scutes"]],
        collection,
        parent,
    )


def _add_scute_plate_from_fn(bm, uv_layer, point_fn, arc_segs=14):
    """Helper to add one solid 4-ring beveled scute plate evaluated via point_fn(si, s_frac, is_inner)."""
    v_uvs = [0.0, 0.42, 0.88, 1.0]
    outer_grid = []
    inner_grid = []
    for si in range(4):
        o_row = []
        i_row = []
        for j in range(arc_segs + 1):
            s_frac = (j / float(arc_segs)) * 2.0 - 1.0
            o_row.append(bm.verts.new(point_fn(si, s_frac, False)))
            i_row.append(bm.verts.new(point_fn(si, s_frac, True)))
        outer_grid.append(o_row)
        inner_grid.append(i_row)

    for si in range(3):
        for j in range(arc_segs):
            try:
                f = bm.faces.new((
                    outer_grid[si][j],
                    outer_grid[si][j + 1],
                    outer_grid[si + 1][j + 1],
                    outer_grid[si + 1][j],
                ))
                f.material_index = 0
                f.smooth = True
                f.loops[0][uv_layer].uv = (j / float(arc_segs), v_uvs[si])
                f.loops[1][uv_layer].uv = ((j + 1) / float(arc_segs), v_uvs[si])
                f.loops[2][uv_layer].uv = ((j + 1) / float(arc_segs), v_uvs[si + 1])
                f.loops[3][uv_layer].uv = (j / float(arc_segs), v_uvs[si + 1])
            except ValueError:
                pass

    for si in range(3):
        for side_j in (0, arc_segs):
            try:
                verts = (
                    (inner_grid[si][0], outer_grid[si][0], outer_grid[si + 1][0], inner_grid[si + 1][0])
                    if side_j == 0
                    else (outer_grid[si][side_j], inner_grid[si][side_j], inner_grid[si + 1][side_j], outer_grid[si + 1][side_j])
                )
                f = bm.faces.new(verts)
                f.material_index = 0
                f.smooth = True
                for l in f.loops:
                    l[uv_layer].uv = (0.05, 0.92)
            except ValueError:
                pass

    for j in range(arc_segs):
        try:
            f_back = bm.faces.new((outer_grid[3][j], outer_grid[3][j + 1], inner_grid[3][j + 1], inner_grid[3][j]))
            f_back.material_index = 0
            f_back.smooth = True
            for l in f_back.loops:
                l[uv_layer].uv = (0.5, 0.98)
        except ValueError:
            pass
        try:
            f_front = bm.faces.new((inner_grid[0][j], inner_grid[0][j + 1], outer_grid[0][j + 1], outer_grid[0][j]))
            f_front.material_index = 0
            f_front.smooth = True
            for l in f_front.loops:
                l[uv_layer].uv = (0.5, 0.02)
        except ValueError:
            pass


def _build_belly_scutes(materials, collection, parent):
    """Build 76 body scutes + 8 tapered throat/jaw scutes reaching under the chin for BOTTOM continuity."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    # 1. Throat / under-jaw scutes bridging u=0.165 forward under mandible (y = -4.42 .. -3.42)
    n_throat = 11
    for k in range(n_throat):
        tk = k / float(n_throat)
        tk1 = (k + 1.18) / float(n_throat)
        y0 = -4.42 + tk * 1.00
        y1 = -4.42 + min(1.02, tk1) * 1.00
        y_samples = [y0, y0 + 0.45 * (y1 - y0), y0 + 0.92 * (y1 - y0), y1]
        offs = [-0.002, 0.015, 0.022, 0.003]

        def throat_pt(si, s_frac, is_inner, _ys=y_samples, _offs=offs):
            yy = _ys[si]
            tf = max(0.0, min(1.0, (yy + 4.48) / 1.08))
            w = 0.11 + 0.20 * tf
            z_base = 1.72 + 0.09 * (tf ** 1.4)
            ang = s_frac * 0.95
            x = math.sin(ang) * w
            z = z_base + (1.0 - math.cos(ang)) * 0.09
            if is_inner:
                return Vector((x * 0.90, yy, z + 0.025))
            edge_taper = math.sqrt(max(0.05, 1.0 - 0.72 * (s_frac ** 2)))
            return Vector((x, yy - 0.025 * (1.0 - s_frac ** 2), z - _offs[si] * edge_taper))

        _add_scute_plate_from_fn(bm, uv_layer, throat_pt, arc_segs=12)

    # 2. Main body & tail scutes (u = 0.165 .. 0.925)
    n_scutes = 76
    u_min = 0.165
    u_max = 0.925
    du_step = (u_max - u_min) / float(n_scutes)

    for k in range(n_scutes):
        u0 = u_min + k * du_step
        u1 = min(0.95, u0 + du_step * 1.18)
        t_k = k / float(n_scutes - 1)
        half_span = 1.00 - 0.18 * (t_k ** 1.6)
        if k < 10:
            half_span -= 0.08 * (1.0 - k / 10.0)

        u_samples = [u0, u0 + 0.45 * (u1 - u0), u0 + 0.92 * (u1 - u0), u1]
        rx_mid, _ = body_radii(u_samples[1])
        r_scale = max(0.25, rx_mid / 0.45)
        outer_offs = [-0.004 * r_scale, 0.022 * r_scale, 0.032 * r_scale, 0.004 * r_scale]
        inner_off = -0.015 * r_scale

        def body_scute_pt(si, s_frac, is_inner, _us=u_samples, _oo=outer_offs, _io=inner_off, _hs=half_span):
            u_s = _us[si]
            theta = -math.pi * 0.5 + s_frac * _hs
            u_curved = u_s - 0.0035 * (1.0 - s_frac * s_frac)
            if is_inner:
                p, _ = body_surface_point(u_curved, theta, offset=_io)
            else:
                edge_taper = math.sqrt(max(0.05, 1.0 - 0.72 * (s_frac ** 2)))
                p, _ = body_surface_point(u_curved, theta, offset=_oo[si] * edge_taper)
            return p

        _add_scute_plate_from_fn(bm, uv_layer, body_scute_pt, arc_segs=14)

    return finalize_mesh_object(
        "Dragon_Belly_Scutes",
        bm,
        [materials["belly_scutes"]],
        collection,
        parent,
    )


def _build_cyan_conduits(materials, collection, parent):
    """Build sleek continuous cyan energy seam + forked neck/shoulder & hip lightning-vein cracks."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    # 1. Crisp continuous undulating lower-flank glowing seam (left & right)
    n_pts = 180
    for side in (-1.0, 1.0):
        pts = []
        rads = []
        for i in range(n_pts):
            t = i / float(n_pts - 1)
            u = 0.162 + t * (0.930 - 0.162)
            rx, _ = body_radii(u)
            scale_r = rx / 0.45
            wave = 0.045 * math.sin(u * 54.0)
            theta = -math.pi * 0.5 + side * (1.03 + wave)
            p, _ = body_surface_point(u, theta, offset=0.008 * scale_r)
            pts.append(p)
            pulse = 0.55 + 0.45 * max(0.0, math.sin(u * 145.0))
            end_taper = smoothstep(0.160, 0.178, u) * (1.0 - smoothstep(0.86, 0.93, u))
            r = (0.007 + 0.006 * pulse) * scale_r * max(0.22, end_taper)
            rads.append(r)

        add_curved_tube(bm, pts, rads, radial_segs=7, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0)

        # 2. Forked cybernetic lightning-vein cracks on neck/chest flanks (FRONT/LEFT) & upper hip flanks (REAR/LEFT)
        vein_specs = [
            # (u0, u1, theta0_from_ventral, theta1_from_ventral, wiggle_freq, base_r)
            (0.168, 0.238, 1.02, 1.72, 24.0, 0.010),
            (0.188, 0.252, 1.24, 1.98, 30.0, 0.0075),
            (0.212, 0.282, 1.04, 1.88, 22.0, 0.0095),
            (0.228, 0.275, 1.36, 1.62, 34.0, 0.0065),
            (0.628, 0.745, 1.03, 1.94, 24.0, 0.010),
            (0.652, 0.728, 1.35, 2.08, 32.0, 0.0075),
            (0.680, 0.775, 1.02, 1.58, 22.0, 0.0090),
            (0.702, 0.768, 1.28, 1.85, 30.0, 0.0065),
        ]
        for v_idx, (u0, u1, th0, th1, w_freq, base_r) in enumerate(vein_specs):
            b_pts = []
            b_rads = []
            steps = 26
            for i in range(steps):
                bt = i / float(steps - 1)
                u = u0 + bt * (u1 - u0)
                rx, _ = body_radii(u)
                scale_r = rx / 0.45
                jitter = 0.06 * math.sin(bt * w_freq + v_idx * 1.3)
                ang_span = th0 + (th1 - th0) * math.sin(bt * math.pi * 0.82) + jitter
                theta = -math.pi * 0.5 + side * ang_span
                p, _ = body_surface_point(u, theta, offset=0.008 * scale_r)
                b_pts.append(p)
                taper = math.sin(bt * math.pi) ** 0.65
                b_rads.append(base_r * scale_r * max(0.12, taper))
            add_curved_tube(bm, b_pts, b_rads, radial_segs=6, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=0)

    return finalize_mesh_object(
        "Dragon_Cyan_Conduits",
        bm,
        [materials["cyan_glow"]],
        collection,
        parent,
    )


def _build_dorsal_crest(materials, collection, parent):
    """Build sleek backward-swept flowing magenta flame wave along dorsal spine (u=0.23..0.88)."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rng = random.Random(42)

    n_central = 72
    for i in range(n_central):
        t = i / float(n_central - 1)
        u = 0.23 + t * (0.875 - 0.23)
        T, R, U = spine_frame(u)
        _, rz = body_radii(u)
        pos, _ = body_surface_point(u, math.pi * 0.5, offset=-0.03 * (rz / 0.40))

        arch_boost = 1.0 + 0.14 * math.sin(t * math.pi * 2.0)
        tail_taper = 1.0 - 0.48 * smoothstep(0.65, 1.0, t)
        length = (0.37 + rng.uniform(-0.03, 0.04)) * arch_boost * tail_taper
        width = 0.145 * (0.55 + 0.45 * tail_taper)
        thickness = 0.072 * (0.55 + 0.45 * tail_taper)

        # Much flatter backward-sweeping wave hugging the spine arches (matches HD LEFT/RIGHT)
        f_dir = (T * 0.90 + U * 0.43).normalized()
        add_flame_tuft(
            bm,
            root=pos,
            forward_dir=f_dir,
            up_dir=U,
            length=length,
            width=width,
            thickness=thickness,
            curl_up=0.06,
            curl_side=rng.uniform(-0.04, 0.04),
            twist=rng.uniform(-0.10, 0.10),
            segs=7,
            uv_layer=uv_layer,
            mat_index=0,
            s_wave=0.08,
        )

        # Dense flanking secondary tufts for continuous TOP/REAR fullness
        for side in (-1.0, 1.0):
            if (i + (0 if side > 0 else 1)) % 2 == 0 or i < 24:
                u_off = u + rng.uniform(-0.004, 0.004)
                T2, R2, U2 = spine_frame(u_off)
                p_lat, _ = body_surface_point(u_off, math.pi * 0.5 - side * 0.20, offset=-0.025)
                lat_dir = (T2 * 0.89 + U2 * 0.36 + R2 * (side * 0.28)).normalized()
                add_flame_tuft(
                    bm,
                    root=p_lat,
                    forward_dir=lat_dir,
                    up_dir=U2,
                    length=length * 0.78,
                    width=width * 0.86,
                    thickness=thickness * 0.82,
                    curl_up=0.10,
                    curl_side=side * 0.08,
                    twist=side * 0.15,
                    segs=6,
                    uv_layer=uv_layer,
                    mat_index=0,
                    s_wave=0.07,
                )

    return finalize_mesh_object(
        "Dragon_Dorsal_Crest",
        bm,
        [materials["magenta_mane"]],
        collection,
        parent,
    )


def _build_tail_plume(materials, collection, parent):
    """Build lush teardrop fiery magenta tail plume from u=0.86 to u=1.00."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rng = random.Random(108)

    n_rings = 12
    for ri in range(n_rings):
        rt = ri / float(n_rings - 1)
        u = 0.865 + rt * (0.985 - 0.865)
        T, R, U = spine_frame(u)
        env = math.sin(max(0.15, rt) * math.pi * 0.88)
        n_around = 7 if ri < n_rings - 1 else 5
        for aj in range(n_around):
            ang = (aj / float(n_around)) * math.tau + ri * 0.55
            p_root, _ = body_surface_point(u, ang, offset=-0.01)
            flare_out = (R * math.cos(ang) + U * math.sin(ang)) * (0.28 + 0.18 * env)
            f_dir = (T * 0.86 + U * 0.18 + flare_out).normalized()
            length = (0.50 + 0.28 * env + rng.uniform(-0.04, 0.05)) * (1.0 - 0.24 * rt)
            width = 0.135 + 0.06 * env
            thickness = 0.075 + 0.03 * env
            add_flame_tuft(
                bm,
                root=p_root,
                forward_dir=f_dir,
                up_dir=(U + flare_out * 0.5).normalized(),
                length=length,
                width=width,
                thickness=thickness,
                curl_up=0.20,
                curl_side=math.cos(ang) * 0.09,
                twist=rng.uniform(-0.18, 0.18),
                segs=7,
                uv_layer=uv_layer,
                mat_index=0,
                s_wave=0.08,
            )

    p_tip = spine_point(0.982)
    T_tip, R_tip, U_tip = spine_frame(0.985)
    for k in range(6):
        off_ang = (k / 6.0) * math.tau
        dir_k = (T_tip * 0.94 + U_tip * 0.20 + R_tip * (0.14 * math.cos(off_ang))).normalized()
        add_flame_tuft(
            bm,
            root=p_tip,
            forward_dir=dir_k,
            up_dir=U_tip,
            length=0.60 - 0.04 * (k % 2),
            width=0.145,
            thickness=0.078,
            curl_up=0.18,
            curl_side=0.08 * math.cos(off_ang),
            segs=7,
            uv_layer=uv_layer,
            mat_index=0,
            s_wave=0.08,
        )

    return finalize_mesh_object(
        "Dragon_Tail_Plume",
        bm,
        [materials["magenta_mane"]],
        collection,
        parent,
    )
