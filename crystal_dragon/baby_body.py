"""Procedural builders for the Crystal Baby Dragon body, chevron belly scutes, dorsal spines, tail plume, and shoulder pauldrons."""

import math
import bpy
import bmesh
from mathutils import Vector

from baby_math import (
    smoothstep,
    hermite_eval,
    catmull_rom_sample,
    loft_sections,
    add_curved_tube,
    add_crystal_frill,
    add_double_walled_membrane,
    finalize_mesh_object,
)


# Spine centerline keys: (u, x, y, z, rx, rz)
_BODY_KEYS = [
    (0.00, 0.00, -0.96, 1.56, 0.28, 0.28),  # inside skull base / upper neck
    (0.14, 0.00, -0.86, 1.24, 0.35, 0.35),  # upright puppy neck
    (0.28, 0.00, -0.70, 0.94, 0.43, 0.42),  # plump chest aligned to muscular shoulder
    (0.46, 0.00, -0.18, 0.85, 0.44, 0.40),  # barrel mid-body
    (0.62, 0.00,  0.38, 0.80, 0.44, 0.39),  # rounded rump
    (0.76, 0.00,  0.85, 0.58, 0.30, 0.28),  # tail root descent
    (0.88, 0.00,  1.45, 0.52, 0.15, 0.14),  # low-sweeping tail mid
    (1.00, 0.00,  2.02, 0.70, 0.055, 0.055),  # gently upturned tail tip
]


def body_point(u: float) -> Vector:
    u_c = max(0.0, min(1.0, float(u)))
    x, y, z = hermite_eval(_BODY_KEYS, u_c, (1, 2, 3))
    return Vector((x, y, z))


def body_radii(u: float):
    u_c = max(0.0, min(1.0, float(u)))
    rx, rz = hermite_eval(_BODY_KEYS, u_c, (4, 5))
    return max(0.03, rx), max(0.03, rz)


def body_frame(u: float):
    eps = 0.002
    u0, u1 = max(0.0, u - eps), min(1.0, u + eps)
    T = (body_point(u1) - body_point(u0))
    T = T.normalized() if T.length > 1e-6 else Vector((0.0, 1.0, 0.0))
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(T.dot(world_up)) > 0.94:
        world_up = Vector((0.0, 1.0, 0.0))
    R = T.cross(world_up)
    R = R.normalized() if R.length > 1e-6 else Vector((1.0, 0.0, 0.0))
    U = R.cross(T).normalized()
    return T, R, U


def body_surface_point(u: float, theta: float, offset: float = 0.0):
    """theta=0 left (+R), +pi/2 dorsal (+U), -pi/2 ventral (-U)."""
    center = body_point(u)
    _, R, U = body_frame(u)
    rx, rz = body_radii(u)
    c, s = math.cos(theta), math.sin(theta)

    r_mod = 1.0
    if s < 0.0:
        # Soft flattened belly bottom for pearl chevron scutes
        r_mod -= 0.10 * ((-s) ** 2.2)
    else:
        # Cute gentle dorsal spine ridge
        r_mod += 0.04 * (s ** 6.0)

    radial = R * (c * rx * r_mod) + U * (s * rz * r_mod)
    out_dir = (R * (c / max(rx, 1e-4)) + U * (s / max(rz, 1e-4))).normalized()
    return center + radial + out_dir * offset, out_dir


def build_body_objects(materials: dict, collection, parent):
    return [
        _build_baby_body(materials, collection, parent),
        _build_belly_scutes(materials, collection, parent),
        _build_dorsal_spines(materials, collection, parent),
        _build_tail_plume(materials, collection, parent),
        _build_pauldrons(materials, collection, parent),
    ]


def _build_baby_body(materials, collection, parent):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    n_u = 90
    n_t = 44
    thetas = [(-math.pi + (j / float(n_t)) * math.tau) for j in range(n_t)]
    rings = []
    u_coords = []

    for i in range(n_u):
        u = i / float(n_u - 1)
        u_coords.append(u * 3.0)
        rx, _ = body_radii(u)
        mask_u = smoothstep(0.02, 0.10, u) * (1.0 - smoothstep(0.92, 1.0, u))
        ring = []
        for theta in thetas:
            ang_ventral = abs(math.atan2(math.cos(theta), -math.sin(theta)))
            flank_mask = smoothstep(0.68, 1.02, ang_ventral)
            # Soft rounded pebble scale relief
            nu = u * 38.0
            nt = (theta / math.tau) * 16.0
            fa = (nu % 1.0) - 0.5
            fb = (nt % 1.0) - 0.5
            fa2 = ((nu + 0.5) % 1.0) - 0.5
            fb2 = ((nt + 0.5) % 1.0) - 0.5
            p1 = max(0.0, 1.0 - (fa * fa + fb * fb) * 4.2)
            p2 = max(0.0, 1.0 - (fa2 * fa2 + fb2 * fb2) * 4.2)
            relief = max(p1, p2)
            pos, _ = body_surface_point(u, theta, offset=0.005 * (rx / 0.42) * mask_u * flank_mask * relief)
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
        "Baby_Body",
        bm,
        [materials["lavender_scales"], materials["pearl_belly"]],
        collection,
        parent,
    )


def _add_scute_plate(bm, uv_layer, point_fn, arc_segs=12):
    v_uvs = [0.0, 0.42, 0.88, 1.0]
    outer_grid, inner_grid = [], []
    for si in range(4):
        o_row, i_row = [], []
        for j in range(arc_segs + 1):
            s_frac = (j / float(arc_segs)) * 2.0 - 1.0
            o_row.append(bm.verts.new(point_fn(si, s_frac, False)))
            i_row.append(bm.verts.new(point_fn(si, s_frac, True)))
        outer_grid.append(o_row)
        inner_grid.append(i_row)

    for si in range(3):
        for j in range(arc_segs):
            try:
                f = bm.faces.new((outer_grid[si][j], outer_grid[si][j + 1], outer_grid[si + 1][j + 1], outer_grid[si + 1][j]))
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
            fb = bm.faces.new((outer_grid[3][j], outer_grid[3][j + 1], inner_grid[3][j + 1], inner_grid[3][j]))
            fb.material_index = 0
            fb.smooth = True
            for l in fb.loops:
                l[uv_layer].uv = (0.5, 0.98)
        except ValueError:
            pass
        try:
            ff = bm.faces.new((inner_grid[0][j], inner_grid[0][j + 1], outer_grid[0][j + 1], outer_grid[0][j]))
            ff.material_index = 0
            ff.smooth = True
            for l in ff.loops:
                l[uv_layer].uv = (0.5, 0.02)
        except ValueError:
            pass


def _build_belly_scutes(materials, collection, parent):
    """Build 38 smooth overlapping chevron pearl-white belly scutes from throat through chest to tail."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    # 1. Under-chin throat chevron plates bridging smoothly under mandible (y=-1.24..-0.96)
    n_chin = 5
    for k in range(n_chin):
        t0 = k / float(n_chin)
        t1 = min(1.04, (k + 1.18) / float(n_chin))
        ys = [-1.24 + (t0 + frac * (t1 - t0)) * 0.28 for frac in (0.0, 0.45, 0.92, 1.0)]
        offs = [-0.002, 0.015, 0.022, 0.004]

        def chin_fn(si, s_frac, is_inner, _ys=ys, _offs=offs):
            yy = _ys[si]
            tf = max(0.0, min(1.0, (yy + 1.24) / 0.30))
            w = 0.14 + 0.11 * tf
            z_b = 1.44 + 0.06 * tf
            ang = s_frac * 0.90
            x = math.sin(ang) * w
            z = z_b + (1.0 - math.cos(ang)) * 0.08
            if is_inner:
                return Vector((x * 0.88, yy, z + 0.022))
            edge_t = math.sqrt(max(0.05, 1.0 - 0.72 * (s_frac ** 2)))
            return Vector((x, yy - 0.042 * (1.0 - abs(s_frac)), z - _offs[si] * edge_t))

        _add_scute_plate(bm, uv_layer, chin_fn, arc_segs=12)

    # 2. Main chest -> belly -> tail chevron plates (u = 0.02 .. 0.94)
    n_scutes = 36
    u_min, u_max = 0.02, 0.94
    du = (u_max - u_min) / float(n_scutes)
    for k in range(n_scutes):
        u0 = u_min + k * du
        u1 = min(0.96, u0 + du * 1.18)
        tk = k / float(n_scutes - 1)
        half_span = 0.94 - 0.20 * (tk ** 1.6)
        u_samples = [u0, u0 + 0.45 * (u1 - u0), u0 + 0.92 * (u1 - u0), u1]
        rx_mid, _ = body_radii(u_samples[1])
        r_scale = max(0.24, rx_mid / 0.42)
        outer_offs = [-0.004 * r_scale, 0.020 * r_scale, 0.028 * r_scale, 0.004 * r_scale]
        inner_off = -0.015 * r_scale

        def body_scute_fn(si, s_frac, is_inner, _us=u_samples, _oo=outer_offs, _io=inner_off, _hs=half_span):
            u_s = _us[si]
            theta = -math.pi * 0.5 + s_frac * _hs
            # V-chevron point forward/down along midline
            u_curved = u_s + 0.022 * abs(s_frac)
            if is_inner:
                p, _ = body_surface_point(u_curved, theta, offset=_io)
            else:
                edge_t = math.sqrt(max(0.05, 1.0 - 0.72 * (s_frac ** 2)))
                p, _ = body_surface_point(u_curved, theta, offset=_oo[si] * edge_t)
            return p

        _add_scute_plate(bm, uv_layer, body_scute_fn, arc_segs=12)

    return finalize_mesh_object(
        "Baby_Belly_Scutes",
        bm,
        [materials["pearl_belly"]],
        collection,
        parent,
    )


def _build_dorsal_spines(materials, collection, parent):
    """Build 28 sculpted lavender-to-rose dorsal fin spikes + flanking neck/rump crystal scales."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    n_central = 28
    for i in range(n_central):
        t = i / float(n_central - 1)
        u = 0.06 + t * (0.93 - 0.06)
        T, R, U = body_frame(u)
        _, rz = body_radii(u)
        pos, _ = body_surface_point(u, math.pi * 0.5, offset=-0.02 * (rz / 0.36))

        # Cute plump dorsal fins: taller along neck/back, tapering toward tail plume
        len_prof = (0.24 - 0.10 * smoothstep(0.65, 1.0, t)) * (1.0 + 0.14 * math.sin(t * math.pi))
        f_dir = (T * 0.74 + U * 0.67).normalized()
        add_crystal_frill(
            bm,
            root=pos,
            forward_dir=f_dir,
            up_dir=U,
            length=len_prof,
            width=0.115 * (1.0 - 0.35 * smoothstep(0.65, 1.0, t)),
            thickness=0.068 * (1.0 - 0.35 * smoothstep(0.65, 1.0, t)),
            curl_up=-0.12,
            segs=6,
            uv_layer=uv_layer,
            mat_index=0,
        )

        # Flanking subtle scale-ridges visible in REAR & TOP
        if i % 2 == 0 and 0.08 < u < 0.85:
            for side in (-1.0, 1.0):
                p_lat, out_rad = body_surface_point(u, math.pi * 0.5 - side * 0.24, offset=-0.015)
                f_lat = (T * 0.82 + U * 0.42 + R * (side * 0.38)).normalized()
                add_crystal_frill(
                    bm,
                    root=p_lat,
                    forward_dir=f_lat,
                    up_dir=out_rad,
                    length=len_prof * 0.65,
                    width=0.085,
                    thickness=0.048,
                    curl_up=0.08,
                    curl_side=side * 0.06,
                    segs=5,
                    uv_layer=uv_layer,
                    mat_index=0,
                )

    return finalize_mesh_object(
        "Baby_Dorsal_Spines",
        bm,
        [materials["frill_gradient"]],
        collection,
        parent,
    )


def _build_tail_plume(materials, collection, parent):
    """Build lush bioluminescent leaf-feather tail plume visible in REAR, LEFT, RIGHT, TOP, BOTTOM."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    n_u = 15
    n_v = 13
    grid = []
    for i in range(n_u):
        u_f = (i / float(n_u - 1)) * 2.0 - 1.0
        col = []
        for j in range(n_v):
            t_len = j / float(n_v - 1)
            w_env = math.sin((max(0.04, t_len) ** 0.72) * math.pi)
            serr = 1.0 - 0.10 * (math.sin(t_len * math.pi * 4.0) ** 2) * (abs(u_f) > 0.7)
            x = u_f * 0.27 * w_env * serr
            y = 1.84 + t_len * 0.68
            z = 0.66 + 0.18 * (t_len ** 1.3) + 0.03 * (1.0 - u_f * u_f) * w_env
            col.append(Vector((x, y, z)))
        grid.append(col)

    add_double_walled_membrane(bm, grid, thickness=0.026, uv_layer=uv_layer, mat_index=0)

    midrib_pts = [
        Vector((0.0, 1.84 + (k / 9.0) * 0.64, 0.675 + 0.178 * ((k / 9.0) ** 1.3)))
        for k in range(10)
    ]
    midrib_r = [0.017 * (1.0 - 0.65 * (k / 9.0)) for k in range(10)]
    add_curved_tube(bm, midrib_pts, midrib_r, radial_segs=7, uv_layer=uv_layer, mat_index=1)

    for side in (-1.0, 1.0):
        for t0, t1, x_reach in [(0.18, 0.45, 0.17), (0.36, 0.68, 0.20), (0.56, 0.84, 0.13)]:
            tm = t0 + 0.5 * (t1 - t0)
            v_pts = catmull_rom_sample([
                (0.0, 1.84 + t0 * 0.68, 0.68 + 0.175 * (t0 ** 1.3)),
                (side * x_reach * 0.55, 1.84 + tm * 0.68, 0.69 + 0.175 * (tm ** 1.3)),
                (side * x_reach, 1.84 + t1 * 0.68, 0.68 + 0.175 * (t1 ** 1.3)),
            ], 8)
            v_r = [0.009 * (1.0 - 0.65 * (k / 7.0)) for k in range(8)]
            add_curved_tube(bm, v_pts, v_r, radial_segs=6, uv_layer=uv_layer, mat_index=1)

    return finalize_mesh_object(
        "Baby_Tail_Plume",
        bm,
        [materials["tail_plume"], materials["ice_cyan_rune"]],
        collection,
        parent,
    )


def _add_curved_armor_plate(bm, uv_layer, center, normal_dir, fwd_dir, width, height, thick, mat_plate, mat_trim, mat_rune=None):
    """Build a chevron-scalloped curved pearl armor plate with 4-edge rose-gold trim & proud cyan rune."""
    n_v = normal_dir.normalized()
    f_v = (fwd_dir - n_v * fwd_dir.dot(n_v)).normalized()
    # Keep vf=+1 consistently pointing world-UP on both left & right sides
    u_v = Vector((0.0, 0.0, 1.0)) - n_v * n_v.z
    u_v = u_v - f_v * u_v.dot(f_v)
    if u_v.length < 1e-6:
        u_v = n_v.cross(f_v)
    u_v.normalize()

    def _plate_pt(uf: float, vf: float, extra_n: float = 0.0) -> Vector:
        px = uf * width * 0.5 * (1.0 - 0.10 * vf)
        cusp_dip = 0.048 * (1.0 - uf * uf) * max(0.0, -vf)
        arch = 1.0 - 0.14 * uf * uf
        py = vf * height * 0.5 * arch - cusp_dip
        bulge = (1.0 - 0.52 * (uf * uf + 0.38 * vf * vf)) * 0.072
        return center + f_v * px + u_v * py + n_v * (bulge + extra_n)

    nu, nv = 9, 7
    grid = []
    for i in range(nu):
        uf = (i / float(nu - 1)) * 2.0 - 1.0
        grid.append([_plate_pt(uf, (j / float(nv - 1)) * 2.0 - 1.0) for j in range(nv)])

    add_double_walled_membrane(bm, grid, thickness=thick, uv_layer=uv_layer, mat_index=mat_plate)

    # Full 4-edge rose-gold metallic border trim (bottom, top, rear, front)
    trim_n = thick * 0.42
    bot_pts = [_plate_pt((i / float(nu - 1)) * 2.0 - 1.0, -1.0, trim_n) for i in range(nu)]
    top_pts = [_plate_pt((i / float(nu - 1)) * 2.0 - 1.0,  1.0, trim_n) for i in range(nu)]
    rear_pts = [_plate_pt(-1.0, (j / float(nv - 1)) * 2.0 - 1.0, trim_n) for j in range(nv)]
    front_pts = [_plate_pt( 1.0, (j / float(nv - 1)) * 2.0 - 1.0, trim_n) for j in range(nv)]
    add_curved_tube(bm, bot_pts, [0.012] * nu, radial_segs=7, uv_layer=uv_layer, mat_index=mat_trim)
    add_curved_tube(bm, top_pts, [0.011] * nu, radial_segs=6, uv_layer=uv_layer, mat_index=mat_trim)
    add_curved_tube(bm, rear_pts, [0.011] * nv, radial_segs=6, uv_layer=uv_layer, mat_index=mat_trim)
    add_curved_tube(bm, front_pts, [0.011] * nv, radial_segs=6, uv_layer=uv_layer, mat_index=mat_trim)

    # Glowing ice-cyan chevron/bar rune strictly proud on the outer plate surface (never clipped inside)
    if mat_rune is not None:
        rune_n = thick * 0.5 + 0.009
        n_r = 7
        rune_pts = []
        for k in range(n_r):
            kt = k / float(n_r - 1)
            uf = -0.48 + kt * 0.96
            vf = 0.08 * (1.0 - 2.0 * uf * uf)
            rune_pts.append(_plate_pt(uf, vf, rune_n))
        rune_r = [0.004, 0.012, 0.015, 0.016, 0.015, 0.012, 0.004]
        add_curved_tube(bm, rune_pts, rune_r, radial_segs=6, uv_layer=uv_layer, mat_index=mat_rune, rx_scale=1.0, rz_scale=0.65)

        dash_pts = [
            _plate_pt(0.55, -0.04, rune_n),
            _plate_pt(0.63, -0.02, rune_n),
            _plate_pt(0.71,  0.00, rune_n),
        ]
        add_curved_tube(bm, dash_pts, [0.004, 0.011, 0.004], radial_segs=6, uv_layer=uv_layer, mat_index=mat_rune, rx_scale=1.0, rz_scale=0.65)


def _build_pauldrons(materials, collection, parent):
    """Build 3-tier sweeping pearl-white + rose-gold + glowing ice-cyan shoulder armor pauldrons."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    # Slots: 0=pearl_armor, 1=rose_gold, 2=ice_cyan_rune

    for side in (-1.0, 1.0):
        # 3 cascading scalloped lames wrapping forward-inward across shoulder arch & new deltoid
        tiers = [
            (Vector((side * 0.41, -0.74, 1.14)), Vector((side * 0.58, -0.45, 0.68)), Vector((-side * 0.18, -0.80, 0.57)), 0.38, 0.20, 0.026, 2),
            (Vector((side * 0.48, -0.73, 1.00)), Vector((side * 0.74, -0.38, 0.55)), Vector((-side * 0.16, -0.82, 0.55)), 0.42, 0.19, 0.024, 2),
            (Vector((side * 0.50, -0.72, 0.87)), Vector((side * 0.85, -0.32, 0.41)), Vector((-side * 0.14, -0.84, 0.52)), 0.38, 0.17, 0.022, None),
        ]
        for ctr, nrm, fwd, w, h, th, rune_slot in tiers:
            _add_curved_armor_plate(bm, uv_layer, ctr, nrm, fwd, w, h, th, mat_plate=0, mat_trim=1, mat_rune=rune_slot)

    return finalize_mesh_object(
        "Baby_Pauldrons",
        bm,
        [materials["pearl_armor"], materials["rose_gold"], materials["ice_cyan_rune"]],
        collection,
        parent,
    )
