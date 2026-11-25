"""Procedural builder for the detachable Crystal Baby Dragon saddle harmonizing with pearl armor, rose-gold trim, cyan runes, and crystal accents."""

import math
import bmesh
from mathutils import Vector

from baby_math import (
    smoothstep,
    add_curved_tube,
    add_crystal_frill,
    add_double_walled_membrane,
    finalize_mesh_object,
)
from baby_body import body_frame, body_surface_point


# Material slot indices on Baby_Saddle
_MAT_PEARL = 0
_MAT_GOLD = 1
_MAT_CYAN = 2
_MAT_VELVET = 3
_MAT_FRILL = 4


def _saddle_bridge_point(u: float, ang: float, extra_off: float = 0.0) -> Vector:
    """Evaluate point on dorsal saddle arch clearing dorsal spines (+0.22m peak) while hugging upper flanks."""
    theta = math.pi * 0.5 + ang
    center_lift = 0.195 * math.exp(-((ang / 0.34) ** 2)) + 0.026
    p, _ = body_surface_point(u, theta, offset=center_lift + extra_off)
    return p


def _add_velvet_seat_deck(bm, uv_layer):
    """Build quilted twilight-lavender seat cushion clearing dorsal spines + perimeter rose-gold piping."""
    u_min, u_max = 0.365, 0.585
    ang_max = 0.78
    nu, na = 14, 15

    grid = []
    for i in range(nu):
        uf = i / float(nu - 1)
        u = u_min + uf * (u_max - u_min)
        row = []
        for j in range(na):
            vf = (j / float(na - 1)) * 2.0 - 1.0
            ang = vf * ang_max
            # Soft quilted diamond relief inside seat cushion + gentle saddle bucket contour
            seat_mask = smoothstep(0.08, 0.25, uf) * (1.0 - smoothstep(0.75, 0.92, uf)) * (1.0 - smoothstep(0.55, 0.95, abs(vf)))
            quilt = 0.006 * abs(math.sin(uf * 6.0 * math.pi) * math.sin(vf * 5.0 * math.pi)) * seat_mask
            bucket_dip = -0.022 * math.sin(uf * math.pi) * math.exp(-((vf / 0.45) ** 2))
            row.append(_saddle_bridge_point(u, ang, extra_off=0.012 + quilt + bucket_dip))
        grid.append(row)

    add_double_walled_membrane(bm, grid, thickness=0.030, uv_layer=uv_layer, mat_index=_MAT_VELVET)

    # Full perimeter rose-gold piping around seat blanket
    border_pts = (
        grid[0]
        + [grid[i][-1] for i in range(1, nu)]
        + list(reversed(grid[-1][:-1]))
        + [grid[i][0] for i in range(nu - 2, -1, -1)]
    )
    add_curved_tube(bm, border_pts, [0.013] * len(border_pts), radial_segs=7, cap_start=False, cap_end=False, uv_layer=uv_layer, mat_index=_MAT_GOLD)

    # Twin sculpted pearl-white structural side rails/bolsters uniting pommel and cantle
    for side in (-1.0, 1.0):
        rail_ang = side * 0.40
        rail_pts = []
        rail_r = []
        for i in range(12):
            uf = i / 11.0
            u = 0.375 + uf * (0.575 - 0.375)
            rail_pts.append(_saddle_bridge_point(u, rail_ang, extra_off=0.036 + 0.018 * (1.0 - math.sin(uf * math.pi))))
            rail_r.append(0.026 - 0.006 * math.sin(uf * math.pi))
        add_curved_tube(bm, rail_pts, rail_r, radial_segs=9, uv_layer=uv_layer, mat_index=_MAT_PEARL)
        trim_pts = [_saddle_bridge_point(0.375 + (i / 11.0) * 0.20, rail_ang + side * 0.03, extra_off=0.056) for i in range(12)]
        add_curved_tube(bm, trim_pts, [0.009] * 12, radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_GOLD)


def _add_pommel_and_cantle(bm, uv_layer):
    """Build front pearl-armor pommel with glowing cyan crystal gem & rear cantle backrest with crystal crest."""
    # 1. Front Swept Crystal Pommel (u ~ 0.37)
    nu_p, nv_p = 11, 7
    p_grid = []
    for i in range(nu_p):
        uf = (i / float(nu_p - 1)) * 2.0 - 1.0  # left..right
        row = []
        for j in range(nv_p):
            vf = j / float(nv_p - 1)  # 0=base -> 1=crest apex
            ang = uf * 0.56 * (1.0 - 0.24 * vf)
            arch_h = 0.26 * (1.0 - 0.62 * (uf * uf)) * (vf ** 0.85)
            u_pt = 0.378 - 0.028 * vf - 0.010 * (1.0 - uf * uf)
            row.append(_saddle_bridge_point(u_pt, ang, extra_off=0.020 + arch_h))
        p_grid.append(row)

    add_double_walled_membrane(bm, p_grid, thickness=0.028, uv_layer=uv_layer, mat_index=_MAT_PEARL)

    # Full rose-gold trim around front pommel crest & side edges
    pom_rim = [p_grid[0][j] for j in range(nv_p)] + [p_grid[i][-1] for i in range(1, nu_p)] + [p_grid[-1][j] for j in range(nv_p - 2, -1, -1)]
    add_curved_tube(bm, pom_rim, [0.013] * len(pom_rim), radial_segs=7, uv_layer=uv_layer, mat_index=_MAT_GOLD)

    # Glowing ice-cyan faceted diamond crystal emblem proud on front pommel face
    T_p, R_p, U_p = body_frame(0.365)
    pom_center = _saddle_bridge_point(0.352, 0.0, extra_off=0.145)
    fwd_p = (-T_p * 0.92 + U_p * 0.38).normalized()
    up_p = (U_p * 0.92 + T_p * 0.38).normalized()
    dia_ring = [
        pom_center + up_p * 0.075,
        pom_center + R_p * 0.052,
        pom_center - up_p * 0.072,
        pom_center - R_p * 0.052,
    ]
    gem_apex = pom_center + fwd_p * 0.036
    gem_back = pom_center - fwd_p * 0.015
    v_ring = [bm.verts.new(p) for p in dia_ring]
    v_apex = bm.verts.new(gem_apex)
    v_back = bm.verts.new(gem_back)
    for k in range(4):
        kn = (k + 1) % 4
        for verts in ((v_ring[k], v_ring[kn], v_apex), (v_ring[kn], v_ring[k], v_back)):
            f = bm.faces.new(verts)
            f.material_index = _MAT_CYAN
            f.smooth = False
            for l in f.loops:
                l[uv_layer].uv = (0.5, 0.5)

    # Flanking lavender-to-rose crystal hornlets crowning pommel shoulders
    for side in (-1.0, 1.0):
        root_h = _saddle_bridge_point(0.365, side * 0.24, extra_off=0.185)
        f_h = (up_p * 0.72 + R_p * (side * 0.45) - T_p * 0.52).normalized()
        add_crystal_frill(
            bm, root=root_h, forward_dir=f_h, up_dir=up_p,
            length=0.105, width=0.052, thickness=0.034, curl_up=0.14, segs=5,
            uv_layer=uv_layer, mat_index=_MAT_FRILL,
        )

    # 2. Rear High-Backed Cantle (u ~ 0.575)
    nu_c, nv_c = 13, 7
    c_grid = []
    c_pad_grid = []
    for i in range(nu_c):
        uf = (i / float(nu_c - 1)) * 2.0 - 1.0
        row = []
        pad_row = []
        for j in range(nv_c):
            vf = j / float(nv_c - 1)
            ang = uf * 0.62 * (1.0 - 0.16 * vf)
            arch_h = 0.25 * (1.0 - 0.45 * (uf * uf)) * (vf ** 0.85)
            u_pt = 0.562 + 0.036 * vf - 0.012 * (1.0 - uf * uf)
            row.append(_saddle_bridge_point(u_pt, ang, extra_off=0.022 + arch_h))
            if vf <= 0.84 and abs(uf) <= 0.84:
                pad_row.append(_saddle_bridge_point(u_pt - 0.008, ang, extra_off=0.020 + arch_h * 0.94))
        c_grid.append(row)
        if pad_row:
            c_pad_grid.append(pad_row)

    add_double_walled_membrane(bm, c_grid, thickness=0.026, uv_layer=uv_layer, mat_index=_MAT_PEARL)
    add_double_walled_membrane(bm, c_pad_grid, thickness=0.018, uv_layer=uv_layer, mat_index=_MAT_VELVET)

    cantle_rim = [c_grid[0][j] for j in range(nv_c)] + [c_grid[i][-1] for i in range(1, nu_c)] + [c_grid[-1][j] for j in range(nv_c - 2, -1, -1)]
    add_curved_tube(bm, cantle_rim, [0.013] * len(cantle_rim), radial_segs=7, uv_layer=uv_layer, mat_index=_MAT_GOLD)

    # Glowing ice-cyan chevron rune inlaid on rear cantle exterior
    chevron_pts = [
        _saddle_bridge_point(0.588 + 0.008 * (1.0 - abs(uf)), uf * 0.34, extra_off=0.145 + 0.045 * (1.0 - abs(uf)))
        for uf in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ]
    add_curved_tube(bm, chevron_pts, [0.004, 0.012, 0.015, 0.012, 0.004], radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_CYAN)

    # 3 small crystal crest fins topping the rear cantle
    _, R_c, U_c = body_frame(0.58)
    for uf_c, flen in ((-0.26, 0.088), (0.0, 0.115), (0.26, 0.088)):
        r_pt = _saddle_bridge_point(0.592, uf_c * 0.50, extra_off=0.225 * (1.0 - 0.35 * uf_c * uf_c))
        f_dir = (U_c * 0.65 + Vector((0.0, 1.0, 0.0)) * 0.72 + R_c * (uf_c * 0.25)).normalized()
        add_crystal_frill(
            bm, root=r_pt, forward_dir=f_dir, up_dir=U_c,
            length=flen, width=0.048, thickness=0.030, curl_up=0.12, segs=5,
            uv_layer=uv_layer, mat_index=_MAT_FRILL,
        )


def _add_side_flaps(bm, uv_layer):
    """Build 2-tier scalloped porcelain-white side skirts with rose-gold borders & cyan chevron runes."""
    for side in (-1.0, 1.0):
        # Tier 0 (upper skirt) and Tier 1 (lower scalloped flap) echoing Baby_Pauldrons
        tiers = [
            (0.415, 0.545, 0.44, 0.68, 0.028, 0.022, True),
            (0.425, 0.535, 0.62, 0.86, 0.020, 0.020, True),
        ]
        nu, nv = 9, 7
        for u0, u1, a0, a1, base_off, thick, has_rune in tiers:
            grid = []
            for i in range(nu):
                uf = i / float(nu - 1)
                row = []
                for j in range(nv):
                    vf = j / float(nv - 1)
                    uf_c = uf * 2.0 - 1.0
                    # Scalloped bottom cusp dipping gracefully downward at center
                    ang_val = a0 + (a1 - a0) * vf + 0.07 * (1.0 - uf_c * uf_c) * (vf ** 1.4)
                    u_eff = (u0 + (u1 - u0) * uf)
                    u_eff = 0.5 * (u0 + u1) + (u_eff - 0.5 * (u0 + u1)) * (1.0 - 0.14 * vf)
                    bulge = (1.0 - 0.45 * (uf_c * uf_c + ((vf - 0.5) * 1.4) ** 2)) * 0.024
                    row.append(_saddle_bridge_point(u_eff, side * ang_val, extra_off=base_off + max(0.0, bulge)))
                grid.append(row)

            add_double_walled_membrane(bm, grid, thickness=thick, uv_layer=uv_layer, mat_index=_MAT_PEARL)

            # 4-edge champagne rose-gold border trim
            trim_r = 0.010
            edges = (
                [grid[i][0] for i in range(nu)],
                [grid[i][-1] for i in range(nu)],
                grid[0],
                grid[-1],
            )
            for pts in edges:
                add_curved_tube(bm, pts, [trim_r] * len(pts), radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_GOLD)

            if has_rune:
                rune_pts = []
                for k in range(7):
                    kt = k / 6.0
                    uf_k = 0.20 + kt * 0.60
                    uf_c = uf_k * 2.0 - 1.0
                    vf_k = 0.52 - 0.08 * (1.0 - 2.0 * uf_c * uf_c)
                    ang_k = a0 + (a1 - a0) * vf_k + 0.04 * (1.0 - uf_c * uf_c)
                    u_k = u0 + (u1 - u0) * uf_k
                    rune_pts.append(_saddle_bridge_point(u_k, side * ang_k, extra_off=base_off + thick * 0.55 + 0.026))
                add_curved_tube(
                    bm, rune_pts, [0.004, 0.010, 0.013, 0.014, 0.013, 0.010, 0.004],
                    radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_CYAN, rx_scale=1.0, rz_scale=0.65,
                )


def _add_girth_straps_and_stirrups(bm, uv_layer):
    """Build twin contoured barrel girth straps clearing belly scutes, cinch buckles, belly clasp, and D-ring stirrups."""
    for u_s in (0.445, 0.518):
        n_th = 38
        band_half = 0.016
        band_thick = 0.014
        grid = [[], [], []]  # front, center, back along u
        for j in range(n_th):
            frac = j / float(n_th - 1)
            # Wrap from right saddle skirt (pi/2 - 0.76) around ventral belly (-pi/2) to left skirt (pi/2 + 0.76)
            theta = (math.pi * 0.5 - 0.76) - frac * (math.tau - 1.52)
            s = math.sin(theta)
            # Tight along flanks (0.018) so wing root clears cleanly, stepping proud over chevron belly scutes (+0.044)
            ventral_t = smoothstep(-0.15, -0.75, s)
            off = 0.018 + 0.026 * ventral_t
            p_f, _ = body_surface_point(u_s - band_half, theta, offset=off)
            p_c, _ = body_surface_point(u_s, theta, offset=off + 0.003)
            p_b, _ = body_surface_point(u_s + band_half, theta, offset=off)
            grid[0].append(p_f)
            grid[1].append(p_c)
            grid[2].append(p_b)

        add_double_walled_membrane(bm, grid, thickness=band_thick, uv_layer=uv_layer, mat_index=_MAT_VELVET)
        add_curved_tube(bm, grid[0], [0.006] * n_th, radial_segs=5, uv_layer=uv_layer, mat_index=_MAT_GOLD)
        add_curved_tube(bm, grid[2], [0.006] * n_th, radial_segs=5, uv_layer=uv_layer, mat_index=_MAT_GOLD)

        # Rose-gold cinch buckle frames on lower lateral flanks
        for side in (-1.0, 1.0):
            th_b = -0.22 if side > 0.0 else (math.pi + 0.22)
            T_b, R_b, U_b = body_frame(u_s)
            b_ctr, b_nrm = body_surface_point(u_s, th_b, offset=0.032)
            b_tan = (b_nrm.cross(T_b)).normalized()
            if b_tan.z < 0:
                b_tan = -b_tan
            du, dt = 0.024, 0.030
            rect = [
                b_ctr - T_b * du - b_tan * dt,
                b_ctr + T_b * du - b_tan * dt,
                b_ctr + T_b * du + b_tan * dt,
                b_ctr - T_b * du + b_tan * dt,
                b_ctr - T_b * du - b_tan * dt,
            ]
            add_curved_tube(bm, rect, [0.008] * 5, radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_GOLD)

    # Ventral pearl-white + rose-gold + cyan-gem clasp uniting the two girth straps beneath belly
    clasp_grid = []
    for i in range(5):
        u_c = 0.435 + (i / 4.0) * (0.528 - 0.435)
        row = []
        for j in range(5):
            th_c = -math.pi * 0.5 + ((j / 4.0) * 2.0 - 1.0) * 0.26
            p_c, _ = body_surface_point(u_c, th_c, offset=0.052)
            row.append(p_c)
        clasp_grid.append(row)
    add_double_walled_membrane(bm, clasp_grid, thickness=0.016, uv_layer=uv_layer, mat_index=_MAT_PEARL)
    gem_c, _ = body_surface_point(0.481, -math.pi * 0.5, offset=0.064)
    add_curved_tube(bm, [gem_c - Vector((0, 0.025, 0)), gem_c, gem_c + Vector((0, 0.025, 0))], [0.005, 0.018, 0.005], radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_CYAN)

    # Hanging velvet stirrup straps + rose-gold D-ring stirrups (safely clear between legs & below wings)
    for side in (-1.0, 1.0):
        p_top = _saddle_bridge_point(0.475, side * 0.72, extra_off=0.036)
        p_mid, n_mid = body_surface_point(0.475, math.pi * 0.5 + side * 1.18, offset=0.032)
        p_bot = Vector((p_mid.x + side * 0.015, p_mid.y + 0.01, p_mid.z - 0.16))
        strap_pts = [p_top, p_top.lerp(p_mid, 0.55) + n_mid * 0.015, p_mid + n_mid * 0.012, p_bot]
        add_curved_tube(bm, strap_pts, [0.013, 0.012, 0.012, 0.011], radial_segs=8, uv_layer=uv_layer, mat_index=_MAT_VELVET, rx_scale=1.35, rz_scale=0.52)

        # Rose-gold arched D-ring stirrup hanging vertically from p_bot
        st_top = p_bot + Vector((0.0, 0.0, -0.005))
        st_l_top = st_top + Vector((side * 0.006, -0.045, -0.040))
        st_l_bot = st_top + Vector((side * 0.008, -0.052, -0.095))
        st_r_bot = st_top + Vector((side * 0.008,  0.052, -0.095))
        st_r_top = st_top + Vector((side * 0.006,  0.045, -0.040))
        arch_pts = [st_r_top, st_top, st_l_top, st_l_bot, st_r_bot, st_r_top]
        add_curved_tube(bm, arch_pts, [0.009, 0.011, 0.009, 0.009, 0.009, 0.009], radial_segs=7, uv_layer=uv_layer, mat_index=_MAT_GOLD)
        # Flat footrest tread + cyan crown jewel
        tread_pts = [st_l_bot + Vector((side * 0.01, 0.005, 0.004)), st_r_bot + Vector((side * 0.01, -0.005, 0.004))]
        add_curved_tube(bm, tread_pts, [0.012, 0.012], radial_segs=7, uv_layer=uv_layer, mat_index=_MAT_PEARL, rx_scale=1.4, rz_scale=0.55)
        add_curved_tube(bm, [st_top + Vector((side * 0.010, -0.012, 0.0)), st_top + Vector((side * 0.010, 0.012, 0.0))], [0.009, 0.009], radial_segs=6, uv_layer=uv_layer, mat_index=_MAT_CYAN)


def build_saddle_object(materials: dict, collection, parent):
    """Build detachable Baby_Saddle (slots: 0=pearl_armor, 1=rose_gold, 2=ice_cyan_rune, 3=saddle_velvet, 4=frill_gradient)."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    _add_velvet_seat_deck(bm, uv_layer)
    _add_pommel_and_cantle(bm, uv_layer)
    _add_side_flaps(bm, uv_layer)
    _add_girth_straps_and_stirrups(bm, uv_layer)

    return finalize_mesh_object(
        "Baby_Saddle",
        bm,
        [
            materials["pearl_armor"],
            materials["rose_gold"],
            materials["ice_cyan_rune"],
            materials["saddle_velvet"],
            materials["frill_gradient"],
        ],
        collection,
        parent,
    )
