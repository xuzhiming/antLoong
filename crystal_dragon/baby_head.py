"""Sculpted Voxel-Remesh + Exact-Boolean chibi head, :D smiling mouth, recessed eyes with eyelids, horns, and frills."""

import math
import random
import bpy
import bmesh
from mathutils import Vector, Matrix

from baby_math import (
    smoothstep,
    catmull_rom_sample,
    loft_sections,
    add_curved_tube,
    add_crystal_frill,
    finalize_mesh_object,
)


def build_head_objects(materials: dict, collection, parent):
    head_obj = _build_sculpted_head(materials, collection, parent)
    return [
        head_obj,
        _build_baby_eyes_with_eyelids(materials, collection, parent),
        _build_baby_horns(materials, collection, parent),
        _build_head_frills(materials, collection, parent, head_obj=head_obj),
    ]


def _add_ellipsoid(bm, center, radii, rot_euler=(0.0, 0.0, 0.0), u_segs=26, v_segs=16, mat_index=0):
    """Add a smooth transformed ellipsoid into bm."""
    mat = (
        Matrix.Translation(Vector(center))
        @ Matrix.Rotation(rot_euler[2], 4, "Z")
        @ Matrix.Rotation(rot_euler[1], 4, "Y")
        @ Matrix.Rotation(rot_euler[0], 4, "X")
        @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    )
    ret = bmesh.ops.create_uvsphere(bm, u_segments=u_segs, v_segments=v_segs, radius=1.0, matrix=mat)
    verts = ret["verts"]
    vert_set = set(verts)
    for f in bm.faces:
        if all(v in vert_set for v in f.verts):
            f.material_index = mat_index
            f.smooth = True


def _add_oriented_ellipsoid(bm, center, ax_x, ax_y, ax_z, radii, u_segs=28, v_segs=18, mat_index=0):
    """Add a transformed ellipsoid aligned to orthonormal basis (ax_x, ax_y, ax_z)."""
    rot_m = Matrix((
        (ax_x.x, ax_y.x, ax_z.x, 0.0),
        (ax_x.y, ax_y.y, ax_z.y, 0.0),
        (ax_x.z, ax_y.z, ax_z.z, 0.0),
        (0.0,    0.0,    0.0,    1.0),
    ))
    mat = Matrix.Translation(Vector(center)) @ rot_m @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    ret = bmesh.ops.create_uvsphere(bm, u_segments=u_segs, v_segments=v_segs, radius=1.0, matrix=mat)
    vert_set = set(ret["verts"])
    for f in bm.faces:
        if all(v in vert_set for v in f.verts):
            f.material_index = mat_index
            f.smooth = True


def _build_mouth_cutter_bmesh():
    """Create watertight manifold :D smile cavity cutter mesh (y=-1.86 front -> y=-1.24 inside cheeks)."""
    bm = bmesh.new()
    # y stations from front open air (-1.86) to rear oral dome (-1.24)
    y_stations = [-1.86, -1.74, -1.62, -1.50, -1.40, -1.33, -1.28, -1.24]
    n_rad = 32
    rings = []
    for y in y_stations:
        # Smile corners terminate around y=-1.34; taper smoothly behind -1.35 into rear palate dome
        if y <= -1.36:
            w_scale = 1.0
            h_scale = 1.0
            z_shift = 0.0
        else:
            t_back = (y - (-1.36)) / (-1.24 - (-1.36))  # 0..1
            w_scale = max(0.18, math.cos(t_back * math.pi * 0.44))
            h_scale = max(0.18, math.cos(t_back * math.pi * 0.44))
            z_shift = 0.035 * t_back

        ring = []
        for j in range(n_rad):
            theta = -math.pi + (j / float(n_rad)) * math.tau
            u_x = math.cos(theta)  # -1..+1 across width
            s = math.sin(theta)    # +1 top lip, -1 bottom lip

            x = u_x * 0.212 * w_scale
            # :D upper lip: gentle Cupid's bow at center (1.655..1.665), upturned smile corners (1.695)
            z_top = 1.665 + 0.030 * (u_x * u_x) - 0.010 * math.exp(-((u_x / 0.25) ** 2))
            # :D lower lip: deep rounded U-arc dipping to 1.505 at center, meeting corners at 1.695
            z_bot = 1.505 + 0.190 * (u_x * u_x)

            z_mid = 0.5 * (z_top + z_bot)
            half_h = 0.5 * (z_top - z_bot) + 0.005  # tiny corner thickness keeps manifold non-degenerate
            z = z_shift + z_mid + s * half_h * h_scale
            ring.append(Vector((x, y, z)))
        rings.append(ring)

    loft_sections(bm, rings, cap_start=True, cap_end=True, uv_layer=None, mat_index=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _make_temp_obj(name, bm, materials, collection):
    mesh = bpy.data.meshes.new(name + "_TempMesh")
    bm.to_mesh(mesh)
    bm.free()
    for m in materials:
        mesh.materials.append(m)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _cleanup_temp_objs(objs):
    for obj in objs:
        if obj is None:
            continue
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _eye_local_frame(side: float):
    """Mirror-symmetrical eye socket local frame: out_dir always points away from nose."""
    eye_center = Vector((side * 0.195, -1.355, 1.885))
    gaze = Vector((side * 0.42, -0.907, 0.01)).normalized()
    raw_right = Vector((0.0, 0.0, 1.0)).cross(gaze).normalized()
    out_dir = raw_right * side
    true_up = gaze.cross(raw_right).normalized()
    return eye_center, gaze, out_dir, true_up


def _almond_eye_aperture(t: float):
    """Return (s, z_upper, z_lower) for happy-arch almond eye (t=0 inner -> t=1 outer corner)."""
    tc = max(0.0, min(1.0, float(t)))
    s = -0.096 + t * 0.194
    z_base = -0.028 + tc * 0.018
    s_sin = math.sin(tc * math.pi)
    z_upper = z_base + 0.110 * (s_sin ** 0.72) * (1.0 + 0.08 * (tc - 0.5))
    z_lower = z_base - 0.032 * s_sin + 0.012 * (s_sin ** 2.4)
    return s, z_upper, z_lower


def _build_eye_socket_cutter_bmesh():
    """Create voxel-safe manifold almond-arch socket Boolean cutter calibrated for post-Boolean smoothing."""
    bm = bmesh.new()
    d_stations = [0.00, 0.030, 0.060, 0.085, 0.115, 0.165]
    scale_stations = [0.92, 0.96, 0.95, 0.98, 1.02, 1.06]
    n_rad = 32

    for side in (-1.0, 1.0):
        eye_center, gaze, out_dir, true_up = _eye_local_frame(side)
        rings = []
        for d, scale in zip(d_stations, scale_stations):
            ring = []
            for j in range(n_rad):
                theta = -math.pi + (j / float(n_rad)) * math.tau
                u_t = 0.5 * (1.0 - math.cos(theta))
                s, z_u, z_l = _almond_eye_aperture(u_t)
                z_mid = 0.5 * (z_u + z_l)
                half_h = 0.5 * (z_u - z_l) + 0.010
                z = z_mid + math.sin(theta) * half_h
                pos = eye_center + gaze * d + out_dir * (s * scale) + true_up * (z_mid + (z - z_mid) * scale)
                ring.append(pos)
            rings.append(ring)
        loft_sections(bm, rings, cap_start=True, cap_end=True, uv_layer=None, mat_index=0)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _build_sculpted_head(materials, collection, parent):
    """Build unified voxel-remeshed & Boolean-carved clay chibi head + tongue, fangs & flush cyan rune."""
    head_mats = [
        materials["face_lavender"],    # 0
        materials["inner_mouth"],      # 1
        materials["pink_pad"],         # 2
        materials["tooth_white"],      # 3
        materials["ice_cyan_rune"],    # 4
        materials["nostril_dark"],     # 5
    ]

    # 1. Positive clay head volumes fused together (sleekly tapered cheeks & jaw around mouth corners)
    bm_clay = bmesh.new()
    # Tall rounded cranium dome
    _add_ellipsoid(bm_clay, (0.0, -1.08, 1.90), (0.42, 0.41, 0.35))
    # Sleeker puppy-dragon muzzle
    _add_ellipsoid(bm_clay, (0.0, -1.49, 1.77), (0.225, 0.265, 0.132))
    # Neck base collar blending cleanly into Baby_Body upright neck
    _add_ellipsoid(bm_clay, (0.0, -0.98, 1.62), (0.28, 0.27, 0.24))
    # Tucked slim lower chin & jaw
    _add_ellipsoid(bm_clay, (0.0, -1.41, 1.525), (0.185, 0.225, 0.095))
    for side in (-1.0, 1.0):
        # Twin upper-lip / whisker pads
        _add_ellipsoid(bm_clay, (side * 0.095, -1.55, 1.725), (0.105, 0.145, 0.068))
        # Slimmed higher cheekbones under outer eye orbit (eliminates puffy mouth-corner cheeks)
        _add_ellipsoid(bm_clay, (side * 0.215, -1.25, 1.755), (0.145, 0.195, 0.118))
        # Soft rounded brow pads
        _add_ellipsoid(bm_clay, (side * 0.17, -1.34, 2.02), (0.135, 0.155, 0.082))
        # Twin slimmed jaw bridges uniting chin and cheekbones
        _add_ellipsoid(bm_clay, (side * 0.135, -1.29, 1.565), (0.095, 0.155, 0.092))
        # 360-degree conformal orbital housing completely wrapping each eyeball before almond carving
        eye_c, gaze, out_dir, true_up = _eye_local_frame(side)
        raw_right = out_dir * side
        _add_oriented_ellipsoid(bm_clay, eye_c, raw_right, -gaze, true_up, (0.156, 0.122, 0.144), u_segs=28, v_segs=18)

    bmesh.ops.recalc_face_normals(bm_clay, faces=bm_clay.faces)
    head_obj = _make_temp_obj("Baby_Head_Temp", bm_clay, head_mats, collection)

    # Stage A: Fuse clay into continuous watertight head, then carve almond sockets & re-voxel smooth corners
    mod_remesh = head_obj.modifiers.new("VoxelRemesh", type="REMESH")
    mod_remesh.mode = "VOXEL"
    mod_remesh.voxel_size = 0.014
    mod_remesh.use_smooth_shade = True

    mod_smooth = head_obj.modifiers.new("ClaySmooth", type="SMOOTH")
    mod_smooth.factor = 0.5
    mod_smooth.iterations = 6

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    baked_clay = bpy.data.meshes.new_from_object(head_obj.evaluated_get(depsgraph))
    head_obj.modifiers.clear()
    old_m = head_obj.data
    head_obj.data = baked_clay
    bpy.data.meshes.remove(old_m)

    bm_sockets = _build_eye_socket_cutter_bmesh()
    socket_cutter = _make_temp_obj("Temp_Socket_Cutter", bm_sockets, [materials["face_lavender"]], collection)

    mod_sock = head_obj.modifiers.new("BoolSockets", type="BOOLEAN")
    mod_sock.operation = "DIFFERENCE"
    mod_sock.solver = "EXACT"
    mod_sock.object = socket_cutter

    mod_srem = head_obj.modifiers.new("SocketRemesh", type="REMESH")
    mod_srem.mode = "VOXEL"
    mod_srem.voxel_size = 0.012
    mod_srem.use_smooth_shade = True

    mod_ssmooth = head_obj.modifiers.new("SocketSmooth", type="SMOOTH")
    mod_ssmooth.factor = 0.45
    mod_ssmooth.iterations = 5

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    baked_sockets = bpy.data.meshes.new_from_object(head_obj.evaluated_get(depsgraph))
    head_obj.modifiers.clear()
    old_m_sock = head_obj.data
    head_obj.data = baked_sockets
    bpy.data.meshes.remove(old_m_sock)
    _cleanup_temp_objs([socket_cutter])

    # Stage B: Carve :D Smile Mouth Cavity and Dark Nostril Pits (preserving exact material slots)
    bm_mouth = _build_mouth_cutter_bmesh()
    mouth_cutter = _make_temp_obj("Temp_Mouth_Cutter", bm_mouth, [materials["inner_mouth"]], collection)

    bm_nostrils = bmesh.new()
    for side in (-1.0, 1.0):
        _add_ellipsoid(
            bm_nostrils,
            (side * 0.092, -1.695, 1.815),
            (0.026, 0.048, 0.020),
            rot_euler=(math.radians(-15.0), 0.0, side * math.radians(-24.0)),
            u_segs=18,
            v_segs=12,
        )
    bmesh.ops.recalc_face_normals(bm_nostrils, faces=bm_nostrils.faces)
    nostril_cutter = _make_temp_obj("Temp_Nostril_Cutter", bm_nostrils, [materials["nostril_dark"]], collection)

    for cname, cobj in [("BoolMouth", mouth_cutter), ("BoolNostrils", nostril_cutter)]:
        bm_mod = head_obj.modifiers.new(cname, type="BOOLEAN")
        bm_mod.operation = "DIFFERENCE"
        bm_mod.solver = "EXACT"
        bm_mod.object = cobj

    cs_mod = head_obj.modifiers.new("CorrectiveSmooth", type="CORRECTIVE_SMOOTH")
    cs_mod.factor = 0.5
    cs_mod.iterations = 2

    dec_mod = head_obj.modifiers.new("DecimateClean", type="DECIMATE")
    dec_mod.decimate_type = "COLLAPSE"
    dec_mod.ratio = 0.48

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    eval_head = head_obj.evaluated_get(depsgraph)
    final_head_mesh = bpy.data.meshes.new_from_object(eval_head)
    head_obj.modifiers.clear()
    old_m2 = head_obj.data
    head_obj.data = final_head_mesh
    bpy.data.meshes.remove(old_m2)

    _cleanup_temp_objs([mouth_cutter, nostril_cutter])

    # Load carved head into BMesh for clean UV projection + intra-oral & surface details
    bm = bmesh.new()
    bm.from_mesh(head_obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    # Smooth spherical/cylindrical UV projection for delicate baby-dragon facial scales
    uv_center = Vector((0.0, -1.16, 1.80))
    for f in bm.faces:
        f.smooth = True
        for loop in f.loops:
            p = loop.vert.co - uv_center
            ang = math.atan2(p.x, -p.y)
            u_uv = (ang / math.pi) * 0.95 + 0.5
            v_uv = (p.z + 0.45) * 1.45
            loop[uv_layer].uv = (u_uv, v_uv)

    # 3. Sculpted Grooved Rosy-Pink Tongue (slot 2) strictly INSIDE carved :D lower jaw cavity
    t_rings = []
    n_ty = 12
    for i in range(n_ty):
        tt = i / float(n_ty - 1)  # 0=front tip inside lower lip, 1=back inside throat
        y = -1.58 + tt * 0.28
        # Gentle dome arch tapering down cleanly behind lower lip
        z_base = 1.525 + 0.045 * math.sin(tt * math.pi * 0.85)
        w_t = 0.115 * math.sin(max(0.10, tt) * math.pi * 0.92)
        h_t = 0.026 * math.sin(max(0.12, tt) * math.pi * 0.95)
        ring = []
        for j in range(16):
            ang = (j / 16.0) * math.tau
            cx, sz = math.cos(ang), math.sin(ang)
            pz = sz * h_t
            # Cute central longitudinal tongue groove on upper surface
            if sz > 0.0:
                groove = 0.011 * math.exp(-((cx / 0.32) ** 2)) * sz
                pz -= groove
            ring.append(Vector((cx * w_t, y, z_base + pz)))
        t_rings.append(ring)
    loft_sections(bm, t_rings, cap_start=True, cap_end=True, uv_layer=uv_layer, mat_index=2)

    # 4. 4 Crisp Cute White Baby Teeth/Fangs (slot 3) anchored cleanly in upper gumline
    for side in (-1.0, 1.0):
        for fx, fy, fz, fl, fr in [
            (0.145, -1.53, 1.675, 0.044, 0.014),  # outer cute baby canine
            (0.068, -1.61, 1.668, 0.024, 0.010),  # tiny inner baby front tooth
        ]:
            p0 = Vector((side * fx, fy, fz + 0.012))
            p1 = Vector((side * fx, fy, fz - fl * 0.45))
            p2 = Vector((side * fx * 0.95, fy - 0.004, fz - fl))
            add_curved_tube(bm, [p0, p1, p2], [fr, fr * 0.65, 0.002], radial_segs=8, uv_layer=uv_layer, mat_index=3)

    # 5. Ray-cast Flush Glowing Ice-Cyan Forehead Diamond Emblem + jewel accents + 4 flanking chevrons (slot 4)
    def _snap_to_forehead(x, y, z_guess, push=0.007):
        origin = Vector((x, y - 0.45, z_guess + 0.45))
        direction = Vector((0.0, 0.72, -0.69)).normalized()
        hit, loc, norm, _ = head_obj.ray_cast(origin, direction)
        if hit:
            return loc + norm * push
        return Vector((x, y, z_guess))

    rune_pts = [
        _snap_to_forehead(0.0, -1.45, 1.96, 0.007),
        _snap_to_forehead(0.0, -1.40, 2.005, 0.008),
        _snap_to_forehead(0.0, -1.35, 2.05, 0.009),
        _snap_to_forehead(0.0, -1.30, 2.09, 0.008),
        _snap_to_forehead(0.0, -1.25, 2.125, 0.007),
    ]
    add_curved_tube(
        bm,
        rune_pts,
        [0.003, 0.022, 0.046, 0.022, 0.003],
        radial_segs=4,
        uv_layer=uv_layer,
        mat_index=4,
        rx_scale=1.25,
        rz_scale=0.35,
    )

    # Small diamond jewel accents immediately above and below main emblem
    for jy, jz in [(-1.21, 2.145), (-1.49, 1.925)]:
        j_pts = [
            _snap_to_forehead(0.0, jy - 0.016, jz - 0.014, 0.007),
            _snap_to_forehead(0.0, jy, jz, 0.008),
            _snap_to_forehead(0.0, jy + 0.016, jz + 0.014, 0.007),
        ]
        add_curved_tube(
            bm,
            j_pts,
            [0.002, 0.015, 0.002],
            radial_segs=4,
            uv_layer=uv_layer,
            mat_index=4,
            rx_scale=1.15,
            rz_scale=0.35,
        )

    for side in (-1.0, 1.0):
        for x0, y0, z0, x1, y1, z1, r_sz in [
            (0.066, -1.37, 2.02, 0.122, -1.32, 2.05, 0.014),
            (0.072, -1.43, 1.98, 0.128, -1.38, 2.01, 0.012),
        ]:
            c_pts = [
                _snap_to_forehead(side * x0, y0, z0, 0.007),
                _snap_to_forehead(side * 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1), 0.007),
                _snap_to_forehead(side * x1, y1, z1, 0.007),
            ]
            add_curved_tube(bm, c_pts, [r_sz, r_sz * 0.72, 0.003], radial_segs=5, uv_layer=uv_layer, mat_index=4)

    _cleanup_temp_objs([head_obj])

    return finalize_mesh_object(
        "Baby_Head",
        bm,
        head_mats,
        collection,
        parent,
    )


def _build_baby_eyes_with_eyelids(materials, collection, parent):
    """Build compact almond-conformal ellipsoid glossy aqua eyes & snug upper eyeliner inside head sockets."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    # Slots: 0=baby_eye, 1=dark_lash

    # Compact ellipsoid seated cleanly INSIDE Baby_Head's 0.156 x 0.144 x 0.122 orbital housing
    r_out = 0.132
    r_up = 0.122
    r_gaze = 0.102

    s_c = 0.002
    z_c = 0.024

    for side in (-1.0, 1.0):
        eye_center, gaze, out_dir, true_up = _eye_local_frame(side)

        # Front-clustered dense ellipsoid mesh + world-space almond iris UV projection
        rings_n = 28
        segs_n = 36
        verts_grid = []
        uv_vals = []
        for i in range(rings_n + 1):
            phi = math.pi * ((i / float(rings_n)) ** 1.25)
            d_g = math.cos(phi) * r_gaze
            sin_p = math.sin(phi)
            v_row = []
            uv_row = []
            for j in range(segs_n):
                ang = (j / float(segs_n)) * math.tau
                s_p = sin_p * math.cos(ang) * r_out
                z_p = sin_p * math.sin(ang) * r_up
                p = eye_center + gaze * d_g + out_dir * s_p + true_up * z_p
                v_row.append(bm.verts.new(p))

                du = ((s_p - s_c) / 0.078) * 0.43 * side
                dv = ((z_p - z_c) / 0.078) * 0.43
                if d_g < 0.0:
                    rad_uv = math.hypot(du, dv)
                    if rad_uv < 0.50:
                        scale_back = 0.50 / max(1e-4, rad_uv)
                        du *= scale_back
                        dv *= scale_back
                u_uv = max(0.01, min(0.99, 0.5 + du))
                v_uv = max(0.01, min(0.99, 0.5 + dv))
                uv_row.append((u_uv, v_uv))
            verts_grid.append(v_row)
            uv_vals.append(uv_row)

        for i in range(rings_n):
            for j in range(segs_n):
                jn = (j + 1) % segs_n
                try:
                    f = bm.faces.new((verts_grid[i][j], verts_grid[i][jn], verts_grid[i + 1][jn], verts_grid[i + 1][j]))
                    f.material_index = 0
                    f.smooth = True
                    f.loops[0][uv_layer].uv = uv_vals[i][j]
                    f.loops[1][uv_layer].uv = uv_vals[i][jn]
                    f.loops[2][uv_layer].uv = uv_vals[i + 1][jn]
                    f.loops[3][uv_layer].uv = uv_vals[i + 1][j]
                except ValueError:
                    pass

        def _proj_eye(s: float, z: float, pad: float = 0.0) -> Vector:
            ro = r_out + pad
            ru = r_up + pad
            rg = r_gaze + pad
            q = max(1e-4, 1.0 - (s / ro) ** 2 - (z / ru) ** 2)
            d_g = rg * math.sqrt(q)
            return eye_center + gaze * d_g + out_dir * s + true_up * z

        # Snug Crisp Dark-Indigo Almond Upper Eyeliner + Outer Wing (slot 1)
        lash_pts = []
        lash_r = []
        for k in range(17):
            kt = k / 16.0
            t = 0.02 + kt * 0.98
            s, z_u, _ = _almond_eye_aperture(t)
            lash_pts.append(_proj_eye(s, z_u, 0.003))
            lash_r.append(0.003 + 0.004 * kt + 0.0065 * math.sin(kt * math.pi))
        lash_pts.append(_proj_eye(0.105, -0.003, 0.006))
        lash_pts.append(_proj_eye(0.114,  0.006, 0.006))
        lash_r.extend([0.006, 0.0025])
        add_curved_tube(bm, lash_pts, lash_r, radial_segs=7, uv_layer=uv_layer, mat_index=1)

        # Subtle dark outer-lower corner lash accent sharpening the almond outer angle
        low_lash_pts = []
        for k in range(6):
            kt = 0.70 + 0.30 * (k / 5.0)
            s, _, z_l = _almond_eye_aperture(kt)
            low_lash_pts.append(_proj_eye(s, z_l, 0.003))
        low_lash_r = [0.0018, 0.0028, 0.0036, 0.0038, 0.0032, 0.0025]
        add_curved_tube(bm, low_lash_pts, low_lash_r, radial_segs=6, uv_layer=uv_layer, mat_index=1)

    return finalize_mesh_object(
        "Baby_Eyes",
        bm,
        [materials["baby_eye"], materials["dark_lash"]],
        collection,
        parent,
    )


def _build_baby_horns(materials, collection, parent):
    """Build sweeping 8-segment ringed crescent horns seated cleanly into cranium."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    for side in (-1.0, 1.0):
        ctrl = [
            (side * 0.21, -1.20, 2.10),  # deeply rooted in upper cranium
            (side * 0.37, -1.07, 2.25),  # sweeping outward-upward-back
            (side * 0.46, -0.92, 2.40),  # widest crescent bow
            (side * 0.40, -0.83, 2.49),  # curving high and inward
            (side * 0.27, -0.86, 2.47),  # inward-curled rounded tip
        ]
        pts = catmull_rom_sample(ctrl, 96)
        rads = []
        for k in range(96):
            u = k / 95.0
            r_base = 0.114 - 0.096 * (u ** 0.88)
            if u < 0.88:
                frac = ((u / 0.88) * 8.0) % 1.0
                ring_dome = math.sin((max(1e-6, frac) ** 0.82) * math.pi) ** 0.52
                fade = smoothstep(0.0, 0.04, u) * (1.0 - smoothstep(0.84, 0.88, u))
                r_eff = r_base * (1.0 + (-0.155 + 0.225 * ring_dome) * fade)
            else:
                r_eff = r_base * (1.0 - 0.55 * smoothstep(0.92, 1.0, u))
            rads.append(max(0.006, r_eff))
        add_curved_tube(
            bm,
            pts,
            rads,
            radial_segs=20,
            ring_ridge_amp=0.0,
            ring_ridge_freq=0.0,
            uv_layer=uv_layer,
            mat_index=0,
        )

    return finalize_mesh_object(
        "Baby_Horns",
        bm,
        [materials["horn_pearl"]],
        collection,
        parent,
    )


def _build_head_frills(materials, collection, parent, head_obj=None):
    """Build plump crown spiky tuft, layered cheek/ear frills, and lush REAR crystal head rosette."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rng = random.Random(2026)

    # 1. Crown spiky tuft rising between the two horns
    crown_specs = [
        ( 0.00, -1.16, 2.18,  0.00,  0.22,  0.98, 0.26, 0.140, 0.074),
        (-0.08, -1.12, 2.16, -0.18,  0.25,  0.95, 0.23, 0.125, 0.066),
        ( 0.08, -1.12, 2.16,  0.18,  0.25,  0.95, 0.23, 0.125, 0.066),
        ( 0.00, -1.04, 2.16,  0.00,  0.48,  0.88, 0.27, 0.140, 0.074),
        (-0.10, -1.00, 2.14, -0.22,  0.52,  0.82, 0.24, 0.128, 0.068),
        ( 0.10, -1.00, 2.14,  0.22,  0.52,  0.82, 0.24, 0.128, 0.068),
        ( 0.00, -0.94, 2.12,  0.00,  0.65,  0.76, 0.26, 0.135, 0.070),
    ]
    for cx, cy, cz, fx, fy, fz, ln, wd, th in crown_specs:
        add_crystal_frill(
            bm,
            root=Vector((cx, cy, cz)),
            forward_dir=Vector((fx, fy, fz)),
            up_dir=Vector((0.0, 0.1, 1.0)),
            length=ln,
            width=wd,
            thickness=th,
            curl_up=-0.08,
            segs=6,
            uv_layer=uv_layer,
            mat_index=0,
        )

    # 2. Radiating Cheek & Ear Frills + secondary inner layered cheek row
    for side in (-1.0, 1.0):
        n_ear = 7
        for k in range(n_ear):
            kt = k / float(n_ear - 1)
            ang = math.radians(52.0) - kt * math.radians(102.0)
            root = Vector((
                side * (0.31 + 0.045 * math.cos(ang) - 0.05 * kt),
                -1.12 + 0.10 * kt,
                1.86 + 0.17 * math.sin(ang),
            ))
            f_dir = Vector((
                side * (0.76 + 0.12 * math.cos(ang)),
                0.46 + 0.10 * kt,
                0.38 * math.sin(ang) - 0.10 * kt,
            )).normalized()
            add_crystal_frill(
                bm,
                root=root,
                forward_dir=f_dir,
                up_dir=Vector((side * math.cos(ang), 0.15, math.sin(ang))).normalized(),
                length=0.29 + 0.06 * math.sin(kt * math.pi),
                width=0.145,
                thickness=0.072,
                curl_up=0.14,
                curl_side=side * 0.04,
                segs=6,
                uv_layer=uv_layer,
                mat_index=0,
            )

        # Secondary forward-offset layered cheek petals for rich pink-tipped crystal fan
        n_inner = 5
        for k in range(n_inner):
            kt = k / float(n_inner - 1)
            ang = math.radians(40.0) - kt * math.radians(82.0)
            root = Vector((
                side * (0.30 + 0.04 * math.cos(ang) - 0.05 * kt),
                -1.17 + 0.08 * kt,
                1.84 + 0.14 * math.sin(ang),
            ))
            f_dir = Vector((
                side * (0.80 + 0.10 * math.cos(ang)),
                0.38 + 0.10 * kt,
                0.34 * math.sin(ang) - 0.10 * kt,
            )).normalized()
            add_crystal_frill(
                bm,
                root=root,
                forward_dir=f_dir,
                up_dir=Vector((side * math.cos(ang), 0.18, math.sin(ang))).normalized(),
                length=0.21 + 0.04 * math.sin(kt * math.pi),
                width=0.12,
                thickness=0.064,
                curl_up=0.12,
                curl_side=side * 0.03,
                segs=6,
                uv_layer=uv_layer,
                mat_index=0,
            )

    # 3. Hemispherical Surface-Snapped Back-of-Head Crystal Rosette (31 overlapping lotus petals)
    cranium_c = Vector((0.0, -1.08, 1.90))
    rings = [
        ( 0.0,  1, 0.22, 0.150, 0.075),
        (20.0,  6, 0.23, 0.145, 0.072),
        (40.0, 10, 0.25, 0.145, 0.072),
        (58.0, 14, 0.24, 0.135, 0.068),
    ]
    for alpha_deg, count, f_len, f_w, f_th in rings:
        alpha = math.radians(alpha_deg)
        for j in range(count):
            ang = (j / float(count)) * math.tau + alpha * 1.4
            probe = cranium_c + Vector((
                math.sin(alpha) * math.cos(ang) * 0.44,
                math.cos(alpha) * 0.44,
                math.sin(alpha) * math.sin(ang) * 0.38,
            ))
            if head_obj is not None:
                ok, loc, nrm, _ = head_obj.closest_point_on_mesh(probe)
                if ok:
                    surf_nrm = nrm.normalized()
                    root = loc - surf_nrm * 0.042
                else:
                    surf_nrm = (probe - cranium_c).normalized()
                    root = probe - surf_nrm * 0.042
            else:
                surf_nrm = (probe - cranium_c).normalized()
                root = probe - surf_nrm * 0.042

            rad_out = Vector((
                math.cos(ang) * math.sin(alpha),
                0.72 * math.cos(alpha) + 0.45,
                math.sin(ang) * math.sin(alpha) * 0.85 - 0.18,
            )).normalized()
            f_dir = (surf_nrm * 0.38 + rad_out * 0.62).normalized()
            add_crystal_frill(
                bm,
                root=root,
                forward_dir=f_dir,
                up_dir=surf_nrm,
                length=f_len + rng.uniform(-0.015, 0.015),
                width=f_w,
                thickness=f_th,
                curl_up=0.06,
                segs=5,
                uv_layer=uv_layer,
                mat_index=0,
            )

    return finalize_mesh_object(
        "Baby_Head_Frills",
        bm,
        [materials["frill_gradient"]],
        collection,
        parent,
    )
