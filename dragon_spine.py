"""Spine math, orthonormal frames, cross-sections, and reusable BMesh builders."""

import math
import bpy
import bmesh
from mathutils import Vector, Matrix


# Control points: (u, x, y, z)
_SPINE_KEYS = [
    (0.00,  0.00, -5.00, 2.04),
    (0.08,  0.00, -4.25, 2.05),
    (0.16,  0.00, -3.50, 1.98),
    (0.19,  0.00, -3.18, 1.70),
    (0.23,  0.00, -2.74, 1.18),
    (0.27,  0.00, -2.30, 0.88),
    (0.32,  0.10, -1.80, 1.12),
    (0.37,  0.15, -1.30, 1.44),
    (0.44,  0.02, -0.58, 1.64),
    (0.50, -0.17,  0.02, 1.50),
    (0.56, -0.02,  0.65, 1.22),
    (0.60,  0.15,  1.06, 1.10),
    (0.64,  0.16,  1.46, 1.22),
    (0.71,  0.01,  2.16, 1.48),
    (0.80,  0.08,  3.06, 1.26),
    (0.85,  0.05,  3.55, 1.14),
    (0.91, -0.03,  4.14, 1.35),
    (0.96, -0.09,  4.62, 1.65),
    (1.00, -0.10,  5.00, 1.92),
]

# Body radii keys: (u, rx, rz)
_RADII_KEYS = [
    (0.15, 0.35, 0.33),
    (0.17, 0.37, 0.35),
    (0.21, 0.45, 0.41),
    (0.25, 0.52, 0.45),
    (0.30, 0.49, 0.43),
    (0.38, 0.48, 0.41),
    (0.45, 0.47, 0.40),
    (0.53, 0.44, 0.38),
    (0.60, 0.41, 0.36),
    (0.66, 0.43, 0.37),
    (0.70, 0.44, 0.38),
    (0.75, 0.38, 0.33),
    (0.80, 0.29, 0.26),
    (0.85, 0.22, 0.20),
    (0.90, 0.16, 0.14),
    (0.95, 0.085, 0.080),
    (0.985, 0.035, 0.035),
    (1.00, 0.020, 0.020),
]


def _hermite_eval(keys, u, value_indices):
    """Evaluate C1 cubic Hermite spline with monotonic-clamped tangents."""
    n = len(keys)
    if u <= keys[0][0]:
        return [keys[0][k] for k in value_indices]
    if u >= keys[-1][0]:
        return [keys[-1][k] for k in value_indices]

    i = 0
    for idx in range(n - 1):
        if keys[idx][0] <= u <= keys[idx + 1][0]:
            i = idx
            break

    u0 = keys[i][0]
    u1 = keys[i + 1][0]
    du = max(u1 - u0, 1e-8)
    t = (u - u0) / du
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    out = []
    for k in value_indices:
        p0 = keys[i][k]
        p1 = keys[i + 1][k]
        if i == 0:
            m0 = (keys[1][k] - keys[0][k]) / max(keys[1][0] - keys[0][0], 1e-8)
        else:
            m0 = (keys[i + 1][k] - keys[i - 1][k]) / max(keys[i + 1][0] - keys[i - 1][0], 1e-8)
        if i + 1 == n - 1:
            m1 = (keys[-1][k] - keys[-2][k]) / max(keys[-1][0] - keys[-2][0], 1e-8)
        else:
            m1 = (keys[i + 2][k] - keys[i][k]) / max(keys[i + 2][0] - keys[i][0], 1e-8)
        val = h00 * p0 + h10 * du * m0 + h01 * p1 + h11 * du * m1
        out.append(val)
    return out


def spine_point(u: float) -> Vector:
    """Return 3D centerline Vector(x, y, z) for u in [0, 1]."""
    u_clamped = max(0.0, min(1.0, float(u)))
    x, y, z = _hermite_eval(_SPINE_KEYS, u_clamped, (1, 2, 3))
    return Vector((x, y, z))


def spine_tangent(u: float, eps: float = 0.002) -> Vector:
    """Return unit forward-to-tail tangent vector along spine."""
    u0 = max(0.0, u - eps)
    u1 = min(1.0, u + eps)
    if u1 - u0 < 1e-6:
        return Vector((0.0, 1.0, 0.0))
    t = spine_point(u1) - spine_point(u0)
    if t.length < 1e-8:
        return Vector((0.0, 1.0, 0.0))
    return t.normalized()


def spine_frame(u: float):
    """Return orthonormal frame (T, R, U) where T points toward tail, R toward dragon's left (+X), U dorsal."""
    T = spine_tangent(u)
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(T.dot(world_up)) > 0.96:
        world_up = Vector((0.0, -1.0, 0.0))
    # For T along +Y and U along +Z: R = U x T -> +Z x +Y = -X, so T x U = +Y x +Z = +X (dragon left)
    R = T.cross(world_up)
    if R.length < 1e-8:
        R = Vector((1.0, 0.0, 0.0))
    else:
        R.normalize()
    U = R.cross(T).normalized()
    return T, R, U


def body_radii(u: float):
    """Return (rx, rz) body cross-section semi-axes for u in [0, 1]."""
    u_clamped = max(0.0, min(1.0, float(u)))
    rx, rz = _hermite_eval(_RADII_KEYS, u_clamped, (1, 2))
    return max(0.015, rx), max(0.015, rz)


def body_surface_point(u: float, theta: float, scale_rx: float = 1.0, scale_rz: float = 1.0, offset: float = 0.0):
    """Compute anatomical rounded-D cross-section surface point and outward normal.

    theta = 0 is dragon left (+R), +pi/2 is dorsal (+U), -pi/2 is ventral (-U), pi is dragon right (-R).
    """
    center = spine_point(u)
    T, R, U = spine_frame(u)
    rx, rz = body_radii(u)
    rx *= scale_rx
    rz *= scale_rz

    c = math.cos(theta)
    s = math.sin(theta)

    # Anatomical shaping:
    # - Slightly flatten ventral belly around theta = -pi/2 (s < 0)
    # - Subtle muscular flank fullness at |s| ~ 0.2..0.5
    # - Subtle dorsal ridge crest around theta = +pi/2
    local_r_mod = 1.0
    if s < 0.0:
        # Flatten belly bottom gently so scutes cradle nicely
        belly_w = (-s) ** 2.2
        local_r_mod -= 0.14 * belly_w
    else:
        # Dorsal ridge peaking softly at theta = +pi/2
        dorsal_w = max(0.0, s) ** 6.0
        local_r_mod += 0.06 * dorsal_w

    # Subtle lower-flank seam tuck where cyan conduit sits (around sin(theta) ~ -0.35)
    seam_dist = (s + 0.34) / 0.14
    local_r_mod -= 0.03 * math.exp(-seam_dist * seam_dist)

    radial_vec = R * (c * rx * local_r_mod) + U * (s * rz * local_r_mod)
    out_dir = (R * (c / max(rx, 1e-4)) + U * (s / max(rz, 1e-4)))
    if out_dir.length < 1e-8:
        out_dir = U
    else:
        out_dir.normalize()
    pos = center + radial_vec + out_dir * offset
    return pos, out_dir


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-8:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def loft_sections(
    bm: bmesh.types.BMesh,
    rings,
    cap_start: bool = True,
    cap_end: bool = True,
    uv_layer=None,
    mat_index: int = 0,
    u_coords=None,
    v_coords=None,
):
    """Loft a list of closed vertex-position rings (each length M) into clean quads.

    Returns (bm_rings, created_faces).
    """
    n_rings = len(rings)
    if n_rings < 2:
        return [], []
    m = len(rings[0])
    if u_coords is None:
        u_coords = [i / max(1, n_rings - 1) for i in range(n_rings)]
    if v_coords is None:
        v_coords = [j / float(m) for j in range(m + 1)]

    bm_rings = []
    for ring in rings:
        bm_rings.append([bm.verts.new(p) for p in ring])

    faces = []
    for i in range(n_rings - 1):
        r0 = bm_rings[i]
        r1 = bm_rings[i + 1]
        u0 = u_coords[i]
        u1 = u_coords[i + 1]
        for j in range(m):
            jn = (j + 1) % m
            v0 = r0[j]
            v1 = r0[jn]
            v2 = r1[jn]
            v3 = r1[j]
            try:
                f = bm.faces.new((v0, v1, v2, v3))
            except ValueError:
                continue
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                vv0 = v_coords[j]
                vv1 = v_coords[j + 1]
                f.loops[0][uv_layer].uv = (u0, vv0)
                f.loops[1][uv_layer].uv = (u0, vv1)
                f.loops[2][uv_layer].uv = (u1, vv1)
                f.loops[3][uv_layer].uv = (u1, vv0)
            faces.append(f)

    if cap_start and m >= 3:
        center_pos = sum((Vector(p) for p in rings[0]), Vector((0.0, 0.0, 0.0))) / m
        vc = bm.verts.new(center_pos)
        r0 = bm_rings[0]
        for j in range(m):
            jn = (j + 1) % m
            try:
                f = bm.faces.new((vc, r0[jn], r0[j]))
                f.material_index = mat_index
                f.smooth = True
                if uv_layer is not None:
                    f.loops[0][uv_layer].uv = (u_coords[0], 0.5)
                    f.loops[1][uv_layer].uv = (u_coords[0], v_coords[j + 1])
                    f.loops[2][uv_layer].uv = (u_coords[0], v_coords[j])
                faces.append(f)
            except ValueError:
                pass

    if cap_end and m >= 3:
        center_pos = sum((Vector(p) for p in rings[-1]), Vector((0.0, 0.0, 0.0))) / m
        vc = bm.verts.new(center_pos)
        r1 = bm_rings[-1]
        for j in range(m):
            jn = (j + 1) % m
            try:
                f = bm.faces.new((vc, r1[j], r1[jn]))
                f.material_index = mat_index
                f.smooth = True
                if uv_layer is not None:
                    f.loops[0][uv_layer].uv = (u_coords[-1], 0.5)
                    f.loops[1][uv_layer].uv = (u_coords[-1], v_coords[j])
                    f.loops[2][uv_layer].uv = (u_coords[-1], v_coords[j + 1])
                faces.append(f)
            except ValueError:
                pass

    return bm_rings, faces


def _compute_parallel_frames(points):
    """Compute smooth parallel-transport frames (T, N, B) along a polyline."""
    n = len(points)
    tangents = []
    for i in range(n):
        if i == 0:
            t = points[1] - points[0]
        elif i == n - 1:
            t = points[-1] - points[-2]
        else:
            t = points[i + 1] - points[i - 1]
        if t.length < 1e-8:
            t = Vector((0.0, 1.0, 0.0))
        else:
            t.normalize()
        tangents.append(t)

    up_ref = Vector((0.0, 0.0, 1.0))
    if abs(tangents[0].dot(up_ref)) > 0.95:
        up_ref = Vector((1.0, 0.0, 0.0))
    n0 = (up_ref - tangents[0] * tangents[0].dot(up_ref)).normalized()
    b0 = tangents[0].cross(n0).normalized()

    frames = [(tangents[0], n0, b0)]
    prev_t = tangents[0]
    prev_n = n0
    for i in range(1, n):
        t = tangents[i]
        axis = prev_t.cross(t)
        if axis.length > 1e-6:
            angle = math.atan2(axis.length, max(-1.0, min(1.0, prev_t.dot(t))))
            rot = Matrix.Rotation(angle, 3, axis.normalized())
            cur_n = (rot @ prev_n).normalized()
        else:
            cur_n = prev_n.copy()
        cur_n = (cur_n - t * cur_n.dot(t))
        if cur_n.length < 1e-6:
            cur_n = Vector((0.0, 0.0, 1.0)).cross(t)
        cur_n.normalize()
        cur_b = t.cross(cur_n).normalized()
        frames.append((t, cur_n, cur_b))
        prev_t = t
        prev_n = cur_n
    return frames


def add_curved_tube(
    bm: bmesh.types.BMesh,
    points,
    radii,
    radial_segs: int = 12,
    cap_start: bool = True,
    cap_end: bool = True,
    ring_ridge_amp: float = 0.0,
    ring_ridge_freq: float = 0.0,
    uv_layer=None,
    mat_index: int = 0,
    rx_scale: float = 1.0,
    rz_scale: float = 1.0,
):
    """Add a smooth 3D tube along points with per-point radii (or scalar) and optional ridges."""
    pts = [Vector(p) for p in points]
    n = len(pts)
    if n < 2:
        return []
    if isinstance(radii, (int, float)):
        r_list = [float(radii)] * n
    else:
        r_list = [float(r) for r in radii]

    frames = _compute_parallel_frames(pts)
    rings = []
    u_coords = []
    for i in range(n):
        u = i / float(n - 1)
        u_coords.append(u)
        _, n_vec, b_vec = frames[i]
        base_r = max(1e-4, r_list[i])
        if ring_ridge_amp != 0.0 and ring_ridge_freq != 0.0:
            base_r *= (1.0 + ring_ridge_amp * math.sin(u * ring_ridge_freq * math.tau))
        ring = []
        for j in range(radial_segs):
            ang = (j / float(radial_segs)) * math.tau
            off = n_vec * (math.cos(ang) * base_r * rz_scale) + b_vec * (math.sin(ang) * base_r * rx_scale)
            ring.append(pts[i] + off)
        rings.append(ring)

    _, faces = loft_sections(
        bm,
        rings,
        cap_start=cap_start,
        cap_end=cap_end,
        uv_layer=uv_layer,
        mat_index=mat_index,
        u_coords=u_coords,
    )
    return faces


def add_flame_tuft(
    bm: bmesh.types.BMesh,
    root,
    forward_dir,
    up_dir,
    length: float,
    width: float,
    thickness: float,
    curl_up: float = 0.25,
    curl_side: float = 0.0,
    twist: float = 0.0,
    segs: int = 8,
    uv_layer=None,
    mat_index: int = 0,
    s_wave: float = 0.0,
):
    """Create a sculptural 3D curved flame-lock blade with diamond/almond cross-section.

    UV.y (v) runs from 0.0 at root to 1.0 at tip for root-to-tip crimson->neon-pink gradients.
    """
    root_v = Vector(root)
    f_dir = Vector(forward_dir).normalized()
    u_raw = Vector(up_dir)
    # Orthogonalize u_dir against f_dir
    u_vec = u_raw - f_dir * u_raw.dot(f_dir)
    if u_vec.length < 1e-6:
        fallback = Vector((1.0, 0.0, 0.0)) if abs(f_dir.z) > 0.9 else Vector((0.0, 0.0, 1.0))
        u_vec = fallback - f_dir * fallback.dot(f_dir)
    u_vec.normalize()
    s_vec = f_dir.cross(u_vec).normalized()

    # Build curved centerline points along tuft
    spine_pts = []
    for i in range(segs + 1):
        t = i / float(segs)
        # S-shaped or J-shaped organic flame curve
        along = t * length
        up_lift = curl_up * length * (t ** 1.65)
        if s_wave != 0.0:
            up_lift += s_wave * length * math.sin(t * math.tau * 0.75) * (t ** 0.8)
        side_sway = curl_side * length * math.sin(t * math.pi * 0.85)
        pt = root_v + f_dir * along + u_vec * up_lift + s_vec * side_sway
        spine_pts.append(pt)

    frames = _compute_parallel_frames(spine_pts)
    radial_segs = 6  # hexagonal diamond profile: sharp lateral edges, keeled top/bottom ridge
    rings = []
    for i in range(segs + 1):
        t = i / float(segs)
        # Width swells briefly near base then tapers to sharp whip tip
        if t < 0.22:
            w_prof = 0.72 + 0.28 * math.sin((t / 0.22) * (math.pi * 0.5))
        else:
            w_prof = max(0.02, ((1.0 - t) / 0.78) ** 0.85)
        th_prof = max(0.02, (1.0 - t) ** 0.95)
        w = 0.5 * width * w_prof
        th = 0.5 * thickness * th_prof

        tan, _, _ = frames[i]
        # Keep blade normal aligned to u_vec / s_vec plus optional twist
        tw = twist * t
        loc_u = u_vec - tan * u_vec.dot(tan)
        if loc_u.length < 1e-6:
            loc_u = frames[i][1]
        else:
            loc_u.normalize()
        loc_s = tan.cross(loc_u).normalized()
        rot_u = loc_u * math.cos(tw) + loc_s * math.sin(tw)
        rot_s = -loc_u * math.sin(tw) + loc_s * math.cos(tw)

        # 6-point diamond-almond cross-section
        profile = [
            ( 1.0,  0.0),
            ( 0.35,  0.85),
            (-0.35,  0.85),
            (-1.0,  0.0),
            (-0.35, -0.75),
            ( 0.35, -0.75),
        ]
        ring = [spine_pts[i] + rot_s * (px * w) + rot_u * (pz * th) for px, pz in profile]
        rings.append(ring)

    # Custom UV assignment: u = circumferential [0..1], v = root-to-tip [0..1]
    bm_rings = [[bm.verts.new(p) for p in ring] for ring in rings]
    faces = []
    for i in range(segs):
        v0_uv = i / float(segs)
        v1_uv = (i + 1) / float(segs)
        r0 = bm_rings[i]
        r1 = bm_rings[i + 1]
        for j in range(radial_segs):
            jn = (j + 1) % radial_segs
            try:
                f = bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
            except ValueError:
                continue
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                u0_uv = j / float(radial_segs)
                u1_uv = (j + 1) / float(radial_segs)
                f.loops[0][uv_layer].uv = (u0_uv, v0_uv)
                f.loops[1][uv_layer].uv = (u1_uv, v0_uv)
                f.loops[2][uv_layer].uv = (u1_uv, v1_uv)
                f.loops[3][uv_layer].uv = (u0_uv, v1_uv)
            faces.append(f)

    # Tip apex cap
    tip_pt = spine_pts[-1] + (spine_pts[-1] - spine_pts[-2]).normalized() * (0.04 * length)
    vt = bm.verts.new(tip_pt)
    r_last = bm_rings[-1]
    for j in range(radial_segs):
        jn = (j + 1) % radial_segs
        try:
            f = bm.faces.new((r_last[j], r_last[jn], vt))
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                f.loops[0][uv_layer].uv = (j / float(radial_segs), 0.96)
                f.loops[1][uv_layer].uv = ((j + 1) / float(radial_segs), 0.96)
                f.loops[2][uv_layer].uv = (0.5, 1.0)
            faces.append(f)
        except ValueError:
            pass

    # Base cap
    r_first = bm_rings[0]
    vb = bm.verts.new(spine_pts[0])
    for j in range(radial_segs):
        jn = (j + 1) % radial_segs
        try:
            f = bm.faces.new((vb, r_first[jn], r_first[j]))
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                f.loops[0][uv_layer].uv = (0.5, 0.0)
                f.loops[1][uv_layer].uv = ((j + 1) / float(radial_segs), 0.0)
                f.loops[2][uv_layer].uv = (j / float(radial_segs), 0.0)
            faces.append(f)
        except ValueError:
            pass

    return faces


def finalize_mesh_object(name: str, bm: bmesh.types.BMesh, materials, collection, parent=None):
    """Recalculate normals, smooth shade, build mesh object, and link into collection."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(mesh)
    bm.free()

    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

    for mat in materials:
        mesh.materials.append(mat)

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj
