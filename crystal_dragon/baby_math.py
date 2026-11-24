"""Shared math, spline, lofting, crystal-frill, and double-walled membrane builders for the Crystal Baby Dragon."""

import math
import bpy
import bmesh
from mathutils import Vector, Matrix


def smoothstep(e0: float, e1: float, x: float) -> float:
    if abs(e1 - e0) < 1e-8:
        return 0.0
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def catmull_rom_sample(ctrl_pts, n_samples: int):
    """Sample a smooth Catmull-Rom spline through ctrl_pts."""
    pts = [Vector(p) for p in ctrl_pts]
    if len(pts) < 2:
        return pts
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


def hermite_eval(keys, u: float, value_indices):
    """Evaluate C1 cubic Hermite spline with finite-difference slopes over non-uniform keys."""
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

    u0, u1 = keys[i][0], keys[i + 1][0]
    du = max(u1 - u0, 1e-8)
    t = (u - u0) / du
    t2, t3 = t * t, t * t * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    out = []
    for k in value_indices:
        p0, p1 = keys[i][k], keys[i + 1][k]
        m0 = (keys[1][k] - keys[0][k]) / max(keys[1][0] - keys[0][0], 1e-8) if i == 0 else (
            (keys[i + 1][k] - keys[i - 1][k]) / max(keys[i + 1][0] - keys[i - 1][0], 1e-8)
        )
        m1 = (keys[-1][k] - keys[-2][k]) / max(keys[-1][0] - keys[-2][0], 1e-8) if i + 1 == n - 1 else (
            (keys[i + 2][k] - keys[i][k]) / max(keys[i + 2][0] - keys[i][0], 1e-8)
        )
        out.append(h00 * p0 + h10 * du * m0 + h01 * p1 + h11 * du * m1)
    return out


def compute_parallel_frames(points):
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
        cur_n = cur_n - t * cur_n.dot(t)
        if cur_n.length < 1e-6:
            cur_n = Vector((0.0, 0.0, 1.0)).cross(t)
        cur_n.normalize()
        cur_b = t.cross(cur_n).normalized()
        frames.append((t, cur_n, cur_b))
        prev_t = t
        prev_n = cur_n
    return frames


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
    """Loft a list of closed vertex-position rings (each length M) into clean quads."""
    n_rings = len(rings)
    if n_rings < 2:
        return [], []
    m = len(rings[0])
    if u_coords is None:
        u_coords = [i / max(1, n_rings - 1) for i in range(n_rings)]
    if v_coords is None:
        v_coords = [j / float(m) for j in range(m + 1)]

    bm_rings = [[bm.verts.new(p) for p in ring] for ring in rings]
    faces = []
    for i in range(n_rings - 1):
        r0, r1 = bm_rings[i], bm_rings[i + 1]
        u0, u1 = u_coords[i], u_coords[i + 1]
        for j in range(m):
            jn = (j + 1) % m
            try:
                f = bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
            except ValueError:
                continue
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                vv0, vv1 = v_coords[j], v_coords[j + 1]
                f.loops[0][uv_layer].uv = (u0, vv0)
                f.loops[1][uv_layer].uv = (u0, vv1)
                f.loops[2][uv_layer].uv = (u1, vv1)
                f.loops[3][uv_layer].uv = (u1, vv0)
            faces.append(f)

    if cap_start and m >= 3:
        vc = bm.verts.new(sum((Vector(p) for p in rings[0]), Vector((0.0, 0.0, 0.0))) / m)
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
        vc = bm.verts.new(sum((Vector(p) for p in rings[-1]), Vector((0.0, 0.0, 0.0))) / m)
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
    """Add a smooth 3D tube along points with per-point radii and optional soft organic growth rings."""
    pts = [Vector(p) for p in points]
    n = len(pts)
    if n < 2:
        return []
    r_list = [float(radii)] * n if isinstance(radii, (int, float)) else [float(r) for r in radii]
    frames = compute_parallel_frames(pts)

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


def add_crystal_frill(
    bm: bmesh.types.BMesh,
    root,
    forward_dir,
    up_dir,
    length: float,
    width: float,
    thickness: float,
    curl_up: float = 0.15,
    curl_side: float = 0.0,
    segs: int = 6,
    uv_layer=None,
    mat_index: int = 0,
):
    """Create a plump sculpted leaf/diamond crystal-frill spike with UV.y=0 at root -> 1 at tip."""
    root_v = Vector(root)
    f_dir = Vector(forward_dir).normalized()
    u_raw = Vector(up_dir)
    u_vec = u_raw - f_dir * u_raw.dot(f_dir)
    if u_vec.length < 1e-6:
        fallback = Vector((1.0, 0.0, 0.0)) if abs(f_dir.z) > 0.9 else Vector((0.0, 0.0, 1.0))
        u_vec = fallback - f_dir * fallback.dot(f_dir)
    u_vec.normalize()
    s_vec = f_dir.cross(u_vec).normalized()

    spine_pts = []
    for i in range(segs + 1):
        t = i / float(segs)
        along = t * length
        up_lift = curl_up * length * (t ** 1.6)
        side_sway = curl_side * length * math.sin(t * math.pi * 0.85)
        spine_pts.append(root_v + f_dir * along + u_vec * up_lift + s_vec * side_sway)

    frames = compute_parallel_frames(spine_pts)
    radial_segs = 6
    rings = []
    for i in range(segs + 1):
        t = i / float(segs)
        # Plump leaf-petal swelling near lower third then curving to crisp soft point
        if t < 0.28:
            w_prof = 0.72 + 0.32 * math.sin((t / 0.28) * math.pi * 0.5)
        else:
            w_prof = max(0.03, ((1.0 - t) / 0.72) ** 0.78)
        th_prof = max(0.03, (1.0 - t) ** 0.85)
        w = 0.5 * width * w_prof
        th = 0.5 * thickness * th_prof

        tan, _, _ = frames[i]
        loc_u = u_vec - tan * u_vec.dot(tan)
        if loc_u.length < 1e-6:
            loc_u = frames[i][1]
        else:
            loc_u.normalize()
        loc_s = tan.cross(loc_u).normalized()

        profile = [
            ( 1.0,  0.0),
            ( 0.42,  0.88),
            (-0.42,  0.88),
            (-1.0,  0.0),
            (-0.42, -0.78),
            ( 0.42, -0.78),
        ]
        ring = [spine_pts[i] + loc_s * (px * w) + loc_u * (pz * th) for px, pz in profile]
        rings.append(ring)

    bm_rings = [[bm.verts.new(p) for p in ring] for ring in rings]
    faces = []
    for i in range(segs):
        v0_uv = i / float(segs)
        v1_uv = (i + 1) / float(segs)
        r0, r1 = bm_rings[i], bm_rings[i + 1]
        for j in range(radial_segs):
            jn = (j + 1) % radial_segs
            try:
                f = bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
            except ValueError:
                continue
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                u0_uv, u1_uv = j / float(radial_segs), (j + 1) / float(radial_segs)
                f.loops[0][uv_layer].uv = (u0_uv, v0_uv)
                f.loops[1][uv_layer].uv = (u1_uv, v0_uv)
                f.loops[2][uv_layer].uv = (u1_uv, v1_uv)
                f.loops[3][uv_layer].uv = (u0_uv, v1_uv)
            faces.append(f)

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


def add_double_walled_membrane(
    bm: bmesh.types.BMesh,
    grid_pts,
    thickness: float = 0.018,
    uv_layer=None,
    mat_index: int = 0,
    uv_grid=None,
):
    """Turn an (M x N) 3D surface grid into a closed watertight thin double-sided membrane.

    By default UV.x = i/(M-1) and UV.y = 1.0 - j/(N-1) (j=0 top edge -> v=1.0, j=N-1 trailing edge -> v=0.0).
    """
    m = len(grid_pts)
    n = len(grid_pts[0])
    if m < 2 or n < 2:
        return []

    pts = [[Vector(grid_pts[i][j]) for j in range(n)] for i in range(m)]

    # Compute smooth normals across grid
    normals = [[Vector((0.0, 0.0, 1.0)) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            i0, i1 = max(0, i - 1), min(m - 1, i + 1)
            j0, j1 = max(0, j - 1), min(n - 1, j + 1)
            du = pts[i1][j] - pts[i0][j]
            dv = pts[i][j1] - pts[i][j0]
            nrm = du.cross(dv)
            if nrm.length < 1e-8:
                nrm = Vector((0.0, 0.0, 1.0))
            else:
                nrm.normalize()
            normals[i][j] = nrm

    top_verts = [[None] * n for _ in range(m)]
    bot_verts = [[None] * n for _ in range(m)]
    for i in range(m):
        ti = i / float(m - 1)
        for j in range(n):
            tj = j / float(n - 1)
            # Taper softly near trailing edge (tj=1) so trailing membrane edge is sleek
            edge_w = 0.28 + 0.72 * (1.0 - tj ** 2)
            half_t = 0.5 * thickness * edge_w
            off = normals[i][j] * half_t
            top_verts[i][j] = bm.verts.new(pts[i][j] + off)
            bot_verts[i][j] = bm.verts.new(pts[i][j] - off)

    def get_uv(i, j):
        if uv_grid is not None:
            return uv_grid[i][j]
        return (i / float(m - 1), 1.0 - j / float(n - 1))

    faces = []
    # Top & Bottom surfaces
    for i in range(m - 1):
        for j in range(n - 1):
            uv00 = get_uv(i, j)
            uv10 = get_uv(i + 1, j)
            uv11 = get_uv(i + 1, j + 1)
            uv01 = get_uv(i, j + 1)
            try:
                ft = bm.faces.new((top_verts[i][j], top_verts[i + 1][j], top_verts[i + 1][j + 1], top_verts[i][j + 1]))
                ft.material_index = mat_index
                ft.smooth = True
                if uv_layer is not None:
                    ft.loops[0][uv_layer].uv = uv00
                    ft.loops[1][uv_layer].uv = uv10
                    ft.loops[2][uv_layer].uv = uv11
                    ft.loops[3][uv_layer].uv = uv01
                faces.append(ft)
            except ValueError:
                pass
            try:
                fb = bm.faces.new((bot_verts[i][j], bot_verts[i][j + 1], bot_verts[i + 1][j + 1], bot_verts[i + 1][j]))
                fb.material_index = mat_index
                fb.smooth = True
                if uv_layer is not None:
                    fb.loops[0][uv_layer].uv = uv00
                    fb.loops[1][uv_layer].uv = uv01
                    fb.loops[2][uv_layer].uv = uv11
                    fb.loops[3][uv_layer].uv = uv10
                faces.append(fb)
            except ValueError:
                pass

    # Stitch all 4 perimeter edges watertight
    def stitch(v0, v1, v2, v3, uv_a, uv_b):
        try:
            f = bm.faces.new((v0, v1, v2, v3))
            f.material_index = mat_index
            f.smooth = True
            if uv_layer is not None:
                f.loops[0][uv_layer].uv = uv_a
                f.loops[1][uv_layer].uv = uv_b
                f.loops[2][uv_layer].uv = uv_b
                f.loops[3][uv_layer].uv = uv_a
            faces.append(f)
        except ValueError:
            pass

    for i in range(m - 1):
        stitch(bot_verts[i][0], bot_verts[i + 1][0], top_verts[i + 1][0], top_verts[i][0], get_uv(i, 0), get_uv(i + 1, 0))
        stitch(top_verts[i][n - 1], top_verts[i + 1][n - 1], bot_verts[i + 1][n - 1], bot_verts[i][n - 1], get_uv(i, n - 1), get_uv(i + 1, n - 1))
    for j in range(n - 1):
        stitch(top_verts[0][j], top_verts[0][j + 1], bot_verts[0][j + 1], bot_verts[0][j], get_uv(0, j), get_uv(0, j + 1))
        stitch(bot_verts[m - 1][j], bot_verts[m - 1][j + 1], top_verts[m - 1][j + 1], top_verts[m - 1][j], get_uv(m - 1, j), get_uv(m - 1, j + 1))

    return faces


def finalize_mesh_object(name: str, bm: bmesh.types.BMesh, materials, collection, parent=None):
    """Recalculate normals, smooth shade, create mesh object, and link into collection."""
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
