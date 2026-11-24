"""Generate 15 bespoke pastel-iridescent PBR PNG textures for the Crystal Baby Dragon."""

import os
import numpy as np
from PIL import Image

TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _save_rgb(arr_float01: np.ndarray, filename: str):
    arr_u8 = np.clip(arr_float01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    path = os.path.join(TEX_DIR, filename)
    Image.fromarray(arr_u8).save(path)
    return path


def _save_gray(arr_float01: np.ndarray, filename: str):
    arr_u8 = np.clip(arr_float01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    path = os.path.join(TEX_DIR, filename)
    Image.fromarray(arr_u8).save(path)
    return path


def _height_to_normal(height: np.ndarray, strength: float = 2.5, periodic_u: bool = True, periodic_v: bool = True) -> np.ndarray:
    """Convert [0,1] heightmap (row=0 -> v=1 top, col=0 -> u=0 left) to OpenGL tangent-space normal RGB."""
    dh_du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 if periodic_u else np.gradient(height, axis=1)
    dh_dv = -(np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5 if periodic_v else -np.gradient(height, axis=0)

    nx = -dh_du * strength
    ny = -dh_dv * strength
    nz = np.ones_like(height)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx * inv_len * 0.5 + 0.5, ny * inv_len * 0.5 + 0.5, nz * inv_len * 0.5 + 0.5], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _periodic_noise(h: int, w: int, seed: int, freq_x: int = 8, freq_y: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = np.zeros((h, w), dtype=np.complex128)
    for ky in range(-freq_y, freq_y + 1):
        for kx in range(-freq_x, freq_x + 1):
            if kx == 0 and ky == 0:
                continue
            r2 = (kx / float(freq_x)) ** 2 + (ky / float(freq_y)) ** 2
            if r2 <= 1.0:
                amp = (1.0 - r2) ** 0.75
                phase = rng.uniform(0.0, 2.0 * np.pi)
                spec[ky % h, kx % w] = amp * np.exp(1j * phase)
    field = np.fft.ifft2(spec).real
    f_min, f_max = field.min(), field.max()
    return (field - f_min) / (f_max - f_min) if (f_max - f_min > 1e-8) else np.full_like(field, 0.5)


def _grid_uv(h: int, w: int):
    """Row 0 = UV.y=1.0 (top), Row H-1 = UV.y=0.0 (bottom)."""
    u = np.broadcast_to(np.linspace(0.0, 1.0, w, endpoint=False)[None, :], (h, w))
    v = np.broadcast_to((1.0 - np.linspace(0.0, 1.0, h, endpoint=False))[:, None], (h, w))
    return u, v


def _cell_hash(gx: np.ndarray, gy: np.ndarray, layer: int, salt: int, seed: int) -> np.ndarray:
    h = (
        (gx.astype(np.uint64) * np.uint64(73856093))
        ^ (gy.astype(np.uint64) * np.uint64(19349663))
        ^ np.uint64(layer * 83492791 + salt * 26544357 + seed * 159733467)
    ) & np.uint64(0xFFFFFFFF)
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(16))) & np.uint64(0xFFFFFFFF)
    return (h & np.uint64(0xFFFF)).astype(np.float64) / 65536.0


def _voronoi_dragon_scales(u: np.ndarray, v: np.ndarray, n_u: int, n_v: int, seed: int, jitter: float = 0.06):
    """Seamlessly tileable staggered Voronoi/hex dragon-scale lattice."""
    x = u * float(n_u)
    y = v * float(n_v)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)

    d1 = np.full_like(x, 1e9)
    d2 = np.full_like(x, 1e9)
    win_dx = np.zeros_like(x)
    win_tint = np.zeros_like(x)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for layer, shift in ((0, 0.0), (1, 0.5)):
        for di, dj in offsets:
            gx = ix + di
            gy = iy + dj
            wx = gx % n_u
            wy = gy % n_v

            jx = (_cell_hash(wx, wy, layer, 1, seed) - 0.5) * (2.0 * jitter)
            jy = (_cell_hash(wx, wy, layer, 2, seed) - 0.5) * (2.0 * jitter)
            tint = _cell_hash(wx, wy, layer, 3, seed)

            cx = gx.astype(np.float64) + shift + jx
            cy = gy.astype(np.float64) + shift + jy
            dx_raw = x - cx
            dx = dx_raw * 0.96
            dy = (y - cy) * 1.04
            d = np.sqrt(dx * dx + dy * dy)

            m1 = d < d1
            d2 = np.where(m1, d1, np.minimum(d2, d))
            d1 = np.where(m1, d, d1)
            win_dx = np.where(m1, dx_raw, win_dx)
            win_tint = np.where(m1, tint, win_tint)

    edge = np.clip(d2 - d1, 0.0, 1.0)
    return edge, win_dx, win_tint


def generate_lavender_scales():
    h, w = 1024, 1024
    u, v = _grid_uv(h, w)
    edge, win_dx, win_tint = _voronoi_dragon_scales(u, v, n_u=14, n_v=28, seed=101, jitter=0.06)

    scale_mask = _smoothstep(0.025, 0.135, edge)
    dome = np.power(_smoothstep(0.0, 0.44, edge), 0.55)
    shingle = np.clip(0.55 - 0.65 * win_dx, 0.0, 1.0)

    macro = _periodic_noise(h, w, seed=101, freq_x=6, freq_y=12)
    micro = _periodic_noise(h, w, seed=103, freq_x=28, freq_y=28)

    c_groove = np.array([68.0, 56.0, 134.0]) / 255.0     # deep rich periwinkle-indigo outline
    c_shadow = np.array([124.0, 112.0, 198.0]) / 255.0   # medium lavender skirt
    c_mid = np.array([156.0, 146.0, 224.0]) / 255.0      # dreamy pastel lavender
    c_pearl = np.array([218.0, 212.0, 248.0]) / 255.0    # pearlescent dome highlight

    irid = (win_tint[..., None] - 0.5) * np.array([0.07, -0.02, 0.04]) + (macro[..., None] - 0.5) * 0.04
    c_body = c_shadow * (1.0 - dome[..., None]) + c_mid * dome[..., None]
    hi_fac = np.power(np.clip(dome * (0.65 + 0.55 * shingle), 0.0, 1.0), 1.6)[..., None]
    c_scale = np.clip(c_body * (1.0 - hi_fac) + c_pearl * hi_fac + irid + 0.025 * (micro[..., None] - 0.5), 0.0, 1.0)
    albedo = np.clip(c_groove * (1.0 - scale_mask[..., None]) + c_scale * scale_mask[..., None], 0.0, 1.0)

    roughness = 0.58 * (1.0 - scale_mask) + (0.24 + 0.08 * micro + 0.06 * (1.0 - dome)) * scale_mask
    height = 0.78 * dome + 0.16 * shingle * scale_mask + 0.06 * micro
    normal = _height_to_normal(height, strength=3.4, periodic_u=True, periodic_v=True)

    _save_rgb(albedo, "lavender_scales_albedo.png")
    _save_rgb(normal, "lavender_scales_normal.png")
    _save_gray(roughness, "lavender_scales_roughness.png")


def generate_face_scales():
    """Velvety, faint organic micro-scale facial lavender texture matching the reference artwork."""
    h, w = 1024, 1024
    u, v = _grid_uv(h, w)
    edge, win_dx, win_tint = _voronoi_dragon_scales(u, v, n_u=32, n_v=20, seed=151, jitter=0.08)

    scale_mask = _smoothstep(0.008, 0.085, edge)
    groove_w = (1.0 - scale_mask)[..., None] * 0.42
    dome = np.power(_smoothstep(0.0, 0.42, edge), 0.58)
    shingle = np.clip(0.55 - 0.55 * win_dx, 0.0, 1.0)

    macro = _periodic_noise(h, w, seed=151, freq_x=6, freq_y=6)
    micro = _periodic_noise(h, w, seed=153, freq_x=24, freq_y=24)

    c_groove = np.array([140.0, 130.0, 206.0]) / 255.0
    c_shadow = np.array([154.0, 144.0, 218.0]) / 255.0
    c_mid = np.array([170.0, 160.0, 232.0]) / 255.0
    c_pearl = np.array([214.0, 208.0, 246.0]) / 255.0

    irid = (win_tint[..., None] - 0.5) * np.array([0.05, -0.015, 0.03]) + (macro[..., None] - 0.5) * 0.035
    c_body = c_shadow * (1.0 - dome[..., None]) + c_mid * dome[..., None]
    hi_fac = np.power(np.clip(dome * (0.55 + 0.35 * shingle), 0.0, 1.0), 1.8)[..., None] * 0.55
    c_scale = np.clip(c_body * (1.0 - hi_fac) + c_pearl * hi_fac + irid * 0.6 + 0.02 * (micro[..., None] - 0.5), 0.0, 1.0)
    albedo = np.clip(c_groove * groove_w + c_scale * (1.0 - groove_w), 0.0, 1.0)

    height = 0.45 * dome + 0.06 * shingle * scale_mask + 0.06 * micro
    normal = _height_to_normal(height, strength=0.65, periodic_u=True, periodic_v=True)

    _save_rgb(albedo, "face_scales_albedo.png")
    _save_rgb(normal, "face_scales_normal.png")


def generate_pearl_belly():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    grain = _periodic_noise(h, w, seed=201, freq_x=10, freq_y=24)

    c_groove = np.array([136.0, 128.0, 184.0]) / 255.0  # deeper periwinkle groove for crisp scutes
    c_pearl = np.array([220.0, 218.0, 240.0]) / 255.0   # soft pearl-lavender plate
    c_edge = np.array([168.0, 160.0, 210.0]) / 255.0

    w_mid = _smoothstep(0.04, 0.40, v) * (1.0 - _smoothstep(0.76, 0.98, v))
    w_edge = _smoothstep(0.76, 0.98, v)
    albedo = c_groove * (1.0 - _smoothstep(0.04, 0.40, v))[..., None] + c_pearl * w_mid[..., None] + c_edge * w_edge[..., None]
    albedo = np.clip(albedo * (0.95 + 0.07 * grain[..., None]), 0.0, 1.0)

    height = 0.85 * _smoothstep(0.03, 0.35, v) * (1.0 - _smoothstep(0.80, 0.98, v)) + 0.04 * grain
    normal = _height_to_normal(height, strength=1.8, periodic_u=True, periodic_v=False)

    _save_rgb(albedo, "pearl_belly_albedo.png")
    _save_rgb(normal, "pearl_belly_normal.png")


def generate_wing_membrane():
    h, w = 512, 512
    u, v = _grid_uv(h, w)
    grain = _periodic_noise(h, w, seed=301, freq_x=12, freq_y=12)

    # Richer dreamy iridescent gradient: upper wing arm (#7C6EE8) -> mid (#CC90E0) -> trailing peach-coral (#FFB6A4)
    c_top = np.array([124.0, 110.0, 232.0]) / 255.0
    c_mid = np.array([204.0, 144.0, 224.0]) / 255.0
    c_bot = np.array([255.0, 182.0, 164.0]) / 255.0

    t_mid = _smoothstep(0.0, 0.55, v)[..., None]
    t_top = _smoothstep(0.45, 1.0, v)[..., None]
    albedo = c_bot * (1.0 - t_mid) + c_mid * t_mid * (1.0 - t_top) + c_top * t_top

    # Delicate branching organic membrane veins
    vein1 = np.exp(-((np.sin((u * 6.0 + 0.25 * np.sin(v * 5.0)) * np.pi)) ** 2) * 32.0) * _smoothstep(0.10, 0.85, v)
    vein2 = np.exp(-((np.sin((v * 8.0 + 0.4 * grain) * np.pi)) ** 2) * 24.0) * 0.45
    veins = np.clip(vein1 * 0.6 + vein2 * 0.4, 0.0, 1.0)

    albedo = np.clip(albedo * (0.94 + 0.09 * grain[..., None]) + 0.05 * veins[..., None], 0.0, 1.0)
    height = 0.22 * veins + 0.10 * grain
    normal = _height_to_normal(height, strength=1.0, periodic_u=False, periodic_v=False)

    _save_rgb(albedo, "wing_membrane_albedo.png")
    _save_rgb(normal, "wing_membrane_normal.png")


def generate_frill_gradient():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    strand = 0.5 + 0.5 * np.sin(u * 2.0 * np.pi * 12.0 + 0.8 * np.sin(v * 2.0 * np.pi * 2.0))
    grain = _periodic_noise(h, w, seed=401, freq_x=16, freq_y=8)

    c_root = np.array([112.0, 96.0, 206.0]) / 255.0
    c_mid = np.array([176.0, 146.0, 230.0]) / 255.0
    c_tip = np.array([255.0, 160.0, 188.0]) / 255.0

    t1 = _smoothstep(0.0, 0.50, v)[..., None]
    t2 = _smoothstep(0.45, 1.0, v)[..., None]
    albedo = c_root * (1.0 - t1) + c_mid * t1 * (1.0 - t2) + c_tip * t2
    albedo = np.clip(albedo * (0.94 + 0.09 * (0.6 * strand + 0.4 * grain)[..., None]), 0.0, 1.0)

    _save_rgb(albedo, "frill_gradient_albedo.png")


def generate_horn_pearl():
    """Synchronized 8-segment ringed horn albedo & normal (512x512, u_ring_end=0.88)."""
    h, w = 512, 512
    u, v = _grid_uv(h, w)
    grain = _periodic_noise(h, w, seed=501, freq_x=18, freq_y=16)

    frac = ((u / 0.88) * 8.0) % 1.0
    s_skew = np.power(np.clip(frac, 1e-6, 1.0), 0.82)
    ring_dome = np.power(np.sin(s_skew * np.pi), 0.52)
    fade = _smoothstep(0.0, 0.04, u) * (1.0 - _smoothstep(0.84, 0.88, u))
    groove_mask = (1.0 - ring_dome) * fade
    striation = 0.5 + 0.5 * np.sin(v * 2.0 * np.pi * 24.0 + 0.35 * grain)

    c_base = np.array([242.0, 232.0, 238.0]) / 255.0    # warm champagne-pearl root
    c_tip = np.array([142.0, 128.0, 160.0]) / 255.0     # dusky mauve-lavender tip
    c_groove = np.array([86.0, 72.0, 108.0]) / 255.0    # deep mauve-plum crease shadow

    t_tip = _smoothstep(0.18, 0.94, u)[..., None]
    col = c_base * (1.0 - t_tip) + c_tip * t_tip
    col = col * (0.94 + 0.10 * ring_dome[..., None]) * (0.96 + 0.05 * striation[..., None]) * (0.97 + 0.05 * grain[..., None])
    g_w = np.power(groove_mask, 1.35)[..., None] * 0.72
    albedo = np.clip(col * (1.0 - g_w) + c_groove * g_w, 0.0, 1.0)

    height = (0.78 * ring_dome * fade + 0.45 * (1.0 - fade)) + 0.06 * striation + 0.06 * grain
    normal = _height_to_normal(height, strength=3.0, periodic_u=False, periodic_v=True)

    _save_rgb(albedo, "horn_pearl_albedo.png")
    _save_rgb(normal, "horn_pearl_normal.png")


def generate_baby_eye():
    """1024x1024 shaded pearl sclera, bug-free glowing aqua iris, large oval pupil & twin catchlights."""
    h, w = 1024, 1024
    u, v = _grid_uv(h, w)
    dx = u - 0.5
    dy = v - 0.5
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    micro = _periodic_noise(h, w, seed=701, freq_x=24, freq_y=24)

    # Shaded pearl sclera with cool upper-lid shadow & peripheral tint
    c_sclera = np.array([244.0, 246.0, 252.0]) / 255.0
    c_sclera_shad = np.array([188.0, 196.0, 226.0]) / 255.0
    lid_shad = _smoothstep(0.12, 0.44, dy) * 0.55
    periph = _smoothstep(0.45, 0.55, r) * 0.35
    sclera_w = np.clip(lid_shad + periph, 0.0, 0.65)[..., None]
    albedo = c_sclera * (1.0 - sclera_w) + c_sclera_shad * sclera_w

    # Deep ocean-teal outer limbal ring -> aqua mid -> upper cast shadow -> lower aquamarine glow crescent
    iris_mask = 1.0 - _smoothstep(0.415, 0.445, r)
    rim_fac = _smoothstep(0.32, 0.425, r)
    upper_shadow = _smoothstep(0.02, 0.34, dy)
    crescent = _smoothstep(-0.04, 0.34, -dy) * (1.0 - _smoothstep(0.32, 0.42, r))

    c_iris_limbal = np.array([13.0, 72.0, 94.0]) / 255.0
    c_iris_top = np.array([11.0, 61.0, 84.0]) / 255.0
    c_iris_mid = np.array([32.0, 172.0, 192.0]) / 255.0
    c_iris_bright = np.array([142.0, 248.0, 255.0]) / 255.0

    fil = 0.5 + 0.25 * np.sin(ang * 24.0) + 0.15 * np.sin(ang * 52.0 + 0.7) + 0.20 * (micro - 0.5)
    iris_col = c_iris_mid * (1.0 - 0.55 * upper_shadow[..., None]) + c_iris_top * (0.55 * upper_shadow[..., None])
    iris_col = iris_col * (1.0 - rim_fac[..., None]) + c_iris_limbal * rim_fac[..., None]
    iris_col = iris_col * (1.0 - 0.68 * crescent[..., None]) + c_iris_bright * (0.68 * crescent[..., None])
    iris_col = np.clip(iris_col * (0.93 + 0.14 * fil[..., None]), 0.0, 1.0)

    albedo = albedo * (1.0 - iris_mask[..., None]) + iris_col * iris_mask[..., None]

    # Larger adorable softly oval midnight-indigo pupil (rx=0.225, ry=0.250)
    r_pupil = np.sqrt((dx / 0.225) ** 2 + (dy / 0.250) ** 2)
    pupil_mask = 1.0 - _smoothstep(0.92, 1.04, r_pupil)
    c_pupil = np.array([10.0, 13.0, 28.0]) / 255.0
    albedo = albedo * (1.0 - pupil_mask[..., None]) + c_pupil * pupil_mask[..., None]

    # Twin crisp specular catchlight sparkles + soft rim highlight arc
    ox, oy = dx + 0.13, dy - 0.14
    ca, sa = np.cos(0.42), np.sin(0.42)
    rx1 = ox * ca + oy * sa
    ry1 = -ox * sa + oy * ca
    d_sp1 = np.sqrt((rx1 / 0.092) ** 2 + (ry1 / 0.072) ** 2)
    sp1 = 1.0 - _smoothstep(0.80, 1.04, d_sp1)

    d_sp2 = np.sqrt((dx - 0.13) ** 2 + (dy + 0.14) ** 2)
    sp2 = 1.0 - _smoothstep(0.028, 0.042, d_sp2)

    rim_arc = (1.0 - _smoothstep(0.014, 0.028, np.abs(r - 0.365))) * _smoothstep(-0.08, 0.28, -dy) * _smoothstep(0.0, 0.22, dx) * 0.22
    sparkle = np.clip(sp1 + sp2 + rim_arc, 0.0, 1.0)[..., None]
    albedo = np.clip(albedo * (1.0 - sparkle) + np.ones_like(albedo) * sparkle, 0.0, 1.0)

    _save_rgb(albedo, "baby_eye_albedo.png")


def generate_tail_plume():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    # u=0..1 across width (0.5 = central midrib), v=0..1 along leaf (1 = base/stem? Wait: v=1 top edge, v=0 trailing tip)
    dist_center = np.abs(u - 0.5) * 2.0  # 0 at center spine, 1 at lateral edge
    grain = _periodic_noise(h, w, seed=601, freq_x=16, freq_y=10)

    c_inner = np.array([158.0, 144.0, 232.0]) / 255.0  # #9E90E8 lavender-violet
    c_outer = np.array([255.0, 196.0, 206.0]) / 255.0  # #FFC4CE warm rose-peach border
    edge_fac = _smoothstep(0.25, 0.90, dist_center)[..., None]
    base_col = c_inner * (1.0 - edge_fac) + c_outer * edge_fac

    # Glowing electric ice-cyan central midrib & branching feather veins (#40F4FF)
    midrib = np.exp(-((dist_center / 0.08) ** 2)) * _smoothstep(0.04, 0.25, v)
    # Branching angled chevron veins radiating out from midrib toward v=0 tip
    branch_phase = ((1.0 - v) * 9.0 - dist_center * 3.5) * np.pi
    branches = np.exp(-((np.sin(branch_phase)) ** 2) * 28.0) * _smoothstep(0.08, 0.35, 1.0 - dist_center) * _smoothstep(0.10, 0.85, v)
    cyan_glow_mask = np.clip(midrib + 0.65 * branches, 0.0, 1.0)[..., None]

    c_cyan = np.array([64.0, 244.0, 255.0]) / 255.0
    albedo = np.clip(base_col * (0.94 + 0.08 * grain[..., None]) * (1.0 - cyan_glow_mask) + c_cyan * cyan_glow_mask, 0.0, 1.0)
    emissive = np.clip(c_cyan * cyan_glow_mask * 1.15 + base_col * 0.08, 0.0, 1.0)

    _save_rgb(albedo, "tail_plume_albedo.png")
    _save_rgb(emissive, "tail_plume_emissive.png")


def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    generate_lavender_scales()
    generate_face_scales()
    generate_pearl_belly()
    generate_wing_membrane()
    generate_frill_gradient()
    generate_horn_pearl()
    generate_baby_eye()
    generate_tail_plume()
    files = sorted(os.listdir(TEX_DIR))
    print(f"Generated {len(files)} textures in {TEX_DIR}:")
    for fn in files:
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
