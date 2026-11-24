"""Generate high-contrast seamless PBR PNG textures for the Cyberpunk Dragon (Blender + glTF/Three.js compatible)."""

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
    if periodic_u:
        dh_du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    else:
        dh_du = np.gradient(height, axis=1)

    if periodic_v:
        dh_dv = -(np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    else:
        dh_dv = -np.gradient(height, axis=0)

    nx = -dh_du * strength
    ny = -dh_dv * strength
    nz = np.ones_like(height)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx *= inv_len
    ny *= inv_len
    nz *= inv_len

    rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _periodic_noise(h: int, w: int, seed: int, freq_x: int = 8, freq_y: int = 8) -> np.ndarray:
    """Deterministic tileable noise in [0, 1] via low-pass Fourier synthesis."""
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
    if f_max - f_min > 1e-8:
        field = (field - f_min) / (f_max - f_min)
    else:
        field = np.full_like(field, 0.5)
    return field


def _grid_uv(h: int, w: int):
    """Return (u, v) meshgrids where row 0 is v=1.0 (top) and row H-1 is v=0.0 (bottom)."""
    u = np.linspace(0.0, 1.0, w, endpoint=False)[None, :]
    v = (1.0 - np.linspace(0.0, 1.0, h, endpoint=False))[:, None]
    u = np.broadcast_to(u, (h, w))
    v = np.broadcast_to(v, (h, w))
    return u, v


def generate_dark_scales():
    h, w = 512, 512
    u, v = _grid_uv(h, w)
    # Larger, crisper overlapping pewter-gunmetal dragon scales (10 x 20 per UV tile)
    nu = u * 10.0
    nv = v * 20.0

    def _scale_cell(a, b):
        fa = (a % 1.0) - 0.5
        fb = (b % 1.0) - 0.5
        d = np.sqrt((fa - 0.08) ** 2 * 1.15 + fb ** 2 * 1.25)
        plate = 1.0 - _smoothstep(0.22, 0.50, d)
        # Bright metallic rim bevel ring around each scale edge
        rim = np.exp(-((d - 0.34) ** 2) * 95.0)
        ridge = np.exp(-((fa + 0.06) ** 2) * 18.0) * (1.0 - _smoothstep(0.24, 0.48, np.abs(fb)))
        return np.clip(0.80 * plate + 0.22 * ridge, 0.0, 1.0), np.clip(rim, 0.0, 1.0)

    p1, r1 = _scale_cell(nu, nv)
    p2, r2 = _scale_cell(nu + 0.5, nv + 0.5)
    use_p1 = p1 >= p2
    height = np.where(use_p1, p1, p2)
    rim_hl = np.where(use_p1, r1, r2)

    grain = _periodic_noise(h, w, seed=11, freq_x=24, freq_y=24)
    macro = _periodic_noise(h, w, seed=23, freq_x=6, freq_y=12)

    c_groove = np.array([6.0, 8.0, 12.0]) / 255.0
    c_low = np.array([44.0, 52.0, 66.0]) / 255.0
    c_high = np.array([88.0, 98.0, 118.0]) / 255.0

    plate_var = np.clip(0.55 * macro + 0.25 * grain + 0.45 * rim_hl, 0.0, 1.0)[..., None]
    c_plate = c_low * (1.0 - plate_var) + c_high * plate_var
    mask = _smoothstep(0.06, 0.30, height)[..., None]
    albedo = c_groove * (1.0 - mask) + c_plate * mask

    roughness = 0.52 * (1.0 - mask[..., 0]) + (0.18 + 0.08 * grain - 0.04 * rim_hl) * mask[..., 0]
    roughness = np.clip(roughness, 0.12, 0.58)
    normal = _height_to_normal(height + 0.03 * grain, strength=4.2, periodic_u=True, periodic_v=True)

    _save_rgb(albedo, "dark_scales_albedo.png")
    _save_rgb(normal, "dark_scales_normal.png")
    _save_gray(roughness, "dark_scales_roughness.png")


def generate_belly_scutes():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    grain = _periodic_noise(h, w, seed=37, freq_x=8, freq_y=28)

    c0 = np.array([64.0, 40.0, 22.0]) / 255.0
    c1 = np.array([204.0, 156.0, 112.0]) / 255.0
    c2 = np.array([82.0, 52.0, 30.0]) / 255.0

    w_mid = _smoothstep(0.04, 0.45, v) * (1.0 - _smoothstep(0.75, 0.98, v))
    w_edge = _smoothstep(0.75, 0.98, v)
    albedo = c0 * (1.0 - _smoothstep(0.04, 0.45, v))[..., None] + c1 * w_mid[..., None] + c2 * w_edge[..., None]
    albedo = np.clip(albedo * (0.93 + 0.14 * grain[..., None]), 0.0, 1.0)

    height = 0.88 * _smoothstep(0.03, 0.35, v) * (1.0 - _smoothstep(0.80, 0.98, v)) + 0.05 * grain
    normal = _height_to_normal(height, strength=2.0, periodic_u=True, periodic_v=False)

    _save_rgb(albedo, "belly_scutes_albedo.png")
    _save_rgb(normal, "belly_scutes_normal.png")


def generate_antique_gold():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    grain = _periodic_noise(h, w, seed=53, freq_x=6, freq_y=20)
    patina = _periodic_noise(h, w, seed=59, freq_x=10, freq_y=10)

    # Organic longitudinal horn-fluting along v (circumferential ridges running along antler length u)
    flute_phase = (v * 12.0 + 0.45 * grain + 0.25 * np.sin(u * 2.0 * np.pi * 2.0)) * 2.0 * np.pi
    flute_wave = 0.5 + 0.5 * np.sin(flute_phase)
    flute_fac = _smoothstep(0.18, 0.82, flute_wave)
    organic_blend = np.clip(0.68 * flute_fac + 0.32 * patina, 0.0, 1.0)

    c_dark = np.array([122.0, 66.0, 16.0]) / 255.0
    c_gold = np.array([236.0, 162.0, 54.0]) / 255.0
    albedo = c_dark * (1.0 - organic_blend[..., None]) + c_gold * organic_blend[..., None]
    albedo = np.clip(albedo * (0.94 + 0.12 * grain[..., None]), 0.0, 1.0)

    height = 0.68 * flute_fac + 0.22 * patina + 0.10 * grain
    normal = _height_to_normal(height, strength=2.0, periodic_u=True, periodic_v=True)

    _save_rgb(albedo, "antique_gold_albedo.png")
    _save_rgb(normal, "antique_gold_normal.png")


def generate_magenta_mane():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    strand = 0.5 + 0.5 * np.sin(u * 2.0 * np.pi * 18.0 + 1.5 * np.sin(v * 2.0 * np.pi * 3.0))
    grain = _periodic_noise(h, w, seed=71, freq_x=24, freq_y=6)
    streak = 0.88 + 0.18 * (0.6 * strand + 0.4 * grain)

    # Hyper-saturated ruby-crimson root -> electric crimson-magenta -> hot neon fuchsia tip (near-zero G)
    c_root = np.array([36.0, 1.0, 16.0]) / 255.0
    c_mid = np.array([206.0, 3.0, 76.0]) / 255.0
    c_tip = np.array([255.0, 14.0, 106.0]) / 255.0

    t1 = _smoothstep(0.0, 0.46, v)[..., None]
    t2 = _smoothstep(0.42, 1.0, v)[..., None]
    albedo = c_root * (1.0 - t1) + c_mid * t1 * (1.0 - t2) + c_tip * t2
    albedo = np.clip(albedo * streak[..., None], 0.0, 1.0)

    em_weight = _smoothstep(0.18, 0.96, v)[..., None]
    c_em_root = np.array([10.0, 0.0, 4.0]) / 255.0
    c_em_tip = np.array([255.0, 12.0, 102.0]) / 255.0
    emissive = np.clip((c_em_root * (1.0 - em_weight) + c_em_tip * em_weight) * (0.9 + 0.14 * strand[..., None]), 0.0, 1.0)

    _save_rgb(albedo, "magenta_mane_albedo.png")
    _save_rgb(emissive, "magenta_mane_emissive.png")


def generate_silver_beard():
    h, w = 256, 256
    u, v = _grid_uv(h, w)
    strand = 0.5 + 0.5 * np.sin(u * 2.0 * np.pi * 18.0 + 1.2 * np.sin(v * 2.0 * np.pi * 2.0))
    grain = _periodic_noise(h, w, seed=89, freq_x=20, freq_y=6)
    streak = 0.92 + 0.14 * (0.6 * strand + 0.4 * grain)

    c_root = np.array([82.0, 92.0, 110.0]) / 255.0
    c_tip = np.array([238.0, 244.0, 252.0]) / 255.0
    t = _smoothstep(0.0, 0.88, v)[..., None]
    albedo = np.clip((c_root * (1.0 - t) + c_tip * t) * streak[..., None], 0.0, 1.0)

    _save_rgb(albedo, "silver_beard_albedo.png")


def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    generate_dark_scales()
    generate_belly_scutes()
    generate_antique_gold()
    generate_magenta_mane()
    generate_silver_beard()
    files = sorted(os.listdir(TEX_DIR))
    print(f"Generated {len(files)} textures in {TEX_DIR}:")
    for fn in files:
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
