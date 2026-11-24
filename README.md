# antLoong · 3D Dragon Studio

Procedural 3D dragon modeling in **Blender 5.2** (with baked PBR textures) and interactive **Three.js** WebGL preview.

**Live 3D Web Preview**: [https://xuzhiming.github.io/antLoong/](https://xuzhiming.github.io/antLoong/)

## Models

1. **Crystal Baby Dragon (`Crystal Baby Dragon`)**
   - Scripts: `crystal_dragon/` (`build_crystal_dragon.py`, `baby_head.py`, `baby_body.py`, `baby_limbs.py`, `baby_wings.py`, `baby_materials.py`, `baby_math.py`, `generate_textures.py`)
   - Assets: `crystal_dragon.blend`, `crystal_dragon.glb`
   - Highlights: Sculpted voxel-remeshed head with seamless 360° almond eye sockets & `:D` smile, 8-segment ringed pearl horns, Voronoi crystal-lavender body scales, soft facial micro-scales, glowing cyan forehead diamond & wing veins, 3-tier pearl/rose-gold shoulder pauldrons, and S-curved forelegs.

2. **Cyberpunk Dragon (`Cyberpunk Dragon`)**
   - Scripts: `build_cyberpunk_dragon.py`, `dragon_head.py`, `dragon_body.py`, `dragon_spine.py`, `dragon_limbs.py`, `dragon_materials.py`, `generate_textures.py`
   - Assets: `cyberpunk_dragon.blend`, `cyberpunk_dragon.glb`

## Interactive Web Preview (`preview/`)

- **Model Switcher**: Switch live between `Crystal Dragon` and `Cyberpunk Dragon`
- **Camera Presets**: `Hero (1)`, `Head (2)`, `Front (3)`, `Rear (4)`, `Left (5)`, `Right (6)`, `Top (7)`, `Bottom (8)`, `Reset (R)`
- **Scene Controls**: Lighting presets (`Obsidian`, `Studio`, `Dark Rim`), `Neon Bloom` & `Exposure` sliders, `Auto-Rotate`, `Wireframe`, `Floor Grid`, `6-View Ref` draggable overlay, and per-part visibility toggles

### Local Preview

```bash
python3 -m http.server 8765 --directory preview
```

Then open [http://localhost:8765/](http://localhost:8765/).
