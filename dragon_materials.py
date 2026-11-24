"""Image-backed PBR material library for the Cyberpunk Dragon (Blender 5.2 + glTF 2.0 compatible)."""

import os
import bpy

TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def _get_or_create_material(name: str):
    """Create a fresh node-based material or reset an existing one with the same name."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (520, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (180, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf, out


def _add_tex_node(nt, filename: str, colorspace: str = "sRGB", extension: str = "REPEAT", loc=(-350, 0)):
    """Load, pack, and create a ShaderNodeTexImage node directly wired for glTF/Cycles compatibility."""
    path = os.path.join(TEX_DIR, filename)
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = colorspace
    img.pack()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = extension
    tex.location = loc
    return tex


def _add_normal_map(nt, filename: str, strength: float = 1.0, extension: str = "REPEAT", loc_tex=(-480, -280), loc_nm=(-120, -240)):
    tex_norm = _add_tex_node(nt, filename, colorspace="Non-Color", extension=extension, loc=loc_tex)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.space = "TANGENT"
    nm.inputs["Strength"].default_value = strength
    nm.location = loc_nm
    nt.links.new(tex_norm.outputs["Color"], nm.inputs["Color"])
    return nm


def build_dark_armor_scales():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Dark_Armor_Scales")
    mat.diffuse_color = (0.07, 0.085, 0.115, 1.0)

    tex_alb = _add_tex_node(nt, "dark_scales_albedo.png", colorspace="sRGB", extension="REPEAT", loc=(-360, 260))
    tex_rough = _add_tex_node(nt, "dark_scales_roughness.png", colorspace="Non-Color", extension="REPEAT", loc=(-360, -20))
    nm = _add_normal_map(nt, "dark_scales_normal.png", strength=1.0, extension="REPEAT", loc_tex=(-480, -300), loc_nm=(-120, -260))

    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Metallic"].default_value = 0.76
    bsdf.inputs["Specular IOR Level"].default_value = 0.65
    bsdf.inputs["Coat Weight"].default_value = 0.28
    bsdf.inputs["Coat Roughness"].default_value = 0.16
    return mat


def build_belly_gold_scutes():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Belly_Gold_Scutes")
    mat.diffuse_color = (0.72, 0.54, 0.38, 1.0)

    tex_alb = _add_tex_node(nt, "belly_scutes_albedo.png", colorspace="sRGB", extension="EXTEND", loc=(-360, 220))
    nm = _add_normal_map(nt, "belly_scutes_normal.png", strength=0.85, extension="EXTEND", loc_tex=(-480, -180), loc_nm=(-120, -180))

    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Metallic"].default_value = 0.48
    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Coat Weight"].default_value = 0.15
    bsdf.inputs["Coat Roughness"].default_value = 0.25
    return mat


def build_antique_gold():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Antique_Gold")
    mat.diffuse_color = (0.85, 0.54, 0.16, 1.0)

    tex_alb = _add_tex_node(nt, "antique_gold_albedo.png", colorspace="sRGB", extension="REPEAT", loc=(-360, 220))
    nm = _add_normal_map(nt, "antique_gold_normal.png", strength=0.90, extension="REPEAT", loc_tex=(-480, -180), loc_nm=(-120, -180))

    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Metallic"].default_value = 0.86
    bsdf.inputs["Roughness"].default_value = 0.24
    bsdf.inputs["Coat Weight"].default_value = 0.18
    return mat


def build_magenta_mane_fur():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Magenta_Mane_Fur")
    mat.diffuse_color = (0.92, 0.03, 0.40, 1.0)

    tex_alb = _add_tex_node(nt, "magenta_mane_albedo.png", colorspace="sRGB", extension="EXTEND", loc=(-360, 220))
    tex_em = _add_tex_node(nt, "magenta_mane_emissive.png", colorspace="sRGB", extension="EXTEND", loc=(-360, -120))

    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex_em.outputs["Color"], bsdf.inputs["Emission Color"])

    bsdf.inputs["Emission Strength"].default_value = 1.05
    bsdf.inputs["Metallic"].default_value = 0.04
    bsdf.inputs["Roughness"].default_value = 0.36
    bsdf.inputs["Sheen Weight"].default_value = 0.40
    bsdf.inputs["Sheen Roughness"].default_value = 0.35
    bsdf.inputs["Sheen Tint"].default_value = (1.0, 0.04, 0.42, 1.0)
    bsdf.inputs["Coat Weight"].default_value = 0.22
    bsdf.inputs["Coat Roughness"].default_value = 0.24
    return mat


def build_magenta_neon_whisker():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Magenta_Neon_Whisker")
    mat.diffuse_color = (0.98, 0.03, 0.42, 1.0)
    bsdf.inputs["Base Color"].default_value = (0.98, 0.03, 0.42, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.22
    bsdf.inputs["Emission Color"].default_value = (0.98, 0.03, 0.40, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 1.45
    return mat


def build_cyan_neon_glow():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Cyan_Neon_Glow")
    mat.diffuse_color = (0.0, 0.86, 1.0, 1.0)
    bsdf.inputs["Base Color"].default_value = (0.0, 0.86, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["Emission Color"].default_value = (0.0, 0.84, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 3.8
    return mat


def build_amber_eye_glow():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Amber_Eye_Glow")
    mat.diffuse_color = (1.0, 0.32, 0.01, 1.0)
    bsdf.inputs["Base Color"].default_value = (1.0, 0.32, 0.01, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.06
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.28, 0.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 5.5
    return mat


def build_pupil_black():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Pupil_Black")
    mat.diffuse_color = (0.008, 0.008, 0.01, 1.0)
    bsdf.inputs["Base Color"].default_value = (0.005, 0.005, 0.008, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    return mat


def build_silver_beard():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Silver_Beard")
    mat.diffuse_color = (0.76, 0.80, 0.86, 1.0)

    tex_alb = _add_tex_node(nt, "silver_beard_albedo.png", colorspace="sRGB", extension="EXTEND", loc=(-360, 180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Metallic"].default_value = 0.22
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Sheen Weight"].default_value = 0.55
    bsdf.inputs["Sheen Tint"].default_value = (0.85, 0.90, 0.98, 1.0)
    return mat


def build_dark_mouth():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Dark_Mouth")
    mat.diffuse_color = (0.08, 0.03, 0.05, 1.0)
    bsdf.inputs["Base Color"].default_value = (0.08, 0.03, 0.05, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = 0.25
    return mat


def create_all_materials():
    """Build and return dict of all Cyberpunk Dragon materials."""
    return {
        "dark_scales": build_dark_armor_scales(),
        "belly_scutes": build_belly_gold_scutes(),
        "antique_gold": build_antique_gold(),
        "magenta_mane": build_magenta_mane_fur(),
        "magenta_whisker": build_magenta_neon_whisker(),
        "cyan_glow": build_cyan_neon_glow(),
        "amber_eye": build_amber_eye_glow(),
        "pupil_black": build_pupil_black(),
        "silver_beard": build_silver_beard(),
        "dark_mouth": build_dark_mouth(),
    }
