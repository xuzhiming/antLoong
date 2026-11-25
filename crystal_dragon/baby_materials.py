"""Image-backed PBR material library for the Crystal-Lavender Winged Baby Dragon (Blender 5.2 + glTF 2.0)."""

import os
import bpy

TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def _get_or_create_material(name: str):
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


def build_lavender_scales():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Lavender_Scales")
    mat.diffuse_color = (0.68, 0.64, 0.88, 1.0)
    tex_alb = _add_tex_node(nt, "lavender_scales_albedo.png", "sRGB", "REPEAT", (-360, 260))
    tex_rough = _add_tex_node(nt, "lavender_scales_roughness.png", "Non-Color", "REPEAT", (-360, -20))
    nm = _add_normal_map(nt, "lavender_scales_normal.png", 1.15, "REPEAT", (-480, -300), (-120, -260))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.08
    bsdf.inputs["Coat Weight"].default_value = 0.24
    bsdf.inputs["Coat Roughness"].default_value = 0.18
    bsdf.inputs["Sheen Weight"].default_value = 0.35
    bsdf.inputs["Sheen Tint"].default_value = (0.95, 0.85, 1.0, 1.0)
    return mat


def build_face_lavender():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Face_Lavender")
    mat.diffuse_color = (0.70, 0.66, 0.89, 1.0)
    tex_alb = _add_tex_node(nt, "face_scales_albedo.png", "sRGB", "REPEAT", (-360, 220))
    nm = _add_normal_map(nt, "face_scales_normal.png", 0.38, "REPEAT", (-480, -180), (-120, -180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.06
    bsdf.inputs["Roughness"].default_value = 0.36
    bsdf.inputs["Coat Weight"].default_value = 0.16
    bsdf.inputs["Sheen Weight"].default_value = 0.35
    bsdf.inputs["Sheen Tint"].default_value = (0.95, 0.85, 1.0, 1.0)
    return mat


def build_pearl_belly():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Pearl_Belly")
    mat.diffuse_color = (0.92, 0.92, 0.97, 1.0)
    tex_alb = _add_tex_node(nt, "pearl_belly_albedo.png", "sRGB", "EXTEND", (-360, 220))
    nm = _add_normal_map(nt, "pearl_belly_normal.png", 0.80, "EXTEND", (-480, -180), (-120, -180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.10
    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Coat Weight"].default_value = 0.25
    return mat


def build_wing_membrane():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Wing_Membrane")
    mat.diffuse_color = (0.88, 0.72, 0.92, 1.0)
    tex_alb = _add_tex_node(nt, "wing_membrane_albedo.png", "sRGB", "EXTEND", (-360, 220))
    nm = _add_normal_map(nt, "wing_membrane_normal.png", 0.65, "EXTEND", (-480, -180), (-120, -180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.46
    bsdf.inputs["Coat Weight"].default_value = 0.05
    bsdf.inputs["Sheen Weight"].default_value = 0.45
    bsdf.inputs["Sheen Tint"].default_value = (1.0, 0.85, 0.92, 1.0)
    return mat


def build_frill_gradient():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Frill_Gradient")
    mat.diffuse_color = (0.82, 0.74, 0.94, 1.0)
    tex_alb = _add_tex_node(nt, "frill_gradient_albedo.png", "sRGB", "EXTEND", (-360, 180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Coat Weight"].default_value = 0.22
    bsdf.inputs["Sheen Weight"].default_value = 0.40
    bsdf.inputs["Sheen Tint"].default_value = (1.0, 0.84, 0.92, 1.0)
    return mat


def build_horn_pearl():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Horn_Pearl")
    mat.diffuse_color = (0.88, 0.84, 0.90, 1.0)
    tex_alb = _add_tex_node(nt, "horn_pearl_albedo.png", "sRGB", "EXTEND", (-360, 220))
    nm = _add_normal_map(nt, "horn_pearl_normal.png", 1.05, "EXTEND", (-480, -180), (-120, -180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.14
    bsdf.inputs["Roughness"].default_value = 0.26
    bsdf.inputs["Coat Weight"].default_value = 0.35
    return mat


def build_baby_eye():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Baby_Eye")
    mat.diffuse_color = (0.20, 0.75, 0.82, 1.0)
    tex_alb = _add_tex_node(nt, "baby_eye_albedo.png", "sRGB", "EXTEND", (-360, 180))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.15
    bsdf.inputs["Roughness"].default_value = 0.04
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.02
    return mat


def build_tail_plume():
    mat, nt, bsdf, _ = _get_or_create_material("Mat_Tail_Plume")
    mat.diffuse_color = (0.68, 0.76, 0.96, 1.0)
    tex_alb = _add_tex_node(nt, "tail_plume_albedo.png", "sRGB", "EXTEND", (-360, 220))
    tex_em = _add_tex_node(nt, "tail_plume_emissive.png", "sRGB", "EXTEND", (-360, -100))
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex_em.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.4
    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Sheen Weight"].default_value = 0.35
    return mat


def _build_solid_mat(name, base_col, metallic=0.0, roughness=0.35, coat=0.0, em_col=None, em_str=0.0, sheen=0.0, sheen_tint=None):
    mat, _, bsdf, _ = _get_or_create_material(name)
    mat.diffuse_color = (*base_col, 1.0)
    bsdf.inputs["Base Color"].default_value = (*base_col, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
    if sheen > 0.0:
        bsdf.inputs["Sheen Weight"].default_value = sheen
        if sheen_tint is not None:
            bsdf.inputs["Sheen Tint"].default_value = sheen_tint
    if em_col is not None and em_str > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*em_col, 1.0)
        bsdf.inputs["Emission Strength"].default_value = em_str
    return mat


def create_all_materials():
    """Build and return dict of all Crystal Baby Dragon materials."""
    return {
        "lavender_scales": build_lavender_scales(),
        "face_lavender": build_face_lavender(),
        "pearl_belly": build_pearl_belly(),
        "wing_membrane": build_wing_membrane(),
        "frill_gradient": build_frill_gradient(),
        "horn_pearl": build_horn_pearl(),
        "claw_pearl": _build_solid_mat("Mat_Claw_Pearl", (0.88, 0.84, 0.91), metallic=0.12, roughness=0.25, coat=0.35),
        "baby_eye": build_baby_eye(),
        "tail_plume": build_tail_plume(),
        "pearl_armor": _build_solid_mat("Mat_Pearl_Armor", (0.90, 0.88, 0.95), metallic=0.12, roughness=0.24, coat=0.40),
        "rose_gold": _build_solid_mat("Mat_Rose_Gold", (0.88, 0.66, 0.48), metallic=0.72, roughness=0.25, coat=0.25),
        "saddle_velvet": _build_solid_mat(
            "Mat_Saddle_Velvet", (0.26, 0.21, 0.42), metallic=0.02, roughness=0.68, coat=0.04,
            sheen=0.65, sheen_tint=(0.86, 0.78, 1.0, 1.0),
        ),
        "ice_cyan_rune": _build_solid_mat(
            "Mat_Ice_Cyan_Rune", (0.06, 0.86, 0.98), metallic=0.0, roughness=0.14,
            em_col=(0.04, 0.88, 1.0), em_str=2.5
        ),
        "pink_pad": _build_solid_mat("Mat_Pink_Paw_Pad", (0.95, 0.64, 0.74), metallic=0.0, roughness=0.42, coat=0.15),
        "inner_mouth": _build_solid_mat("Mat_Inner_Mouth", (0.26, 0.06, 0.14), metallic=0.0, roughness=0.30, coat=0.30),
        "nostril_dark": _build_solid_mat("Mat_Nostril_Dark", (0.24, 0.20, 0.40), metallic=0.0, roughness=0.45),
        "dark_lash": _build_solid_mat("Mat_Dark_Lash", (0.09, 0.08, 0.16), metallic=0.0, roughness=0.22),
        "tooth_white": _build_solid_mat("Mat_Tooth_White", (0.96, 0.95, 0.98), metallic=0.0, roughness=0.15, coat=0.25),
    }
