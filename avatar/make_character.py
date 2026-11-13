#!/usr/bin/env python3
"""make_character.py - build a talking-avatar character with MPFB and export it as .glb

Runs inside Blender 4.5 (or the `bpy` 4.5 Python module) with the MPFB add-on
(MakeHuman for Blender) and these CC0 MakeHuman asset packs installed:

    makehuman_system_assets, skins01, skins02, eyebrows01, eyelashes01, hair01,
    faceunits01 (the ARKit face-unit targets used by FaceService)

Usage:
    blender -b -P make_character.py -- --kind maya --out maya.glb --texture-size 1024
    python  make_character.py --kind maya --out maya.glb --texture-size 1024   (bpy module)

Optional extras (the three flags above are the stable interface):
    --preview-dir DIR   render front / three-quarter / expression-strip PNGs of the exported file
    --list              list the available --kind values and exit

What the app relies on, and what this script guarantees and then verifies on the written file:
    * one binary .glb, Y-up, metres, feet at the origin, facing +Z, height 1.6-1.85 m
    * the body mesh carries all 52 ARKit blendshapes (FaceService.load_targets +
      FaceService.interpolate_targets); eyes, teeth, tongue, eyebrows and eyelashes carry them too
    * skin / eyes / teeth / tongue / clothing OPAQUE, hair MASK (cutoff 0.5),
      eyebrows / eyelashes BLEND  (patch_alpha_modes)
    * no Draco, no KTX2/Basis; embedded PNG/JPEG only; < 8 MB; < 60,000 triangles
"""

import argparse
import importlib
import json
import math
import os
import struct
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

# --------------------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------------------
# Every asset is "<pack dir>/<file>" below the MPFB data dir for its type. Only assets
# that are marked CC0 in the installed pack metadata are accepted (see check_licenses).
#
# macro   : MPFB macro targets, 0..1.  gender 0 = female, 1 = male.  age 0.5 = 25 y, 1.0 = 90 y.
# targets : MPFB detail targets and their weights. A leading "lr-" loads the l- and r- pair.
# skin_grade / hair_grade / clothes_grade : colour grading applied to the texture pixels in
#           fix_materials (glTF has no node graph, so the look has to live in the pixels).

PRESETS = {
    # Early-30s woman, South Asian, warm brown skin, shoulder-length dark hair,
    # subtle make-up, navy blouse. A generic face: no likeness of a real person.
    "maya": {
        "macro": {
            "gender": 0.0,
            "age": 0.53,
            "muscle": 0.42,
            "weight": 0.50,
            "proportions": 0.62,
            "height": 0.60,
            "cupsize": 0.45,
            "firmness": 0.55,
            "race": {"asian": 0.30, "caucasian": 0.50, "african": 0.20},
        },
        "targets": {
            # head / jaw: soft oval, slimmer lower face, lower hairline
            "head-oval": 0.45,
            "chin-width-decr": 0.30,
            "chin-height-decr": 0.10,
            "chin-jaw-drop-decr": 0.10,
            # cheeks: light cheekbones, keeps the face from looking flat
            "lr-cheek-bones-incr": 0.25,
            "lr-cheek-volume-incr": 0.12,
            # eyes: slightly larger and a touch upturned = attentive, friendly
            "lr-eye-scale-incr": 0.10,
            "lr-eye-corner2-up": 0.10,
            "lr-eye-height2-decr": 0.05,
            "lr-eye-bag-decr": 0.30,
            # brows: relaxed, outer end slightly lifted
            "eyebrows-trans-up": 0.10,
            "eyebrows-angle-up": 0.45,
            # nose: a little narrower and shorter, soft tip
            "nose-scale-horiz-decr": 0.20,
            "nose-width3-decr": 0.15,
            "nose-point-up": 0.10,
            "nose-scale-vert-decr": 0.08,
            # mouth: natural lips, corners resting slightly up
            "mouth-upperlip-volume-incr": 0.30,
            "mouth-lowerlip-volume-incr": 0.20,
            "mouth-angles-up": 0.55,
            "mouth-cupidsbow-incr": 0.15,
            # neck
            "neck-scale-horiz-decr": 0.15,
        },
        "skin": "cutoff3d_indian_female_skin/cutoff3d_indian_female_skin.mhmat",     # skins01
        "eyes": "high-poly/high-poly.mhclo",                                        # system
        "eye_material": "materials/brown.mhmat",                                    # system
        "eyebrows": "eyebrow012/eyebrow012.mhclo",                                   # system
        "eyelashes": "eyelashes01/eyelashes01.mhclo",                                # system
        "teeth": "teeth_base/teeth_base.mhclo",                                      # system
        "tongue": "tongue01/tongue01.mhclo",                                         # system
        "hair": "toigo_blunt_bob/toigo_blunt_bob.mhclo",                             # hair01
        "clothes": ["female_elegantsuit01/female_elegantsuit01.mhclo"],              # system
        "skin_grade": {"gain": (0.90, 0.86, 0.83), "saturation": 0.95, "gamma": 1.12},
        "lips": {"color": (0.96, 0.70, 0.70), "strength": 0.55},
        "hair_grade": {"gain": (0.24, 0.19, 0.17), "saturation": 0.55, "gamma": 1.0},
        "brow_grade": {"gain": (0.30, 0.24, 0.21), "saturation": 0.6, "gamma": 1.0},
        "eye_grade": {"gain": (0.92, 0.84, 0.78), "saturation": 1.0, "gamma": 1.1},
        # three.js draws no shadow inside the mouth, so the tongue is darkened in the texture
        "tongue_grade": {"gain": (0.50, 0.38, 0.38), "saturation": 0.9, "gamma": 1.0},
        "clothes_grade": {"hue_to": 0.615, "contrast": 0.22, "saturation": 0.85, "gain": (0.80, 0.85, 1.0), "gamma": 1.0},
        "face_texture_size": 2048,
    },
}

# glTF material name, alpha mode and cutoff per role. patch_alpha_modes() enforces this on
# the written file, because the Blender exporter derives alpha modes from the node graph.
ALPHA_MODES = {
    "skin": ("OPAQUE", None),
    "eyes": ("OPAQUE", None),
    "teeth": ("OPAQUE", None),
    "tongue": ("OPAQUE", None),
    "clothes": ("OPAQUE", None),
    "hair": ("MASK", 0.5),
    "eyebrows": ("BLEND", None),
    "eyelashes": ("BLEND", None),
}

# Surface values per role (the knobs to turn when the look is off).
SURFACE = {
    "skin": {"roughness": 0.52, "specular": 0.40, "subsurface": 0.06,
             "subsurface_color": (0.80, 0.36, 0.24), "double_sided": False},
    "eyes": {"roughness": 0.10, "specular": 0.80, "double_sided": False},
    "teeth": {"roughness": 0.25, "specular": 0.55, "double_sided": False},
    "tongue": {"roughness": 0.35, "specular": 0.50, "double_sided": False},
    "clothes": {"roughness": 0.80, "specular": 0.25, "double_sided": True},
    "hair": {"roughness": 0.50, "specular": 0.30, "double_sided": True},
    "eyebrows": {"roughness": 0.70, "specular": 0.10, "double_sided": True},
    "eyelashes": {"roughness": 0.70, "specular": 0.10, "double_sided": True},
}

# Roles whose mesh must carry every ARKit key (moving parts of the face).
MORPH_ROLES = ("skin", "eyes", "teeth", "tongue", "eyebrows", "eyelashes")

REQUIRED_KEYS = [
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "mouthRollLower", "tongueOut", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
]

MAX_BYTES = 8 * 1024 * 1024
MAX_TRIANGLES = 60000
HEIGHT_RANGE = (1.60, 1.85)


def log(*args):
    print("[make_character]", *args, flush=True)


# --------------------------------------------------------------------------------------
# MPFB access
# --------------------------------------------------------------------------------------

class Mpfb:
    """Finds the MPFB add-on however it was installed and exposes the services we use."""

    CANDIDATES = ("bl_ext.user_default.mpfb", "bl_ext.blender_org.mpfb", "bl_ext.system.mpfb", "mpfb")

    def __init__(self):
        import addon_utils
        self.package = None
        for name in self.CANDIDATES:
            try:
                if name not in bpy.context.preferences.addons:
                    # default_set registers the add-on in this session's preferences, which
                    # MPFB reads while it loads (needed for the bpy module / a fresh profile).
                    addon_utils.enable(name, default_set=True)
                importlib.import_module(name + ".services.humanservice")
                self.package = name
                break
            except Exception:  # not installed under this name
                continue
        if not self.package:
            raise SystemExit("MPFB is not installed/enabled in this Blender. Install the MPFB extension first.")

        def svc(module, cls):
            return getattr(importlib.import_module(self.package + ".services." + module), cls)

        self.HumanService = svc("humanservice", "HumanService")
        self.TargetService = svc("targetservice", "TargetService")
        self.AssetService = svc("assetservice", "AssetService")
        self.FaceService = svc("faceservice", "FaceService")
        self.ExportService = svc("exportservice", "ExportService")
        self.LocationService = svc("locationservice", "LocationService")
        self.ObjectService = svc("objectservice", "ObjectService")
        self.ARKIT_FACEUNITS = list(
            importlib.import_module(self.package + ".services.faceservice").ARKIT_FACEUNITS)

    def asset(self, fragment, subdir):
        path = self.AssetService.find_asset_absolute_path(fragment, subdir)
        if not path or not os.path.exists(path):
            raise SystemExit("Asset not found: %s/%s - is the asset pack installed?" % (subdir, fragment))
        return path

    def pack_license(self, asset_name):
        """License string for an asset from the installed pack metadata, or None if unknown."""
        for root in self.AssetService.get_available_data_roots():
            packs = os.path.join(root, "packs")
            if not os.path.isdir(packs):
                continue
            for file_name in sorted(os.listdir(packs)):
                if not file_name.endswith(".json"):
                    continue
                with open(os.path.join(packs, file_name), "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                if asset_name in meta:
                    return meta[asset_name].get("license"), file_name[:-5]
        return None, None


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def parse_mhmat(path):
    """Read the handful of .mhmat fields we need. Texture paths come back absolute."""
    out = {"path": path}
    base = os.path.dirname(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts[0], parts[1].strip()
            if key.endswith("Texture") and key != "litsphereTexture":
                out[key] = os.path.normpath(os.path.join(base, value))
            elif key == "diffuseColor":
                out[key] = tuple(float(x) for x in value.split()[:3])
    return out


def mhclo_material(mhclo_path):
    with open(mhclo_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("material "):
                return os.path.normpath(os.path.join(os.path.dirname(mhclo_path), line.split(None, 1)[1]))
            if line.startswith("verts "):
                break
    return None


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def triangle_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------

def check_licenses(mpfb, preset):
    """Refuse anything that the installed pack metadata does not mark as CC0."""
    used = []
    for key in ("skin", "eyes", "eye_material", "eyebrows", "eyelashes", "teeth", "tongue", "hair"):
        if preset.get(key):
            used.append((key, preset[key]))
    for fragment in preset.get("clothes", []):
        used.append(("clothes", fragment))
    report = []
    for role, fragment in used:
        name = os.path.splitext(os.path.basename(fragment))[0]
        lic, pack = mpfb.pack_license(name)
        if lic is None:  # some packs key the metadata on the directory name
            lic, pack = mpfb.pack_license(os.path.basename(os.path.dirname(fragment)))
        report.append((role, fragment, pack, lic))
        if lic != "CC0":
            raise SystemExit("Asset %s (%s) is not marked CC0 in the pack metadata: %r" % (fragment, role, lic))
    return report


def build_human(mpfb, preset):
    """Base mesh with macro + detail targets, baked so no modelling shape keys remain."""
    macro = mpfb.TargetService.get_default_macro_info_dict()
    for key, value in preset["macro"].items():
        if key == "race":
            macro["race"].update(value)
        else:
            macro[key] = value

    basemesh = mpfb.HumanService.create_human(
        mask_helpers=True, detailed_helpers=True, extra_vertex_groups=True,
        feet_on_ground=True, scale=0.1, macro_detail_dict=macro)

    for name, weight in preset.get("targets", {}).items():
        names = ["l-" + name[3:], "r-" + name[3:]] if name.startswith("lr-") else [name]
        for target in names:
            path = mpfb.TargetService.target_full_path(target)
            if not path:
                raise SystemExit("Unknown MPFB target: " + target)
            mpfb.TargetService.load_target(basemesh, path, weight=float(weight), name=target)

    # Bake the modelling targets into the mesh: only face units may reach the export.
    mpfb.TargetService.bake_targets(basemesh)

    # Feet on the ground after the final shape is known.
    lowest = min((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices
                 if _in_group(basemesh, v, "body"))
    basemesh.location.z -= lowest
    activate(basemesh)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    basemesh["mc_role"] = "skin"
    return basemesh


def _in_group(obj, vert, group_name):
    index = obj.vertex_groups[group_name].index
    return any(g.group == index and g.weight > 0.5 for g in vert.groups)


def add_assets(mpfb, basemesh, preset):
    """Skin, eyes, brows, lashes, teeth, tongue, hair and clothing through MPFB."""
    objects = {"skin": [basemesh]}

    skin_mhmat = mpfb.asset(preset["skin"], "skins")
    mpfb.HumanService.set_character_skin(skin_mhmat, basemesh, skin_type="GAMEENGINE", material_instances=False)
    basemesh["mc_mhmat"] = skin_mhmat

    plan = [
        ("eyes", "eyes", "Eyes"), ("eyebrows", "eyebrows", "Eyebrows"), ("eyelashes", "eyelashes", "Eyelashes"),
        ("teeth", "teeth", "Teeth"), ("tongue", "tongue", "Tongue"), ("hair", "hair", "Hair"),
    ]
    jobs = [(role, subdir, atype, preset[role]) for role, subdir, atype in plan if preset.get(role)]
    jobs += [("clothes", "clothes", "Clothes", fragment) for fragment in preset.get("clothes", [])]

    for role, subdir, atype, fragment in jobs:
        mhclo = mpfb.asset(fragment, subdir)
        obj = mpfb.HumanService.add_mhclo_asset(
            mhclo, basemesh, asset_type=atype, subdiv_levels=0, material_type="GAMEENGINE",
            set_up_rigging=False, interpolate_weights=False, import_subrig=False, import_weights=False)
        obj["mc_role"] = role
        mhmat = mhclo_material(mhclo)
        if role == "eyes" and preset.get("eye_material"):
            mhmat = mpfb.asset(preset["eye_material"], "eyes")
        obj["mc_mhmat"] = mhmat or ""
        objects.setdefault(role, []).append(obj)
        log("added", role, fragment, "tris=%d" % triangle_count(obj))
    return objects


def add_face_units(mpfb, basemesh, objects):
    """All 52 ARKit face units on the body, interpolated to the meshes that follow the face."""
    mpfb.FaceService.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=False,
                                  load_arkit_faceunits=True)
    mpfb.FaceService.interpolate_targets(basemesh)

    names = mpfb.ARKIT_FACEUNITS
    for role, objs in objects.items():
        for obj in objs:
            keys = obj.data.shape_keys
            if role not in MORPH_ROLES:
                # Hair and clothing stay rigid: a jaw-driven bob looks wrong, and it saves bytes.
                if keys:
                    activate(obj)
                    bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
                continue
            # interpolate_targets skips keys that do not move a mesh. Pad them so every face
            # mesh has the same 52 names in the same order (apps often index by position).
            if not keys:
                obj.shape_key_add(name="Basis", from_mix=False)
            for name in names:
                if name not in obj.data.shape_keys.key_blocks:
                    obj.shape_key_add(name=name, from_mix=False)
            _order_shape_keys(obj, names)
            for block in obj.data.shape_keys.key_blocks:
                block.value = 0.0


def _order_shape_keys(obj, names):
    blocks = obj.data.shape_keys.key_blocks
    current = [b.name for b in blocks][1:]
    if current == list(names):
        return
    activate(obj)
    for position, name in enumerate(names, start=1):
        index = blocks.find(name)
        while index > position:
            obj.active_shape_key_index = index
            bpy.ops.object.shape_key_move(type="UP")
            index -= 1


def stage_for_export(mpfb, basemesh, objects):
    """Bake delete-group masks, drop helper geometry, give objects clean names."""
    mpfb.ExportService.bake_modifiers_remove_helpers(
        basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=False)

    counters = {}
    for role, objs in objects.items():
        for obj in objs:
            for modifier in list(obj.modifiers):
                obj.modifiers.remove(modifier)
            obj.parent = None
            activate(obj)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            bpy.ops.object.shade_smooth()
            for group in list(obj.vertex_groups):
                obj.vertex_groups.remove(group)
            name = "body" if role == "skin" else role
            counters[name] = counters.get(name, 0) + 1
            if counters[name] > 1:
                name = "%s%d" % (name, counters[name])
            obj.name = name
            obj.data.name = name


# --------------------------------------------------------------------------------------
# Textures and materials
# --------------------------------------------------------------------------------------

def _read_pixels(path, max_side):
    """Load an image file as float RGBA (h, w, 4), scaled so the longest side is <= max_side."""
    image = bpy.data.images.load(path, check_existing=False)
    image.colorspace_settings.name = "sRGB"
    width, height = image.size
    scale = min(1.0, float(max_side) / max(width, height))
    if scale < 1.0:
        width, height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        image.scale(width, height)
    flat = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(flat)
    bpy.data.images.remove(image)
    return flat.reshape(height, width, 4)


def _write_pixels(pixels, path, name, non_color=False, quality=90):
    """Save float RGBA as PNG (if it needs alpha) or JPEG and return it as a file-backed image.

    Image.save() writes the byte values as they are. (save_render() would push them through
    the scene's display transform and brighten every texture.)
    """
    height, width = pixels.shape[:2]
    is_jpeg = path.lower().endswith(".jpg")
    image = bpy.data.images.new(name + "_tmp", width, height, alpha=not is_jpeg)
    image.pixels.foreach_set(np.clip(pixels, 0.0, 1.0).astype(np.float32).ravel())
    image.file_format = "JPEG" if is_jpeg else "PNG"
    image.filepath_raw = path
    image.save(filepath=path, quality=quality)
    bpy.data.images.remove(image)
    saved = bpy.data.images.load(path, check_existing=False)
    saved.name = name
    saved.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return saved


def _grade(pixels, grade):
    """Colour-grade RGB in place: optional hue replacement, saturation, gamma, per-channel gain."""
    if not grade:
        return pixels
    rgb = pixels[..., :3]
    luma = (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1, keepdims=True)
    if grade.get("hue_to") is not None:
        # Replace the hue, keep each pixel's own saturation and brightness (keeps greys grey).
        maxc, minc = rgb.max(axis=-1, keepdims=True), rgb.min(axis=-1, keepdims=True)
        sat = np.where(maxc > 1e-5, (maxc - minc) / np.maximum(maxc, 1e-5), 0.0)
        hue = float(grade["hue_to"]) * 6.0
        k = (np.array([5.0, 3.0, 1.0], dtype=np.float32) + hue) % 6.0
        base = 1.0 - np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)      # hue colour at s = 1, v = 1
        rgb = maxc * (1.0 - sat * (1.0 - base))
        luma = (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1, keepdims=True)
    if grade.get("contrast") is not None:
        # Pull values towards the mean brightness (tones down loud stripes / prints).
        mean = float(luma.mean())
        scaled = mean + (luma - mean) * float(grade["contrast"])
        rgb = rgb * (scaled / np.maximum(luma, 1e-4))
        luma = scaled
    rgb = luma + (rgb - luma) * float(grade.get("saturation", 1.0))
    rgb = np.clip(rgb, 0.0, 1.0) ** float(grade.get("gamma", 1.0))
    rgb = rgb * np.array(grade.get("gain", (1.0, 1.0, 1.0)), dtype=np.float32)
    pixels[..., :3] = np.clip(rgb, 0.0, 1.0)
    return pixels


def _uv_group_mask(obj, group_name, width, height, feather):
    """Rasterise the faces of a vertex group into a soft UV-space mask (h, w) in 0..1."""
    mask = np.zeros((height, width), dtype=np.float32)
    if group_name not in obj.vertex_groups:
        return mask
    index = obj.vertex_groups[group_name].index
    members = {v.index for v in obj.data.vertices if any(g.group == index and g.weight > 0.5 for g in v.groups)}
    uvs = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        if not all(v in members for v in poly.vertices):
            continue
        pts = np.array([(uvs[l].uv[0] * width, uvs[l].uv[1] * height) for l in poly.loop_indices])
        for a in range(1, len(pts) - 1):
            tri = pts[[0, a, a + 1]]
            x0, y0 = np.floor(tri.min(axis=0)).astype(int)
            x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width - 1), min(y1, height - 1)
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, ay), (bx, by), (cx, cy) = tri
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-9:
                continue
            w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
            w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
            inside = (w0 >= -0.01) & (w1 >= -0.01) & (w0 + w1 <= 1.01)
            mask[y0:y1 + 1, x0:x1 + 1][inside] = 1.0
    for _ in range(3):  # three box blurs ~ a gaussian
        if feather > 0:
            mask = _box_blur(mask, feather)
    return mask


def _box_blur(mask, radius):
    size = 2 * radius + 1
    for axis in (0, 1):
        padded = np.pad(mask, [(radius + 1, radius) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
        if axis == 0:
            mask = ((csum[size:, :] - csum[:-size, :]) / size).astype(np.float32)
        else:
            mask = ((csum[:, size:] - csum[:, :-size]) / size).astype(np.float32)
    return mask


def capture_uv_masks(basemesh, preset, size):
    """Lip mask for the make-up pass. Must run while the MPFB vertex groups still exist."""
    masks = {}
    if preset.get("lips"):
        masks["lips"] = _uv_group_mask(basemesh, "lips", size, size, feather=max(1, size // 512))
    return masks


def fix_materials(objects, preset, texture_size, work_dir, uv_masks=None):
    """Replace MPFB's materials with plain glTF-ready Principled materials.

    One material per role, named after the role (skin, eyes, teeth, tongue, hair, eyebrows,
    eyelashes, clothes). Textures are graded, resized and written as PNG (alpha) or JPEG
    (opaque) so the exporter embeds exactly those files. Tweak SURFACE and the *_grade
    entries of the preset to change the look.
    """
    uv_masks = uv_masks or {}
    grades = {"skin": "skin_grade", "hair": "hair_grade", "eyebrows": "brow_grade", "eyelashes": "brow_grade",
              "eyes": "eye_grade", "clothes": "clothes_grade", "tongue": "tongue_grade"}
    materials = {}

    for role, objs in objects.items():
        for number, obj in enumerate(objs):
            mhmat_path = obj.get("mc_mhmat") or ""
            mhmat = parse_mhmat(mhmat_path) if mhmat_path and os.path.exists(mhmat_path) else {}
            mode, cutoff = ALPHA_MODES[role]
            surface = SURFACE[role]
            name = role if number == 0 else "%s%d" % (role, number + 1)
            size = texture_size
            if role == "skin":
                size = max(texture_size, int(preset.get("face_texture_size", texture_size)))

            material = bpy.data.materials.new(name)
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            out = tree.nodes.new("ShaderNodeOutputMaterial")
            bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
            tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            bsdf.inputs["Roughness"].default_value = surface["roughness"]
            bsdf.inputs["Specular IOR Level"].default_value = surface["specular"]
            bsdf.inputs["Metallic"].default_value = 0.0
            if surface.get("subsurface"):
                # Used by the Blender preview only; glTF core has no subsurface term.
                bsdf.inputs["Subsurface Weight"].default_value = surface["subsurface"]
                bsdf.inputs["Subsurface Radius"].default_value = surface["subsurface_color"]
                bsdf.inputs["Subsurface Scale"].default_value = 0.01

            diffuse = mhmat.get("diffuseTexture")
            if diffuse and os.path.exists(diffuse):
                pixels = _read_pixels(diffuse, size)
                pixels = _grade(pixels, preset.get(grades.get(role, "")))
                if role == "skin" and preset.get("lips") and "lips" in uv_masks:
                    pixels = _apply_lips(pixels, uv_masks["lips"], preset["lips"])
                if mode == "OPAQUE":
                    if role == "eyes":
                        # The high-poly eye has a see-through cornea shell mapped to a transparent
                        # patch of the texture. The app needs the eye OPAQUE, so drop that shell.
                        _delete_transparent_faces(obj, pixels[..., 3])
                    pixels[..., 3] = 1.0
                    path = os.path.join(work_dir, name + "_color.jpg")
                else:
                    path = os.path.join(work_dir, name + "_color.png")
                image = _write_pixels(pixels, path, name + "_color")
                tex = tree.nodes.new("ShaderNodeTexImage")
                tex.image = image
                tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                if mode != "OPAQUE":
                    tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            else:
                colour = mhmat.get("diffuseColor", (0.8, 0.8, 0.8))
                bsdf.inputs["Base Color"].default_value = (colour[0], colour[1], colour[2], 1.0)

            normal = mhmat.get("normalmapTexture")
            if normal and os.path.exists(normal) and role in ("clothes", "hair"):
                pixels = _read_pixels(normal, texture_size)
                pixels[..., 3] = 1.0
                image = _write_pixels(pixels, os.path.join(work_dir, name + "_normal.jpg"), name + "_normal",
                                      non_color=True, quality=92)
                tex = tree.nodes.new("ShaderNodeTexImage")
                tex.image = image
                node = tree.nodes.new("ShaderNodeNormalMap")
                node.inputs["Strength"].default_value = 1.0
                tree.links.new(tex.outputs["Color"], node.inputs["Color"])
                tree.links.new(node.outputs["Normal"], bsdf.inputs["Normal"])

            material.use_backface_culling = not surface["double_sided"]
            if mode == "MASK":
                material.surface_render_method = "DITHERED"
                material.alpha_threshold = cutoff
            elif mode == "BLEND":
                material.surface_render_method = "BLENDED"

            obj.data.materials.clear()
            obj.data.materials.append(material)
            materials[name] = (mode, cutoff, surface["double_sided"])
    return materials


def _delete_transparent_faces(obj, alpha, threshold=0.5):
    """Remove faces whose UV centre samples a transparent texel (keeps the shape keys)."""
    import bmesh
    height, width = alpha.shape
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    layer = bm.loops.layers.uv.active
    doomed = []
    for face in bm.faces:
        u = sum(loop[layer].uv[0] for loop in face.loops) / len(face.loops)
        v = sum(loop[layer].uv[1] for loop in face.loops) / len(face.loops)
        x = min(width - 1, max(0, int(u * width)))
        y = min(height - 1, max(0, int(v * height)))
        if alpha[y, x] < threshold:
            doomed.append(face)
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    log("removed %d transparent faces from %s" % (len(doomed), obj.name))


def _apply_lips(pixels, mask, lips):
    """Subtle lip colour: multiply the skin by a tint inside the (feathered) lip mask."""
    if mask.shape != pixels.shape[:2]:
        ys = (np.arange(pixels.shape[0]) * mask.shape[0] / pixels.shape[0]).astype(int)
        xs = (np.arange(pixels.shape[1]) * mask.shape[1] / pixels.shape[1]).astype(int)
        mask = mask[ys][:, xs]
    amount = (mask * float(lips.get("strength", 0.5)))[..., None]
    rgb = pixels[..., :3]
    tinted = rgb * np.array(lips["color"], dtype=np.float32)
    pixels[..., :3] = rgb * (1.0 - amount) + tinted * amount
    return pixels


# --------------------------------------------------------------------------------------
# Export and post-processing
# --------------------------------------------------------------------------------------

def export_glb(objects, out_path):
    bpy.ops.object.select_all(action="DESELECT")
    for objs in objects.values():
        for obj in objs:
            obj.select_set(True)
    bpy.context.view_layer.objects.active = objects["skin"][0]
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLB",
        use_selection=True,
        export_yup=True,                      # Blender -Y forward / Z up  ->  glTF +Z forward / Y up
        export_apply=False,                   # applying modifiers would drop the shape keys
        export_texcoords=True,
        export_normals=True,
        export_tangents=False,
        export_materials="EXPORT",
        export_image_format="AUTO",           # keeps our PNG / JPEG files as they are
        export_draco_mesh_compression_enable=False,
        export_morph=True,
        export_morph_normal=False,            # halves the morph data; lips shade fine without
        export_morph_tangent=False,
        export_try_sparse_sk=True,            # store only the vertices a face unit moves
        export_try_omit_sparse_sk=True,
        export_animations=False,
        export_skins=False,
        export_cameras=False,
        export_lights=False,
        export_extras=False,
    )


def _read_glb(path):
    with open(path, "rb") as handle:
        data = handle.read()
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise SystemExit("Not a binary glTF 2.0 file: " + path)
    json_len, json_type = struct.unpack_from("<I4s", data, 12)
    if json_type != b"JSON":
        raise SystemExit("First GLB chunk is not JSON")
    doc = json.loads(data[20:20 + json_len].decode("utf-8"))
    rest = data[20 + json_len:]
    return doc, rest


def _write_glb(path, doc, rest):
    payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload) + len(rest)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<I4s", len(payload), b"JSON"))
        handle.write(payload)
        handle.write(rest)


def patch_alpha_modes(glb_path, materials):
    """Force alphaMode / alphaCutoff / doubleSided per material in the written .glb.

    The Blender exporter guesses the alpha mode from the node graph (and has changed how
    between versions), so the app's contract is enforced on the file itself:
    skin, eyes, teeth, tongue, clothing OPAQUE; hair MASK 0.5; eyebrows, eyelashes BLEND.
    `materials` maps glTF material name -> (mode, cutoff, double_sided).
    """
    doc, rest = _read_glb(glb_path)
    for material in doc.get("materials", []):
        spec = materials.get(material.get("name"))
        if spec is None:
            raise SystemExit("Exported material without an alpha rule: %r" % material.get("name"))
        mode, cutoff, double_sided = spec
        material.pop("alphaCutoff", None)
        if mode == "OPAQUE":
            material.pop("alphaMode", None)       # OPAQUE is the glTF default
        else:
            material["alphaMode"] = mode
            if mode == "MASK":
                material["alphaCutoff"] = float(cutoff)
        material["doubleSided"] = bool(double_sided)
    _write_glb(glb_path, doc, rest)


def verify_glb(glb_path, expect_all_keys):
    """Check the written file against the app's hard requirements; print what was found."""
    doc, rest = _read_glb(glb_path)
    problems = []
    size = os.path.getsize(glb_path)

    used_ext = set(doc.get("extensionsUsed", [])) | set(doc.get("extensionsRequired", []))
    for banned in ("KHR_draco_mesh_compression", "KHR_texture_basisu", "EXT_meshopt_compression"):
        if banned in used_ext:
            problems.append("uses " + banned)

    accessors = doc.get("accessors", [])
    mesh_node = {node["mesh"]: node for node in doc.get("nodes", []) if "mesh" in node}
    triangles = 0
    low, high = [1e9] * 3, [-1e9] * 3
    log("shape keys per exported mesh:")
    for index, mesh in enumerate(doc.get("meshes", [])):
        names = mesh.get("extras", {}).get("targetNames", [])
        tris = 0
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4:
                problems.append("mesh %s is not a triangle list" % mesh.get("name"))
            tris += accessors[prim["indices"]]["count"] // 3
            pos = accessors[prim["attributes"]["POSITION"]]
            offset = mesh_node.get(index, {}).get("translation", [0, 0, 0])
            for axis in range(3):
                low[axis] = min(low[axis], pos["min"][axis] + offset[axis])
                high[axis] = max(high[axis], pos["max"][axis] + offset[axis])
            if len(prim.get("targets", [])) != len(names):
                problems.append("mesh %s: %d targets but %d names" % (mesh.get("name"), len(prim.get("targets", [])), len(names)))
        triangles += tris
        log("  %-10s tris=%-6d keys=%d %s" % (mesh.get("name"), tris, len(names), ", ".join(names)))
        if mesh.get("name") in expect_all_keys:
            missing = [key for key in expect_all_keys[mesh["name"]] if key not in names]
            if missing:
                problems.append("mesh %s lacks shape keys: %s" % (mesh["name"], ", ".join(missing)))

    log("materials:")
    for material in doc.get("materials", []):
        mode = material.get("alphaMode", "OPAQUE")
        log("  %-10s %-6s cutoff=%s doubleSided=%s" % (material.get("name"), mode,
                                                      material.get("alphaCutoff", "-"), material.get("doubleSided", False)))
        want = ALPHA_MODES.get(material.get("name", "").rstrip("0123456789"))
        if want and want[0] != mode:
            problems.append("material %s is %s, expected %s" % (material.get("name"), mode, want[0]))
        if mode == "MASK" and abs(material.get("alphaCutoff", 0.5) - 0.5) > 1e-6:
            problems.append("material %s cutoff is not 0.5" % material.get("name"))

    log("images:")
    views = doc.get("bufferViews", [])
    bin_data = rest[8:] if rest else b""
    for image in doc.get("images", []):
        mime = image.get("mimeType")
        if "uri" in image or mime not in ("image/png", "image/jpeg"):
            problems.append("image %s is not an embedded PNG/JPEG" % image.get("name"))
            continue
        view = views[image["bufferView"]]
        blob = bin_data[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
        width, height = _image_size(blob, mime)
        log("  %-18s %-10s %4dx%-4d %7.1f kB" % (image.get("name"), mime, width, height, len(blob) / 1024.0))
        limit = 2048 if image.get("name", "").startswith("skin") else 1024
        if max(width, height) > limit:
            problems.append("image %s is %dx%d (limit %d)" % (image.get("name"), width, height, limit))

    height_m = high[1] - low[1]
    log("bounds  x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f   height %.3f m" % (low[0], high[0], low[1], high[1], low[2], high[2], height_m))
    log("totals  %d triangles, %.2f MB, %d meshes, %d materials, %d images" % (
        triangles, size / 1048576.0, len(doc.get("meshes", [])), len(doc.get("materials", [])), len(doc.get("images", []))))
    if not HEIGHT_RANGE[0] <= height_m <= HEIGHT_RANGE[1]:
        problems.append("height %.3f m outside %.2f-%.2f" % (height_m, HEIGHT_RANGE[0], HEIGHT_RANGE[1]))
    if abs(low[1]) > 0.01:
        problems.append("feet are not at y = 0 (lowest point %.3f)" % low[1])
    if abs((low[0] + high[0]) * 0.5) > 0.03:
        problems.append("character is not centred on x")
    if triangles >= MAX_TRIANGLES:
        problems.append("%d triangles (budget %d)" % (triangles, MAX_TRIANGLES))
    if size >= MAX_BYTES:
        problems.append("%.2f MB (budget 8 MB)" % (size / 1048576.0))
    if doc.get("skins") or doc.get("animations"):
        problems.append("unexpected skins/animations in the file")
    return problems


def _image_size(blob, mime):
    if mime == "image/png":
        return struct.unpack(">II", blob[16:24])
    position = 2
    while position < len(blob):
        marker, length = struct.unpack(">HH", blob[position:position + 4])
        if 0xFFC0 <= marker <= 0xFFCF and marker not in (0xFFC4, 0xFFC8, 0xFFCC):
            height, width = struct.unpack(">HH", blob[position + 5:position + 9])
            return width, height
        position += 2 + length
    return 0, 0


# --------------------------------------------------------------------------------------
# Preview renders (of the exported file, not of the working scene)
# --------------------------------------------------------------------------------------

def render_previews(glb_path, out_dir, kind):
    """Front, three-quarter and a 4-up expression strip, rendered from the written .glb."""
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_homefile(use_empty=True, use_factory_startup=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]
    body = next(o for o in meshes if o.name.startswith("body"))
    top = max((body.matrix_world @ Vector(c)).z for c in body.bound_box)
    for obj in meshes:
        for material in obj.data.materials:
            name = material.name.rstrip("0123456789.")
            if name == "skin":
                bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
                bsdf.inputs["Subsurface Weight"].default_value = SURFACE["skin"]["subsurface"]
                bsdf.inputs["Subsurface Radius"].default_value = SURFACE["skin"]["subsurface_color"]
                bsdf.inputs["Subsurface Scale"].default_value = 0.01

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("preview")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.74, 0.77, 0.82, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.40
    scene.world = world

    def light(name, location, energy, size, colour):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size, data.color = energy, size, colour
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        target = Vector((0.0, 0.0, top - 0.12))
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()

    light("key", (0.9, -1.6, top + 0.45), 55.0, 0.9, (1.0, 0.96, 0.90))
    light("fill", (-1.3, -1.2, top + 0.05), 14.0, 1.4, (0.88, 0.93, 1.0))
    light("rim", (-0.5, 1.4, top + 0.5), 45.0, 0.7, (1.0, 1.0, 1.0))

    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens = 85.0
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    def aim(angle_deg, distance, look_z, lens=85.0):
        camera_data.lens = lens
        angle = math.radians(angle_deg)
        camera.location = (math.sin(angle) * distance, -math.cos(angle) * distance, look_z + 0.02)
        target = Vector((0.0, 0.0, look_z))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()

    def set_keys(values):
        for obj in meshes:
            keys = obj.data.shape_keys
            if not keys:
                continue
            for block in keys.key_blocks[1:]:
                block.value = values.get(block.name, 0.0)

    def render(path, width, height):
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        return path

    outputs = []
    set_keys({})
    aim(0.0, 1.55, top - 0.20)
    outputs.append(render(os.path.join(out_dir, kind + "_front.png"), 900, 1100))
    aim(32.0, 1.55, top - 0.20)
    outputs.append(render(os.path.join(out_dir, kind + "_three_quarter.png"), 900, 1100))

    tiles = []
    aim(0.0, 1.45, top - 0.145, lens=110.0)
    for key in ("jawOpen", "mouthPucker", "mouthClose", "eyeBlinkLeft"):
        set_keys({key: 1.0})
        tiles.append(render(os.path.join(out_dir, "_%s_%s.png" % (kind, key)), 520, 560))
    set_keys({})

    arrays = []
    for tile in tiles:
        image = bpy.data.images.load(tile)
        width, height = image.size
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
        arrays.append(flat.reshape(height, width, 4))
        bpy.data.images.remove(image)
        os.remove(tile)
    strip = np.concatenate(arrays, axis=1)
    image = bpy.data.images.new("strip", strip.shape[1], strip.shape[0], alpha=True)
    image.pixels.foreach_set(strip.ravel())
    image.file_format = "PNG"
    strip_path = os.path.join(out_dir, kind + "_strip_jawOpen_mouthPucker_mouthClose_eyeBlinkLeft.png")
    image.filepath_raw = strip_path
    image.save()
    outputs.append(strip_path)
    for path in outputs:
        log("preview", path)
    return outputs


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Build a talking-avatar character (.glb) with MPFB")
    parser.add_argument("--kind", default="maya", help="preset name, one of: " + ", ".join(sorted(PRESETS)))
    parser.add_argument("--out", default="character.glb", help="output .glb path")
    parser.add_argument("--texture-size", type=int, default=1024, help="longest texture side in pixels")
    parser.add_argument("--preview-dir", default=None, help="also render preview PNGs of the exported file here")
    parser.add_argument("--list", action="store_true", help="list presets and exit")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.list:
        print("\n".join(sorted(PRESETS)))
        return 0
    if args.kind not in PRESETS:
        raise SystemExit("Unknown --kind %r. Available: %s" % (args.kind, ", ".join(sorted(PRESETS))))
    preset = PRESETS[args.kind]
    texture_size = max(64, min(int(args.texture_size), 2048))
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Empty scene, but keep the user preferences (they hold the enabled MPFB add-on).
    bpy.ops.wm.read_homefile(use_empty=True, use_factory_startup=True)
    mpfb = Mpfb()
    log("Blender", bpy.app.version_string, "| MPFB package", mpfb.package)

    log("assets (all must be CC0):")
    for role, fragment, pack, lic in check_licenses(mpfb, preset):
        log("  %-12s %-62s pack=%-24s %s" % (role, fragment, pack, lic))

    basemesh = build_human(mpfb, preset)
    objects = add_assets(mpfb, basemesh, preset)
    add_face_units(mpfb, basemesh, objects)
    mask_size = max(texture_size, int(preset.get("face_texture_size", texture_size)))
    uv_masks = capture_uv_masks(basemesh, preset, mask_size)
    stage_for_export(mpfb, basemesh, objects)

    with tempfile.TemporaryDirectory(prefix="make_character_") as work_dir:
        materials = fix_materials(objects, preset, texture_size, work_dir, uv_masks)
        export_glb(objects, out_path)
    patch_alpha_modes(out_path, materials)

    expect = {obj.name: mpfb.ARKIT_FACEUNITS for role in MORPH_ROLES for obj in objects.get(role, [])}
    for key in REQUIRED_KEYS:
        assert key in mpfb.ARKIT_FACEUNITS, key
    problems = verify_glb(out_path, expect)
    if args.preview_dir:
        render_previews(out_path, os.path.abspath(args.preview_dir), args.kind)
    if problems:
        for problem in problems:
            log("FAILED:", problem)
        return 1
    log("OK", out_path)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as error:
        if error.code not in (None, 0):
            print("[make_character] ERROR:", error.code, flush=True)
        code = 0 if error.code in (None, 0) else 1
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard exit: `blender -b` would otherwise report success after a Python error, and the
    # bpy module can hang on interpreter shutdown.
    os._exit(code)
