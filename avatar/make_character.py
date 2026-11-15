#!/usr/bin/env python3
"""make_character.py - build a talking-avatar character with MPFB and export it as .glb

Runs inside Blender 4.5 (or the `bpy` 4.5 Python module) with the MPFB add-on
(MakeHuman for Blender) and these CC0 MakeHuman asset packs installed:

    makehuman_system_assets, skins01, skins02, eyebrows01, eyelashes01, hair01,
    faceunits01 (the ARKit face-unit targets used by FaceService)

Usage:
    blender -b -P make_character.py -- --kind maya --out maya.glb --texture-size 1024
    python  make_character.py --kind maya --out maya.glb --texture-size 1024   (bpy module)

Presets: `maya` is the unrigged head-and-shoulders character. `maya_pro` is the same person
as a full-length character: Mixamo-named skeleton, arms-down rest pose, shoes, an "idle"
animation clip, skin micro-normal map and morph normals.

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
from mathutils import Matrix, Quaternion, Vector

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

# Full-length version of the same character. Adds (all optional keys, read by the functions
# named in brackets):
#   rig            MPFB built-in rig; "mixamo" gives mixamorig:* bone names          [add_rig]
#   rest_pose      arms lowered from the A-pose and baked in as the bind pose        [apply_rest_pose]
#   idle           a looping "idle" clip (breathing, weight shift) in the .glb       [build_idle_animation]
#   clothes        entries may be dicts with their own grade / surface / paint
#   hair_follows   "head" (default, rigid: right for short hair) or "body" (weights taken from
#                  the body underneath, so long hair stays on the shoulders)      [add_assets]
#   skin_normal    strength of a micro-normal map derived from the skin texture      [fix_materials]
#   morph_normals  export shape-key normals (better lip and lid shading, more bytes) [export_glb]
FACE_REFINED = {
    # head and jaw: oval, slimmer, narrower jaw with a soft point at the chin
    "head-oval": 0.50, "head-invertedtriangular": 0.20, "head-fat-decr": 0.30, "head-scale-horiz-decr": 0.08,
    "chin-width-decr": 0.50, "chin-bones-decr": 0.45, "chin-triangle": 0.30,
    "chin-prominent-incr": 0.10, "chin-height-decr": 0.10, "chin-jaw-drop-decr": 0.10,
    # cheeks: high cheekbones, a little less fullness below them
    "lr-cheek-bones-incr": 0.35, "lr-cheek-trans-up": 0.15, "lr-cheek-inner-decr": 0.15,
    "lr-cheek-volume-decr": 0.15,
    # eyes: larger, almond, outer corners lifted, no bags
    "lr-eye-scale-incr": 0.18, "lr-eye-corner2-up": 0.18, "lr-eye-height2-incr": 0.08,
    "lr-eye-bag-decr": 0.50,
    # brows: soft lifted arch
    "eyebrows-trans-up": 0.08, "eyebrows-angle-up": 0.35,
    # nose: slim, straight, small refined tip
    "nose-scale-horiz-decr": 0.30, "nose-point-width-decr": 0.35, "nose-flaring-decr": 0.30,
    "nose-nostrils-width-decr": 0.25, "nose-width1-decr": 0.20, "nose-width2-decr": 0.20,
    "nose-width3-decr": 0.20, "nose-point-up": 0.15, "nose-hump-decr": 0.30,
    "nose-volume-decr": 0.20, "nose-scale-vert-decr": 0.10,
    # mouth: full lips, defined cupid's bow, corners resting slightly up
    "mouth-upperlip-volume-incr": 0.35, "mouth-lowerlip-volume-incr": 0.32,
    "mouth-cupidsbow-incr": 0.35, "mouth-scale-horiz-decr": 0.05, "mouth-angles-up": 0.22,
    # forehead, neck, ears
    "forehead-scale-vert-decr": 0.30, "neck-scale-horiz-decr": 0.25, "neck-scale-vert-incr": 0.10,
    "lr-ear-scale-decr": 0.20,
}

PRESETS["maya_pro"] = dict(PRESETS["maya"])
PRESETS["maya_pro"].update({
    # Late 20s, slim, refined: soft V-shaped jaw, high cheekbones, slim straight nose,
    # larger almond eyes with lifted outer corners, full lips with a defined cupid's bow,
    # long neck. A generic face, no likeness of a real person.
    "macro": dict(PRESETS["maya"]["macro"], **{
        "age": 0.50, "muscle": 0.34, "weight": 0.42, "proportions": 0.78, "firmness": 0.62,
        "race": {"asian": 0.25, "caucasian": 0.55, "african": 0.20},
    }),
    "targets": FACE_REFINED,
    "hair": "culturalibre_hair_01/culturalibre_hair_01.mhclo",                       # hair01
    # the pack's hair is jet black: lift it to a very dark warm brown
    "hair_grade": {"gain": (1.0, 0.86, 0.76), "saturation": 0.7, "gamma": 0.72},
    "hair_follows": "body",
    "rig": "mixamo",
    "rest_pose": {"shoulder_drop": 3.0, "upper_arm": (10.0, 1.0), "forearm": (4.0, 10.0), "hand": (1.0, 14.0),
                  "finger_curl": 9.0, "leg_in": 2.5},
    "idle": {"seconds": 8.0, "fps": 30},
    "clothes": [
        {"asset": "female_elegantsuit01/female_elegantsuit01.mhclo",                 # system
         # plain navy blouse, charcoal skirt: the stripes are flattened out
         "grade": {"solid": {"color": (0.085, 0.115, 0.23), "threshold": 0.20, "shading": 0.55},
                   "hue_to": 0.615, "saturation": 0.55, "gain": (0.50, 0.52, 0.60), "gamma": 1.0}},
        {"asset": "shoes04/shoes04.mhclo",                                           # system
         # the sock part of the texture is painted black: reads as ankle boots
         "paint": [{"uv_box": (0.80, 0.70, 1.0, 1.0), "color": (0.035, 0.035, 0.04)}],
         "grade": {"saturation": 0.3, "gain": (0.75, 0.75, 0.78), "gamma": 1.0},
         "surface": {"roughness": 0.42, "specular": 0.5, "double_sided": True}},
    ],
    # Smoother, even-toned skin (callharvey3d's face texture is the cleanest in the packs),
    # graded from mid-tone to warm brown, then retouched and lightly made up.
    "skin": "callharvey3d_midtoned_female/callharvey3d_midtoned_female.mhmat",       # skins01
    "skin_grade": {"gain": (1.04, 1.02, 1.00), "saturation": 0.92, "gamma": 0.98},
    "skin_retouch": {"smooth": 0.5, "detail_clip": 0.03, "even_tone": 0.4, "tone_clip": 0.04},
    "skin_normal": 0.12,
    "skin_roughness": {"base": 0.52, "lips": 0.30},
    "makeup": {
        "concealer": {"strength": 0.35},
        "blush": {"color": (1.0, 0.80, 0.78), "strength": 0.55},
        "eyeshadow": {"color": (0.80, 0.74, 0.74), "strength": 0.38},
        "eyeliner": {"color": (0.06, 0.04, 0.04), "strength": 0.75},
    },
    "lips": {"color": (0.92, 0.62, 0.62), "strength": 0.65},
    "eyebrows": "eyebrow001/eyebrow001.mhclo",                                       # system
    "brow_grade": {"gain": (0.36, 0.29, 0.25), "saturation": 0.6, "gamma": 1.0},
    "eyelashes": "eyelashes02/eyelashes02.mhclo",                                    # system
    "eye_grade": {"gain": (0.84, 0.76, 0.68), "saturation": 0.72, "gamma": 1.12},
    "morph_normals": True,
})

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
    for entry in preset.get("clothes", []):
        used.append(("clothes", _clothes_entry(entry)["asset"]))
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


def _clothes_entry(entry):
    """Clothes may be listed as a plain asset fragment or as a dict with per-item overrides."""
    return entry if isinstance(entry, dict) else {"asset": entry}


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
        if not weight:
            continue
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


def add_assets(mpfb, basemesh, preset, rig=None):
    """Skin, eyes, brows, lashes, teeth, tongue, hair and clothing through MPFB."""
    objects = {"skin": [basemesh]}

    skin_mhmat = mpfb.asset(preset["skin"], "skins")
    mpfb.HumanService.set_character_skin(skin_mhmat, basemesh, skin_type="GAMEENGINE", material_instances=False)
    basemesh["mc_mhmat"] = skin_mhmat

    plan = [
        ("eyes", "eyes", "Eyes"), ("eyebrows", "eyebrows", "Eyebrows"), ("eyelashes", "eyelashes", "Eyelashes"),
        ("teeth", "teeth", "Teeth"), ("tongue", "tongue", "Tongue"), ("hair", "hair", "Hair"),
    ]
    jobs = [(role, subdir, atype, {"asset": preset[role]}) for role, subdir, atype in plan if preset.get(role)]
    jobs += [("clothes", "clothes", "Clothes", _clothes_entry(entry)) for entry in preset.get("clothes", [])]

    for role, subdir, atype, entry in jobs:
        fragment = entry["asset"]
        mhclo = mpfb.asset(fragment, subdir)
        obj = mpfb.HumanService.add_mhclo_asset(
            mhclo, basemesh, asset_type=atype, subdiv_levels=0, material_type="GAMEENGINE",
            set_up_rigging=bool(rig), interpolate_weights=bool(rig), import_subrig=False, import_weights=bool(rig))
        obj["mc_role"] = role
        obj["mc_overrides"] = json.dumps({k: v for k, v in entry.items() if k != "asset"})
        rigid = role in HEAD_RIGID_ROLES and not (role == "hair" and preset.get("hair_follows") == "body")
        if rig and rigid:
            _bind_to_bone(obj, rig, _bone(rig, "Head").name)
        mhmat = mhclo_material(mhclo)
        if role == "eyes" and preset.get("eye_material"):
            mhmat = mpfb.asset(preset["eye_material"], "eyes")
        obj["mc_mhmat"] = mhmat or ""
        objects.setdefault(role, []).append(obj)
        log("added", role, fragment, "tris=%d" % triangle_count(obj))
    return objects


# --------------------------------------------------------------------------------------
# Skeleton, rest pose, idle animation
# --------------------------------------------------------------------------------------

# These follow the head as one rigid piece (no smeared weights on hair tips or eyeballs).
HEAD_RIGID_ROLES = ("eyes", "eyebrows", "eyelashes", "teeth", "tongue", "hair")


def add_rig(mpfb, basemesh, preset):
    """Add an MPFB built-in rig with weights. Returns the armature, or None if the preset has no rig."""
    if not preset.get("rig"):
        return None
    rig = mpfb.HumanService.add_builtin_rig(basemesh, preset["rig"], import_weights=True)
    if rig is None:
        raise SystemExit("MPFB could not create the rig %r" % preset["rig"])
    rig.name = rig.data.name = "Armature"
    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"
    log("rig", preset["rig"], "bones=%d" % len(rig.data.bones))
    return rig


def _bone(rig, key):
    """Pose bone by its short name ("LeftArm"), whatever prefix the rig uses ("mixamorig:")."""
    for bone in rig.pose.bones:
        if bone.name.split(":")[-1].lower() == key.lower():
            return bone
    raise SystemExit("The rig has no %r bone - presets with a rest pose or idle need a Mixamo-named rig" % key)


def _bind_to_bone(obj, rig, bone_name):
    for group in list(obj.vertex_groups):
        obj.vertex_groups.remove(group)
    group = obj.vertex_groups.new(name=bone_name)
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    if not any(m.type == "ARMATURE" for m in obj.modifiers):
        obj.modifiers.new("Armature", "ARMATURE").object = rig


def _rotate_pose_bone(pose_bone, axis, degrees):
    """Rotate a pose bone about an armature-space axis through its head (exact, parent first)."""
    head = pose_bone.matrix.to_translation()
    turn = Matrix.Translation(head) @ Matrix.Rotation(math.radians(degrees), 4, axis) @ Matrix.Translation(-head)
    pose_bone.matrix = turn @ pose_bone.matrix
    bpy.context.view_layer.update()


def _aim_pose_bone(pose_bone, towards, direction):
    """Swing a pose bone so that the line from its head to `towards` points along `direction`.

    A pure swing (shortest arc): the bone's twist is left alone, which is what keeps the palms
    facing the thighs when the arms come down from the A-pose.
    """
    head = pose_bone.matrix.to_translation()
    current = (towards.matrix.to_translation() - head).normalized()
    swing = current.rotation_difference(Vector(direction).normalized()).to_matrix().to_4x4()
    pose_bone.matrix = Matrix.Translation(head) @ swing @ Matrix.Translation(-head) @ pose_bone.matrix
    bpy.context.view_layer.update()


def apply_rest_pose(rig, objects, preset):
    """Bring the arms down from MakeHuman's A-pose, narrow the stance, and make that the bind pose.

    Runs before the face units are loaded, so no mesh has shape keys yet and the armature
    modifier can simply be applied. Armature axes: X = character's left, -Y = forward, Z = up.
    Angles are degrees away from straight down: `out` to the side, `forward` to the front.
    """
    pose = preset.get("rest_pose")
    if not rig or not pose:
        return

    def down(sign, angles):
        return (sign * math.tan(math.radians(angles[0])), -math.tan(math.radians(angles[1])), -1.0)

    for side in ("Left", "Right"):
        arm, forearm, hand = (_bone(rig, side + part) for part in ("Arm", "ForeArm", "Hand"))
        sign = 1.0 if (hand.head - arm.head).x > 0 else -1.0
        _rotate_pose_bone(_bone(rig, side + "Shoulder"), "Y", sign * float(pose.get("shoulder_drop", 0.0)))
        _aim_pose_bone(arm, forearm, down(sign, pose.get("upper_arm", (10.0, 2.0))))
        _aim_pose_bone(forearm, hand, down(sign, pose.get("forearm", (5.0, 10.0))))
        _aim_pose_bone(hand, _bone(rig, side + "HandMiddle1"), down(sign, pose.get("hand", (2.0, 14.0))))
        curl = float(pose.get("finger_curl", 0.0))
        if curl:
            for finger in ("Index", "Middle", "Ring", "Pinky"):
                for joint, share in ((1, 0.8), (2, 1.2), (3, 0.8)):
                    _rotate_pose_bone(_bone(rig, "%sHand%s%d" % (side, finger, joint)), "Y", sign * curl * share)
        # Narrower stance: swing the leg in at the hip and turn the foot back flat.
        leg_in = float(pose.get("leg_in", 0.0))
        if leg_in:
            _rotate_pose_bone(_bone(rig, side + "UpLeg"), "Y", sign * leg_in)
            _rotate_pose_bone(_bone(rig, side + "Foot"), "Y", -sign * leg_in)

    meshes = [obj for objs in objects.values() for obj in objs]
    for obj in meshes:
        activate(obj)
        for modifier in [m for m in obj.modifiers if m.type == "ARMATURE"]:
            bpy.ops.object.modifier_move_to_index(modifier=modifier.name, index=0)
            bpy.ops.object.modifier_apply(modifier=modifier.name)
    activate(rig)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for obj in meshes:
        obj.modifiers.new("Armature", "ARMATURE").object = rig


def build_idle_animation(rig, preset):
    """A looping "idle" action: breathing, a slow weight shift, a little arm and head drift.

    Small rotations about armature-space axes (X pitch, Y roll, Z yaw), written as bone-local
    quaternions. Every channel completes whole cycles, so the clip loops without a seam.
    Hip motion is cancelled in the thighs so the feet stay planted.
    """
    spec = preset.get("idle")
    if not rig or not spec:
        return None
    fps = int(spec.get("fps", 30))
    frames = int(round(float(spec.get("seconds", 8.0)) * fps))
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, frames + 1

    # bone, axis, amplitude in degrees, cycles per clip, phase (0..1)
    channels = [
        ("Hips", "Y", 0.55, 1, 0.00), ("LeftUpLeg", "Y", -0.55, 1, 0.00), ("RightUpLeg", "Y", -0.55, 1, 0.00),
        ("Hips", "Z", 0.70, 1, 0.20), ("LeftUpLeg", "Z", -0.70, 1, 0.20), ("RightUpLeg", "Z", -0.70, 1, 0.20),
        ("Spine", "Y", -0.35, 1, 0.00), ("Spine", "Z", -0.40, 1, 0.20),
        ("Spine1", "X", 0.45, 2, 0.00), ("Spine2", "X", 0.60, 2, 0.02),          # breathing, 4 s
        ("LeftShoulder", "Y", -0.70, 2, 0.04), ("RightShoulder", "Y", 0.70, 2, 0.04),
        ("Neck", "X", -0.55, 2, 0.03), ("Neck", "Y", 0.20, 1, 0.05),
        ("Head", "X", -0.35, 2, 0.10), ("Head", "Z", 0.60, 1, 0.35), ("Head", "Y", 0.25, 1, 0.60),
        ("LeftArm", "X", 0.70, 1, 0.15), ("RightArm", "X", 0.70, 1, 0.65),
        ("LeftArm", "Y", 0.35, 2, 0.05), ("RightArm", "Y", -0.35, 2, 0.05),
        ("LeftForeArm", "X", 0.90, 2, 0.20), ("RightForeArm", "X", 0.90, 2, 0.70),
        ("LeftHand", "X", 0.80, 1, 0.40), ("RightHand", "X", 0.80, 1, 0.90),
    ]
    axes = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}
    local = {}
    for key, axis, _amp, _cycles, _phase in channels:
        bone = _bone(rig, key)
        local[(key, axis)] = (bone, bone.bone.matrix_local.to_3x3().inverted() @ axes[axis])

    action = bpy.data.actions.new("idle")
    rig.animation_data_create()
    rig.animation_data.action = action
    animated = {bone for bone, _axis in local.values()}
    for frame in range(0, frames + 1, 3):
        phase = frame / frames
        for bone in animated:
            bone.rotation_quaternion = Quaternion()
        for key, axis, amplitude, cycles, offset in channels:
            bone, local_axis = local[(key, axis)]
            angle = math.radians(amplitude) * math.sin(2.0 * math.pi * (cycles * phase + offset))
            bone.rotation_quaternion = bone.rotation_quaternion @ Quaternion(local_axis, angle)
        for bone in animated:
            bone.keyframe_insert("rotation_quaternion", frame=frame + 1)
    for bone in animated:
        bone.rotation_quaternion = Quaternion()
    log("idle animation: %d frames at %d fps, %d bones" % (frames, fps, len(animated)))
    return action


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


def stage_for_export(mpfb, basemesh, objects, rig=None):
    """Bake delete-group masks, drop helper geometry, give objects clean names."""
    mpfb.ExportService.bake_modifiers_remove_helpers(
        basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=False)

    bones = {bone.name for bone in rig.data.bones} if rig else set()
    counters = {}
    for role, objs in objects.items():
        for obj in objs:
            for modifier in list(obj.modifiers):
                obj.modifiers.remove(modifier)
            world = obj.matrix_world.copy()
            obj.parent = rig
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_world = world
            activate(obj)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            bpy.ops.object.shade_smooth()
            for group in list(obj.vertex_groups):
                if group.name not in bones:
                    obj.vertex_groups.remove(group)
            if rig:
                # Every vertex needs a bone; strays (if any) follow the hips.
                if not obj.vertex_groups:
                    _bind_to_bone(obj, rig, _bone(rig, "Hips").name)
                else:
                    loose = [v.index for v in obj.data.vertices if not v.groups]
                    if loose:
                        hips = _bone(rig, "Hips").name
                        group = obj.vertex_groups.get(hips) or obj.vertex_groups.new(name=hips)
                        group.add(loose, 1.0, "REPLACE")
                obj.modifiers.new("Armature", "ARMATURE").object = rig
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
    solid_mask = None
    if grade.get("solid"):
        # Printed (saturated) cloth becomes one flat colour; only its broad shading is kept.
        # The mask is taken from the blurred saturation, so thin pale stripes are covered too.
        spec = grade["solid"]
        maxc, minc = rgb.max(axis=-1), rgb.min(axis=-1)
        radius = max(2, rgb.shape[0] // 96)
        saturation = _box_blur(_box_blur(np.where(maxc > 1e-5, (maxc - minc) / np.maximum(maxc, 1e-5), 0.0).astype(np.float32), radius), radius)
        threshold = float(spec.get("threshold", 0.2))
        solid_mask = np.clip((saturation - threshold * 0.6) / (threshold * 0.4), 0.0, 1.0)[..., None]
        broad = _box_blur(_box_blur(luma[..., 0], radius), radius)[..., None]
        mean = float((broad * solid_mask).sum() / max(float(solid_mask.sum()), 1.0))
        shade = 1.0 + (broad / max(mean, 1e-4) - 1.0) * float(spec.get("shading", 0.5))
        solid_rgb = np.array(spec["color"], dtype=np.float32) * np.clip(shade, 0.4, 1.8)
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
    if solid_mask is not None:
        rgb = rgb * (1.0 - solid_mask) + solid_rgb * solid_mask
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
    """UV masks for the make-up pass. Must run while the MPFB vertex groups and helpers exist."""
    masks = {}
    if preset.get("lips"):
        masks["lips"] = _uv_group_mask(basemesh, "lips", size, size, feather=max(1, size // 512))
    if preset.get("makeup"):
        masks.update(_makeup_masks(basemesh, size))
    return masks


def _vertex_coords(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _group_members(obj, name):
    members = np.zeros(len(obj.data.vertices), dtype=bool)
    if name not in obj.vertex_groups:
        return members
    index = obj.vertex_groups[name].index
    for vert in obj.data.vertices:
        for g in vert.groups:
            if g.group == index and g.weight > 0.5:
                members[vert.index] = True
                break
    return members


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _makeup_masks(basemesh, size):
    """Soft per-vertex weights for blush, eyeshadow, eyeliner and concealer, rasterised to UV.

    Placed from landmarks on the MakeHuman base mesh: the eyeball helpers (eye centre and
    radius), the open rim of the eye sockets (the lash line), and the lips group.
    Blender axes: X = character's left, -Y = forward, Z = up.
    """
    co = _vertex_coords(basemesh)
    body = _group_members(basemesh, "body")
    eyes = []
    for helper in ("helper-l-eye", "helper-r-eye"):
        pts = co[_group_members(basemesh, helper)]
        centre = pts.mean(axis=0)
        eyes.append((centre, float(np.linalg.norm(pts - centre, axis=1).mean())))
    mouth = co[_group_members(basemesh, "lips")].mean(axis=0)
    ipd = float(np.linalg.norm(eyes[0][0] - eyes[1][0]))
    mid_x = (eyes[0][0][0] + eyes[1][0][0]) / 2.0

    # Open boundary of the body surface around each eye = the lid rim / lash line.
    edge_faces = {}
    for poly in basemesh.data.polygons:
        if not all(body[v] for v in poly.vertices):
            continue
        for key in poly.edge_keys:
            edge_faces[key] = edge_faces.get(key, 0) + 1
    rim = np.array(sorted({v for key, n in edge_faces.items() if n == 1 for v in key}), dtype=np.int64)

    weights = {name: np.zeros(len(co)) for name in ("blush", "eyeshadow", "eyeliner", "concealer")}
    for centre, radius in eyes:
        d = np.linalg.norm(co - centre, axis=1)
        near = body & (d < radius * 3.2) & (co[:, 1] < centre[1] + radius * 0.6)
        above = _smoothstep(centre[2] - radius * 0.25, centre[2] + radius * 0.15, co[:, 2])
        below = _smoothstep(centre[2] + radius * 0.1, centre[2] - radius * 0.5, co[:, 2])
        outward = np.sign(centre[0] - mid_x)
        side = _smoothstep(-radius * 0.6, radius * 0.9, (co[:, 0] - centre[0]) * outward)    # stronger at the outer corner

        lid_rim = rim[(np.linalg.norm(co[rim] - centre, axis=1) < radius * 2.2)]
        if len(lid_rim):
            rim_dist = np.full(len(co), 1e9)
            for v in lid_rim:
                rim_dist = np.minimum(rim_dist, np.linalg.norm(co - co[v], axis=1))
            liner = np.exp(-(rim_dist / (radius * 0.16)) ** 2) * above * (0.55 + 0.45 * side)
            weights["eyeliner"] = np.maximum(weights["eyeliner"], np.where(near, liner, 0.0))
        shadow = (1.0 - _smoothstep(radius * 1.15, radius * 2.0, d)) * above * (0.6 + 0.4 * side)
        weights["eyeshadow"] = np.maximum(weights["eyeshadow"], np.where(near, shadow, 0.0))
        conceal = (1.0 - _smoothstep(radius * 1.2, radius * 2.7, d)) * below
        weights["concealer"] = np.maximum(weights["concealer"], np.where(near, conceal, 0.0))

        # Blush on the apple of the cheek: below and slightly outside the eye, on the surface.
        target = centre + np.array([outward * ipd * 0.16, 0.0, -(centre[2] - mouth[2]) * 0.55])
        lane = body & (np.abs(co[:, 0] - target[0]) < ipd * 0.12) & (np.abs(co[:, 2] - target[2]) < ipd * 0.12)
        if lane.any():
            surface = co[np.where(lane)[0][np.argmin(co[lane, 1])]]
            blush = np.exp(-np.sum((co - surface) ** 2, axis=1) / (2.0 * (ipd * 0.26) ** 2))
            blush *= co[:, 1] < surface[1] + ipd * 0.5
            weights["blush"] = np.maximum(weights["blush"], np.where(body, blush, 0.0))

    feather = max(1, size // 1024)
    return {"makeup_" + name: _uv_vertex_mask(basemesh, w, size, size, body, feather) for name, w in weights.items()}


def _uv_vertex_mask(obj, weights, width, height, allowed, feather=1):
    """Rasterise per-vertex weights into UV space with barycentric interpolation."""
    mask = np.zeros((height, width), dtype=np.float32)
    uvs = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        verts = list(poly.vertices)
        if not all(allowed[v] for v in verts):
            continue
        w = weights[verts]
        if w.max() < 1e-3:
            continue
        pts = np.array([(uvs[l].uv[0] * width, uvs[l].uv[1] * height) for l in poly.loop_indices])
        for a in range(1, len(pts) - 1):
            tri, tw = pts[[0, a, a + 1]], w[[0, a, a + 1]]
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
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
            value = (w0 * tw[0] + w1 * tw[1] + w2 * tw[2]).astype(np.float32)
            region = mask[y0:y1 + 1, x0:x1 + 1]
            region[inside] = np.maximum(region[inside], value[inside])
    if feather > 0:
        mask = _box_blur(mask, feather)
    return np.clip(mask, 0.0, 1.0)


def _retouch_skin(pixels, spec):
    """Edge-preserving skin smoothing (a cheap frequency-separation retouch).

    Fine detail (pores, speckle) and mid-size blotches are reduced, but only where the
    difference is small: strong features such as lids, nostrils, lip edges and brows exceed
    the clip and keep their contrast.
    """
    size = pixels.shape[0]
    rgb = pixels[..., :3]

    def blur(img, radius):
        return np.stack([_box_blur(_box_blur(img[..., c], radius), radius) for c in range(3)], axis=-1)

    fine = blur(rgb, max(1, size // 512))
    detail = rgb - fine
    limit = float(spec.get("detail_clip", 0.035))
    rgb = rgb - np.clip(detail, -limit, limit) * float(spec.get("smooth", 0.7))
    broad = blur(rgb, max(2, size // 40))
    middle = blur(rgb, max(1, size // 160)) - broad
    limit = float(spec.get("tone_clip", 0.04))
    rgb = rgb - np.clip(middle, -limit, limit) * float(spec.get("even_tone", 0.5))
    pixels[..., :3] = np.clip(rgb, 0.0, 1.0)
    return pixels


def _apply_makeup(pixels, masks, makeup):
    """Blush, eyeshadow, concealer (tints) and eyeliner (paint) from the UV masks."""
    def mask(name):
        m = masks.get("makeup_" + name)
        if m is None:
            return None
        if m.shape != pixels.shape[:2]:
            ys = (np.arange(pixels.shape[0]) * m.shape[0] / pixels.shape[0]).astype(int)
            xs = (np.arange(pixels.shape[1]) * m.shape[1] / pixels.shape[1]).astype(int)
            m = m[ys][:, xs]
        return m[..., None]

    rgb = pixels[..., :3]
    for name in ("concealer", "blush", "eyeshadow"):
        spec, m = makeup.get(name), mask(name)
        if not spec or m is None:
            continue
        amount = m * float(spec.get("strength", 0.3))
        if name == "concealer":
            luma = (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1, keepdims=True)
            lifted = luma + (rgb - luma) * 0.85
            rgb = rgb * (1.0 - amount) + np.clip(lifted * 1.18, 0, 1) * amount
        else:
            rgb = rgb * (1.0 - amount) + rgb * np.array(spec["color"], dtype=np.float32) * amount
    spec, m = makeup.get("eyeliner"), mask("eyeliner")
    if spec and m is not None:
        amount = m * float(spec.get("strength", 0.6))
        rgb = rgb * (1.0 - amount) + np.array(spec["color"], dtype=np.float32) * amount
    pixels[..., :3] = np.clip(rgb, 0.0, 1.0)
    return pixels


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
            overrides = json.loads(obj.get("mc_overrides") or "{}")
            surface = dict(SURFACE[role], **overrides.get("surface", {}))
            grade = overrides.get("grade") or preset.get(grades.get(role, ""))
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
                for patch in overrides.get("paint", []):
                    _paint_uv_box(pixels, patch["uv_box"], patch["color"])
                pixels = _grade(pixels, grade)
                if role == "skin" and preset.get("skin_retouch"):
                    pixels = _retouch_skin(pixels, preset["skin_retouch"])
                if role == "skin":
                    detail_source = pixels.copy()        # pore normals come from skin, not make-up
                if role == "skin" and preset.get("makeup"):
                    pixels = _apply_makeup(pixels, uv_masks, preset["makeup"])
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
                    # Transparent texels keep whatever colour the artist left (often white). Texture
                    # filtering mixes it into the strand edges, so fill them with nearby colour.
                    pixels = _bleed_alpha(pixels)
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
            normal_pixels = None
            if normal and os.path.exists(normal) and role in ("clothes", "hair"):
                normal_pixels = _read_pixels(normal, texture_size)
            elif role == "skin" and preset.get("skin_normal") and diffuse and os.path.exists(diffuse):
                normal_pixels = _micro_normal(detail_source, float(preset["skin_normal"]))
            if normal_pixels is not None:
                pixels = normal_pixels
                pixels[..., 3] = 1.0
                image = _write_pixels(pixels, os.path.join(work_dir, name + "_normal.jpg"), name + "_normal",
                                      non_color=True, quality=92)
                tex = tree.nodes.new("ShaderNodeTexImage")
                tex.image = image
                node = tree.nodes.new("ShaderNodeNormalMap")
                node.inputs["Strength"].default_value = 1.0
                tree.links.new(tex.outputs["Color"], node.inputs["Color"])
                tree.links.new(node.outputs["Normal"], bsdf.inputs["Normal"])

            rough = preset.get("skin_roughness") if role == "skin" else None
            if rough and "lips" in uv_masks:
                # glTF metallicRoughness texture: G = roughness. Lips a little glossier than skin.
                lips = uv_masks["lips"]
                lips = np.clip(_box_blur(lips, max(1, lips.shape[0] // 512)) * 1.3, 0.0, 1.0)
                if lips.shape[0] != texture_size:
                    step = lips.shape[0] / float(texture_size)
                    idx = (np.arange(texture_size) * step).astype(int)
                    lips = lips[idx][:, idx]
                value = float(rough["base"]) * (1.0 - lips) + float(rough["lips"]) * lips
                mr = np.zeros((texture_size, texture_size, 4), dtype=np.float32)
                mr[..., 1], mr[..., 3] = value, 1.0
                image = _write_pixels(mr, os.path.join(work_dir, name + "_roughness.jpg"), name + "_roughness",
                                      non_color=True, quality=90)
                tex = tree.nodes.new("ShaderNodeTexImage")
                tex.image = image
                split = tree.nodes.new("ShaderNodeSeparateColor")
                tree.links.new(tex.outputs["Color"], split.inputs["Color"])
                tree.links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
                tree.links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])

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


def _bleed_alpha(pixels, cutoff=0.5):
    """Replace the RGB of transparent texels with the average colour of nearby opaque ones."""
    alpha = pixels[..., 3]
    solid = (alpha >= cutoff).astype(np.float32)
    if solid.all() or not solid.any():
        return pixels
    rgb = pixels[..., :3].copy()
    filled = solid.copy()
    for radius in (2, 6, 18, 54, max(pixels.shape[:2]) // 4):
        weight = _box_blur(_box_blur(solid, radius), radius)
        colour = np.stack([_box_blur(_box_blur(pixels[..., c] * solid, radius), radius) for c in range(3)], axis=-1)
        take = (filled < 0.5) & (weight > 1e-4)
        rgb[take] = colour[take] / weight[take][:, None]
        filled[take] = 1.0
        if filled.all():
            break
    pixels[..., :3] = np.where(solid[..., None] > 0, pixels[..., :3], rgb)
    return pixels


def _paint_uv_box(pixels, uv_box, colour):
    """Fill a UV rectangle (u0, v0, u1, v1) with a flat colour."""
    height, width = pixels.shape[:2]
    u0, v0, u1, v1 = uv_box
    pixels[int(v0 * height):int(v1 * height), int(u0 * width):int(u1 * width), :3] = np.array(colour, dtype=np.float32)


def _micro_normal(pixels, strength):
    """Tangent-space normal map from the fine detail of a colour texture (pores, creases).

    High-pass the luminance (so shading painted into the texture does not become relief),
    treat the result as height and take its gradient.
    """
    luma = (pixels[..., :3] * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1)
    radius = max(2, pixels.shape[0] // 256)
    detail = luma - _box_blur(_box_blur(luma, radius), radius)
    detail = _box_blur(detail, 1) if pixels.shape[0] >= 2048 else detail
    scale = strength * pixels.shape[0] / 64.0
    dx = (np.roll(detail, -1, axis=1) - np.roll(detail, 1, axis=1)) * scale
    dy = (np.roll(detail, -1, axis=0) - np.roll(detail, 1, axis=0)) * scale
    length = np.sqrt(dx * dx + dy * dy + 1.0)
    out = np.empty(pixels.shape, dtype=np.float32)
    out[..., 0] = (-dx / length) * 0.5 + 0.5
    out[..., 1] = (-dy / length) * 0.5 + 0.5
    out[..., 2] = (1.0 / length) * 0.5 + 0.5
    out[..., 3] = 1.0
    return out


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

def export_glb(objects, out_path, rig=None, preset=None):
    preset = preset or {}
    bpy.ops.object.select_all(action="DESELECT")
    for objs in objects.values():
        for obj in objs:
            obj.select_set(True)
    if rig:
        rig.select_set(True)
    bpy.context.view_layer.objects.active = rig or objects["skin"][0]
    options = dict(
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
        export_morph_normal=bool(preset.get("morph_normals")),
        export_morph_tangent=False,
        export_try_sparse_sk=True,            # store only the vertices a face unit moves
        export_try_omit_sparse_sk=True,
        export_animations=False,
        export_skins=False,
        export_cameras=False,
        export_lights=False,
        export_extras=False,
    )
    if rig:
        options.update(
            export_skins=True,
            export_all_influences=False,      # 4 bone influences per vertex
            export_def_bones=False,
            export_rest_position_armature=True,
            export_animations=bool(rig.animation_data and rig.animation_data.action),
            export_animation_mode="ACTIVE_ACTIONS",
            export_force_sampling=True,
            export_frame_step=1,
            export_optimize_animation_size=True,
            export_optimize_animation_keep_anim_armature=False,
            export_morph_animation=False,
            export_anim_slide_to_zero=True,
        )
    bpy.ops.export_scene.gltf(**options)


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


def name_single_animation(glb_path, name):
    """The exporter names a lone armature action "Animation"; give it the clip name apps look for."""
    doc, rest = _read_glb(glb_path)
    animations = doc.get("animations", [])
    if len(animations) == 1 and animations[0].get("name") != name:
        animations[0]["name"] = name
        _write_glb(glb_path, doc, rest)


_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def slim_morph_normals(glb_path, keep_meshes=("body",), threshold=0.02):
    """Shrink the shape-key normals the exporter writes, then repack the binary chunk.

    Morph normals fix the shading of lips and lids while they move, but the exporter stores
    a full normal delta per vertex and key (most of them ~0). Here they are kept only on
    `keep_meshes`, and only where the normal really changes, as sparse accessors.
    """
    doc, rest = _read_glb(glb_path)
    if not rest:
        return
    binary = rest[8:8 + struct.unpack_from("<I", rest, 0)[0]]
    views, accessors = doc["bufferViews"], doc["accessors"]

    def read_vec3(index):
        accessor = accessors[index]
        count = accessor["count"]
        data = np.zeros((count, 3), dtype=np.float32)
        if "bufferView" in accessor:
            view = views[accessor["bufferView"]]
            start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            stride = view.get("byteStride", 12)
            raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + 12, offset=start)
            data = np.lib.stride_tricks.as_strided(raw, shape=(count, 12), strides=(stride, 1)).copy().view(np.float32).reshape(count, 3)
        if "sparse" in accessor:
            sparse = accessor["sparse"]
            fmt = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[sparse["indices"]["componentType"]]
            iview, vview = views[sparse["indices"]["bufferView"]], views[sparse["values"]["bufferView"]]
            rows = np.frombuffer(binary, dtype=fmt, count=sparse["count"],
                                 offset=iview.get("byteOffset", 0) + sparse["indices"].get("byteOffset", 0))
            values = np.frombuffer(binary, dtype=np.float32, count=sparse["count"] * 3,
                                   offset=vview.get("byteOffset", 0) + sparse["values"].get("byteOffset", 0)).reshape(-1, 3)
            data = data.copy()
            data[rows] = values
        return data

    extra = []          # new bufferView payloads, appended after the existing ones

    def add_view(payload):
        views.append({"buffer": 0, "byteLength": len(payload), "_payload": payload})
        return len(views) - 1

    dropped = kept = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            for target in prim.get("targets", []):
                if "NORMAL" not in target:
                    continue
                if mesh.get("name") not in keep_meshes:
                    del target["NORMAL"]
                    dropped += 1
                    continue
                data = read_vec3(target["NORMAL"])
                rows = np.nonzero(np.linalg.norm(data, axis=1) > threshold)[0]
                accessor = {"componentType": 5126, "type": "VEC3", "count": int(data.shape[0])}
                if len(rows):
                    wide = data.shape[0] > 65535
                    accessor["sparse"] = {
                        "count": int(len(rows)),
                        "indices": {"bufferView": add_view(rows.astype(np.uint32 if wide else np.uint16).tobytes()),
                                    "componentType": 5125 if wide else 5123},
                        "values": {"bufferView": add_view(data[rows].astype(np.float32).tobytes())},
                    }
                accessors.append(accessor)
                target["NORMAL"] = len(accessors) - 1
                kept += 1

    # Repack: keep only the bufferViews something still points at.
    used_accessors = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            used_accessors.update(prim["attributes"].values())
            if "indices" in prim:
                used_accessors.add(prim["indices"])
            for target in prim.get("targets", []):
                used_accessors.update(target.values())
    for skin in doc.get("skins", []):
        if "inverseBindMatrices" in skin:
            used_accessors.add(skin["inverseBindMatrices"])
    for animation in doc.get("animations", []):
        for sampler in animation["samplers"]:
            used_accessors.update((sampler["input"], sampler["output"]))
    accessor_map = {old: new for new, old in enumerate(sorted(used_accessors))}

    def remap_accessors(value):
        return accessor_map[value]

    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            prim["attributes"] = {k: remap_accessors(v) for k, v in prim["attributes"].items()}
            if "indices" in prim:
                prim["indices"] = remap_accessors(prim["indices"])
            if "targets" in prim:
                prim["targets"] = [{k: remap_accessors(v) for k, v in t.items()} for t in prim["targets"]]
    for skin in doc.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = remap_accessors(skin["inverseBindMatrices"])
    for animation in doc.get("animations", []):
        for sampler in animation["samplers"]:
            sampler["input"], sampler["output"] = remap_accessors(sampler["input"]), remap_accessors(sampler["output"])
    doc["accessors"] = [accessors[old] for old in sorted(used_accessors)]

    used_views = set()
    for accessor in doc["accessors"]:
        if "bufferView" in accessor:
            used_views.add(accessor["bufferView"])
        if "sparse" in accessor:
            used_views.add(accessor["sparse"]["indices"]["bufferView"])
            used_views.add(accessor["sparse"]["values"]["bufferView"])
    for image in doc.get("images", []):
        if "bufferView" in image:
            used_views.add(image["bufferView"])
    view_map, new_views, chunks, offset = {}, [], [], 0
    for old in sorted(used_views):
        view = dict(views[old])
        payload = view.pop("_payload", None)
        if payload is None:
            start = view.get("byteOffset", 0)
            payload = binary[start:start + view["byteLength"]]
        padding = (4 - offset % 4) % 4
        chunks.append(b"\x00" * padding + payload)
        offset += padding
        view["byteOffset"] = offset
        offset += len(payload)
        view_map[old] = len(new_views)
        new_views.append(view)
    for accessor in doc["accessors"]:
        if "bufferView" in accessor:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
        if "sparse" in accessor:
            accessor["sparse"]["indices"]["bufferView"] = view_map[accessor["sparse"]["indices"]["bufferView"]]
            accessor["sparse"]["values"]["bufferView"] = view_map[accessor["sparse"]["values"]["bufferView"]]
    for image in doc.get("images", []):
        if "bufferView" in image:
            image["bufferView"] = view_map[image["bufferView"]]
    doc["bufferViews"] = new_views
    blob = b"".join(chunks)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)
    doc["buffers"] = [{"byteLength": len(blob)}]
    _write_glb(glb_path, doc, struct.pack("<I4s", len(blob), b"BIN\x00") + blob)
    log("morph normals: %d kept as sparse, %d dropped; binary %.2f -> %.2f MB" % (
        kept, dropped, len(binary) / 1048576.0, len(blob) / 1048576.0))


def verify_glb(glb_path, expect_all_keys, expect_rig=False):
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
    if expect_rig:
        if len(doc.get("skins", [])) != 1:
            problems.append("expected exactly one skin, found %d" % len(doc.get("skins", [])))
        else:
            joints = doc["skins"][0]["joints"]
            names = [doc["nodes"][j].get("name", "") for j in joints]
            log("skeleton %d joints: %s" % (len(joints), ", ".join(names)))
        for mesh in doc.get("meshes", []):
            for prim in mesh["primitives"]:
                if "JOINTS_0" not in prim["attributes"] or "WEIGHTS_0" not in prim["attributes"]:
                    problems.append("mesh %s is not skinned" % mesh.get("name"))
        for animation in doc.get("animations", []):
            length = max(accessors[sampler["input"]]["max"][0] for sampler in animation["samplers"])
            log("animation %-8s %.2f s, %d channels" % (animation.get("name"), length, len(animation["channels"])))
    elif doc.get("skins") or doc.get("animations"):
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
    rig = add_rig(mpfb, basemesh, preset)
    objects = add_assets(mpfb, basemesh, preset, rig)
    apply_rest_pose(rig, objects, preset)
    add_face_units(mpfb, basemesh, objects)
    mask_size = max(texture_size, int(preset.get("face_texture_size", texture_size)))
    uv_masks = capture_uv_masks(basemesh, preset, mask_size)
    stage_for_export(mpfb, basemesh, objects, rig)
    build_idle_animation(rig, preset)

    with tempfile.TemporaryDirectory(prefix="make_character_") as work_dir:
        materials = fix_materials(objects, preset, texture_size, work_dir, uv_masks)
        export_glb(objects, out_path, rig, preset)
    patch_alpha_modes(out_path, materials)
    if preset.get("morph_normals"):
        slim_morph_normals(out_path)
    if rig:
        name_single_animation(out_path, "idle")

    expect = {obj.name: mpfb.ARKIT_FACEUNITS for role in MORPH_ROLES for obj in objects.get(role, [])}
    for key in REQUIRED_KEYS:
        assert key in mpfb.ARKIT_FACEUNITS, key
    problems = verify_glb(out_path, expect, expect_rig=bool(rig))
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
