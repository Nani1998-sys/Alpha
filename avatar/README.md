# Avatar generator

`make_character.py` builds a talking-avatar character with Blender 4.5 + MPFB and writes one `.glb`
for three.js, with all 52 ARKit blendshapes for lip sync.

    blender -b -P make_character.py -- --kind maya --out out/maya.glb --texture-size 1024
    # optional: --preview-dir previews   (renders the exported file)   --list

Needs the MPFB extension and these CC0 MakeHuman asset packs: `makehuman_system_assets`, `skins01`,
`hair01`, `faceunits01` (ARKit targets). `skins02`, `eyebrows01`, `eyelashes01` may be installed but
the `maya` preset does not use them.

- `out/maya.glb` – the built character (4.2 MB, 37,984 triangles)
- `out/maya_build_log.txt` – assets, licences, shape keys per mesh, materials, images, bounds
- `previews/` – Cycles renders of the exported file, plus a real three.js r170 render

The script checks the written file (size, triangles, shape keys, alpha modes, image formats, height,
feet at origin) and exits non-zero if any check fails. Add characters as new `PRESETS` entries.
