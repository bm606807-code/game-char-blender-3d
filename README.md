# game-char-blender-3d

Reconstruction of a cyberpunk female fighter from a four-view reference sheet into an editable 3D character built in Blender (5.2 LTS, `bpy`), with orthographic four-view Cycles renders and GLB/OBJ/FBX exports.

> **How the geometry was produced.** No image-to-3D service could run in the build environment: Hugging Face, Tripo, Meshy and Rodin were all blocked, and there was no GPU. With the owner's approval, the character was therefore **modelled procedurally**:
>
> - anatomy and costume were sculpted as signed-distance fields (SDFs) and polygonised with marching cubes;
> - Blender handled assembly, materials, strand geometry and hard-surface parts.
>
> It is genuine, editable 3D geometry, but it is **not** an AI reconstruction from the image. See `docs/QUALITY_INSPECTION.md` for an honest assessment against the reference.

## Layout

| Path | Contents |
|---|---|
| `reference/` | Original sheet and per-view cut-outs (`reference/views/`) |
| `scripts/01_prepare_reference_views.py` | Stage 2: split the sheet, remove the background, square-pad each view |
| `scripts/02_generate_textures.py` | Tileable PBR maps (albedo/normal/roughness) → `assets/textures/` |
| `scripts/03_build_character.py` | Stages 4–7: SDF sculpt → meshes → Blender assembly → `output/blend/character_model.blend` |
| `scripts/04_render_views.py` | Stages 8–9: studio lights, four ortho cameras, Cycles renders, turnaround sheet |
| `scripts/05_export.py` | Stage 10: backup `.blend`, then GLB, OBJ+MTL and FBX exports |
| `scripts/charlib/` | SDF toolkit (`sdf.py`), character definition (`anatomy.py`), Blender helpers (`bl.py`) |
| `output/` | `.blend`, renders, exports |
| `docs/` | Stage reports and the quality inspection |

## Reproduce

```bash
pip install bpy==5.2.3 scipy scikit-image pillow numpy
```

```bash
python3 scripts/01_prepare_reference_views.py
```

```bash
python3 scripts/02_generate_textures.py
```

```bash
python3 scripts/03_build_character.py
```

```bash
python3 scripts/04_render_views.py
```

```bash
python3 scripts/05_export.py
```

`03_build_character.py --reuse` skips the SDF phase and reuses the cached raw meshes in `build/cache/` (not committed).
`04_render_views.py --preview` renders quick low-sample checks.

## Coordinate conventions

- Blender: Z up, metres, character faces −Y, boot soles at z = 0, pedestal below. Height is 1.75 m to the top of the skull; the bun reaches about 1.81 m.
- Cameras rotate about the vertical axis: 0° front, 90° her right side, 180° back, 270° her left side. The character never moves.
- Exports use the Y-up convention (glTF/OBJ/FBX).

## Deliverables

| # | Deliverable | File |
|---|---|---|
| 1 | Editable Blender project | `output/blend/character_model.blend` (collections: CHAR_*, PEDESTAL, CAMERAS, LIGHTS) |
| 2 | 3D character mesh | 154 mesh objects, about 1.24 M faces, in the `.blend` and every export |
| 3 | GLB | `output/export/character.glb` (textures embedded) |
| 4 | OBJ + MTL | `output/export/character_obj.zip` (OBJ, MTL with PBR extensions, textures) |
| — | FBX | `output/export/character.fbx` (textures embedded) |
| 5–6 | Textures and PBR materials | `assets/textures/*.png`, 24 Principled BSDF materials |
| 7 | Automation scripts | `scripts/` |
| 8 | Four camera renders | `output/renders/Camera_0{1..4}_*.png` (1536×2048 RGBA) |
| 9 | Turnaround sheet | `output/renders/turnaround_sheet.jpg` (labelled), `turnaround_sheet_rgba.png` (transparent) |
| 10 | Quality inspection | `docs/QUALITY_INSPECTION.md`, `docs/model_stats.md` |

Backups: the raw marching-cubes meshes are cached untouched in `build/cache/raw_meshes.npz`. `05_export.py` copies the `.blend` to `output/blend/backup/` before exporting (local only); git history keeps every committed version of the `.blend`.
