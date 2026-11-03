"""STAGE 10 - Export deliverables from the finished .blend.

  output/export/character.glb               glTF binary, textures embedded, Y-up (glTF convention)
  output/export/obj/character.obj + .mtl    OBJ/MTL with PBR extensions, textures copied alongside
  output/export/character.fbx               FBX (textures copied/embedded)
  output/blend/backup/character_model_<ts>.blend   backup before export-time changes

Usage: python3 scripts/05_export.py
"""
import sys
import shutil
import time
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
BLEND = ROOT / "output" / "blend" / "character_model.blend"
EXP = ROOT / "output" / "export"
(EXP / "obj").mkdir(parents=True, exist_ok=True)
(ROOT / "output" / "blend" / "backup").mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
stamp = time.strftime("%Y%m%d_%H%M%S")
backup = ROOT / "output" / "blend" / "backup" / f"character_model_{stamp}.blend"
shutil.copy2(BLEND, backup)
print("backup:", backup.relative_to(ROOT))

# export only the character + pedestal (no cameras / lights / rig)
for o in bpy.data.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = next(o for o in bpy.data.objects if o.type == "MESH")

bpy.ops.export_scene.gltf(filepath=str(EXP / "character.glb"), export_format="GLB", use_selection=True,
                          export_apply=True, export_yup=True, export_image_format="AUTO",
                          export_materials="EXPORT", export_cameras=False, export_lights=False)

bpy.ops.wm.obj_export(filepath=str(EXP / "obj" / "character.obj"), export_selected_objects=True,
                      export_uv=True, export_normals=True, export_materials=True, export_pbr_extensions=True,
                      path_mode="COPY", forward_axis="NEGATIVE_Z", up_axis="Y", export_object_groups=True)

bpy.ops.export_scene.fbx(filepath=str(EXP / "character.fbx"), use_selection=True, object_types={"MESH"},
                         apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS", path_mode="COPY",
                         embed_textures=True, axis_forward="-Z", axis_up="Y", mesh_smooth_type="FACE")

# OBJ text is large: ship it zipped together with its MTL and textures (GitHub 100 MB file limit)
import zipfile
with zipfile.ZipFile(EXP / "character_obj.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted((EXP / "obj").iterdir()):
        z.write(p, f"character_obj/{p.name}")
shutil.rmtree(EXP / "obj")

for p in sorted(EXP.rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 2**20:.1f} MB")
