"""STAGES 8-9 - Studio lighting, four orthographic cameras, Cycles renders, turnaround sheet.

Cameras (rotation about the character's vertical axis; the character never moves):
  Camera_01_Front   0 deg   looks at her front (+Y direction)
  Camera_02_Right  90 deg   looks at her right side
  Camera_03_Back  180 deg   looks at her back
  Camera_04_Left  270 deg   looks at her left side
All four share one orthographic scale and the same vertical framing. The key/fill/rim rig is
parented to an empty that turns with the active camera, so every view gets identical lighting.

Usage:
  python3 scripts/04_render_views.py            # final: 1536x2048, 128 spp + OIDN denoise
  python3 scripts/04_render_views.py --preview  # quick check: 384x512, 16 spp
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = "--preview" in sys.argv
BLEND_IN = ROOT / "output" / "blend" / "character_model.blend"
OUT = ROOT / "output" / ("preview" if PREVIEW else "renders")
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = [("Camera_01_Front", 0), ("Camera_02_Right", 90), ("Camera_03_Back", 180), ("Camera_04_Left", 270)]
only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]

T0 = time.time()
bpy.ops.wm.open_mainfile(filepath=str(BLEND_IN))
sc = bpy.context.scene


def coll(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    sc.collection.children.link(c)
    return c


cam_coll, light_coll = coll("CAMERAS"), coll("LIGHTS")
for c in (cam_coll, light_coll):
    for o in list(c.objects):
        bpy.data.objects.remove(o, do_unlink=True)

# --------------------------------------------------------------------------- framing
Z_MIN, Z_MAX = -0.20, 1.85          # pedestal bottom .. top of the hair bun
Z_CENTER = (Z_MIN + Z_MAX) / 2
ORTHO = (Z_MAX - Z_MIN) * 1.08      # identical for all cameras
TARGET = Vector((0.0, 0.0, Z_CENTER))
DIST = 6.0

for name, ang in VIEWS:
    cd = bpy.data.cameras.new(name)
    cd.type = "ORTHO"
    cd.ortho_scale = ORTHO
    cd.clip_start, cd.clip_end = 0.1, 20
    cam = bpy.data.objects.new(name, cd)
    cam_coll.objects.link(cam)
    a = math.radians(ang)
    # 0 deg = in front of her (-Y); 90 deg = her right side (-X); 180 = behind; 270 = her left (+X)
    cam.location = (-DIST * math.sin(a), -DIST * math.cos(a), Z_CENTER)
    cam.rotation_euler = (math.pi / 2, 0.0, -a)

# --------------------------------------------------------------------------- lights
rig = bpy.data.objects.new("LightRig", None)
light_coll.objects.link(rig)


def area(name, loc, energy, size, color, size_y=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.color = color
    ld.shape = "RECTANGLE"
    ld.size = size
    ld.size_y = size_y or size
    ob = bpy.data.objects.new(name, ld)
    light_coll.objects.link(ob)
    ob.location = loc
    d = Vector((0, 0, 0.95)) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    ob.parent = rig
    return ob


# positions are relative to the front camera; the rig rotates with each camera
area("Key_Softbox", (-1.9, -2.3, 2.5), 480, 1.6, (1.0, 0.95, 0.90))
area("Fill_Softbox", (2.4, -1.9, 1.3), 120, 2.2, (0.86, 0.91, 1.0))
area("Rim_Left", (-2.0, 2.0, 2.1), 320, 0.6, (1.0, 0.92, 0.85), 2.2)
area("Rim_Right", (2.0, 2.1, 1.8), 320, 0.6, (0.85, 0.93, 1.0), 2.2)
area("Top_Light", (0.0, 0.3, 3.4), 110, 1.2, (1.0, 1.0, 1.0))
area("Kicker_Low", (0.0, -2.5, 0.05), 45, 1.5, (1.0, 0.95, 0.9), 0.4)

world = bpy.data.worlds.get("Studio") or bpy.data.worlds.new("Studio")
sc.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
tc = nt.nodes.new("ShaderNodeTexCoord")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.055, 1)
ramp.color_ramp.elements[1].color = (0.42, 0.43, 0.45, 1)
mr = nt.nodes.new("ShaderNodeMapRange")
mr.inputs["From Min"].default_value = -0.3
mr.inputs["From Max"].default_value = 1.0
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 0.2
wo = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(tc.outputs["Generated"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], wo.inputs["Surface"])

# --------------------------------------------------------------------------- render settings
sc.render.engine = "CYCLES"
cy = sc.cycles
cy.device = "CPU"
cy.samples = 16 if PREVIEW else 128
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.02
cy.use_denoising = True
cy.denoiser = "OPENIMAGEDENOISE"
cy.max_bounces = 8
cy.transparent_max_bounces = 8
cy.caustics_reflective = False
cy.caustics_refractive = False
sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = (384, 512) if PREVIEW else (1536, 2048)
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.image_settings.color_depth = "8"
try:
    sc.view_settings.view_transform = "Khronos PBR Neutral"  # keeps saturated orange/cyan
except TypeError:
    sc.view_settings.view_transform = "AgX"
sc.render.threads_mode = "AUTO"

files = []
for name, ang in VIEWS:
    if only and name not in only:
        continue
    sc.camera = bpy.data.objects[name]
    rig.rotation_euler = (0, 0, -math.radians(ang))
    path = OUT / f"{name}.png"
    sc.render.filepath = str(path)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"rendered {name} in {time.time() - t:.0f}s", flush=True)
    files.append((name, path))

if not PREVIEW:
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_IN), compress=True)

# --------------------------------------------------------------------------- turnaround sheet
if len(files) == 4:
    ims = [Image.open(p).convert("RGBA") for _, p in files]
    w, h = ims[0].size
    pad = w // 24
    sheet = Image.new("RGBA", (4 * w + 5 * pad, h + 2 * pad), (0, 0, 0, 0))
    for i, im in enumerate(ims):
        sheet.alpha_composite(im, (pad + i * (w + pad), pad))
    sheet.save(OUT / "turnaround_sheet_rgba.png")
    # labelled presentation version on a studio backdrop
    bgimg = Image.new("RGBA", sheet.size, (46, 48, 54, 255))
    grad = Image.linear_gradient("L").resize(sheet.size)
    bgimg = Image.composite(Image.new("RGBA", sheet.size, (82, 85, 92, 255)), bgimg, grad.point(lambda v: 255 - v))
    bgimg.alpha_composite(sheet)
    dr = ImageDraw.Draw(bgimg)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, h // 48))
    except OSError:
        font = ImageFont.load_default()
    for i, (name, _) in enumerate(files):
        label = name.replace("_", " ")
        dr.text((pad + i * (w + pad) + pad // 2, pad // 3), label, fill=(225, 228, 232, 255), font=font)
    bgimg.convert("RGB").save(OUT / "turnaround_sheet.jpg", quality=92)
print(f"done in {time.time() - T0:.0f}s")
