"""STAGE 2 - Split the four-view reference sheet into individual, background-free views.

Usage: python3 scripts/01_prepare_reference_views.py [reference/character_4view_reference.png]

Outputs (reference/views/):
  view_XX_<name>.png          tight RGBA crop, background removed
  view_XX_<name>_sq1024.png   centred on a 1024x1024 transparent square (image-to-3D input)
  contact_sheet.png           all four RGBA crops on a dark backdrop for inspection
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "reference/character_4view_reference.png")
OUT = Path("reference/views")
OUT.mkdir(parents=True, exist_ok=True)

# Column ranges measured from the sheet (foreground column-occupancy runs), padded.
VIEWS = [
    ("01_front_3q_right", 56, 392),   # front / three-quarter, gesturing arm on viewer's right
    ("02_right_profile", 432, 724),   # side profile
    ("03_back", 784, 1156),           # rear: backpack, hood
    ("04_front_3q_left", 1177, 1492), # opposite three-quarter, gesturing arm toward viewer
]
PAD = 12

img = np.asarray(Image.open(SRC).convert("RGB")).astype(np.int16)
H, W, _ = img.shape
mx, mn = img.max(2), img.min(2)
# Fake-transparency checkerboard: very bright and unsaturated.
bg_candidate = (mn >= 232) & ((mx - mn) <= 14)
# Only background connected to the image border counts (keeps light highlights inside the figure).
lab, _ = ndimage.label(bg_candidate)
border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
bg = np.isin(lab, border[border > 0])
alpha = (~bg).astype(np.uint8) * 255
# Clean up: fill pinholes, soften edge by 1px.
alpha = ndimage.binary_fill_holes(alpha > 0).astype(np.uint8) * 255
alpha = ndimage.binary_opening(alpha > 0, iterations=1).astype(np.uint8) * 255
alpha = np.asarray(Image.fromarray(alpha).filter(__import__("PIL.ImageFilter").ImageFilter.GaussianBlur(0.6)))

rgba = np.dstack([img.astype(np.uint8), alpha])
crops = []
for name, x0, x1 in VIEWS:
    sub = rgba[:, max(0, x0 - PAD):min(W, x1 + PAD)]
    ys = np.where(sub[..., 3].max(1) > 0)[0]
    sub = sub[max(0, ys.min() - PAD):min(H, ys.max() + PAD)]
    im = Image.fromarray(sub, "RGBA")
    im.save(OUT / f"view_{name}.png")
    # square, character fills ~90% of the height
    s = 1024
    scale = 0.9 * s / im.height
    r = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sq.paste(r, ((s - r.width) // 2, (s - r.height) // 2), r)
    sq.save(OUT / f"view_{name}_sq1024.png")
    crops.append(im)
    print(f"{name}: {im.size}, opaque px {int((sub[...,3]>128).sum())}")

# contact sheet on mid-grey so halo / leftover checkerboard is visible
gap = 20
cw = sum(c.width for c in crops) + gap * (len(crops) + 1)
ch = max(c.height for c in crops) + 2 * gap
sheet = Image.new("RGBA", (cw, ch), (60, 62, 68, 255))
x = gap
for c in crops:
    sheet.alpha_composite(c, (x, gap)); x += c.width + gap
sheet.save(OUT / "contact_sheet.png")
