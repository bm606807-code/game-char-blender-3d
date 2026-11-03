# Quality inspection: final four-view renders

Inspected: `output/renders/Camera_01_Front.png`, `Camera_02_Right.png`, `Camera_03_Back.png`, `Camera_04_Left.png` (1536×2048 RGBA, Cycles, 128 spp with adaptive sampling, OpenImageDenoise) and `turnaround_sheet.jpg`. Measured figures are in `docs/model_stats.md`.

## Bottom line

The deliverable is a **real, editable, watertight 3D character** in Blender. It reproduces the reference's:

- costume layout and colour scheme;
- gesture;
- accessories (goggles, hood, cropped grey jacket with rolled sleeves, harness, belts, thigh pouches with cyan LEDs, knee pads, fingerless gloves, combat boots, backpack with cyan chevrons);
- rock pedestal.

It does **not** reach the brief's AAA, semi-photorealistic target. The reference was not reconstructed by an image-to-3D model: every such service was blocked in this environment, and the owner chose procedural modelling as the fallback. The anatomy and costume are sculpted from signed-distance fields, which is a big step up from primitive shapes, but:

- the **face is mannequin-like**, not the reference's realistic face;
- **cloth folds are mild**;
- **detailing is simpler** than the reference.

## Checklist

| Check | Result | Notes |
|---|---|---|
| Height and scale | **Pass** | Boot sole to top of skull is 1.745 m (target 1.75 m); the bun reaches 1.84 m |
| Same pose in all four views | **Pass** | The character never moves: four cameras orbit her at 0/90/180/270°, with identical ortho scale (2.21 m), framing and camera-locked lighting |
| Hand gesture | **Pass (with caveat)** | Left arm extended forward and slightly raised, palm up, five separate fingers with a slight progressive curl, thumb abducted. Right arm hangs relaxed with curled fingers. The pose follows view 01; the reference itself flips the gesturing arm in the back view (see `STAGE_01-03_REPORT.md`) |
| Front: facial accuracy | **Fail vs. reference** | Correct anatomy (eye sockets, lids, irises, brows, lashes, nose with alae/nostrils, lips, ears, jaw), but the proportions and shading are stylised and mannequin-like. It does not match the reference's face, makeup or expression |
| Side profile | **Partial** | Readable profile with bust, glutes, calves, backpack side wing and LED. The reference's curves (hips and glutes) are fuller |
| Rear backpack | **Partial** | Housing, yoke, side wings, orange V panel, cyan chevrons, vents and screws are all present. Shapes are simpler, boxier hard surfaces than the reference |
| Costume consistency | **Pass** | Same materials and placement in every view: pouch on the right thigh, holster straps, knee pads only at the front |
| Body proportions | **Partial** | Athletic adult female, 7.5 heads tall. The thighs and hips are slimmer than the reference's stylised figure |
| Hair | **Partial** | Volumetric cap and large messy bun plus about 2,400 strand meshes (scalp, bun wraps, loose bun loops, face-framing strands). Reads as dark brown hair, but it's tidier and more helmet-like at the hairline than the reference |
| Boots | **Pass (simplified)** | Separate upper, toe cap, orange midsole, black lugged outsole, straps, padded collar, ankle discs with orange rings, metal eyelets and orange laces. Less sculpted detail than the reference |
| Pedestal | **Pass** | Faceted basalt with cracks and rust-orange mineral texture. The boots make contact with slight sinking, so there is no floating |
| Mesh intersections | **Minor** | The clothing shells are offsets of the body field, so they don't pierce it. The hand meshes overlap the gloved forearm ends (hidden). Some hard-surface items (pouches, buckles) are placed by coordinates and slightly embed in the leather |
| Missing geometry | **Pass** | No holes in sculpted meshes: 0 open edges on body, head, clothing, hair cap, bun and pedestal. A few open edges are left over from decimation on the boot uppers (289 each) and hand skin (57 and 384), hidden inside the gloves and boots |
| Texture and material quality | **Partial** | 24 PBR materials using generated tileable albedo, normal and roughness maps (leather grain, fabric twill, skin pores, brushed metal, basalt, iris, hair). Subsurface-scattered skin, clear-coated eyes, emissive cyan LEDs. Not the hand-painted, wear-detailed texturing of the reference |

## Known visual flaws (visible in the renders)

1. The face is clearly CG and mannequin-like. This is the largest gap to the reference.
2. The harness straps end in rounded caps under the bust instead of terminating at buckles.
3. The hood's front lobes sit as two lumps over the collarbones; the reference hood drapes more naturally.
4. The jacket is a smooth offset shell with only fine displacement wrinkles, with no major folds at the elbows or armpits.
5. The goggle strap and the hairline edge form a straight band across the forehead.
6. The back-of-glove guard is a flat rounded plate rather than the reference's articulated mechanical parts.

## What would close the gap

The brief's quality target needs either:

- **An image-to-3D pass on the prepared views.** The cut-outs are in `reference/views/*_sq1024.png`. This needs Rodin, Tripo or Meshy (paid credits plus an API key, with their hosts allowed in the environment), or Hunyuan3D-2mv (open weights, which needs `huggingface.co` allowed, ideally on a GPU machine). The resulting GLB can be dropped into this Blender pipeline for scale, cameras, lighting, rendering and export, replacing `03_build_character.py`.
- **Or manual sculpting** in Blender/ZBrush on top of this procedural base, especially the face, hands and cloth folds.
