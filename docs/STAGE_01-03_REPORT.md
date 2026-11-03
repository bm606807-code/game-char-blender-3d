# Stages 1–3 report: environment, reference preparation, tool selection

Date: 2026-11-03 · Status: **Stages 1–3 complete.** Decision taken: procedural Blender model, and `bpy` 5.2.3 installed from PyPI (owner-approved). Stages 4–10 are documented in `README.md` and `docs/QUALITY_INSPECTION.md`.

## Stage 1: Environment inspection

| Item | Result |
|---|---|
| Machine | Cloud container, 4 vCPU, 15 GB RAM, **no GPU** (no `nvidia-smi`), ~30 GB free disk |
| Blender application | **Not installed.** No `blender` binary; no Blender MCP server connected to this session |
| download.blender.org / mirrors | **Blocked** by the environment network policy |
| PyPI | Reachable. `bpy 5.2.3` (Blender 5.2 as a Python module, includes Cycles CPU) has a `cp313 manylinux` wheel (384 MB) matching the local Python 3.13 |
| Python imaging | Pillow 12.3, NumPy 2.5; SciPy installed for Stage 2 |
| huggingface.co, cdn-lfs.huggingface.co, hf-mirror.com, modelscope.cn | **Blocked** (no open-source model weights can be downloaded) |
| download.pytorch.org | **Blocked** |
| Tripo (api.tripo3d.ai), Meshy (api.meshy.ai), Rodin (hyperhuman.deemos.com), Stability (api.stability.ai), Replicate | **Blocked** |
| GitHub | Reachable (source code only; no model weights for these tools are hosted there) |

## Stage 2: Reference preparation (done)

Script: `scripts/01_prepare_reference_views.py`

- Source sheet saved as `reference/character_4view_reference.png` (1536×1024).
- The checkerboard "fake transparency" background was removed with a border-connected flood fill into a real alpha channel.
- Each figure was cropped independently, including the full boots and pedestal:

| File | View | Crop |
|---|---|---|
| `view_01_front_3q_right.png` | Front / three-quarter | 360×996 |
| `view_02_right_profile.png` | Right-side profile | 316×1003 |
| `view_03_back.png` | Back | 396×1005 |
| `view_04_front_3q_left.png` | Opposite three-quarter | 339×999 |

- `*_sq1024.png`: each view centred on a 1024² transparent square (the standard input for image-to-3D tools).
- `contact_sheet.png`: all four cut-outs on a dark backdrop for checking the alpha. A faint light halo remains on a few loose hair strands.

### Cross-view analysis: inconsistencies in the reference sheet

The sheet appears AI-generated and is **not** fully consistent between views. These points must be resolved by choosing one canonical version before or during reconstruction:

1. **The gesturing arm changes sides.** In view 01 (front) the raised open hand is on the viewer's right, so it is the character's **left** arm. In view 03 (back) the raised hand is also on the viewer's right, which from behind is her **right** arm.
2. **The profile hides the gesture.** In view 02 (right profile) no forward-raised arm is visible, although one should show in front of the torso.
3. **The thigh pouch is consistent.** In views 01, 02 and 03 the pouch is on her **right** thigh.
4. **The backpack changes size.** It reads as a compact rear housing in view 03 but bulkier at the shoulder in views 02 and 04.
5. **Knee pads are front-only.** They are visible in views 01, 02 and 04 and correctly absent from the back.

**Proposed canonical version:** view 01 (front), which is the clearest image. It has the left arm extended forward and slightly raised with the palm up, the right arm hanging, the pouch on the right thigh, and the backpack design from view 03.

## Stage 3: Reconstruction tool evaluation

| Tool | Multi-view input | Runs here? | Notes |
|---|---|---|---|
| Hunyuan3D-2mv (open weights) | Yes (front/left/back/right) | **No.** Weights are on Hugging Face (blocked). Shape generation is feasible but slow on CPU; the texture stage needs CUDA | Best open multi-view option if HF access is allowed |
| Hunyuan3D-2 / 2mini | Single image | No, same reasons | |
| Stable Fast 3D | Single image | No. Gated HF weights need a token; HF is blocked | Low detail for a full character |
| TripoSR | Single image | No, HF blocked. Would run on CPU | Low resolution, vertex colours only: far below the target quality |
| Tripo API | Multi-view supported | No. Host blocked; needs an API key and **paid credits** | Strong quality for characters |
| Meshy API | Multi-image supported | No. Host blocked; needs an API key and **paid credits** | |
| Rodin (Hyper3D) | Multi-view supported | No. Host blocked; needs an API key and **paid credits** | Highest-fidelity option of the group |
| Procedural modelling in Blender | n/a | Yes, once `bpy` is installed | Explicitly rejected in the brief as too simplistic for the target quality |

## Blockers (need the owner's decision)

1. **No image-to-3D tool can run in this environment** under the current network policy and without a GPU.
2. **Blender isn't installed.** The only reachable route is `pip install bpy==5.2.3`: free, but it adds 384 MB of software, and the brief asks for approval first.

**Resolution:** the owner chose procedural modelling in Blender and approved installing `bpy`.
