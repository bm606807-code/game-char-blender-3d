"""Blender (bpy) helpers: fast mesh import, materials, modifiers, UVs, curves."""
from pathlib import Path
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets" / "textures"


def srgb(hexstr, a=1.0):
    h = hexstr.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return (*lin, a)


# --------------------------------------------------------------------------- scene
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0
    return sc


def collection(name, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def link(obj, coll):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


# --------------------------------------------------------------------------- meshes
def mesh_from_arrays(name, verts, faces, coll, mat=None):
    me = bpy.data.meshes.new(name)
    nv, nf = len(verts), len(faces)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    k = faces.shape[1]
    me.loops.add(nf * k)
    me.loops.foreach_set("vertex_index", np.asarray(faces, np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, nf * k, k, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    for p in me.polygons:
        pass
    me.shade_smooth()
    return ob


def activate(ob):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)


def add_mod(ob, kind, name=None, **props):
    m = ob.modifiers.new(name or kind.title(), kind)
    for k, v in props.items():
        setattr(m, k, v)
    return m


def apply_mods(ob):
    activate(ob)
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def recalc_normals(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def cleanup(ob, smooth_iter=2, smooth_factor=0.5, decimate=None):
    """Light smoothing of marching-cubes terracing + optional collapse decimation."""
    if smooth_iter:
        add_mod(ob, "SMOOTH", factor=smooth_factor, iterations=smooth_iter)
    if decimate and decimate < 1.0:
        add_mod(ob, "DECIMATE", ratio=decimate, use_collapse_triangulate=False)
    apply_mods(ob)
    recalc_normals(ob)
    ob.data.shade_smooth()


def face_centers(ob):
    me = ob.data
    c = np.zeros(len(me.polygons) * 3, np.float32)
    me.polygons.foreach_get("center", c)
    return c.reshape(-1, 3)


def assign_by_mask(ob, mask, mat):
    me = ob.data
    if mat.name not in [m.name for m in me.materials if m]:
        me.materials.append(mat)
    idx = [m.name for m in me.materials].index(mat.name)
    cur = np.zeros(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", cur)
    cur[mask] = idx
    me.polygons.foreach_set("material_index", cur)


def world_uv(ob, tile_m=0.25):
    """Smart-project UVs, then rescale so 1 UV unit == tile_m metres (consistent texel density)."""
    activate(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=np.radians(55), island_margin=0.002, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    me = ob.data
    uv = me.uv_layers.active.data
    n = len(uv)
    a = np.zeros(n * 2, np.float32)
    uv.foreach_get("uv", a)
    a = a.reshape(-1, 2)
    # areas
    area3d = sum(p.area for p in me.polygons)
    area_uv = 0.0
    loops = np.zeros(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", loops)
    tot = np.zeros(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", tot)
    for ls, lt in zip(loops, tot):
        p = a[ls:ls + lt]
        x, y = p[:, 0], p[:, 1]
        area_uv += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    s = np.sqrt(area3d / max(area_uv, 1e-12)) / tile_m
    uv.foreach_set("uv", (a * s).ravel())


# --------------------------------------------------------------------------- materials
_img_cache = {}


def img(name, colorspace="Non-Color"):
    key = (name, colorspace)
    if key not in _img_cache:
        im = bpy.data.images.load(str(TEX / name), check_existing=True)
        im.colorspace_settings.name = colorspace
        _img_cache[key] = im
    return _img_cache[key]


def pbr(name, base, rough=0.5, metal=0.0, normal_map=None, normal_strength=1.0, rough_map=None,
        albedo_map=None, albedo_is_mask=False, sss=0.0, sss_radius=(1.0, 0.4, 0.25), coat=0.0,
        emission=None, emission_strength=0.0, sheen=0.0, transmission=0.0, alpha=1.0, spec=0.5,
        aniso=0.0):
    """Principled BSDF material driven by the generated texture set (glTF-exportable layout)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    out = nt.nodes["Material Output"]
    bsdf.inputs["Base Color"].default_value = srgb(base) if isinstance(base, str) else base
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if aniso:
        bsdf.inputs["Anisotropic"].default_value = aniso
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.location = (-1100, 0)
    x = -800
    if albedo_map:
        t = nt.nodes.new("ShaderNodeTexImage"); t.location = (x, 300)
        t.image = img(albedo_map, "sRGB")
        nt.links.new(uvn.outputs["UV"], t.inputs["Vector"])
        if albedo_is_mask:
            # tint grey detail map by the base colour (exports as baseColorTexture * factor)
            mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
            mix.location = (-400, 300)
            mix.inputs["Factor"].default_value = 1.0
            mix.inputs["A"].default_value = srgb(base) if isinstance(base, str) else base
            nt.links.new(t.outputs["Color"], mix.inputs["B"])
            nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
        else:
            nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if rough_map:
        t = nt.nodes.new("ShaderNodeTexImage"); t.location = (x, 0)
        t.image = img(rough_map)
        nt.links.new(uvn.outputs["UV"], t.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMath"); mr.operation = "MULTIPLY"; mr.location = (-400, 0)
        mr.inputs[1].default_value = rough / 0.6
        nt.links.new(t.outputs["Color"], mr.inputs[0])
        nt.links.new(mr.outputs[0], bsdf.inputs["Roughness"])
    if normal_map:
        t = nt.nodes.new("ShaderNodeTexImage"); t.location = (x, -300)
        t.image = img(normal_map)
        nt.links.new(uvn.outputs["UV"], t.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.location = (-400, -300)
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if sss:
        bsdf.inputs["Subsurface Weight"].default_value = sss
        bsdf.inputs["Subsurface Radius"].default_value = sss_radius
        bsdf.inputs["Subsurface Scale"].default_value = 0.004
    if coat:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.03
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
        bsdf.inputs["Sheen Roughness"].default_value = 0.4
    if transmission:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = srgb(emission)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
    return m


# --------------------------------------------------------------------------- curves / strands
def strands_object(name, coll, polylines, radius, mat, radius_taper=True, resolution=0):
    """One curve object holding many poly splines (hair, laces, piping, eyebrows)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = resolution
    cu.use_fill_caps = True
    for pts in polylines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        flat = np.c_[np.asarray(pts, np.float32), np.ones(len(pts), np.float32)].ravel()
        sp.points.foreach_set("co", flat)
        if radius_taper:
            n = len(pts)
            r = np.linspace(1.0, 0.25, n).astype(np.float32)
            sp.points.foreach_set("radius", r)
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    cu.materials.append(mat)
    return ob


def bvh(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(ob, dg)


def project_polyline(tree, pts, lift=0.0015):
    """Snap points onto the nearest surface of a BVH and lift them along the normal."""
    out = []
    for p in pts:
        hit = tree.find_nearest(Vector(p))
        if hit[0] is None:
            out.append(p); continue
        loc, nrm = hit[0], hit[1]
        out.append(tuple(loc + nrm * lift))
    return out


# --------------------------------------------------------------------------- primitives
def box(name, coll, size, loc, rot=(0, 0, 0), mat=None, bevel=0.004, segs=3):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=np.radians(rot))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    link(ob, coll)
    activate(ob)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        add_mod(ob, "BEVEL", width=bevel, segments=segs, limit_method="ANGLE")
    if mat:
        ob.data.materials.append(mat)
    ob.data.shade_smooth()
    add_mod(ob, "WEIGHTED_NORMAL", keep_sharp=True)
    return ob


def cylinder(name, coll, r, depth, loc, rot=(0, 0, 0), mat=None, verts=32, bevel=0.0015):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc,
                                        rotation=np.radians(rot))
    ob = bpy.context.active_object
    ob.name = name
    link(ob, coll)
    if bevel:
        add_mod(ob, "BEVEL", width=bevel, segments=2, limit_method="ANGLE")
    if mat:
        ob.data.materials.append(mat)
    ob.data.shade_smooth()
    add_mod(ob, "WEIGHTED_NORMAL", keep_sharp=True)
    return ob


def uvsphere(name, coll, r, loc, scale=(1, 1, 1), rot=(0, 0, 0), mat=None, seg=48, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings, radius=r, location=loc,
                                         rotation=np.radians(rot))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = scale
    link(ob, coll)
    if mat:
        ob.data.materials.append(mat)
    ob.data.shade_smooth()
    return ob


def torus_obj(name, coll, R, r, loc, rot=(0, 0, 0), mat=None, major=48, minor=12):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=major,
                                     minor_segments=minor, location=loc, rotation=np.radians(rot))
    ob = bpy.context.active_object
    ob.name = name
    link(ob, coll)
    if mat:
        ob.data.materials.append(mat)
    ob.data.shade_smooth()
    return ob
