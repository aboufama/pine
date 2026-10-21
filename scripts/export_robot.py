"""Pack Unitree's H2 visual meshes and kinematic tree for the browser.

Reads the official MJCF (unitree_mujoco/unitree_robots/h2/h2_mujoco.xml) and its STL meshes, keeps the visual
geoms, simplifies each mesh by vertex clustering (shape kept, triangle count capped), and writes
demo/h2.bin (Float32 positions, Uint16 indices) and demo/h2.json (bodies, joints, mesh offsets).
"""
from __future__ import annotations
import json, os, struct, sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.g1 import SPECS, RAW  # noqa: E402

ROBOT = "h2"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
MAX_TRIS = 4000


def read_stl(path):
    b = open(path, "rb").read()
    if b[:5].lower() == b"solid" and b"facet" in b[:400]:
        vs = [list(map(float, l.split()[1:4])) for l in b.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
        return np.array(vs, np.float64).reshape(-1, 3, 3)
    n = struct.unpack("<I", b[80:84])[0]
    dt = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
    return np.frombuffer(b[84:84 + n * 50], dt)["v"].reshape(-1, 3, 3).astype(np.float64)


def simplify(tri, cell):
    v = tri.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    pos = np.zeros((inv.max() + 1, 3)); np.add.at(pos, inv, v); pos /= np.bincount(inv)[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, 1), axis=0, return_index=True)       # drop duplicate faces, keep winding
    f = f[np.sort(first)]
    used = np.unique(f); remap = -np.ones(len(pos), np.int64); remap[used] = np.arange(len(used))
    return pos[used].astype(np.float32), remap[f].astype(np.uint16)


def main():
    sp = SPECS[ROBOT]
    path = os.path.join(RAW, sp["mjcf"]); mdir = os.path.join(os.path.dirname(path), "meshes")
    root = ET.parse(path).getroot()
    files = {m.get("name") or os.path.splitext(m.get("file"))[0]: m.get("file") for m in root.find("asset").findall("mesh")}
    bodies, blob, meshes = [], bytearray(), {}

    def add_mesh(name):
        if name in meshes:
            return
        tri = read_stl(os.path.join(mdir, files[name]))
        cell = 0.003
        pos, idx = simplify(tri, cell)
        while len(idx) > MAX_TRIS:
            cell *= 1.25; pos, idx = simplify(tri, cell)
        m = dict(pos=len(blob), n_verts=len(pos))
        blob.extend(pos.tobytes())
        m["idx"] = len(blob); m["n_tris"] = len(idx)
        blob.extend(idx.tobytes())
        while len(blob) % 4: blob.append(0)
        m["source_tris"] = len(tri); meshes[name] = m

    def walk(el, parent):
        for b in el.findall("body"):
            i = len(bodies)
            j = b.find("joint")
            joint = None
            if j is not None and j.get("type") in (None, "hinge"):
                joint = dict(index=sp["joints"].index(j.get("name").replace("_joint", "")),
                             axis=[float(x) for x in j.get("axis").split()])
            vis = [g for g in b.findall("geom") if g.get("mesh") and g.get("group") == "1"] or \
                  [g for g in b.findall("geom") if g.get("mesh")][:1]
            geoms = []
            for g in vis:
                add_mesh(g.get("mesh"))
                geoms.append(dict(mesh=g.get("mesh"), rgba=[float(x) for x in (g.get("rgba") or "0.7 0.7 0.7 1").split()],
                                  pos=[float(x) for x in (g.get("pos") or "0 0 0").split()],
                                  quat=[float(x) for x in (g.get("quat") or "1 0 0 0").split()]))
            bodies.append(dict(name=b.get("name"), parent=parent,
                               pos=[float(x) for x in (b.get("pos") or "0 0 0").split()] if parent >= 0 else [0, 0, 0],
                               quat=[float(x) for x in (b.get("quat") or "1 0 0 0").split()], joint=joint, geoms=geoms))
            walk(b, i)

    walk(root.find("worldbody"), -1)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"{ROBOT}.bin"), "wb").write(bytes(blob))
    json.dump(dict(robot=sp["name"], joints=sp["joints"], bodies=bodies, meshes=meshes,
                   source="unitreerobotics/unitree_mujoco, unitree_robots/h2/h2_mujoco.xml"),
              open(os.path.join(OUT, f"{ROBOT}.json"), "w"), separators=(",", ":"))
    st = sum(m["source_tris"] for m in meshes.values()); kt = sum(m["n_tris"] for m in meshes.values())
    print(f"{len(bodies)} bodies, {len(meshes)} meshes, triangles {st:,} -> {kt:,}, bin {len(blob)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
