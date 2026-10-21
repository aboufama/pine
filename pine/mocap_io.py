"""Readers for raw motion-capture formats -> world-space joint positions.

Every reader returns a `Take`: joint names, parent indices, world positions in metres
(Z up), and the frame rate. Nothing downstream ever sees a source-specific format.
EXO-1 logs will get a reader here too; the rest of the pipeline does not change.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # +90 deg about X


@dataclass
class Take:
    source: str            # dataset id, e.g. "cmu", "bandai2"
    clip: str              # clip id, unique within source
    subject: str           # performer id (used for held-out splits)
    fps: float
    names: list[str]
    parents: list[int]
    pos: np.ndarray        # (N, J, 3) metres, Z up
    meta: dict = field(default_factory=dict)


def _rot(axis: str, ang: np.ndarray) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    R = np.zeros(ang.shape + (3, 3))
    if axis in "xX":
        R[..., 0, 0] = 1; R[..., 1, 1] = c; R[..., 1, 2] = -s; R[..., 2, 1] = s; R[..., 2, 2] = c
    elif axis in "yY":
        R[..., 1, 1] = 1; R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 2, 0] = -s; R[..., 2, 2] = c
    else:
        R[..., 2, 2] = 1; R[..., 0, 0] = c; R[..., 0, 1] = -s; R[..., 1, 0] = s; R[..., 1, 1] = c
    return R


# --------------------------------------------------------------------------- BVH
def read_bvh(path: str, unit_to_m: float, source: str, clip: str, subject: str) -> Take:
    text = open(path, "r", errors="replace").read()
    head, _, motion = text.partition("MOTION")
    names, parents, offsets, chans = [], [], [], []
    stack: list[int] = []
    pending = -1
    for line in head.splitlines():
        t = line.split()
        if not t:
            continue
        k = t[0]
        if k in ("ROOT", "JOINT"):
            names.append(t[1]); parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0]); chans.append([]); pending = len(names) - 1
        elif k == "End":
            names.append(names[stack[-1]] + "_end"); parents.append(stack[-1])
            offsets.append([0.0, 0.0, 0.0]); chans.append([]); pending = len(names) - 1
        elif k == "{":
            stack.append(pending)
        elif k == "}":
            stack.pop()
        elif k == "OFFSET":
            offsets[pending] = [float(x) for x in t[1:4]]
        elif k == "CHANNELS":
            chans[pending] = t[2:2 + int(t[1])]
    m = re.search(r"Frames:\s*(\d+)", motion)
    ft = re.search(r"Frame Time:\s*([0-9.eE+-]+)", motion)
    n = int(m.group(1)); dt = float(ft.group(1))
    body = motion[ft.end():]
    vals = np.array(body.split(), dtype=np.float64)
    ncols = sum(len(c) for c in chans)
    n = min(n, vals.size // ncols)
    data = vals[: n * ncols].reshape(n, ncols)

    J = len(names)
    gpos = np.zeros((n, J, 3)); grot = np.zeros((n, J, 3, 3))
    col = 0
    for j in range(J):
        t_loc = np.tile(np.array(offsets[j]), (n, 1))
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for c in chans[j]:
            v = data[:, col]; col += 1
            if c.lower().endswith("position"):
                t_loc[:, "XYZ".index(c[0].upper())] = v
            else:
                R = R @ _rot(c[0], np.deg2rad(v))
        p = parents[j]
        if p < 0:
            grot[:, j] = R; gpos[:, j] = t_loc
        else:
            grot[:, j] = grot[:, p] @ R
            gpos[:, j] = gpos[:, p] + np.einsum("nij,nj->ni", grot[:, p], t_loc)
    pos = (gpos * unit_to_m) @ Y_UP_TO_Z_UP.T
    return Take(source, clip, subject, 1.0 / dt, names, parents, pos.astype(np.float32))


# --------------------------------------------------------------------- ASF / AMC
def _euler_sxyz(a: np.ndarray) -> np.ndarray:
    """Static-frame x, then y, then z: R = Rz @ Ry @ Rx (Acclaim convention)."""
    return _rot("z", a[..., 2]) @ _rot("y", a[..., 1]) @ _rot("x", a[..., 0])


def read_asf(path: str) -> dict:
    lines = open(path, "r", errors="replace").read().splitlines()
    length_unit = 1.0
    bones = {"root": dict(direction=np.zeros(3), length=0.0, axis=np.zeros(3), dof=["rx", "ry", "rz"])}
    hierarchy: dict[str, list[str]] = {}
    section = None; cur = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            section = line.split()[0][1:]
            if section == "units":
                pass
            continue
        t = line.split()
        if section == "units" and t[0] == "length":
            length_unit = float(t[1])
        elif section == "bonedata":
            if t[0] == "begin":
                cur = dict(direction=np.zeros(3), length=0.0, axis=np.zeros(3), dof=[])
            elif t[0] == "end":
                bones[cur["name"]] = cur
            elif t[0] == "name":
                cur["name"] = t[1]
            elif t[0] == "direction":
                cur["direction"] = np.array([float(x) for x in t[1:4]])
            elif t[0] == "length":
                cur["length"] = float(t[1])
            elif t[0] == "axis":
                cur["axis"] = np.deg2rad(np.array([float(x) for x in t[1:4]]))
            elif t[0] == "dof":
                cur["dof"] = t[1:]
        elif section == "hierarchy":
            if t[0] in ("begin", "end"):
                continue
            hierarchy[t[0]] = t[1:]
    return dict(bones=bones, hierarchy=hierarchy, length_unit=length_unit)


def read_amc(path: str, asf: dict, fps: float, source: str, clip: str, subject: str) -> Take:
    frames: list[dict] = []
    cur = None
    for raw in open(path, "r", errors="replace"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue
        if line.isdigit():
            cur = {}; frames.append(cur); continue
        t = line.split()
        cur[t[0]] = np.array([float(x) for x in t[1:]])
    n = len(frames)
    bones, hier = asf["bones"], asf["hierarchy"]
    order, parents = ["root"], [-1]
    queue = ["root"]
    while queue:
        b = queue.pop(0)
        for c in hier.get(b, []):
            order.append(c); parents.append(order.index(b)); queue.append(c)
    J = len(order)
    pos = np.zeros((n, J, 3)); mat = np.zeros((n, J, 3, 3))
    for j, name in enumerate(order):
        bone = bones[name]
        C = _euler_sxyz(bone["axis"]); Cinv = C.T
        if name == "root":
            vals = np.stack([f.get("root", np.zeros(6)) for f in frames])
            M = _euler_sxyz(np.deg2rad(vals[:, 3:6]))
            mat[:, j] = C @ M @ Cinv
            pos[:, j] = vals[:, :3]
            continue
        ang = np.zeros((n, 3))
        dofs = bone["dof"]
        if dofs:
            vals = np.stack([f.get(name, np.zeros(len(dofs))) for f in frames])
            for i, d in enumerate(dofs):
                if d in ("rx", "ry", "rz") and i < vals.shape[1]:
                    ang[:, "xyz".index(d[1])] = vals[:, i]
        M = _euler_sxyz(np.deg2rad(ang))
        p = parents[j]
        mat[:, j] = mat[:, p] @ C @ M @ Cinv
        pos[:, j] = pos[:, p] + bone["length"] * np.einsum("nij,j->ni", mat[:, j], bone["direction"])
    unit_to_m = (1.0 / asf["length_unit"]) * 0.0254
    pos = (pos * unit_to_m) @ Y_UP_TO_Z_UP.T
    return Take(source, clip, subject, fps, order, parents, pos.astype(np.float32))
