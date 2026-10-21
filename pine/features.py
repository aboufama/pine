"""Robot-space frame features, channel groups (what a suit can measure), splits and window sampling.

The robot is chosen with PINE_ROBOT (default "h2", the full-size Unitree H2). G1 and H2 both solve 23 joints in
the same group layout (legs 12, waist 3, arms 8), so every index below holds for either robot.
"""
from __future__ import annotations
import json, os, zlib
import numpy as np
from .g1 import SPECS, active_joints

ROBOT = os.environ.get("PINE_ROBOT", "h2")
JOINTS = SPECS[ROBOT]["joints"]
ACTIVE = np.array(active_joints(ROBOT))              # 23 joints (wrists, and H2's head, held at zero)
NAMES = [JOINTS[j] for j in ACTIVE] + ["grav_x", "grav_y", "grav_z", "yaw_rate", "vel_fwd", "vel_left", "vel_up", "height"]
D = len(NAMES)                                      # 31
GROUPS = {"legs": list(range(0, 12)), "waist": list(range(12, 15)), "arms": list(range(15, 23)),
          "imu": list(range(23, 27)), "root": list(range(27, 31))}
# EXO-1 assumption for this proof of concept: joint encoders at hips, knees and ankles plus a pelvis IMU.
EXO = sorted(GROUPS["legs"] + GROUPS["imu"])
HIDDEN = [i for i in range(D) if i not in EXO]
ANGLE_DIMS = list(range(0, 23))


def g1_features(pr, R, q, fps=20.0):
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    dp = np.diff(pr, axis=0); dp = np.concatenate([dp, dp[-1:]]) * fps
    c, s = np.cos(-yaw), np.sin(-yaw)
    v = np.stack([c * dp[:, 0] - s * dp[:, 1], s * dp[:, 0] + c * dp[:, 1], dp[:, 2]], 1)
    dyaw = np.diff(yaw); dyaw = np.concatenate([dyaw, dyaw[-1:]]) * fps
    grav = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -1.0]))
    return np.concatenate([q[:, ACTIVE], grav, dyaw[:, None], v, pr[:, 2:3]], 1).astype(np.float32)


class MotionData:
    """All frames of all takes as one array, plus clip boundaries and the split of each clip."""

    def __init__(self, canon_dir, g1_dir):
        man = json.load(open(os.path.join(canon_dir, "manifest.json")))
        by_index = {c["index"]: c for c in man["clips"] if "reject" not in c}
        feats, clips, errs = [], [], []
        for name in sorted(os.listdir(g1_dir)):
            if not name.startswith("part"):
                continue
            z = dict(np.load(os.path.join(g1_dir, name)))      # read each array once; per-clip access re-reads it
            off = z["offsets"]
            for j, idx in enumerate(z["idx"]):
                a, b = off[j], off[j + 1]
                feats.append(g1_features(z["pr"][a:b].astype(np.float64), z["R"][a:b].astype(np.float64), z["q"][a:b]))
                errs.append(z["err"][a:b].copy())
                clips.append(by_index[int(idx)])
        order = np.argsort([c["index"] for c in clips])
        self.clips = [clips[i] for i in order]
        feats = [feats[i] for i in order]; self.err = [errs[i] for i in order]
        self.lens = np.array([len(f) for f in feats])
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.raw = np.concatenate(feats)
        self.split = np.array([c["split"] for c in self.clips])
        tr = np.concatenate([np.arange(s, s + n) for s, n, sp in zip(self.starts, self.lens, self.split) if sp == "train"])
        self.mean = self.raw[tr].mean(0)
        self.std = np.maximum(self.raw[tr].std(0), 1e-3)
        self.X = ((self.raw - self.mean) / self.std).astype(np.float32)

    def subset(self, split, fraction=1.0, seed=0):
        ids = np.where(self.split == split)[0]
        if fraction < 1.0:
            r = np.random.default_rng(seed); ids = r.permutation(ids)
            total = self.lens[ids].sum(); keep = np.cumsum(self.lens[ids]) <= fraction * total
            keep[0] = True; ids = np.sort(ids[keep])
        return ids

    def hours(self, ids, fps=20.0):
        return float(self.lens[ids].sum() / fps / 3600)


class Windows:
    def __init__(self, data: MotionData, ids, T=32, K=4, seed=0):
        self.d, self.T, self.K = data, T, K
        L = T + K
        ok = [i for i in ids if data.lens[i] >= L]
        self.ids = np.array(ok)
        self.n_starts = data.lens[self.ids] - L + 1
        self.p = self.n_starts / self.n_starts.sum()
        self.rng = np.random.default_rng(seed)
        self.frames = int(data.lens[self.ids].sum())

    def sample(self, B):
        c = self.rng.choice(len(self.ids), size=B, p=self.p)
        off = (self.rng.random(B) * self.n_starts[c]).astype(int)
        s = self.d.starts[self.ids[c]] + off
        idx = s[:, None] + np.arange(self.T + self.K)[None]
        return self.d.X[idx]                               # (B, T+K, D)

    def fixed(self, n, seed=123):
        r = self.rng; self.rng = np.random.default_rng(seed)
        w = self.sample(n); self.rng = r
        return w


def make_masks(B, mode, rng):
    m = np.ones((B, D), np.float32)
    if mode == "exo":
        m[:, HIDDEN] = 0
    elif mode == "mixed":
        u = rng.random(B)
        exo = (u >= 0.5) & (u < 0.8)
        m[np.ix_(np.where(exo)[0], HIDDEN)] = 0
        rnd = np.where(u >= 0.8)[0]
        for g in GROUPS.values():
            drop = rnd[rng.random(len(rnd)) < 0.5]
            m[np.ix_(drop, g)] = 0
    return m


def batch(w, T, K, mask):
    x = w[:, :T] * mask[:, None, :]
    inp = np.concatenate([x, np.broadcast_to(mask[:, None, :], x.shape)], -1)
    tgt = np.concatenate([w[:, 1 + j:T + 1 + j] for j in range(K)], -1)
    return inp.astype(np.float32), tgt.astype(np.float32)
