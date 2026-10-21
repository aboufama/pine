"""Batched human -> robot retargeting on the GPU: a line-by-line port of pine.g1.Humanoid (fk, solve, retarget_many)
for any robot in pine.g1.SPECS (H2 by default, G1, ...): nq, ACTIVE, kp_body/kp_off, anc, lo/hi and sole all come
from the pine.g1.Humanoid instance, nothing is hard-coded.

Same maths as pine/g1.py: targets from Humanoid.targets (NumPy, per take), analytic Jacobian (position, SO(3) and
joint columns), keypoint weights, Levenberg-Marquardt damping 1e-2, joint regulariser 1e-3 (waist 5e-2), projection
onto joint limits, 8 iterations, [1 4 6 4 1]/16 smoothing per take with edge padding, then 3 iterations pulled
toward the smoothed joints (mu 0.5).

The solver is written once against a tiny array backend, so the same code runs on NumPy (gpu/tests checks it against
pine/g1.py to ~1e-9 m) and on torch (check_against_numpy: < 1 mm mean keypoint difference on 2,000 frames, run by
gpu/build_shards.py before it trusts the GPU; otherwise it falls back to pine/g1.py on all CPU cores).
"""
from __future__ import annotations
import time
import numpy as np
from pine.canon import KP, KEYPOINTS


# ---------------------------------------------------------------------------------------------------- backends
class NumpyOps:
    name = "numpy"

    def __init__(self):
        self.device = "cpu"

    def asarray(self, x):
        return np.array(x, dtype=np.float64)

    def numpy(self, x):
        return np.asarray(x)

    def zeros(self, *shape):
        return np.zeros(shape)

    def eye(self, n):
        return np.eye(n)

    def stack(self, xs, axis):
        return np.stack(xs, axis)

    def cat(self, xs, axis):
        return np.concatenate(xs, axis)

    def cross(self, a, b):
        return np.cross(a, b)

    def solve(self, H, g):
        return np.linalg.solve(H, g[..., None])[..., 0]

    def norm(self, x, axis, keepdims=False):
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    def sum(self, x, axis, keepdims=False):
        return x.sum(axis, keepdims=keepdims)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def floor_max(self, x, v):
        return np.maximum(x, v)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def permute(self, x, perm):
        return np.transpose(x, perm)

    def einsum(self, eq, *xs):
        return np.einsum(eq, *xs)

    def copy(self, x):
        return x.copy()

    def index(self, idx):
        return np.asarray(idx, dtype=np.int64)

    def broadcast(self, x, shape):
        return np.broadcast_to(x, shape)

    def sync(self):
        pass


class TorchOps:
    name = "torch"

    def __init__(self, device="cuda", dtype=None):
        import torch
        self.t = torch
        self.device = torch.device(device)
        self.dtype = dtype or torch.float64

    def asarray(self, x):
        return self.t.as_tensor(np.asarray(x, dtype=np.float64), device=self.device).to(self.dtype)

    def numpy(self, x):
        return x.detach().to("cpu", self.t.float64).numpy()

    def zeros(self, *shape):
        return self.t.zeros(shape, device=self.device, dtype=self.dtype)

    def eye(self, n):
        return self.t.eye(n, device=self.device, dtype=self.dtype)

    def stack(self, xs, axis):
        return self.t.stack(list(xs), axis)

    def cat(self, xs, axis):
        return self.t.cat(list(xs), axis)

    def cross(self, a, b):
        shape = self.t.broadcast_shapes(a.shape, b.shape)
        return self.t.linalg.cross(a.expand(shape), b.expand(shape), dim=-1)

    def solve(self, H, g):
        return self.t.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)

    def norm(self, x, axis, keepdims=False):
        return self.t.linalg.vector_norm(x, dim=axis, keepdim=keepdims)

    def sum(self, x, axis, keepdims=False):
        return x.sum(axis, keepdim=keepdims)

    def sin(self, x):
        return self.t.sin(x)

    def cos(self, x):
        return self.t.cos(x)

    def floor_max(self, x, v):
        return self.t.clamp(x, min=v)

    def clip(self, x, lo, hi):
        return self.t.minimum(self.t.maximum(x, lo), hi)

    def permute(self, x, perm):
        return x.permute(*perm)

    def einsum(self, eq, *xs):
        return self.t.einsum(eq, *xs)

    def copy(self, x):
        return x.clone()

    def index(self, idx):
        return self.t.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)

    def broadcast(self, x, shape):
        return x.expand(shape)

    def sync(self):
        if self.device.type == "cuda":
            self.t.cuda.synchronize(self.device)


def make_ops(backend: str = "torch", device: str = "cuda"):
    return NumpyOps() if backend == "numpy" else TorchOps(device)


# ---------------------------------------------------------------------------------------------------- solver
class BatchedIK:
    """pine.g1.Humanoid's fk / solve / retarget_many on a backend. `robot` is a pine.g1.Humanoid (any robot)."""

    def __init__(self, robot, ops):
        o = self.o = ops
        self.robot = robot
        self.nq, self.Bn = int(robot.nq), int(robot.B)
        self.parent = [int(p) for p in robot.parent]
        self.jidx = [int(j) for j in robot.jidx]
        self.pos = [o.asarray(p) for p in robot.pos]
        self.rot = [o.asarray(r) for r in robot.rot]
        # jaxis uses the axis as written in the MJCF; the rotation uses it normalised (exactly as pine.g1)
        self.axis = [o.asarray(a) if a is not None else None for a in robot.axis]
        self.K_axis = []
        for a in robot.axis:
            if a is None:
                self.K_axis.append(None); continue
            k = np.asarray(a, float); k = k / np.linalg.norm(k)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            self.K_axis.append((o.asarray(K), o.asarray(K @ K)))
        A = np.asarray(robot.ACTIVE, dtype=np.int64)
        self.A_np, self.A = A, o.index(A)
        self.nA = len(A)
        self.lo_A, self.hi_A = o.asarray(robot.lo[A]), o.asarray(robot.hi[A])
        self.kp_body = o.index(robot.kp_body)
        self.kp_off = o.asarray(robot.kp_off)
        self.W = o.asarray(np.repeat(robot.w, 3))
        self.anc_A = o.asarray(robot.anc[:, A].astype(np.float64))[None, :, :, None]
        reg = np.full(self.nq, 1e-3); reg[[12, 13, 14]] = 5e-2        # waist (same indices on G1 and H2)
        self.reg = o.asarray(reg[A])
        self.eye3 = o.eye(3)
        self.lift = o.asarray([0.0, 0.0, 0.12])

    def axis_rot(self, b, ang):
        K, KK = self.K_axis[b]
        s, c = self.o.sin(ang)[:, None, None], self.o.cos(ang)[:, None, None]
        return self.eye3 + s * K + (1 - c) * KK

    def so3_exp(self, w):
        o = self.o
        th = o.norm(w, -1, keepdims=True)
        k = w / o.floor_max(th, 1e-12)
        k0, k1, k2 = k[..., 0], k[..., 1], k[..., 2]
        z = k0 * 0
        K = o.stack([o.stack([z, -k2, k1], -1), o.stack([k2, z, -k0], -1), o.stack([-k1, k0, z], -1)], -2)
        th = th[..., None]
        return self.eye3 + o.sin(th) * K + (1 - o.cos(th)) * (K @ K)

    def fk(self, root_p, root_R, q):
        o = self.o
        N = q.shape[0]
        P, R = [None] * self.Bn, [None] * self.Bn
        jaxis, jorg = o.zeros(N, self.nq, 3), o.zeros(N, self.nq, 3)
        for b in range(self.Bn):
            p = self.parent[b]
            if p < 0:
                Rb = root_R @ self.rot[b]; Pb = root_p + 0 * self.pos[b]
            else:
                Rb = R[p] @ self.rot[b]; Pb = P[p] + R[p] @ self.pos[b]
            ji = self.jidx[b]
            if ji >= 0:
                jaxis[:, ji] = Rb @ self.axis[b]; jorg[:, ji] = Pb
                Rb = Rb @ self.axis_rot(b, q[:, ji])
            R[b] = Rb; P[b] = Pb
        P, R = o.stack(P, 1), o.stack(R, 1)
        kp = P[:, self.kp_body] + o.einsum("nkij,kj->nki", R[:, self.kp_body], self.kp_off)
        return dict(P=P, R=R, kp=kp, jaxis=jaxis, jorg=jorg)

    def solve(self, tgt, iters=8, init=None, smooth_to=None, mu=0.0):
        o = self.o
        N = tgt.shape[0]; A = self.A; nv = 6 + self.nA
        nK = len(KEYPOINTS)
        if init is None:
            q = o.zeros(N, self.nq)
            y = tgt[:, KP["l_hip"]] - tgt[:, KP["r_hip"]]; y = y / o.norm(y, 1, keepdims=True)
            up = (tgt[:, KP["neck"]] - tgt[:, KP["pelvis"]]); up = up - o.sum(up * y, 1, keepdims=True) * y
            z = up / o.norm(up, 1, keepdims=True)
            Rr = o.stack([o.cross(y, z), y, z], 2)
            pr = 0.5 * (tgt[:, KP["l_hip"]] + tgt[:, KP["r_hip"]]) + self.lift
        else:
            pr, Rr, q = (o.copy(a) for a in init)
        lam = 1e-2
        D = o.cat([o.zeros(6), self.reg], 0)
        if smooth_to is not None:
            D = D + o.cat([o.zeros(6), o.zeros(self.nA) + mu], 0)
        I = o.eye(nv)[None]
        Jp = o.broadcast(self.eye3, (N, nK, 3, 3))
        e3 = self.eye3[None, None, :, :]
        for it in range(iters):
            f = self.fk(pr, Rr, q)
            r = (tgt - f["kp"]).reshape(N, -1)
            ja, jo = f["jaxis"][:, A][:, None], f["jorg"][:, A][:, None]          # (N, 1, nA, 3)
            Jq = o.cross(ja, f["kp"][:, :, None, :] - jo)
            Jq = Jq * self.anc_A
            Jw = o.cross(e3, (f["kp"] - pr[:, None])[:, :, None, :])
            J = o.permute(o.cat([Jp, Jw, Jq], 2), (0, 1, 3, 2)).reshape(N, -1, nv)
            JW = J * self.W[None, :, None]
            JWt = o.permute(JW, (0, 2, 1))
            H = JWt @ J
            g = (JWt @ r[..., None])[..., 0]
            qA = q[:, A]
            g[:, 6:] -= self.reg * qA
            if smooth_to is not None:
                g[:, 6:] += mu * (smooth_to[:, A] - qA)
            H = H + (D + lam)[None, :, None] * I
            dlt = o.solve(H, g)
            pr = pr + dlt[:, :3]; Rr = self.so3_exp(dlt[:, 3:6]) @ Rr
            q[:, A] = o.clip(qA + dlt[:, 6:], self.lo_A, self.hi_A)
        f = self.fk(pr, Rr, q)
        err = o.norm(tgt - f["kp"], 2)
        return (pr, Rr, q), err

    def smooth(self, q, lens):
        """Per take: np.convolve(np.pad(q[:, j], 2, mode='edge'), [1,4,6,4,1]/16, mode='valid')."""
        o = self.o
        lens = np.asarray(lens, dtype=np.int64)
        start = np.repeat(np.cumsum(lens) - lens, lens)
        end = np.repeat(np.cumsum(lens) - 1, lens)
        i = np.arange(int(lens.sum()), dtype=np.int64)
        k = np.array([1, 4, 6, 4, 1], float); k /= k.sum()
        out = None
        for j, w in zip(range(-2, 3), k):
            term = q[o.index(np.clip(i + j, start, end))] * float(w)
            out = term if out is None else out + term
        return out

    def retarget_targets(self, tgts, iters=(8, 3)):
        """tgts: list of (n_i, 19, 3) NumPy target arrays (from Humanoid.targets). Returns NumPy per take."""
        o = self.o
        lens = [len(t) for t in tgts]
        tgt = o.asarray(np.concatenate(tgts))
        (pr, Rr, q), _ = self.solve(tgt, iters=iters[0])
        qs = self.smooth(q, lens)
        (pr, Rr, q), err = self.solve(tgt, iters=iters[1], init=(pr, Rr, qs), smooth_to=qs, mu=0.5)
        cut = np.cumsum(lens)[:-1]
        arrs = [np.split(o.numpy(x), cut) for x in (pr, Rr, q, err)]
        return [tuple(a[i] for a in arrs) for i in range(len(lens))]

    def retarget_many(self, takes, iters=(8, 3)):
        """Same contract as pine.g1.Humanoid.retarget_many: [((pr, R, q), err, tgt) per take]."""
        tgts = [self.robot.targets(h.astype(np.float64)) for h in takes]
        out = self.retarget_targets(tgts, iters)
        return [((pr, R, q), e, t) for (pr, R, q, e), t in zip(out, tgts)]


def chunks_by_frames(lengths, max_frames: int):
    """Group take indices into chunks of whole takes with at most max_frames frames (a longer take is alone)."""
    out, cur, n = [], [], 0
    for i, L in enumerate(lengths):
        if cur and n + L > max_frames:
            out.append(cur); cur, n = [], 0
        cur.append(i); n += int(L)
    if cur:
        out.append(cur)
    return out


def keypoint_difference(robot, sol_a, sol_b) -> dict:
    """Mean / max distance (m) between the FK keypoints of two solutions (lists of (pr, R, q))."""
    d = []
    dq = 0.0
    for (pa, Ra, qa), (pb, Rb, qb) in zip(sol_a, sol_b):
        ka = robot.fk(pa, Ra, qa)["kp"]; kb = robot.fk(pb, Rb, qb)["kp"]
        d.append(np.linalg.norm(ka - kb, axis=-1).ravel())
        dq = max(dq, float(np.abs(qa - qb).max()))
    d = np.concatenate(d)
    return dict(frames=int(sum(len(s[2]) for s in sol_a)), mean_mm=float(d.mean() * 1e3), max_mm=float(d.max() * 1e3),
                max_joint_diff_deg=float(np.degrees(dq)))


def check_against_numpy(robot, takes, ik: BatchedIK, frames: int = 2000, limit_mm: float = 1.0) -> dict:
    """Retarget the same `frames` frames with pine/g1.py (reference) and with `ik`; pass if mean difference < limit."""
    sel, n = [], 0
    for h in takes:
        if n >= frames:
            break
        take = h[: frames - n]
        if len(take) >= 10:
            sel.append(take); n += len(take)
    if n < frames:
        return dict(passed=False, reason=f"only {n} frames available for the check", frames=n)
    t0 = time.time()
    ref = robot.retarget_many(sel)
    t_ref = time.time() - t0
    t0 = time.time()
    got = ik.retarget_many(sel)
    ik.o.sync()
    t_ik = time.time() - t0
    res = keypoint_difference(robot, [r[0] for r in ref], [g[0] for g in got])
    res.update(passed=bool(res["mean_mm"] < limit_mm), limit_mm=limit_mm, backend=ik.o.name,
               seconds_numpy=t_ref, seconds_backend=t_ik,
               residual_mean_cm_numpy=float(np.concatenate([r[1].ravel() for r in ref]).mean() * 100),
               residual_mean_cm_backend=float(np.concatenate([g[1].ravel() for g in got]).mean() * 100))
    return res
