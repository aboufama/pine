"""Unitree G1 (29 DoF) kinematics and human -> G1 retargeting.

The robot is read from Unitree's own MJCF (unitree_mujoco/unitree_robots/g1/g1_29dof.xml).
Retargeting is a batched Levenberg-Marquardt inverse kinematics over every frame at once:
scaled human keypoints are targets, the variables are the floating base (6) and the 29 joints,
Jacobians are analytic, joint limits are enforced by projection, and a second pass pulls each
frame toward its smoothed neighbours so the result has no jitter.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
import os
from .canon import KP, KEYPOINTS

RAW = os.environ.get("PINE_RAW", "/home/claude/data/raw")

JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
LEG_JOINTS = list(range(12))          # what a lower-body exoskeleton measures directly
JOINTS_H2 = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_roll", "left_ankle_pitch",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_roll", "right_ankle_pitch",
    "waist_yaw", "waist_roll", "waist_pitch", "head_pitch", "head_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]

# keypoint -> (G1 body, offset in that body's frame)
G1_POINTS = {
    "pelvis": ("pelvis", (0, 0, 0)),
    "l_hip": ("left_hip_roll_link", (0, 0, 0)), "r_hip": ("right_hip_roll_link", (0, 0, 0)),
    "spine": ("torso_link", (0, 0, 0.06)), "chest": ("torso_link", (0, 0, 0.20)),
    "neck": ("torso_link", (0, 0, 0.30)), "head": ("torso_link", (0, 0, 0.40)),
    "l_shoulder": ("left_shoulder_roll_link", (0, 0, 0)), "r_shoulder": ("right_shoulder_roll_link", (0, 0, 0)),
    "l_elbow": ("left_elbow_link", (0, 0, 0)), "r_elbow": ("right_elbow_link", (0, 0, 0)),
    "l_wrist": ("left_wrist_roll_link", (0, 0, 0)), "r_wrist": ("right_wrist_roll_link", (0, 0, 0)),
    "l_knee": ("left_knee_link", (0, 0, 0)), "r_knee": ("right_knee_link", (0, 0, 0)),
    "l_ankle": ("left_ankle_roll_link", (0, 0, 0)), "r_ankle": ("right_ankle_roll_link", (0, 0, 0)),
    "l_toe": ("left_ankle_roll_link", (0.11, 0, -0.03)), "r_toe": ("right_ankle_roll_link", (0.11, 0, -0.03)),
}
H2_POINTS = {
    "pelvis": ("pelvis", (0, 0, 0)),
    "l_hip": ("left_hip_roll_link", (0, 0, 0)), "r_hip": ("right_hip_roll_link", (0, 0, 0)),
    "spine": ("torso_link", (0, 0, 0.10)), "chest": ("torso_link", (0, 0, 0.26)),
    "neck": ("torso_link", (0, 0, 0.37)), "head": ("head_pitch_link", (0, 0.0189, 0.02)),
    "l_shoulder": ("left_shoulder_roll_link", (0, 0, 0)), "r_shoulder": ("right_shoulder_roll_link", (0, 0, 0)),
    "l_elbow": ("left_elbow_link", (0, 0, 0)), "r_elbow": ("right_elbow_link", (0, 0, 0)),
    "l_wrist": ("left_wrist_roll_link", (0.07, 0, 0)), "r_wrist": ("right_wrist_roll_link", (0.07, 0, 0)),
    "l_knee": ("left_knee_link", (0, 0, 0)), "r_knee": ("right_knee_link", (0, 0, 0)),
    "l_ankle": ("left_ankle_roll_link", (0, 0, 0)), "r_ankle": ("right_ankle_roll_link", (0, 0, 0)),
    "l_toe": ("left_ankle_pitch_link", (0.12, 0.0365, -0.05)), "r_toe": ("right_ankle_pitch_link", (0.12, -0.0365, -0.05)),
}
WRISTS = ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
# Joints the public data cannot determine (no hand or face markers) are held at zero.
SPECS = {
    "g1": dict(name="Unitree G1", dof=29, height_m=1.32, mjcf="unitree_mujoco/repo/unitree_robots/g1/g1_29dof.xml",
               joints=JOINTS, points=None, sole=0.035, held=WRISTS),
    "h2": dict(name="Unitree H2", dof=31, height_m=1.80, mjcf="unitree_mujoco/repo/unitree_robots/h2/h2_mujoco.xml",
               joints=JOINTS_H2, points=H2_POINTS, sole=0.058, held=WRISTS + ["head_pitch", "head_yaw"]),
}
SPECS["g1"]["points"] = None  # filled below once G1_POINTS exists


def active_joints(robot: str) -> list[int]:
    sp = SPECS[robot]
    return [i for i, j in enumerate(sp["joints"]) if j not in sp["held"]]


WEIGHTS = dict(pelvis=0.0, l_hip=2.0, r_hip=2.0, spine=0.4, chest=0.6, neck=0.6, head=0.4,
               l_shoulder=1.0, r_shoulder=1.0, l_elbow=1.0, r_elbow=1.0, l_wrist=1.0, r_wrist=1.0,
               l_knee=2.0, r_knee=2.0, l_ankle=3.0, r_ankle=3.0, l_toe=2.0, r_toe=2.0)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_rot(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rodrigues, batched: axis (3,), ang (N,) -> (N, 3, 3)."""
    k = np.asarray(axis, float); k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def so3_exp(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = w / np.maximum(th, 1e-12)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


SPECS["g1"]["points"] = G1_POINTS


class Humanoid:
    """Any MJCF humanoid with one hinge per body; `robot` names an entry in SPECS."""

    def __init__(self, robot: str = "h2", mjcf_path: str | None = None):
        sp = SPECS[robot]
        self.robot, self.spec, self.joints = robot, sp, sp["joints"]
        self.nq = len(self.joints)
        self.ACTIVE = np.array(active_joints(robot))
        self.sole = sp["sole"]
        root = ET.parse(mjcf_path or os.path.join(RAW, sp["mjcf"])).getroot()
        self.bodies, self.parent, self.pos, self.rot, self.axis, self.jidx = [], [], [], [], [], []
        self.lo = np.zeros(self.nq); self.hi = np.zeros(self.nq)
        points = sp["points"]

        def walk(el, parent):
            for b in el.findall("body"):
                i = len(self.bodies)
                self.bodies.append(b.get("name")); self.parent.append(parent)
                self.pos.append(np.array([float(v) for v in (b.get("pos") or "0 0 0").split()]))
                self.rot.append(quat_to_mat([float(v) for v in (b.get("quat") or "1 0 0 0").split()]))
                j = b.find("joint")
                if j is not None and j.get("type") in (None, "hinge"):
                    name = j.get("name").replace("_joint", "")
                    ji = self.joints.index(name)
                    self.jidx.append(ji); self.axis.append(np.array([float(v) for v in j.get("axis").split()]))
                    lo, hi = (float(v) for v in j.get("range").split()); self.lo[ji], self.hi[ji] = lo, hi
                else:
                    self.jidx.append(-1); self.axis.append(None)
                walk(b, i)
        walk(root.find("worldbody"), -1)
        self.B = len(self.bodies)
        self.kp_body = np.array([self.bodies.index(points[k][0]) for k in KEYPOINTS])
        self.kp_off = np.array([points[k][1] for k in KEYPOINTS], float)
        self.w = np.array([WEIGHTS[k] for k in KEYPOINTS])
        # which joints move which keypoints (ancestor mask), keypoint x joint
        self.anc = np.zeros((len(KEYPOINTS), self.nq), bool)
        for k, b in enumerate(self.kp_body):
            while b >= 0:
                if self.jidx[b] >= 0:
                    self.anc[k, self.jidx[b]] = True
                b = self.parent[b]
        z = self.fk(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, self.nq)))
        self.rest = z
        kp = z["kp"][0]
        self.leg = (np.linalg.norm(kp[KP["l_hip"]] - kp[KP["l_knee"]]) + np.linalg.norm(kp[KP["l_knee"]] - kp[KP["l_ankle"]]))

    def fk(self, root_p, root_R, q):
        N = q.shape[0]
        P = np.zeros((N, self.B, 3)); R = np.zeros((N, self.B, 3, 3))
        jaxis = np.zeros((N, self.nq, 3)); jorg = np.zeros((N, self.nq, 3))
        for b in range(self.B):
            p = self.parent[b]
            if p < 0:
                Rb = root_R @ self.rot[b]; Pb = root_p + 0 * self.pos[b]
            else:
                Rb = R[:, p] @ self.rot[b]; Pb = P[:, p] + R[:, p] @ self.pos[b]
            ji = self.jidx[b]
            if ji >= 0:
                jaxis[:, ji] = Rb @ self.axis[b]; jorg[:, ji] = Pb
                Rb = Rb @ axis_rot(self.axis[b], q[:, ji])
            R[:, b] = Rb; P[:, b] = Pb
        kp = P[:, self.kp_body] + np.einsum("nkij,kj->nki", R[:, self.kp_body], self.kp_off)
        return dict(P=P, R=R, kp=kp, jaxis=jaxis, jorg=jorg)

    # ------------------------------------------------------------------ retarget
    def targets(self, human: np.ndarray) -> np.ndarray:
        """Scale a human take (N, 19, 3) to the robot's proportions and put its feet on the robot's floor."""
        leg = np.median(np.linalg.norm(human[:, KP["l_hip"]] - human[:, KP["l_knee"]], axis=1)
                        + np.linalg.norm(human[:, KP["l_knee"]] - human[:, KP["l_ankle"]], axis=1))
        s = self.leg / leg
        t = human * s
        ank = np.minimum(t[:, KP["l_ankle"], 2], t[:, KP["r_ankle"], 2])
        t[..., 2] += self.sole - np.percentile(ank, 5)      # ankle axis height above the sole
        # Proportions differ above the hips (G1 has a long pelvis and a short neck), so each torso
        # point is lifted by the static height difference measured on upright frames; arms move
        # with their shoulder. Motion is kept, only the fixed offset changes.
        hip = 0.5 * (t[:, KP["l_hip"]] + t[:, KP["r_hip"]])
        rest = self.rest["kp"][0]; rest_hip = 0.5 * (rest[KP["l_hip"]] + rest[KP["r_hip"]])
        up = t[:, KP["neck"]] - hip; upright = up[:, 2] / np.linalg.norm(up, axis=1) > 0.9
        if upright.sum() >= 10:
            groups = {"spine": ["spine"], "chest": ["chest"], "neck": ["neck"], "head": ["head"],
                      "l_shoulder": ["l_shoulder", "l_elbow", "l_wrist"], "r_shoulder": ["r_shoulder", "r_elbow", "r_wrist"]}
            for k, members in groups.items():
                dz = (rest[KP[k], 2] - rest_hip[2]) - np.median((t[upright, KP[k], 2] - hip[upright, 2]))
                for m in members:
                    t[:, KP[m], 2] += dz
        return t

    def solve(self, tgt, iters=8, init=None, smooth_to=None, mu=0.0):
        N = tgt.shape[0]; A = self.ACTIVE; nv = 6 + len(A)
        if init is None:
            q = np.zeros((N, self.nq))
            y = tgt[:, KP["l_hip"]] - tgt[:, KP["r_hip"]]; y = y / np.linalg.norm(y, axis=1, keepdims=True)
            up = (tgt[:, KP["neck"]] - tgt[:, KP["pelvis"]]); up -= (up * y).sum(1, keepdims=True) * y
            z = up / np.linalg.norm(up, axis=1, keepdims=True)
            Rr = np.stack([np.cross(y, z), y, z], axis=2)
            pr = 0.5 * (tgt[:, KP["l_hip"]] + tgt[:, KP["r_hip"]]) + np.array([0, 0, 0.12])
        else:
            pr, Rr, q = (a.copy() for a in init)
        W = np.repeat(self.w, 3)
        lam = 1e-2
        reg = np.full(self.nq, 1e-3); reg[[12, 13, 14]] = 5e-2   # waist (same indices on G1 and H2)
        reg = reg[A]
        I = np.eye(nv)[None]
        for it in range(iters):
            f = self.fk(pr, Rr, q)
            r = (tgt - f["kp"]).reshape(N, -1)
            Jq = np.cross(f["jaxis"][:, None, A, :], f["kp"][:, :, None, :] - f["jorg"][:, None, A, :])
            Jq = Jq * self.anc[None][:, :, A, None]
            Jw = np.cross(np.eye(3)[None, None, :, :], (f["kp"] - pr[:, None])[:, :, None, :])
            Jp = np.broadcast_to(np.eye(3), (N, len(KEYPOINTS), 3, 3))
            J = np.concatenate([Jp, Jw, Jq], axis=2).transpose(0, 1, 3, 2).reshape(N, -1, nv)
            JW = J * W[None, :, None]
            JWt = JW.transpose(0, 2, 1)
            H = np.matmul(JWt, J)
            g = np.matmul(JWt, r[..., None])[..., 0]
            D = np.concatenate([np.zeros(6), reg])
            g[:, 6:] -= reg * q[:, A]
            if smooth_to is not None:
                D = D + np.concatenate([np.zeros(6), np.full(len(A), mu)])
                g[:, 6:] += mu * (smooth_to[:, A] - q[:, A])
            H = H + (D + lam)[None, :, None] * I
            dlt = np.linalg.solve(H, g[..., None])[..., 0]
            pr = pr + dlt[:, :3]; Rr = so3_exp(dlt[:, 3:6]) @ Rr
            q[:, A] = np.clip(q[:, A] + dlt[:, 6:], self.lo[A], self.hi[A])
        f = self.fk(pr, Rr, q)
        err = np.linalg.norm(tgt - f["kp"], axis=2)
        return (pr, Rr, q), err

    def retarget(self, human, iters=(8, 3)):
        return self.retarget_many([human], iters)[0]

    def retarget_many(self, takes, iters=(8, 3)):
        """Retarget a list of takes in one batch (frames are independent until smoothing)."""
        tgts = [self.targets(h.astype(np.float64)) for h in takes]
        lens = np.array([len(t) for t in tgts]); cut = np.cumsum(lens)[:-1]
        tgt = np.concatenate(tgts)
        (pr, Rr, q), _ = self.solve(tgt, iters=iters[0])
        k = np.array([1, 4, 6, 4, 1], float); k /= k.sum()
        qs = np.concatenate([np.stack([np.convolve(np.pad(qq[:, j], 2, mode="edge"), k, mode="valid")
                                       for j in range(self.nq)], 1) for qq in np.split(q, cut)])
        (pr, Rr, q), err = self.solve(tgt, iters=iters[1], init=(pr, Rr, qs), smooth_to=qs, mu=0.5)
        return [((a, b, c), e, t) for a, b, c, e, t in zip(np.split(pr, cut), np.split(Rr, cut), np.split(q, cut),
                                                           np.split(err, cut), np.split(tgt, cut))]


class G1(Humanoid):
    ACTIVE = np.array(active_joints("g1"))   # kept as a class attribute for older callers

    def __init__(self, mjcf_path: str | None = None):
        super().__init__("g1", mjcf_path)


def make(robot: str) -> Humanoid:
    return G1() if robot == "g1" else Humanoid(robot)
