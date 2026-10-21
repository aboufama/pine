"""Canonical human keypoints: every source is mapped onto the same 19 points, the same
frame rate (20 fps), metres, Z up, feet on the floor. Quality checks run here, once,
so the model never sees a glitch twice.
"""
from __future__ import annotations
import numpy as np
from .mocap_io import Take

KEYPOINTS = ["pelvis", "l_hip", "r_hip", "spine", "chest", "neck", "head",
             "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
             "l_knee", "r_knee", "l_ankle", "r_ankle", "l_toe", "r_toe"]
KP = {k: i for i, k in enumerate(KEYPOINTS)}
EDGES = [("pelvis", "l_hip"), ("pelvis", "r_hip"), ("pelvis", "spine"), ("spine", "chest"),
         ("chest", "neck"), ("neck", "head"), ("chest", "l_shoulder"), ("chest", "r_shoulder"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe")]

# Source skeleton -> canonical keypoint. Joint positions in BVH are where a bone starts;
# in ASF/AMC (CMU) they are where a bone ends, hence the different names.
MAPS = {
    "bandai": dict(pelvis="Hips", l_hip="UpperLeg_L", r_hip="UpperLeg_R", spine="Spine", chest="Chest",
                   neck="Neck", head="Head", l_shoulder="UpperArm_L", r_shoulder="UpperArm_R",
                   l_elbow="LowerArm_L", r_elbow="LowerArm_R", l_wrist="Hand_L", r_wrist="Hand_R",
                   l_knee="LowerLeg_L", r_knee="LowerLeg_R", l_ankle="Foot_L", r_ankle="Foot_R",
                   l_toe="Toes_L", r_toe="Toes_R"),
    "cmu": dict(pelvis="root", l_hip="lhipjoint", r_hip="rhipjoint", spine="lowerback", chest="upperback",
                neck="thorax", head="upperneck", l_shoulder="lclavicle", r_shoulder="rclavicle",
                l_elbow="lhumerus", r_elbow="rhumerus", l_wrist="lradius", r_wrist="rradius",
                l_knee="lfemur", r_knee="rfemur", l_ankle="ltibia", r_ankle="rtibia",
                l_toe="lfoot", r_toe="rfoot"),
    # Mixamo / LAFAN1 / Noitom-style names (LAFAN1, HiPHI and most studio BVH)
    "mixamo": dict(pelvis="Hips", l_hip="LeftUpLeg", r_hip="RightUpLeg", spine="Spine", chest="Spine2",
                   neck="Neck", head="Head", l_shoulder="LeftArm", r_shoulder="RightArm",
                   l_elbow="LeftForeArm", r_elbow="RightForeArm", l_wrist="LeftHand", r_wrist="RightHand",
                   l_knee="LeftLeg", r_knee="RightLeg", l_ankle="LeftFoot", r_ankle="RightFoot",
                   l_toe="LeftToe", r_toe="RightToe"),
}
ALIASES = {"LeftToe": ["LeftToeBase", "LeftToe_End"], "RightToe": ["RightToeBase", "RightToe_End"],
           "Spine2": ["Spine1", "Chest", "Spine"]}

FPS = 20.0


def to_canonical(take: Take, skel: str) -> np.ndarray:
    """(N, J, 3) source joints -> (N, 19, 3) canonical keypoints."""
    idx = {n.split(":")[-1]: i for i, n in enumerate(take.names)}
    out = []
    for k in KEYPOINTS:
        name = MAPS[skel][k]
        cands = [name] + ALIASES.get(name, [])
        j = next((idx[c] for c in cands if c in idx), None)
        if j is None:
            raise KeyError(f"{take.source}/{take.clip}: no joint for {k} ({name})")
        out.append(take.pos[:, j])
    return np.stack(out, 1).astype(np.float64)


def resample(x: np.ndarray, fps_in: float, fps_out: float = FPS) -> np.ndarray:
    n = x.shape[0]
    t_in = np.arange(n) / fps_in
    t_out = np.arange(0, t_in[-1] + 1e-9, 1.0 / fps_out)
    i = np.clip(np.searchsorted(t_in, t_out) - 1, 0, n - 2)
    w = ((t_out - t_in[i]) * fps_in)[:, None, None]
    return x[i] * (1 - w) + x[i + 1] * w


def facing(kp: np.ndarray) -> np.ndarray:
    """Per-frame unit forward vector in the ground plane (from the hip and shoulder axes)."""
    across = (kp[:, KP["l_hip"]] - kp[:, KP["r_hip"]]) + (kp[:, KP["l_shoulder"]] - kp[:, KP["r_shoulder"]])
    fwd = np.stack([across[:, 1], -across[:, 0], np.zeros(len(kp))], 1)   # left x up
    return fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)


def qa_and_normalise(kp: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Floor the take, check it, and return (keypoints, report). None means rejected."""
    rep: dict = {"frames": int(len(kp))}
    if len(kp) < 2 * FPS:
        rep["reject"] = "shorter than 2 s"; return None, rep
    if not np.isfinite(kp).all():
        rep["reject"] = "non-finite values"; return None, rep
    feet = kp[:, [KP["l_toe"], KP["r_toe"], KP["l_ankle"], KP["r_ankle"]], 2]
    kp = kp.copy(); kp[..., 2] -= np.percentile(feet.min(1), 5)
    thigh = np.linalg.norm(kp[:, KP["l_hip"]] - kp[:, KP["l_knee"]], axis=1)
    shin = np.linalg.norm(kp[:, KP["l_knee"]] - kp[:, KP["l_ankle"]], axis=1)
    rep["leg_m"] = float(np.median(thigh + shin))
    rep["bone_cv"] = float(max(thigh.std() / thigh.mean(), shin.std() / shin.mean()))
    if rep["bone_cv"] > 0.05:
        rep["reject"] = "bone lengths drift (marker swap or glitch)"; return None, rep
    if not 0.55 < rep["leg_m"] < 1.15:
        rep["reject"] = f"implausible leg length {rep['leg_m']:.2f} m"; return None, rep
    v = np.linalg.norm(np.diff(kp, axis=0), axis=2) * FPS
    rep["max_speed"] = float(v.max())
    if rep["max_speed"] > 20.0:
        rep["reject"] = f"teleport ({rep['max_speed']:.0f} m/s)"; return None, rep
    # left must be on the left: (l_hip - r_hip) . (up x forward) > 0 on most frames
    f = facing(kp)
    left = np.cross(np.array([0, 0, 1.0]), f)
    side = np.einsum("ni,ni->n", kp[:, KP["l_hip"]] - kp[:, KP["r_hip"]], left)
    rep["lr_ok"] = float((side > 0).mean())
    rep["pelvis_h"] = float(np.median(kp[:, 0, 2]))
    rep["speed"] = float(np.median(np.linalg.norm(np.diff(kp[:, 0, :2], axis=0), axis=1)) * FPS)
    return kp.astype(np.float32), rep
