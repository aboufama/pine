"""Double the training data by mirroring every take, verify the mirror on the robot itself, then retrain.

Mirroring is exact for a left-right symmetric robot: swap left and right joints, negate roll and yaw angles,
negate the sideways velocity, turn rate and sideways gravity. The check: forward kinematics of the mirrored
joint angles must equal the reflection of the original keypoints (with left and right labels swapped).
Training uses the same budget as the sweep (900 steps), the same normalisation, and the same held-out sets.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine import features as F, train_np            # noqa: E402
from pine.features import MotionData               # noqa: E402
from pine.g1 import make                           # noqa: E402
from pine.canon import KEYPOINTS                   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "more")
os.makedirs(OUT, exist_ok=True)


def joint_mirror(names):
    perm, sign = [], []
    for n in names:
        m = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        perm.append(names.index(m)); sign.append(-1.0 if ("roll" in n or "yaw" in n) else 1.0)
    return np.array(perm), np.array(sign)


def verify(robot, z):
    """FK(mirror(q)) vs reflect(FK(q)) on 2,000 real frames."""
    pr, R, q = z["pr"][:2000].astype(float), z["R"][:2000].astype(float), z["q"][:2000].astype(float)
    perm, sign = joint_mirror(robot.joints)
    M = np.diag([1.0, -1.0, 1.0])
    qm = q[:, perm] * sign
    km = robot.fk(pr @ M, M @ R @ M, qm)["kp"]
    k = robot.fk(pr, R, q)["kp"] @ M
    kperm = [KEYPOINTS.index(n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")) for n in KEYPOINTS]
    return float(np.abs(km - k[:, kperm]).max())


def feature_mirror(X):
    names = F.NAMES
    perm, sign = joint_mirror(names[:23])
    Y = X.copy()
    Y[:, :23] = X[:, :23][:, perm] * sign
    for n in ("grav_y", "yaw_rate", "vel_left"):
        Y[:, names.index(n)] *= -1
    return Y


def main():
    robot = make(F.ROBOT)
    z = dict(np.load(sorted(glob.glob(f"/home/claude/data/{F.ROBOT}/part*.npz"))[0]))
    err = verify(robot, z)
    print(f"mirror check on {robot.spec['name']}: max keypoint difference {err*1000:.3f} mm", flush=True)
    assert err < 1e-3, "robot is not mirror-symmetric under this mapping"
    d = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    tr = d.subset("train")
    starts, lens = list(d.starts), list(d.lens)
    raws = [d.raw]; split = list(d.split); clips = list(d.clips); off = len(d.raw)
    for i in tr:
        s, n = d.starts[i], d.lens[i]
        raws.append(feature_mirror(d.raw[s:s + n])); starts.append(off); lens.append(n); off += n
        split.append("train"); clips.append(dict(d.clips[i], clip=d.clips[i]["clip"] + "_mirror"))
    d.raw = np.concatenate(raws).astype(np.float32); d.X = ((d.raw - d.mean) / d.std).astype(np.float32)
    d.starts, d.lens, d.split, d.clips = np.array(starts), np.array(lens), np.array(split), clips
    hours = d.hours(d.subset("train"))
    print(f"training hours with mirrors: {hours:.1f}", flush=True)
    res = dict(check_mm=err * 1000, train_hours=hours, runs={})
    for size in ("S", "M"):
        rec, _ = train_np.run(d, size=size, fraction=1.0, steps=900, log=lambda s: print(s, flush=True))
        base = json.load(open(os.path.join(OUT, "..", f"run_{size}_1.0000.json")))
        res["runs"][size] = dict(params=rec["params"], val=rec["val"]["full"], test=rec["test"]["full"],
                                 hidden_deg=rec["test"]["hidden_joint_rmse_deg"],
                                 base_val=base["val"]["full"], base_test=base["test"]["full"],
                                 base_hidden_deg=base["test"]["hidden_joint_rmse_deg"])
        r = res["runs"][size]
        print(f"   {size}: val {r['base_val']:.4f} -> {r['val']:.4f} | new people {r['base_test']:.4f} -> {r['test']:.4f} | "
              f"hidden joints {r['base_hidden_deg']:.1f} -> {r['hidden_deg']:.1f} deg", flush=True)
        json.dump(res, open(os.path.join(OUT, "mirror.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
