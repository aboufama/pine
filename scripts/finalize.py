"""After the sweep: scaling fits (runs/fits.json) and the EXO-1 demo clip (runs/exo_demo.json)."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine import scaling, features as F, train_np          # noqa: E402
from pine.features import MotionData                        # noqa: E402
from pine.model_np import MotionTransformer                 # noqa: E402
from pine.g1 import make                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def runs():
    return [json.load(open(p)) for p in glob.glob(os.path.join(RUNS, "run_*.json"))]


def fits(rs):
    pred = json.load(open(os.path.join(RUNS, "prediction.json")))
    lad = sorted([r for r in rs if r["fraction"] == 1.0 and r["size"] in pred["fit_on"]], key=lambda r: r["params"])
    xl = next((r for r in rs if r["size"] == pred["size"] and r["fraction"] == 1.0), None)
    model = dict(form=pred["form"], E=pred["E"], A=pred["A"], alpha=pred["alpha"], fit_on=pred["fit_on"],
                 registered_at=pred["registered_at"],
                 points=[dict(size=r["size"], params=r["params"], loss=r["val"]["full"]) for r in lad],
                 prediction=dict(size=pred["size"], params=pred["params"], loss=pred["loss"], lo=pred["lo"], hi=pred["hi"]),
                 observed=dict(size=pred["size"], params=xl["params"], loss=xl["val"]["full"]) if xl else None)
    dl = sorted([r for r in rs if r["size"] == "S"], key=lambda r: r["train_hours"])
    x = [r["train_hours"] for r in dl]; y = [r["val"]["full"] for r in dl]
    E, B, b = scaling.fit(x, y)
    hrs = np.geomspace(min(x), 10000, 40)
    data = dict(form="L(D) = E + B * D^-beta", E=float(E), B=float(B), beta=float(b), size="S",
                points=[dict(hours=h, loss=l, fraction=r["fraction"]) for h, l, r in zip(x, y, dl)],
                curve=[dict(hours=float(h), loss=float(scaling.law(h, E, B, b))) for h in hrs])
    json.dump(dict(model=model, data=data), open(os.path.join(RUNS, "fits.json"), "w"), indent=1)
    return model, data


def exo_demo(size):
    data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    w = dict(np.load(os.path.join(RUNS, f"weights_{size}.npz")))
    d, L, H = train_np.SIZES[size]
    m = MotionTransformer(2 * F.D, 4 * F.D, d, L, H, 32); m.p = w; m._std = data.std
    robot = make(F.ROBOT)
    # a held-out person walking: arm swing has to be inferred from the legs alone
    cands = [i for i, c in enumerate(data.clips) if c["split"] == "test" and "walk" in c["desc"].lower() and data.lens[i] >= 140]
    i = max(cands[:40], key=lambda j: data.raw[data.starts[j]:data.starts[j] + data.lens[j], 15:23].std())
    s, n = data.starts[i], min(data.lens[i], 200)
    X = data.X[s:s + n]
    mask = np.ones(F.D, np.float32); mask[F.HIDDEN] = 0
    pred = X.copy()
    for t in range(1, n):                      # predict frame t from frames < t (suit channels only)
        a = max(0, t - 32); win = X[a:t] * mask
        inp = np.concatenate([win, np.broadcast_to(mask, win.shape)], -1)[None]
        y, _ = m.forward(inp)
        pred[t, F.HIDDEN] = y[0, -1, :F.D][F.HIDDEN]
    raw_t = X * data.std + data.mean; raw_p = pred * data.std + data.mean
    for p in sorted(glob.glob(f"/home/claude/data/{F.ROBOT}/part*.npz")):   # root pose from the stored solution
        z = np.load(p); hit = np.where(z["idx"] == data.clips[i]["index"])[0]
        if len(hit):
            o = z["offsets"]; a, b = o[hit[0]], o[hit[0]] + n
            pr, R = z["pr"][a:b].astype(np.float64), z["R"][a:b].astype(np.float64); break
    def q_of(raw):
        q = np.zeros((n, robot.nq)); q[:, F.ACTIVE] = raw[:, :23]; return q
    kt = robot.fk(pr, R, q_of(raw_t))["kp"]; kp_ = robot.fk(pr, R, q_of(raw_p))["kp"]
    hid = [j for j in F.HIDDEN if j < 23]
    err = np.degrees(raw_p[1:, hid] - raw_t[1:, hid])
    base = np.degrees(data.mean[hid] - raw_t[1:, hid])
    out = dict(robot=robot.spec["name"], clip_id=f"{data.clips[i]['source']}/{data.clips[i]['clip']}", desc=data.clips[i]["desc"], frames=int(n),
               model=size, observed=[F.NAMES[j] for j in F.EXO], hidden=[F.NAMES[j] for j in hid],
               truth_g1=np.round(kt * 100).astype(int).ravel().tolist(), pred_g1=np.round(kp_ * 100).astype(int).ravel().tolist(),
               rmse_deg=float(np.sqrt((err ** 2).mean())), meanpose_rmse_deg=float(np.sqrt((base ** 2).mean())))
    json.dump(out, open(os.path.join(RUNS, "exo_demo.json"), "w"))
    return out


def rot_from(yaw, grav):
    """Root rotation from heading and the gravity vector seen in the pelvis frame."""
    u = -grav / np.linalg.norm(grav); ez = np.array([0.0, 0.0, 1.0])
    k = np.cross(u, ez); s_, c_ = np.linalg.norm(k), float(u @ ez)
    K = np.zeros((3, 3)) if s_ < 1e-9 else (lambda a: np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]))(k / s_)
    Rrp = np.eye(3) + s_ * K + (1 - c_) * (K @ K)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]]) @ Rrp


def rollout_demo(size, context=32, horizon=40):
    """The model continues a held-out take on its own for 2 s, from 1.6 s of real motion."""
    data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    w = dict(np.load(os.path.join(RUNS, f"weights_{size}.npz")))
    d, L, H = train_np.SIZES[size]
    m = MotionTransformer(2 * F.D, 4 * F.D, d, L, H, 32); m.p = w
    robot = make(F.ROBOT)
    n = context + horizon
    cands = [i for i, c in enumerate(data.clips) if c["split"] == "test" and data.lens[i] >= n
             and any(k in c["desc"].lower() for k in ("walk", "run", "jog"))]
    i = max(cands[:60], key=lambda j: data.raw[data.starts[j]:data.starts[j] + n, 27].mean())
    s = data.starts[i]
    X = data.X[s:s + n].copy(); seq = X[:context].copy()
    ones = np.ones(F.D, np.float32)
    for t in range(horizon):
        win = seq[-32:]
        inp = np.concatenate([win, np.broadcast_to(ones, win.shape)], -1)[None]
        y, _ = m.forward(inp)
        seq = np.concatenate([seq, y[0, -1, :F.D][None]], 0)
    raw_t = X * data.std + data.mean; raw_p = seq * data.std + data.mean
    for p in sorted(glob.glob(f"/home/claude/data/{F.ROBOT}/part*.npz")):
        z = np.load(p); hit = np.where(z["idx"] == data.clips[i]["index"])[0]
        if len(hit):
            o = z["offsets"]; a = o[hit[0]]
            pr_t, R_t = z["pr"][a:a + n].astype(np.float64), z["R"][a:a + n].astype(np.float64); break
    # integrate the predicted root from the last real frame
    pr_p, R_p = pr_t.copy(), R_t.copy()
    yaw = float(np.arctan2(R_t[context - 1, 1, 0], R_t[context - 1, 0, 0])); pos = pr_t[context - 1].copy()
    for t in range(context, n):
        f = raw_p[t - 1]
        yaw += f[26] / 20.0
        c, s_ = np.cos(yaw), np.sin(yaw)
        pos = pos + np.array([c * f[27] - s_ * f[28], s_ * f[27] + c * f[28], 0.0]) / 20.0
        pos[2] = raw_p[t, 30]
        pr_p[t] = pos; R_p[t] = rot_from(yaw, raw_p[t, 23:26])
    def q_of(raw):
        q = np.zeros((n, robot.nq)); q[:, F.ACTIVE] = raw[:, :23]; return q
    kt = robot.fk(pr_t, R_t, q_of(raw_t))["kp"]; kp_ = robot.fk(pr_p, R_p, q_of(raw_p))["kp"]
    hold = np.repeat(kt[context - 1:context], horizon, 0)          # baseline: freeze the last real pose
    def err(a, b, k):  # pose error with the pelvis aligned, cm, at horizon frame k
        return float(np.linalg.norm((a[k] - a[k, :1]) - (b[k] - b[k, :1]), axis=-1).mean() * 100)
    ev = {f"{sec}s": dict(model_cm=err(kp_[context:], kt[context:], int(sec * 20) - 1),
                          freeze_cm=err(hold, kt[context:], int(sec * 20) - 1)) for sec in (0.5, 1.0, 2.0)}
    out = dict(robot=robot.spec["name"], model=size, params=int(m.n_params), clip_id=f"{data.clips[i]['source']}/{data.clips[i]['clip']}",
               desc=data.clips[i]["desc"], fps=20, context=context, horizon=horizon,
               truth=np.round(kt * 100).astype(int).ravel().tolist(), pred=np.round(kp_ * 100).astype(int).ravel().tolist(),
               pose_error=ev)
    json.dump(out, open(os.path.join(RUNS, "rollout_demo.json"), "w"))
    return out


def channel_report(size, n_windows=512):
    """What each channel is, its real range, and how much the trained model needs it."""
    data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    w = dict(np.load(os.path.join(RUNS, f"weights_{size}.npz")))
    d, L, H = train_np.SIZES[size]
    m = MotionTransformer(2 * F.D, 4 * F.D, d, L, H, 32); m.p = w
    tr = data.subset("train")
    idx = np.concatenate([np.arange(data.starts[i], data.starts[i] + data.lens[i]) for i in tr])
    raw = data.raw[idx]
    def unit(j):
        if j < 23: return "deg", np.degrees
        n = F.NAMES[j]
        if n.startswith("grav"): return "g", lambda x: x
        if n == "yaw_rate": return "deg/s", np.degrees
        if n.startswith("vel"): return "m/s", lambda x: x
        return "m", lambda x: x
    group_of = {j: g for g, js in F.GROUPS.items() for j in js}
    chans = []
    for j, name in enumerate(F.NAMES):
        u, f = unit(j)
        p = f(np.percentile(raw[:, j], [5, 50, 95]))
        chans.append(dict(name=name, group=group_of[j], unit=u, p5=float(p[0]), p50=float(p[1]), p95=float(p[2]),
                          suit=j in F.EXO, source="derived from the solved pose" if j >= 23 else "joint angle solved by retargeting"))
    vw = F.Windows(data, data.subset("val"), 32, 4).fixed(n_windows, seed=7)
    def loss_with(mask):
        tot = 0.0
        for i in range(0, len(vw), 128):
            b = vw[i:i + 128]
            inp, tgt = F.batch(b, 32, 4, np.broadcast_to(mask, (len(b), F.D)).astype(np.float32))
            y, _ = m.forward(inp); tot += float(((y - tgt) ** 2).mean()) * len(b)
        return tot / len(vw)
    full = np.ones(F.D, np.float32)
    base = loss_with(full)
    hidden = []
    for g, js in F.GROUPS.items():
        mk = full.copy(); mk[js] = 0
        hidden.append(dict(group=g, channels=len(js), loss=loss_with(mk)))
    def only(groups):
        mk = np.zeros(F.D, np.float32)
        for g in groups: mk[F.GROUPS[g]] = 1
        return loss_with(mk)
    suits = [dict(name="EXO-1 as assumed: leg encoders + pelvis IMU", groups=["legs", "imu"]),
             dict(name="Leg encoders only", groups=["legs"]), dict(name="Pelvis IMU only", groups=["imu"]),
             dict(name="Legs + IMU + arm encoders", groups=["legs", "imu", "arms"]),
             dict(name="Everything except the root", groups=["legs", "imu", "arms", "waist"])]
    for sv in suits:
        sv["loss"] = only(sv["groups"]); sv["channels"] = int(sum(len(F.GROUPS[g]) for g in sv["groups"]))
    out = dict(robot=make(F.ROBOT).spec["name"], model=size, fps=20, channels=chans, full_loss=base, nothing_loss=only([]),
               hide_one_group=hidden, suits=suits,
               groups={g: [F.NAMES[j] for j in js] for g, js in F.GROUPS.items()})
    json.dump(out, open(os.path.join(RUNS, "channels.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    rs = runs()
    mo, da = fits(rs)
    print("model fit", {k: mo[k] for k in ("E", "A", "alpha")}, "pred", mo["prediction"], "obs", mo["observed"])
    print("data fit", {k: da[k] for k in ("E", "B", "beta")})
    best = json.load(open(os.path.join(RUNS, "prediction.json")))["size"]
    ex = exo_demo(best)
    print("exo", ex["clip_id"], ex["desc"], f"rmse {ex['rmse_deg']:.1f} deg vs mean pose {ex['meanpose_rmse_deg']:.1f}")
    ch = channel_report(best)
    print("channels: full", round(ch["full_loss"], 4), {h["group"]: round(h["loss"], 4) for h in ch["hide_one_group"]},
          {sv["name"][:18]: round(sv["loss"], 4) for sv in ch["suits"]})
    ro = rollout_demo(best)
    print("rollout", ro["clip_id"], ro["desc"], ro["pose_error"])
