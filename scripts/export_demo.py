"""Collect everything the demo page shows into demo/data.json (real numbers only).

    python scripts/export_demo.py
Sections that have not been produced yet are written as null so the page can say so.
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine import canon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON, G1DIR, RUNS = "/home/claude/data/canon", "/home/claude/data/g1", os.path.join(ROOT, "runs")
SOURCES = {
    "cmu": dict(name="CMU Graphics Lab Motion Capture Database", fmt="BVH (B. Hahne conversion of ASF/AMC)", fps_in=120,
                license="Free to use, including in commercial products; the data itself may not be resold",
                url="http://mocap.cs.cmu.edu"),
    "bandai1": dict(name="Bandai Namco Research Motion Dataset 1", fmt="BVH", fps_in=30, license="CC BY-NC 4.0",
                    url="https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset"),
    "bandai2": dict(name="Bandai Namco Research Motion Dataset 2", fmt="BVH", fps_in=30, license="CC BY-NC 4.0",
                    url="https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset"),
}
PICKS = [("cmu", "walk"), ("cmu", "run"), ("bandai2", "wave-both-hands"), ("cmu", "jump"),
         ("bandai2", "walk_elderly"), ("cmu", "dance"), ("bandai1", "dance"), ("cmu", "box")]


def cm(a):
    return np.round(np.asarray(a) * 100).astype(int).ravel().tolist()


PI_ROWS = [("direct4", "Predict the next 200 ms directly (original)", "this run"),
           ("resid4", "Predict offsets from the current frame", "Martinez et al. 2017"),
           ("chunk16", "Offsets as 0.8 s chunks: play 0.4 s, re-plan", "π0, Black et al. 2024; ACT, Zhao et al. 2023"),
           ("dct16", "The same chunk as 6 cosine coefficients per channel", "π0-FAST, Pertsch et al. 2025")]


def pi_pass():
    d = os.path.join(RUNS, "pi_pass")
    recs = {n: json.load(open(os.path.join(d, n + ".json"))) for n, _, _ in PI_ROWS if os.path.exists(os.path.join(d, n + ".json"))}
    if not recs:
        return None
    hold = next(iter(recs.values()))["hold_pose"]
    rows = [dict(key="hold", label="Hold the last pose", **{"from": "baseline"}, **{k: hold[k] for k in ("0.5s", "1.0s", "2.0s")},
                 mean=hold["mean_2s"], real200=None)]
    for key, label, src in PI_ROWS:
        if key not in recs:
            continue
        r = recs[key]; e = r[f"execute_{r['execute']}"]
        rows.append(dict(key=key, label=label, **{"from": src}, **{k: e[k] for k in ("0.5s", "1.0s", "2.0s")},
                         mean=e["mean_2s"], real200=r["from_real_200ms"]))
    best = min(rows[1:], key=lambda r: r["mean"]); best["best"] = True
    base = next((r for r in rows if r["key"] == "direct4"), None)
    text = None
    runner = sorted(rows[1:], key=lambda r: r["mean"])[1] if len(rows) > 2 else None
    if base and best["key"] == "dct16" and runner and runner["key"] == "chunk16" and abs(runner["mean"] - best["mean"]) < 0.5:
        text = (f"Predicting offsets as 0.8 s chunks (π0) cut the free-running error over 2 s from {base['mean']:.1f}° to "
                f"{runner['mean']:.1f}°, {100 * (1 - runner['mean'] / hold['mean_2s']):.0f}% better than holding the pose. Storing each chunk "
                f"as 6 cosine coefficients per channel (π0-FAST) matched it at {best['mean']:.1f}° with 62% fewer outputs.")
    elif base and best["key"] != "direct4":
        text = (f"{best['label']} ({best['from']}) cut the free-running error over 2 s from {base['mean']:.1f}° to "
                f"{best['mean']:.1f}°, now {100 * (1 - best['mean'] / hold['mean_2s']):.0f}% better than holding the pose, "
                f"under the same training budget.")
    return dict(variants=rows, best=text, complete=len(recs) == len(PI_ROWS),
                caption="Size S (114k parameters), 900 steps, same data, seed and sensor masking for every row. Error is the "
                        "joint-angle RMSE of the model running on its own, on 200 windows from people never trained on; the last "
                        "column predicts 200 ms from real frames. FAST's tokenizer and π0's flow-matching head are left for the GPU run.")


def main():
    man = json.load(open(os.path.join(CANON, "manifest.json")))
    s = man["summary"]; clips = man["clips"]
    kept = [c for c in clips if "reject" not in c]
    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), skeleton=dict(keypoints=canon.KEYPOINTS,
               edges=[[canon.KP[a], canon.KP[b]] for a, b in canon.EDGES]))
    split_h = {sp: sum(c["frames"] for c in kept if c["split"] == sp) / canon.FPS / 3600 for sp in ("train", "val", "test")}
    out["dataset"] = dict(takes_in=s["takes_in"], takes_kept=s["takes_kept"], frames=s["frames"], hours=s["hours"],
                          fps=canon.FPS, people=len({c["subject"] for c in kept if c["source"] == "cmu"}) + 3,
                          people_note="CMU subjects plus the 3 Bandai Namco actors", rejects=s["rejects"],
                          ingest_seconds=s["seconds_wall"], splits_hours=split_h,
                          sources=[dict(id=k, **SOURCES[k], **s["per_source"][k]) for k in ("cmu", "bandai1", "bandai2")])
    d = np.load(os.path.join(CANON, "keypoints.npz")); kp, off = d["kp"], d["offsets"]
    # robot solutions (primary H2, second embodiment G1), if retargeting has finished
    from pine.g1 import make, SPECS
    out["robot"] = dict(id="h2", name=SPECS["h2"]["name"], dof=SPECS["h2"]["dof"], height_m=SPECS["h2"]["height_m"], solved_joints=23)
    out["robot2"] = dict(id="g1", name=SPECS["g1"]["name"], dof=SPECS["g1"]["dof"], height_m=SPECS["g1"]["height_m"], solved_joints=23)
    sols, robots, out["retarget"] = {}, {}, {}
    for rid in ("h2", "g1"):
        parts = sorted(glob.glob(os.path.join(f"/home/claude/data/{rid}", "part*.npz")))
        if not parts:
            out["retarget"][rid] = None; continue
        robots[rid] = make(rid); sols[rid] = {}; errs = []
        for p in parts:
            z = dict(np.load(p)); o = z["offsets"]; errs.append(z["err"])
            for j, idx in enumerate(z["idx"]):
                sols[rid][int(idx)] = (z["pr"][o[j]:o[j + 1]], z["R"][o[j]:o[j + 1]], z["q"][o[j]:o[j + 1]])  # views of one load
        e = np.concatenate(errs)
        r = dict(robot=SPECS[rid]["name"], joints=SPECS[rid]["dof"], solved_joints=23, frames=int(len(e)),
                 mean_err_cm=float(e.mean() * 100), feet_err_cm=float(e[:, 15:19].mean() * 100),
                 hands_err_cm=float(e[:, 11:13].mean() * 100), p95_err_cm=float(np.percentile(e, 95) * 100))
        tp = os.path.join(f"/home/claude/data/{rid}", "timing.json")
        if os.path.exists(tp):
            r.update({k: v for k, v in json.load(open(tp)).items() if k in ("ms_per_frame", "wall_s", "workers")})
        out["retarget"][rid] = r
    # playback clips: held-out people first, so what you watch was never trained on
    shown, used = [], set()
    for src, key in PICKS:
        cand = [c for c in kept if c["source"] == src and key.replace("_", " ") in (c["desc"] + " " + c["clip"]).replace("_", " ").lower()
                and c["index"] not in used and c["frames"] >= 80]
        cand.sort(key=lambda c: (c["split"] != "test", c["split"] != "val", -min(c["frames"], 200)))
        if not cand:
            continue
        c = cand[0]; used.add(c["index"])
        h = kp[off[c["index"]]:off[c["index"] + 1]][:200]
        item = dict(id=f"{c['source']}/{c['clip']}", source=c["source"], desc=c["desc"], split=c["split"],
                    frames=len(h), human=cm(h))
        for rid, key in (("h2", "robot"), ("g1", "g1")):
            if rid in sols and c["index"] in sols[rid]:
                pr, R, q = sols[rid][c["index"]]
                f = robots[rid].fk(pr[:200].astype(np.float64), R[:200].astype(np.float64), q[:200].astype(np.float64))
                item[key] = cm(f["kp"])
                if rid == "h2":   # what the 3D robot needs: root position, root quaternion (w x y z), 31 joint angles
                    Rm = R[:200].astype(np.float64)
                    w = np.sqrt(np.maximum(1e-12, 1 + Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2])) / 2
                    quat = np.stack([w, (Rm[:, 2, 1] - Rm[:, 1, 2]) / (4 * w), (Rm[:, 0, 2] - Rm[:, 2, 0]) / (4 * w),
                                     (Rm[:, 1, 0] - Rm[:, 0, 1]) / (4 * w)], 1)
                    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
                    item["h2_root"] = np.round(np.concatenate([pr[:200], quat], 1), 4).ravel().tolist()
                    item["h2_q"] = np.round(q[:200], 4).ravel().tolist()
        shown.append(item)
    # annotations: per-take labels and per-frame contacts for the clips on the page
    ap = os.path.join(RUNS, "annotations.json")
    out["annotations"] = json.load(open(ap)) if os.path.exists(ap) else None
    if out["annotations"]:
        fa = np.load("/home/claude/data/annotations/frames.npz"); pos = {int(i): k for k, i in enumerate(fa["index"])}
        tk = {t["id"]: t for t in json.load(open("/home/claude/data/annotations/takes.json"))}
        for c in shown:
            t = tk.get(c["id"])
            if t:
                k = pos[t["index"]]; a = fa["offsets"][k]
                c["contacts"] = fa["contacts"][a:a + c["frames"]].ravel().tolist()
                c["activity"], c["grade"], c["style"] = t["activity"], t["grade"], t["style"]
    out["clips"] = shown
    # the wall: many real takes at once, each trimmed to its most active 4 s
    wall, seen = [], set(c["index"] for c in [] )
    K = canon.KP
    for kw in ["cartwheel", "dance", "boxing", "jump", "run", "kick", "golf", "basketball", "salsa", "sword",
               "climb", "crawl", "stretch", "flip", "throw", "punch", "march", "skip", "balance", "sit",
               "walk, elderly", "wave both hands", "walk, exhausted", "run, active", "jog", "soccer", "swing", "drink"]:
        cand = [c for c in kept if kw in c["desc"].lower() and c["frames"] >= 80 and c["index"] not in seen]
        if not cand:
            continue
        def energy(c):
            h = kp[off[c["index"]]:off[c["index"] + 1]]
            v = np.linalg.norm(np.diff(h[:, [K["l_wrist"], K["r_wrist"], K["l_ankle"], K["r_ankle"]]], axis=0), axis=2).sum(1)
            return v
        c = max(cand[:30], key=lambda c: energy(c).mean())
        v = energy(c); win = np.convolve(v, np.ones(80), "valid"); a = int(np.argmax(win))
        h = kp[off[c["index"]]:off[c["index"] + 1]][a:a + 80]
        seen.add(c["index"])
        wall.append(dict(id=f"{c['source']}/{c['clip']}", source=c["source"], desc=c["desc"], frames=len(h), human=cm(h)))
        if len(wall) == 24:
            break
    out["wall"] = wall
    # training runs and fits
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, "run_*.json")))]
    out["runs"] = runs or None
    fp = os.path.join(RUNS, "fits.json")
    out["fits"] = json.load(open(fp)) if os.path.exists(fp) else None
    xp = os.path.join(RUNS, "exo_demo.json")
    out["exo"] = json.load(open(xp)) if os.path.exists(xp) else None
    cp = os.path.join(RUNS, "channels.json")
    out["channels"] = json.load(open(cp)) if os.path.exists(cp) else None
    rp = os.path.join(RUNS, "rollout_demo.json")
    out["rollout"] = json.load(open(rp)) if os.path.exists(rp) else None
    out["pi_pass"] = pi_pass()
    mp = os.path.join(RUNS, "more", "mirror.json")
    out["mirror"] = json.load(open(mp)) if os.path.exists(mp) else None
    gp = os.path.join(ROOT, "gpu", "plan.json")
    out["gpu_plan"] = json.load(open(gp)) if os.path.exists(gp) else None
    os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
    path = os.path.join(ROOT, "demo", "data.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(path, f"{os.path.getsize(path)/1e3:.0f} KB", "clips:", len(shown), "retarget:", {k: bool(v) for k, v in out["retarget"].items()}, "runs:", len(runs))


if __name__ == "__main__":
    main()
