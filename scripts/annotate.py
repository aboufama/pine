"""Annotate every take and frame, and check the annotations against independent evidence.

Per take:  activity (taxonomy from the capture notes), style (when the source gives one), quality grade
           (retarget error, joint-limit saturation, foot skate) and the QA history.
Per frame: left/right foot contact, pelvis speed, locomotion state (still / walking / running).
Checks:    contacts are computed twice, on the human and on the retargeted robot, from different skeletons;
           their agreement says whether retargeting kept the footfalls. A seeded random sample of activity
           labels is printed beside its source description for review.
"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.canon import KP, FPS                 # noqa: E402
from pine.g1 import make                       # noqa: E402
from pine import features as F                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAXONOMY = [  # first match wins
    ("unlabeled", r"^\s*$|^unknown"),
    ("dance", r"danc|salsa|waltz|ballet|jete|pirouette|tango|charleston|lambada|break ?sequence|indian"),
    ("pantomime and story", r"story|vacation|nursery|rhyme|prairie|bear|snake|\bcat\b|elephant|mouse|\bdog\b|chicken|whale|penguin|dragon|hummingbird|pterosaur|\bfish\b|animal|motorcycle|mime|charade"),
    ("fighting", r"box|punch|kick(?! ball)|fight|martial|karate|kung|sword|taekwondo|attack|slash|block|jab"),
    ("sport", r"basketball|soccer|football|golf|baseball|tennis|throw|catch|volley|frisbee|bowl|dribbl|kick ball|pitch|swim|stroke|free style|putt|\btee\b|fishing|swing"),
    ("acrobatics", r"cartwheel|flip|handstand|somersault|tumbl|acrobat|balance|beam"),
    ("exercise", r"yoga|stretch|range of motion|squat|push ?up|sit ?up|jumping jack|exercise|lunge|tai chi"),
    ("climb and crawl", r"climb|crawl|ladder|playground|hang|monkey"),
    ("jump", r"jump|hop|leap|skip|\b180\b|\b360\b"),
    ("run", r"run|jog|sprint|dash"),
    ("walk", r"walk|stroll|march|step|turn|navigate|stairs|limp|slope"),
    ("gesture and arms", r"wave|raise|gesture|point|clap|conduct|hand|arm|salute|beckon|bow|bye|call|respond|guide|signal|direct"),
    ("everyday", r"sit|stand|lift|carry|pick|drink|eat|wash|sweep|open|push|pull|chair|stool|vignette|yawn|bend|kneel|lie|get up|mop|clean|cook|phone|read|write|lbs|look|laugh|scared|sneak|reach"),
]
STYLES = r"active|elderly|exhausted|feminine|masculine|normal|youthful|angry|anger|happy|sad|tired|proud|careful|drunk|childlike|zombie"


def activity(desc):
    d = desc.lower().replace("-", " ")
    for name, pat in TAXONOMY:
        if re.search(pat, d):
            return name
    return "other"


def foot_down(kp, toe, ank):
    """A foot is down when its lowest point is within 6 cm of that foot's floor level and its ankle is nearly still."""
    low = np.minimum(kp[:, toe, 2], kp[:, ank, 2])
    v = np.r_[0, np.linalg.norm(np.diff(kp[:, ank, :2], axis=0), axis=1) * FPS]
    return (low - np.percentile(low, 5) < 0.06) & (v < 0.6)


def both_feet(kp):
    return np.stack([foot_down(kp, KP["l_toe"], KP["l_ankle"]), foot_down(kp, KP["r_toe"], KP["r_ankle"])], 1)


def skate(kp, c, ank):
    v = np.r_[0, np.linalg.norm(np.diff(kp[:, ank, :2], axis=0), axis=1) * FPS]
    return float(v[c].mean() * 100) if c.any() else 0.0


def main():
    man = json.load(open("/home/claude/data/canon/manifest.json"))
    kept = {c["index"]: c for c in man["clips"] if "reject" not in c}
    d = np.load("/home/claude/data/canon/keypoints.npz"); kp, off = d["kp"], d["offsets"]
    robot = make(F.ROBOT)
    lo, hi = robot.lo[robot.ACTIVE], robot.hi[robot.ACTIVE]
    takes, agree_n, agree_d, sk_h, sk_r = [], 0, 0, [], []
    frames_contact, frames_state = {}, {}
    for p in sorted(glob.glob(f"/home/claude/data/{F.ROBOT}/part*.npz")):
        z = dict(np.load(p)); o = z["offsets"]
        for j, idx in enumerate(z["idx"]):
            c = kept[int(idx)]; a, b = o[j], o[j + 1]
            h = kp[off[idx]:off[idx + 1]].astype(np.float64)
            pr, R, q = z["pr"][a:b].astype(np.float64), z["R"][a:b].astype(np.float64), z["q"][a:b].astype(np.float64)
            rk = robot.fk(pr, R, q)["kp"]
            ch_, cr = both_feet(h), both_feet(rk)
            agree_n += int((ch_ == cr).sum()); agree_d += ch_.size
            sh = np.mean([skate(h, ch_[:, 0], KP["l_ankle"]), skate(h, ch_[:, 1], KP["r_ankle"])])
            sr = np.mean([skate(rk, cr[:, 0], KP["l_ankle"]), skate(rk, cr[:, 1], KP["r_ankle"])])
            sk_h.append(sh); sk_r.append(sr)
            speed = np.r_[0, np.linalg.norm(np.diff(pr[:, :2], axis=0), axis=1) * FPS]
            state = np.where(speed < 0.25, 0, np.where(speed < 2.0, 1, 2))
            qa = q[:, robot.ACTIVE]
            pinned = (qa <= lo + np.radians(1)) | (qa >= hi - np.radians(1))
            ank = [k for k, jn in enumerate(np.array(robot.joints)[robot.ACTIVE]) if "ankle_roll" in jn]
            ankle_pinned = float(pinned[:, ank].any(1).mean())
            sat = float(np.delete(pinned, ank, axis=1).any(1).mean())   # the ankle-roll limit is reported on its own
            e = z["err"][a:b]; err, p95 = float(e.mean() * 100), float(np.percentile(e, 95) * 100)
            grade = "A" if (err < 4.5 and sr < 15 and sat < 0.05) else ("B" if (err < 6.5 and sr < 30 and sat < 0.2) else "C")
            sm = re.search(STYLES, c["desc"].lower())
            takes.append(dict(index=int(idx), id=f"{c['source']}/{c['clip']}", source=c["source"], subject=c["subject"], split=c["split"],
                              desc=c["desc"], activity=activity(c["desc"]), style=sm.group(0) if sm else None,
                              frames=int(b - a), err_cm=err, p95_cm=p95, limit_saturation=sat, ankle_roll_pinned=ankle_pinned, skate_human=sh, skate_robot=sr,
                              contact_rate=float(cr.mean()), still=float((state == 0).mean()), walking=float((state == 1).mean()),
                              running=float((state == 2).mean()), grade=grade))
            frames_contact[int(idx)] = cr; frames_state[int(idx)] = state
    os.makedirs("/home/claude/data/annotations", exist_ok=True)
    json.dump(takes, open("/home/claude/data/annotations/takes.json", "w"))
    order = sorted(frames_contact)
    np.savez_compressed("/home/claude/data/annotations/frames.npz", index=np.array(order),
                        contacts=np.concatenate([frames_contact[i] for i in order]).astype(np.uint8),
                        state=np.concatenate([frames_state[i] for i in order]).astype(np.uint8),
                        offsets=np.r_[0, np.cumsum([len(frames_state[i]) for i in order])])
    hrs = lambda ts: sum(t["frames"] for t in ts) / FPS / 3600
    acts = {}
    for t in takes:
        acts.setdefault(t["activity"], []).append(t)
    grades = {g: [t for t in takes if t["grade"] == g] for g in "ABC"}
    rng = np.random.default_rng(42); sample = [takes[i] for i in rng.choice(len(takes), 24, replace=False)]
    summary = dict(
        takes=len(takes), hours=hrs(takes),
        activity=[dict(name=k, takes=len(v), hours=hrs(v)) for k, v in sorted(acts.items(), key=lambda kv: -hrs(kv[1]))],
        styles=sorted({t["style"] for t in takes if t["style"]}), styled_takes=sum(1 for t in takes if t["style"]),
        grades={g: dict(takes=len(v), hours=hrs(v)) for g, v in grades.items()},
        contact_agreement=agree_n / agree_d, skate_human_cm_s=float(np.median(sk_h)), skate_robot_cm_s=float(np.median(sk_r)),
        err_cm_median=float(np.median([t["err_cm"] for t in takes])), saturation_median=float(np.median([t["limit_saturation"] for t in takes])),
        ankle_roll_pinned=float(np.average([t["ankle_roll_pinned"] for t in takes], weights=[t["frames"] for t in takes])),
        locomotion=dict(still=float(np.average([t["still"] for t in takes], weights=[t["frames"] for t in takes])),
                        walking=float(np.average([t["walking"] for t in takes], weights=[t["frames"] for t in takes])),
                        running=float(np.average([t["running"] for t in takes], weights=[t["frames"] for t in takes]))),
        spot_check=[dict(desc=t["desc"], activity=t["activity"], style=t["style"]) for t in sample],
        per_frame=["23 robot joint angles", "gravity direction at the pelvis (3)", "turn rate", "pelvis velocity (3)", "pelvis height",
                   "19 human keypoints (x, y, z)", "left and right foot contact", "locomotion state"],
        per_take=["activity", "style", "source description", "person id", "quality grade", "retarget error", "joint-limit saturation",
                  "foot skate", "QA history", "split"])
    json.dump(summary, open(os.path.join(ROOT, "runs", "annotations.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("spot_check", "per_frame", "per_take")}, indent=1))
    for s in summary["spot_check"]:
        print(f"  {s['activity']:16s} | {s['style'] or '':10s} | {s['desc'][:70]}")


if __name__ == "__main__":
    main()
