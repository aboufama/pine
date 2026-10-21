"""Compare the research variants with pi_pass chunk16 and with a second-seed re-run of chunk16 (the control).

  * vs pi_pass chunk16 (seed 0, same seed as the variants): aggregate numbers only (pi_pass kept no weights).
  * vs the control (seed 1): paired over the same 200 held-out windows, bootstrap over windows.
  * seed noise: control (seed 1) minus pi_pass chunk16 (seed 0), the run-to-run spread of the metric itself.
Light compute only: one forward pass per model for the 200 ms-from-real-frames numbers.
Writes runs/research/compare.json.
"""
import json, os, sys
import numpy as np
ROOT = "/home/claude/pine-engine"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pi_pass                                  # noqa: E402
from pine.features import MotionData            # noqa: E402
from pine import features as F, train_np        # noqa: E402
from pine.model_np import MotionTransformer     # noqa: E402

OUT = os.path.join(ROOT, "runs", "research")
OBJ = next(o for o in pi_pass.VARIANTS if o.name == "chunk16")
CONTROL = "none_s1"
data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
wins = F.Windows(data, data.subset("test"), 32 + 40 + 16 - 4, 4)
W = wins.fixed(200, seed=21)                                         # exactly pi_pass.main's windows
r = np.random.default_rng(21)                                        # replay the draw to learn each window's take
c = r.choice(len(wins.ids), size=200, p=wins.p)
src = np.array([data.clips[i]["source"] for i in wins.ids[c]])
std = data.std[:23]


def real200(name):
    d, L, H = train_np.SIZES["S"]
    m = MotionTransformer(2 * F.D, OBJ.out_dim, d, L, H, 32); m.p = dict(np.load(os.path.join(OUT, f"weights_{name}.npz")))
    y, _ = m.forward(pi_pass.inputs(W[:, :32], 32, np.ones((len(W), F.D), np.float32)))
    fut = OBJ.decode(y[:, -1], W[:, 31]) * data.std
    return ((fut[:, 3, :23] - W[:, 35, :23] * std) ** 2).mean(1)      # per-window, rad^2


def rmse_deg(sq):
    return float(np.degrees(np.sqrt(sq.mean())))


def paired(a, b, n=4000, seed=0):
    """RMSE(a) - RMSE(b) in degrees, 95% bootstrap interval over windows, and the share of draws below 0."""
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    d = np.degrees(np.sqrt(a[idx].mean(1))) - np.degrees(np.sqrt(b[idx].mean(1)))
    return dict(diff=rmse_deg(a) - rmse_deg(b), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_better=float((d < 0).mean()))


ref = json.load(open(os.path.join(ROOT, "runs", "pi_pass", "chunk16.json")))
names = [v for v in (CONTROL, "noisyhist", "selfhist") if os.path.exists(os.path.join(OUT, f"{v}.json"))]
recs = {v: json.load(open(os.path.join(OUT, f"{v}.json"))) for v in names}
pick = lambda rec: dict(execute_1=rec["execute_1"]["mean_2s"], execute_8=rec["execute_8"]["mean_2s"],
                        execute_8_1s=rec["execute_8"]["1.0s"], execute_8_2s=rec["execute_8"]["2.0s"],
                        from_real_200ms=rec["from_real_200ms"], best_val=rec["best_val"])
out = dict(windows=200, by_source={s: int((src == s).sum()) for s in sorted(set(src))},
           headline={"chunk16 (pi_pass, seed 0)": pick(ref), **{v: pick(rec) for v, rec in recs.items()}})
if CONTROL in recs:
    out["seed_noise_control_minus_pi_pass"] = {k: pick(recs[CONTROL])[k] - pick(ref)[k] for k in pick(ref)}
pw = {v: {ex: np.array(rec["detail"][ex]["per_window_mse_rad2"]) for ex in ("execute_1", "execute_8")}
      for v, rec in recs.items()}
for v in recs:
    pw[v]["real200"] = real200(v)
    assert abs(rmse_deg(pw[v]["real200"]) - recs[v]["from_real_200ms"]) < 1e-3, v   # same number pi_pass.score gave
hold = np.array(recs[names[0]]["detail"]["hold_pose"]["per_window_mse_rad2"])
out["hold_by_source"] = {s: rmse_deg(hold[src == s]) for s in sorted(set(src))}
for v in recs:
    out.setdefault("by_source", {})
    out[f"{v}_by_source_execute_8"] = {s: rmse_deg(pw[v]["execute_8"][src == s]) for s in sorted(set(src))}
    if v == CONTROL or CONTROL not in recs:
        continue
    out[f"{v}_minus_control_paired"] = {k: paired(pw[v][k], pw[CONTROL][k]) for k in ("execute_1", "execute_8", "real200")}
if "noisyhist" in recs and "selfhist" in recs:
    out["selfhist_minus_noisyhist_paired"] = {k: paired(pw["selfhist"][k], pw["noisyhist"][k])
                                              for k in ("execute_1", "execute_8", "real200")}
json.dump(out, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
