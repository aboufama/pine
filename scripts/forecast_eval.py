"""Score the model's free-running forecast on many held-out windows (joint-angle error, degrees)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine import features as F, train_np      # noqa: E402
from pine.features import MotionData          # noqa: E402
from pine.model_np import MotionTransformer   # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
size = json.load(open(os.path.join(RUNS, "prediction.json")))["size"]
data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
d, L, H = train_np.SIZES[size]
m = MotionTransformer(2 * F.D, 4 * F.D, d, L, H, 32); m.p = dict(np.load(os.path.join(RUNS, f"weights_{size}.npz")))
ctx, hor, n = 32, 40, 200
W = F.Windows(data, data.subset("test"), ctx + hor - 4, 4).fixed(n, seed=21)     # (n, 72, D) held-out people
seq = W[:, :ctx].copy(); ones = np.ones((n, 1, F.D), np.float32)
for t in range(hor):
    win = seq[:, -32:]
    y, _ = m.forward(np.concatenate([win, np.broadcast_to(ones, win.shape)], -1))
    seq = np.concatenate([seq, y[:, -1:, :F.D]], 1)
std = data.std[:23]
truth = W[:, ctx:, :23] * std; pred = seq[:, ctx:, :23] * std; freeze = np.repeat(W[:, ctx - 1:ctx, :23] * std, hor, 1)
out = {}
for sec in (0.25, 0.5, 1.0, 2.0):
    k = int(sec * 20) - 1
    out[f"{sec}s"] = dict(model_deg=float(np.degrees(np.sqrt(((pred[:, k] - truth[:, k]) ** 2).mean()))),
                          freeze_deg=float(np.degrees(np.sqrt(((freeze[:, k] - truth[:, k]) ** 2).mean()))))
r = json.load(open(os.path.join(RUNS, "rollout_demo.json")))
r["eval"] = dict(windows=n, split="test", metric="joint-angle RMSE over 23 joints, degrees", horizons=out)
json.dump(r, open(os.path.join(RUNS, "rollout_demo.json"), "w"))
print(json.dumps(out, indent=1))

# One step at a time from real frames (what the model was trained to do): horizons 50-200 ms.
inp, tgt = F.batch(W[:, :36], 32, 4, np.ones((n, F.D), np.float32))
y, _ = m.forward(inp)
tf = {}
for k in range(4):
    p = y[:, :, k * F.D:k * F.D + 23] * std; t_ = tgt[:, :, k * F.D:k * F.D + 23] * std
    hold = W[:, :32, :23] * std
    tf[f"{(k + 1) * 50}ms"] = dict(model_deg=float(np.degrees(np.sqrt(((p - t_) ** 2).mean()))),
                                   freeze_deg=float(np.degrees(np.sqrt(((hold - t_) ** 2).mean()))))
r = json.load(open(os.path.join(RUNS, "rollout_demo.json")))
r["eval"]["from_real_frames"] = tf
r["eval"]["note"] = ("Free-running, the model is fed its own predictions and drifts past holding the last pose; "
                     "from real frames it beats holding the pose at every horizon it was trained on.")
json.dump(r, open(os.path.join(RUNS, "rollout_demo.json"), "w"))
print(json.dumps(tf, indent=1))
