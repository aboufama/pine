"""CPU scaling sweep with a pre-registered prediction.

1. Model ladder 2XS, XS, S, M on all training data.
2. Fit L(N) = E + A N^-alpha, bootstrap an interval for L, and write it to runs/prediction.json
   (timestamped) BEFORE L trains. Then train L.
3. Data ladder: size S on 1/64, 1/16, 1/4 of the training hours (plus the full run from step 1).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.features import MotionData   # noqa: E402
from pine import train_np, scaling, features as F   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs"); os.makedirs(RUNS, exist_ok=True)
STEPS = int(os.environ.get("PINE_STEPS", 900))
LADDER, HELD_OUT = ("2XS", "XS", "S", "M"), "L"      # fit on four sizes, predict the fifth
DATA_SIZE, FRACTIONS = "S", (1 / 64, 1 / 16, 1 / 4)
LOG = open(os.path.join(RUNS, "sweep.log"), "a")


def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


def save(rec, weights=None):
    name = f"run_{rec['size']}_{rec['fraction']:.4f}"
    json.dump(rec, open(os.path.join(RUNS, name + ".json"), "w"))
    if weights is not None:
        np.savez(os.path.join(RUNS, f"weights_{rec['size']}.npz"), **weights)


def load(size, frac):
    p = os.path.join(RUNS, f"run_{size}_{frac:.4f}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def train(data, size, frac, keep=False):
    rec = load(size, frac)
    if rec:
        log(f"skip {size} {frac} (done)"); return rec
    log(f"== {size} fraction {frac}")
    rec, w = train_np.run(data, size=size, fraction=frac, steps=STEPS, log=log)
    save(rec, w if keep else None)
    log(f"   {size} f={frac}: params {rec['params']:,} best val {rec['val']['full']:.4f} test {rec['test']['full']:.4f} "
        f"hidden {rec['val']['hidden_joint_rmse_deg']:.1f} deg, {rec['wall_s']:.0f}s")
    return rec


def main():
    t0 = time.time()
    data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    log(f"data: {len(data.clips)} takes, {data.raw.shape[0]:,} frames, {F.D} channels; load {time.time()-t0:.0f}s")
    ladder = [train(data, s, 1.0) for s in LADDER]
    pred_path = os.path.join(RUNS, "prediction.json")
    if not os.path.exists(pred_path):
        from pine.model_np import count_params
        d, L, H = train_np.SIZES[HELD_OUT]
        n_xl = count_params(2 * F.D, 4 * F.D, d, L, 32)
        bp = scaling.bootstrap_prediction([r["params"] for r in ladder], [r["val"]["per_window"] for r in ladder], n_xl)
        pred = dict(size=HELD_OUT, params=n_xl, steps=STEPS, registered_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    fit_on=[r["size"] for r in ladder], form="L(N) = E + A * N^-alpha", **bp)
        json.dump(pred, open(pred_path, "w"), indent=1)
        log(f"PRE-REGISTERED: {HELD_OUT} ({n_xl:,} params) val loss {bp['loss']:.4f} [{bp['lo']:.4f}, {bp['hi']:.4f}]")
    train(data, HELD_OUT, 1.0, keep=True)
    for frac in FRACTIONS:
        train(data, DATA_SIZE, frac)
    log(f"sweep done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
