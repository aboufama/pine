"""Suit-mode sanity check for the chunk16-family models (residual targets), light compute only.

The residual objective (pi_pass resid4/chunk16/dct16) predicts future - current for every channel, including the
channels a leg exoskeleton never measures. Decoding a hidden channel then needs its current value, which a suit
does not have. This script measures, on the same 200 held-out windows as pi_pass and with EXO-1 masks on the
whole context, the error of the joints the suit does not measure (waist + arms) 200 ms ahead:
  oracle   decode with the true current hidden values (not available on a real suit)
  no_info  decode with the hidden channels at their training mean (0 in normalized units)
  mean     baseline: always predict the mean pose for hidden joints
Also reports the measured leg joints at 200 ms in suit mode. Writes runs/research/exo_check.json.
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
data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
W = F.Windows(data, data.subset("test"), 32 + 40 + 16 - 4, 4).fixed(200, seed=21)
hid = [j for j in F.HIDDEN if j in F.ANGLE_DIMS]                 # waist + arms (11 joints)
legs = F.GROUPS["legs"]
mask = F.make_masks(len(W), "exo", np.random.default_rng(0))
deg = lambda e: float(np.degrees(np.sqrt((e ** 2).mean())))
truth = W[:, 35] * data.std                                       # frame 200 ms after the last context frame
out = dict(windows=200, hidden_joints=[F.NAMES[j] for j in hid],
           mean_pose_baseline_hidden_deg=deg(W[:, 35, hid] * data.std[hid]),      # normalized 0 = training mean
           hold_true_current_hidden_deg=deg((W[:, 31, hid] - W[:, 35, hid]) * data.std[hid]))
d, L, H = train_np.SIZES["S"]
for name in ("none_s1", "noisyhist", "selfhist"):
    m = MotionTransformer(2 * F.D, OBJ.out_dim, d, L, H, 32); m.p = dict(np.load(os.path.join(OUT, f"weights_{name}.npz")))
    y, _ = m.forward(pi_pass.inputs(W[:, :32], 32, mask))
    oracle = OBJ.decode(y[:, -1], W[:, 31])[:, 3] * data.std
    blind = OBJ.decode(y[:, -1], W[:, 31] * mask[:, :])[:, 3] * data.std           # hidden current value unknown -> 0
    out[name] = dict(hidden_oracle_deg=deg(oracle[:, hid] - truth[:, hid]),
                     hidden_no_info_deg=deg(blind[:, hid] - truth[:, hid]),
                     legs_deg=deg(oracle[:, legs] - truth[:, legs]))
json.dump(out, open(os.path.join(OUT, "exo_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
