# Pine Engine

Pine's data engine, end to end, on public data: full-body motion capture in, a pretrained humanoid
motion model out. EXO-1 recordings are meant to enter the same pipe at step 1.

```
raw takes ──► canonical keypoints ──► Unitree G1 joint space ──► shards ──► pretraining ──► scaling fits
(BVH, ASF/AMC,   19 points, 20 fps,      29-DoF IK, analytic        31 channels   causal transformer   pre-registered
 EXO-1 logs)      metres, QA'd            Jacobians, joint limits    per frame     + suit-channel masks  predictions
```

| step | code | what it does |
|---|---|---|
| read | `pine/mocap_io.py` | BVH and ASF/AMC readers, forward kinematics, units to metres, Z up |
| canonicalise | `pine/canon.py` | 19 named keypoints per frame, 20 fps, floor, QA (length, glitches, bone drift, left/right) |
| retarget | `pine/g1.py` | batched Levenberg-Marquardt IK onto Unitree's own G1 MJCF, two passes with smoothing |
| features | `pine/features.py` | 31 channels: 23 joints, gravity in the pelvis frame, yaw rate, velocity, height |
| model | `pine/model_np.py` | hand-written NumPy transformer (forward + backward, gradient-checked) and AdamW |
| train | `pine/train_np.py` | next-K-frame prediction with random suit-channel masking |
| scale | `pine/scaling.py`, `scripts/sweep.py` | ladder of sizes, power-law fit, bootstrap interval, prediction written before the held-out run |
| GPU | `gpu/`, `runpod/` | PyTorch DDP mirror checked against the NumPy reference, dataset fetchers, RunPod launcher |

## Reproduce on a CPU

```sh
python3 scripts/ingest.py      # ~3 min on 2 cores: 5,625 takes -> 4,958 kept, 13.4 h
python3 scripts/retarget.py    # ~20 min on 2 cores: every frame onto the G1
python3 scripts/sweep.py       # scaling sweep, writes runs/
python3 scripts/finalize.py    # fits + EXO-1 demo clip
python3 scripts/export_demo.py # demo/data.json for the page
```

Raw data is cloned from GitHub: `una-dinosauria/cmu-mocap` (CMU, BVH conversion by B. Hahne),
`BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset`, and `unitreerobotics/unitree_mujoco` for the robot.

## Licences

CMU data is free to use, including in commercial products, but may not be resold. Both Bandai Namco sets are
CC BY-NC 4.0 (non-commercial). The GPU plan adds sets with their own terms (HiPHI is non-commercial research
only; AMASS derivatives inherit AMASS's licence). Train any commercial model on data whose licence allows it.

## Assumptions to replace with real EXO-1 facts

- Suit channels: hip, knee and ankle joint encoders plus a pelvis IMU (`pine/features.py: EXO`).
- Wrists are not solved (no hand markers in the public sets), so they are held at zero.
