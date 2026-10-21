"""Write gpu/plan.json: the machine-readable GPU plan the demo page shows and runpod/launch.py reads.

    python3 gpu/make_plan.py            (numpy only; run it again after changing gpu/plan_lib.py)
"""
from __future__ import annotations
import glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("PINE_ROBOT", "h2")

import plan_lib as P                                    # noqa: E402
from pine import features as F                          # noqa: E402
from pine.g1 import SPECS                               # noqa: E402
from pine.model_np import count_params                  # noqa: E402
from pine.train_np import SIZES                         # noqa: E402

IMAGE = "runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404"
DATASETS = [
    dict(name="CMU Graphics Lab Motion Capture Database (BVH conversion by B. Hahne)", id="cmu", hours=9.74,
         hours_note="kept after QA in the CPU ingest (2,333 of 2,548 takes)", format="BVH, 120 fps, Y-up",
         license="Free to use, including in commercial products; the data itself may not be resold",
         url="https://github.com/una-dinosauria/cmu-mocap", access="public (git clone)", robots=["h2", "g1"],
         path="BVH -> canonical -> IK"),
    dict(name="Bandai Namco Research Motion Dataset 1 and 2", id="bandai", hours=3.69,
         hours_note="kept after QA in the CPU ingest (169 + 2,456 takes)", format="BVH, 30 fps, cm, Y-up",
         license="CC BY-NC 4.0", url="https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset",
         access="public (git clone)", robots=["h2", "g1"], path="BVH -> canonical -> IK"),
    dict(name="Ubisoft La Forge Animation Dataset (LAFAN1)", id="lafan1", hours=4.6,
         hours_note="README: 77 sequences, 496,672 frames at 30 fps, 5 subjects", format="BVH, 30 fps, cm, Mixamo-style names",
         license="CC BY-NC-ND 4.0", url="https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
         access="public (git clone + git LFS for lafan1/lafan1.zip)", robots=["h2", "g1"], path="BVH -> canonical -> IK"),
    dict(name="HiPHI (Noitom Robotics)", id="hiphi", hours=617.5,
         hours_note="dataset card: 371.8 h body-only + 245.7 h human-object interaction, 132 performers, 215 GB. Every "
                    "motion also ships a __mirror copy; build_shards.py skips those, so if the card's total counts them "
                    "about half is used. The hours actually built are in stats.json.",
         format="BVH, 90 Hz, cm, Y-up, 55 joints (32 .tar.zst archives)",
         license="ModalityNet Open Research License (research, non-commercial; commercial licensing via info@modalitynet.com)",
         url="https://huggingface.co/datasets/noitomrobotics/HiPHI",
         access="gated: request access on the dataset page, then run the launcher with HF_TOKEN set; skipped otherwise",
         robots=["h2", "g1"], path="BVH -> canonical -> IK"),
    dict(name="Unitree LAFAN1_Retargeting_Dataset (G1)", id="lafan1_g1", hours=4.6,
         hours_note="the LAFAN1 motions, retargeted by Unitree", format="CSV, 30 fps: root xyz, quaternion xyzw "
         "(column label root_joint(XYZQXQYQZQW)), 29 joints in pine.g1.JOINTS order",
         license="CC BY-NC-ND 4.0 (data), MIT (code)", url="https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset",
         access="public", robots=["g1"], path="G1-native: no IK; used only when PINE_ROBOT=g1 (replaces our IK of LAFAN1)"),
    dict(name="Retargeted AMASS for robotics (G1, fleaven)", id="amass_g1", hours=None,
         hours_note="AMASS is roughly 40+ h; the retargeted hours are measured at build time. Its CMU subset is skipped "
                    "when the CMU BVH is present (same motions).",
         format="npy rows [-1,36]: root pos, quaternion xyzw, 29 joints; fps not documented: read from the file name "
                "(..._poses_<fps>_jpos.npy, as the dataset's own visualize.py does), else 30 fps assumed and logged",
         license="tagged CC BY 4.0; the card says each AMASS sub-dataset's own licence applies",
         url="https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics", access="public (6.95 GB)",
         robots=["g1"], path="G1-native: no IK; used only when PINE_ROBOT=g1"),
]


def cpu_baseline():
    runs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "run_*.json"))):
        try:
            r = json.load(open(p))
            runs.append(dict(size=r["size"], params=r["params"], fraction=r["fraction"], train_hours=round(r["train_hours"], 3),
                             steps=r["steps"], batch=r["batch"], val=round(r["val"]["full"], 5), test=round(r["test"]["full"], 5),
                             hidden_joint_rmse_deg=round(r["val"]["hidden_joint_rmse_deg"], 2), wall_s=round(r["wall_s"])))
        except (OSError, KeyError, ValueError):
            continue
    pred = None
    pp = os.path.join(ROOT, "runs", "prediction.json")
    if os.path.exists(pp):
        q = json.load(open(pp))
        pred = {k: q.get(k) for k in ("size", "params", "registered_at", "fit_on", "loss", "lo", "hi")}
    return dict(note="CPU NumPy sweep in this repo (CMU + Bandai Namco, 900 steps x 64 windows). The GPU sweep uses more data, "
                     "so its losses are on a different validation set and are not directly comparable.",
                runs=sorted(runs, key=lambda r: (r["fraction"] != 1.0, r["params"], r["fraction"])), prediction=pred)


def main():
    robot = F.ROBOT
    sp = SPECS[robot]
    din, dout = 2 * F.D, P.K * F.D
    ladder = []
    for s in P.LADDER:
        d, L, H = SIZES[s]
        n = count_params(din, dout, d, L, P.T)
        assert n == P.count_params(din, dout, d, L, P.T), "plan_lib.count_params drifted from pine.model_np"
        ladder.append(dict(size=s, params=n, d_model=d, layers=L, heads=H))
    gpus, best = [], None
    for g in P.GPUS:
        e = P.estimate(SIZES, din, dout, g, P.DEFAULT_GPU_COUNT, hiphi=True)
        e0 = P.estimate(SIZES, din, dout, g, P.DEFAULT_GPU_COUNT, hiphi=False)
        row = dict(id=g["id"], name=g["name"], usd_per_hr=g["usd_per_hr"], bf16_tflops=g["bf16_tflops"],
                   est_gpu_hours=P.ceil2(e["gpu_hours"]), est_cost_usd=P.ceil2(e["cost_usd"]),
                   est_gpu_hours_without_hiphi=P.ceil2(e0["gpu_hours"]), est_cost_usd_without_hiphi=P.ceil2(e0["cost_usd"]))
        gpus.append(row)
        best = best or (row, e)
    first, est = best
    plan = dict(
        comment=("Prices: on-demand per GPU-hour from " + P.PRICE_SOURCE + ". Hours: 6*N*tokens FLOPs at "
                 f"{P.MFU:.0%} MFU of the GPU's dense BF16 peak, at least {P.STEP_FLOOR_S*1e3:.0f} ms per step, plus "
                 f"{P.RUN_OVERHEAD_S} s per run and the setup time (download, BVH parsing, IK)."),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        robot=dict(id=robot, name=sp["name"], dof=sp["dof"], height_m=sp["height_m"], solved_joints=len(F.ACTIVE),
                   held_at_zero=sp["held"], channels=F.D, model="https://github.com/unitreerobotics/unitree_mujoco",
                   mjcf=sp["mjcf"], model_license="BSD-3-Clause"),
        data_path=("For the H2 every source goes through BVH -> 19 canonical keypoints -> batched IK on the GPU "
                   "(CMU, Bandai Namco, LAFAN1, HiPHI). The two G1-native Hugging Face sets (Unitree's LAFAN1 retarget, "
                   "Retargeted AMASS) are used only when PINE_ROBOT=g1."),
        datasets=DATASETS,
        ladder=ladder,
        sweep=dict(model_ladder=P.LADDER, fit_on=P.LADDER[:-1], predicted=P.LADDER[-1], data_ladder_size=P.DATA_SIZE,
                   data_fractions=P.DATA_FRACTIONS + [1.0],
                   steps=f"{P.EPOCHS} epochs of the full training set, clamped to [{P.STEPS_MIN}, {P.STEPS_MAX}]; same for every run",
                   est_steps=dict(without_hiphi=P.steps_for(int(P.TRAIN_HOURS["base"] * 72000)),
                                  with_hiphi=P.steps_for(int(P.TRAIN_HOURS["hiphi"] * 72000))), batch=P.BATCH, context=P.T, horizon=P.K,
                   masking="50% full, 30% EXO (legs + IMU only), 20% random channel groups", loss="MSE",
                   lr="min(3e-3, 0.02/sqrt(d_model)), warmup 5%, cosine to 10%", optimizer="AdamW wd 0.1, clip 1.0",
                   precision="bf16 autocast, fp32 weights; evaluation in fp32", val_windows=P.VAL_WINDOWS,
                   order="model ladder -> pre-registered prediction (timestamped) -> predicted run -> data ladder"),
        runs=[dict(r, est_seconds_first_gpu=s["est_seconds"]) for r, s in zip(P.runs(SIZES), est["runs"])],
        gpus=gpus,
        gpu=first["id"], gpu_count=P.DEFAULT_GPU_COUNT,
        est_gpu_hours=first["est_gpu_hours"], est_cost_usd=first["est_cost_usd"],
        est_gpu_hours_without_hiphi=first["est_gpu_hours_without_hiphi"], est_cost_usd_without_hiphi=first["est_cost_usd_without_hiphi"],
        max_gpu_hours=P.DEFAULT_MAX_GPU_HOURS,
        max_cost_usd=P.ceil2(P.DEFAULT_MAX_GPU_HOURS * first["usd_per_hr"]
                             + P.DEFAULT_MAX_GPU_HOURS * (P.CONTAINER_DISK_GB + P.VOLUME_GB) * P.STORAGE_USD_PER_GB_MONTH / 730),
        cost_guard=("PINE_MAX_GPU_HOURS (default %.0f): before each run the sweep estimates its length from the measured "
                    "tokens/s and does not start it if the GPU-hours used since the pod started would exceed the cap; "
                    "every run also gets a hard deadline. The pod stops itself PINE_AUTOSTOP_MIN minutes after finishing "
                    "(terminates instead if the results were already fetched)." % P.DEFAULT_MAX_GPU_HOURS),
        runpod=dict(api="https://rest.runpod.io/v1", image=IMAGE,
                    image_source="https://hub.docker.com/r/runpod/pytorch/tags (pushed 2026-09-30; CUDA 12.8.1, torch 2.8.0, Ubuntu 24.04)",
                    docs=["https://docs.runpod.io/api-reference/pods/POST/pods", "https://docs.runpod.io/api-reference/pods/GET/pods/podId",
                          "https://docs.runpod.io/api-reference/pods/DELETE/pods/podId", "https://docs.runpod.io/pods/configuration/use-ssh"],
                    container_disk_gb=P.CONTAINER_DISK_GB, volume_gb=P.VOLUME_GB, ports=["22/tcp"]),
        command="python3 runpod/launch.py up --yes",
        parity=dict(script="gpu/parity.py", reference="pine/model_np.py (hand-written NumPy forward/backward and AdamW)",
                    dtype="float64 on CPU (plus an fp32 check on the GPU)",
                    compares=["loss", "every parameter gradient",
                              "one AdamW step (decoupled weight decay on 2-D params except pos, clip 1.0, bias-corrected, b1 0.9, b2 0.95)"],
                    tolerance="max relative error < 1e-6", gate="run_all.sh stops before any download or training if it fails",
                    status="not run yet: torch is not installed where this package was written; it runs first on the pod",
                    ik_check=("gpu/retarget_torch.py vs pine/g1.py on 2,000 frames: mean keypoint difference must be < 1 mm, "
                              "otherwise build_shards.py uses the NumPy solver on all CPU cores")),
        cpu_baseline=cpu_baseline(),
    )
    out = os.path.join(HERE, "plan.json")
    with open(out, "w") as f:
        json.dump(plan, f, indent=1)
    print(out, f"gpu={first['name']} est {first['est_gpu_hours']} GPU-h ${first['est_cost_usd']}, cap {P.DEFAULT_MAX_GPU_HOURS} h")


if __name__ == "__main__":
    main()
