"""Sweep definition and cost model for the GPU scale-up. Standard library only.

Used by make_plan.py (writes gpu/plan.json), sweep_gpu.py (what to train, budget guard) and, through plan.json,
by runpod/launch.py. Change the experiment here, then re-run `python3 gpu/make_plan.py`.
"""
from __future__ import annotations
import math

# ----------------------------------------------------------------------------------------------- the experiment
LADDER = ["M", "L", "XL", "2XL", "3XL", "4XL"]   # model ladder on all training data; L(N) is fit on all but the last
DATA_SIZE = "2XL"                                 # data ladder: this size on fractions of the training hours ...
DATA_FRACTIONS = [1 / 64, 1 / 16, 1 / 4]          # ... plus its full-data run, which the model ladder already has
# Optimizer steps per run, the same for every run of a sweep (as on CPU): EPOCHS passes over the full training set,
# clamped. ~16 h of BVH without HiPHI -> STEPS_MIN; with HiPHI (~300 h) -> ~STEPS_MAX. PINE_STEPS overrides.
EPOCHS = 20
STEPS_MIN, STEPS_MAX = 4_000, 40_000
BATCH = 512                                       # global batch, in windows of T + K frames
T, K = 32, 4                                      # context frames, predicted horizon (as on CPU)
VAL_WINDOWS = 2048                                # fixed validation / test windows per run (CPU used 512)
EVALS_PER_RUN = 20

# ----------------------------------------------------------------------------------------------- cost model
MFU = 0.25              # conservative model-FLOPs utilisation for the estimate (what the brief asks for)
STEP_FLOOR_S = 0.012    # small models are launch-bound, not FLOP-bound: no step is assumed faster than this
RUN_OVERHEAD_S = 120    # per run: torchrun start, loading shards onto the GPU, 20 evaluations
SETUP_HOURS = {"base": 0.6, "hiphi": 1.6}  # image pull + fetch + BVH parse + IK; HiPHI adds ~215 GB download + parse

PRICE_SOURCE = ("https://www.runpod.io/pricing and https://www.runpod.io/gpu-pricing, fetched 2026-10-21 "
                "(page says 'Updated October 8, 2026'); on-demand $/GPU-hour, storage $0.10/GB/month while running")
# REST gpuTypeIds (https://docs.runpod.io/api-reference/pods/POST/pods), peak dense BF16 tensor TFLOPS, on-demand price.
GPUS = [
    dict(id="NVIDIA H100 80GB HBM3", name="H100 SXM 80GB", bf16_tflops=989.0, usd_per_hr=3.99),
    dict(id="NVIDIA H100 NVL", name="H100 NVL 94GB", bf16_tflops=835.0, usd_per_hr=3.19),
    dict(id="NVIDIA H100 PCIe", name="H100 PCIe 80GB", bf16_tflops=756.0, usd_per_hr=2.89),
    dict(id="NVIDIA A100-SXM4-80GB", name="A100 SXM 80GB", bf16_tflops=312.0, usd_per_hr=1.79),
    dict(id="NVIDIA A100 80GB PCIe", name="A100 PCIe 80GB", bf16_tflops=312.0, usd_per_hr=1.79),
    dict(id="NVIDIA L40S", name="L40S 48GB", bf16_tflops=362.0, usd_per_hr=1.09),
]
GPU_BY_ID = {g["id"]: g for g in GPUS}
DEFAULT_GPU_COUNT = 1    # these models are small: one GPU is the cheapest way to run the sweep (DDP works for N > 1)
DEFAULT_MAX_GPU_HOURS = 8.0
CONTAINER_DISK_GB = 60
VOLUME_GB = 400          # HiPHI alone is ~215 GB compressed; one archive is extracted at a time next to it
STORAGE_USD_PER_GB_MONTH = 0.10


def steps_for(train_frames: int, batch: int = None, T_: int = None) -> int:
    batch, T_ = batch or BATCH, T_ or T
    return int(min(STEPS_MAX, max(STEPS_MIN, math.ceil(EPOCHS * train_frames / (batch * T_)))))


def count_params(din: int, dout: int, d: int, L: int, T_: int) -> int:
    """Same formula as pine.model_np.count_params (checked in gpu/tests)."""
    return din * d + d + T_ * d + L * (12 * d * d + 13 * d) + 2 * d + d * dout + dout


def runs(sizes: dict, ladder=None, data_size=None, fractions=None) -> list[dict]:
    """The ordered list of runs. `sizes` is pine.train_np.SIZES: name -> (d_model, layers, heads)."""
    ladder = list(ladder or LADDER)
    data_size = data_size or DATA_SIZE
    fractions = list(DATA_FRACTIONS if fractions is None else fractions)
    for s in ladder + [data_size]:
        if s not in sizes:
            raise KeyError(f"size {s!r} is not in pine.train_np.SIZES ({', '.join(sizes)})")
    out = [dict(size=s, fraction=1.0, role="model_ladder") for s in ladder[:-1]]
    out.append(dict(size=ladder[-1], fraction=1.0, role="predicted"))     # before the data ladder: if the budget
    out += [dict(size=data_size, fraction=float(f), role="data_ladder") for f in fractions]   # runs out, these go
    return out


def run_flops(params: int, steps: int = STEPS_MAX, batch: int = BATCH, T_: int = T) -> float:
    return 6.0 * params * steps * batch * T_          # same accounting as pine.train_np.run


def run_seconds(params: int, gpu: dict, n_gpus: int = 1, steps: int = STEPS_MAX, batch: int = BATCH,
                mfu: float = MFU) -> float:
    compute = run_flops(params, steps, batch) / (mfu * gpu["bf16_tflops"] * 1e12 * n_gpus)
    return max(compute, steps * STEP_FLOOR_S) + RUN_OVERHEAD_S


# training hours by scenario, for the estimate only (the pod uses what it actually built)
TRAIN_HOURS = {"base": 16.0, "hiphi": 16.0 + 0.9 * 309.0}


def estimate(sizes: dict, din: int, dout: int, gpu: dict, n_gpus: int = 1, hiphi: bool = True,
             steps: int | None = None, batch: int = BATCH) -> dict:
    if steps is None:
        steps = steps_for(int(TRAIN_HOURS["hiphi" if hiphi else "base"] * 3600 * 20), batch)
    total_s, per = 0.0, []
    for r in runs(sizes):
        d, L, H = sizes[r["size"]]
        n = count_params(din, dout, d, L, T)
        s = run_seconds(n, gpu, n_gpus, steps, batch)
        total_s += s
        per.append(dict(r, params=n, flops=run_flops(n, steps, batch), est_seconds=round(s)))
    setup_h = SETUP_HOURS["hiphi" if hiphi else "base"]
    wall_h = setup_h + total_s / 3600
    storage = (CONTAINER_DISK_GB + VOLUME_GB) * STORAGE_USD_PER_GB_MONTH / 730.0
    return dict(gpu=gpu["id"], n_gpus=n_gpus, steps=steps, setup_hours=setup_h, train_hours=total_s / 3600,
                wall_hours=wall_h, gpu_hours=wall_h * n_gpus,
                cost_usd=wall_h * (gpu["usd_per_hr"] * n_gpus + storage), runs=per)


def ceil2(x: float) -> float:
    return math.ceil(x * 100) / 100
