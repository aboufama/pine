"""Add the GPU run (branch engine-results) to demo/data.json as D.gpu, without recomputing the CPU-side export.

    python scripts/export_gpu.py <clone of engine-results>

D.gpu = shards (hours, takes, sources, IK), ladder (model sizes on all data), prediction + fits, data ladder,
extra (dct16mix run), phase2 (people vs hours, recipes, flagship), cost. Missing pieces stay null, so the page can be
rebuilt at any point of the run.
"""
from __future__ import annotations
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo", "data.json")


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def slim_run(r: dict) -> dict:
    curve = r.get("curve") or []
    step = max(1, len(curve) // 20)
    return dict(size=r["size"], params=r["params"], fraction=r["fraction"], train_hours=r.get("train_hours"),
                val=r["val"]["full"], test=r["test"]["full"], val_exo=r["val"].get("exo"), test_exo=r["test"].get("exo"),
                hidden_deg=r["val"].get("hidden_joint_rmse_deg"), hidden_deg_test=r["test"].get("hidden_joint_rmse_deg"),
                meanpose_deg=r["val"].get("hidden_joint_rmse_deg_meanpose"), steps=r.get("steps"), best_step=r.get("best_step"),
                epochs=r.get("epochs"), wall_s=r.get("wall_s"), stopped_early=r.get("stopped_early"),
                tokens_per_s=(r.get("device") or {}).get("tokens_per_s"), objective=(r.get("objective") or {}).get("name"),
                curve=[dict(step=c["step"], train=c["train"], val=c["val"]) for c in curve[::step]],
                pi_pass=r.get("pi_pass_score"), split=(r.get("split") or {}).get("meta"))


def main(res_dir: str) -> None:
    D = json.load(open(DEMO))
    st = load(os.path.join(res_dir, "status.json")) or {}
    sh = load(os.path.join(res_dir, "shards_stats.json")) or {}
    runs = [load(p) for p in sorted(glob.glob(os.path.join(res_dir, "runs", "run_*.json")))]
    runs = [r for r in runs if r]
    ladder = sorted([slim_run(r) for r in runs if r["fraction"] == 1.0], key=lambda r: r["params"])
    data_ladder = sorted([slim_run(r) for r in runs if r["size"] == "2XL"], key=lambda r: r["train_hours"])
    extra = [slim_run(r) for r in (load(p) for p in glob.glob(os.path.join(res_dir, "runs", "extra_*.json"))) if r]
    p2 = [slim_run(r) for r in (load(p) for p in sorted(glob.glob(os.path.join(res_dir, "runs", "phase2", "phase2_*.json")))) if r]
    p2sum = load(os.path.join(res_dir, "runs", "phase2", "summary.json"))
    seed_stats = None
    det = ((p2sum or {}).get("seed_status") or {})
    if det.get("status") == "ok":   # SEED_OK holds the merge summary as JSON (cut at 500 characters): take the numbers out
        import re
        m = {k: re.search(rf'"{k}": ([0-9.]+)', det.get("detail", "")) for k in ("takes", "hours", "actors")}
        seed_stats = {k: float(v.group(1)) for k, v in m.items() if v}
    D["gpu"] = dict(
        updated=st.get("updated"), stage=st.get("stage"), gpu=(st.get("gpus") or [{}])[0].get("name"),
        gpu_hours=st.get("gpu_hours"), cost_usd=st.get("est_cost_usd"),
        shards=dict(hours=sh.get("hours"), frames=sh.get("frames"), takes=sh.get("takes"), takes_in=sh.get("takes_in"),
                    hours_by_split=sh.get("hours_by_split"), per_source=sh.get("per_source"), ik=sh.get("ik"),
                    rejects=sh.get("rejects")) if sh else None,
        ladder=ladder, data_ladder=data_ladder, extra=extra,
        prediction=load(os.path.join(res_dir, "runs", "prediction.json")), fits=load(os.path.join(res_dir, "runs", "fits.json")),
        phase2=dict(runs=p2, seed=(p2sum or {}).get("seed_status"), seed_stats=seed_stats) if (p2 or p2sum) else None)
    json.dump(D, open(DEMO, "w"), separators=(",", ":"))
    g = D["gpu"]
    print(f"D.gpu: {len(ladder)} ladder runs, {len(data_ladder)} data-ladder runs, {len(extra)} extra, "
          f"{len(p2)} phase-2 runs, shards {(g['shards'] or {}).get('hours')} h, prediction {bool(g['prediction'])}, fits {bool(g['fits'])}")


if __name__ == "__main__":
    main(sys.argv[1])
