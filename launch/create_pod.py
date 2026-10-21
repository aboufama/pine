"""Create the Pine Engine pod on RunPod (run by GitHub Actions; the key comes from the RUNPOD_API_KEY repository secret).

Settings match the hand-deploy instructions: H100 first, on-demand (not spot), RunPod PyTorch 2.8 image, 60 GB container
disk, a large /workspace volume, the two RunPod secrets as environment variables, and the bootstrap start command.
It never creates a second pod while one named pine-engine is already running. The key is never printed.
"""
import json, os, sys, time, urllib.error, urllib.request

API = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
NAME = "pine-engine"
START = ('git clone -q -b engine-code --depth 1 https://github.com/aboufama/pine /workspace/pine-engine || true; '
         'bash /workspace/pine-engine/gpu/bootstrap.sh')
# Most preferred first. The models are small, so any of these trains them; the GPU-hour cap still applies.
GPUS = ["NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA RTX A6000",
        "NVIDIA A40", "NVIDIA GeForce RTX 4090"]
IMAGES = [("runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404", ["12.8", "12.9", "13.0"]),
          ("runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "13.0"])]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_launch.json")


def http(method, path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, method=method)
    req.add_header("Authorization", "Bearer " + KEY); req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "pine-engine-launcher/1.0")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, {"error": txt[:500]}


def record(**kw):
    kw["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(kw, open(OUT, "w"), indent=1); print(json.dumps(kw, indent=1))


def main():
    if not KEY:
        record(ok=False, step="secret", message="RUNPOD_API_KEY repository secret is not set; nothing was created"); return 0
    acts = {}
    ap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actions.json")
    if os.path.exists(ap):
        acts = json.load(open(ap))
    st, pods = http("GET", "/pods")
    if st == 200:   # terminate pods listed in actions.json (only ours, by name), e.g. one on a host with a broken GPU
        for p in pods or []:
            if p.get("id") in acts.get("terminate", []) and p.get("name") == NAME:
                ts, tp = http("DELETE", f"/pods/{p['id']}")
                print(f"terminate {p['id']}: HTTP {ts}")
        st, pods = http("GET", "/pods")
    if acts.get("start"):   # restart stopped pods (keeps their /workspace volume), e.g. for phase 2
        res = []
        for pid in acts["start"]:
            ss, sp = http("POST", f"/pods/{pid}/start")
            res.append(dict(pod_id=pid, http=ss, reply=str(sp)[:300]))
        ok = all(r["http"] in (200, 201, 202) for r in res)
        record(ok=ok, step="start", pods=res, message="pod(s) restarted" if ok else "could not restart"); return 0 if ok else 1
    if acts.get("create") is False:
        record(ok=True, step="no-create", message="actions.json says not to create a pod", pods=[{k: p.get(k) for k in ("id", "name", "desiredStatus")} for p in pods or []]); return 0
    if st != 200:
        record(ok=False, step="auth", status=st, message=str(pods)[:300]); return 1
    live = [p for p in (pods or []) if p.get("name") == NAME and p.get("desiredStatus") in ("RUNNING", "CREATED")]
    if live:
        p = live[0]; record(ok=True, step="exists", pod_id=p.get("id"), status=p.get("desiredStatus"),
                            message="a pine-engine pod is already running; not creating another"); return 0
    env = {"HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}", "GH_TOKEN": "{{ RUNPOD_SECRET_gh_token }}", "PINE_MAX_GPU_HOURS": "8",
           "PINE_ROBOT": "h2"}
    errors = []
    # Data-centre (SECURE) hosts first, with smaller volumes and no CUDA filter on the CUDA 12.4 image (every RunPod driver
    # runs it); community hosts last, because the ones free right now have repeatedly had GPUs that CUDA cannot open.
    plans = [("SECURE", IMAGES[1][0], None, v) for v in (300, 200)] + [("SECURE", IMAGES[0][0], IMAGES[0][1], v) for v in (300, 200)] \
          + [("COMMUNITY", IMAGES[1][0], None, 300), ("COMMUNITY", IMAGES[0][0], IMAGES[0][1], 300)]
    for cloud, image, cuda, volume in plans:
        body = dict(name=NAME, imageName=image, cloudType=cloud, computeType="GPU",
                    gpuTypeIds=GPUS, gpuTypePriority="custom", gpuCount=1, interruptible=False,
                    containerDiskInGb=60, volumeInGb=volume, volumeMountPath="/workspace", ports=["22/tcp"], env=env,
                    dockerStartCmd=["bash", "-c", START])
        if cuda:
            body["allowedCudaVersions"] = cuda
        st, p = http("POST", "/pods", body)
        if st in (200, 201) and isinstance(p, dict) and p.get("id"):
            m = p.get("machine") or {}
            record(ok=True, step="created", pod_id=p["id"], image=image, cloud=cloud, volume_gb=volume,
                   gpu=m.get("gpuTypeId") or p.get("gpuTypeId") or (p.get("gpu") or {}).get("id"),
                   cost_per_hr=p.get("costPerHr") or p.get("adjustedCostPerHr"), status=p.get("desiredStatus"), tried_before=errors)
            return 0
        errors.append(dict(cloud=cloud, image=image.split(":")[1], volume_gb=volume, status=st, error=str(p)[:160]))
    record(ok=False, step="create", attempts=errors, message="no listed GPU available right now; nothing is billing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
