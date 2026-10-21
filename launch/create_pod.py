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
GPUS = ["NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
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
    st, pods = http("GET", "/pods")
    if st != 200:
        record(ok=False, step="auth", status=st, message=str(pods)[:300]); return 1
    live = [p for p in (pods or []) if p.get("name") == NAME and p.get("desiredStatus") in ("RUNNING", "CREATED")]
    if live:
        p = live[0]; record(ok=True, step="exists", pod_id=p.get("id"), status=p.get("desiredStatus"),
                            message="a pine-engine pod is already running; not creating another"); return 0
    env = {"HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}", "GH_TOKEN": "{{ RUNPOD_SECRET_gh_token }}", "PINE_MAX_GPU_HOURS": "8",
           "PINE_ROBOT": "h2"}
    errors = []
    for volume in (500, 400, 700, 1000):   # HiPHI is extracted one archive at a time; 500 GB fits, and an idle volume bills by size
        for cloud in ("SECURE", "COMMUNITY"):
            body = dict(name=NAME, imageName="runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404", cloudType=cloud, computeType="GPU",
                        gpuTypeIds=GPUS, gpuTypePriority="custom", gpuCount=1, interruptible=False,
                        containerDiskInGb=60, volumeInGb=volume, volumeMountPath="/workspace", ports=["22/tcp"], env=env,
                        dockerStartCmd=["bash", "-lc", START], allowedCudaVersions=["12.8", "12.9", "13.0"])
            st, p = http("POST", "/pods", body)
            if st in (200, 201) and isinstance(p, dict) and p.get("id"):
                record(ok=True, step="created", pod_id=p["id"], cloud=cloud, volume_gb=volume, gpu=(p.get("machine") or {}).get("gpuTypeId") or p.get("gpuTypeId"),
                       cost_per_hr=p.get("costPerHr") or p.get("adjustedCostPerHr"), status=p.get("desiredStatus"))
                return 0
            errors.append(dict(cloud=cloud, volume_gb=volume, status=st, error=str(p)[:300]))
    record(ok=False, step="create", attempts=errors, message="no H100/A100 80GB available with these settings; nothing is billing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
