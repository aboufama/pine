#!/usr/bin/env python3
"""Launch the Pine Engine GPU scale-up on RunPod, from your own machine. Python standard library only.

    python3 runpod/launch.py plan                 GPU choice, hours and cost; no network call
    python3 runpod/launch.py up --yes             create the pod (billing starts), upload this folder, start gpu/run_all.sh
    python3 runpod/launch.py status               pod state, cost so far, progress on the pod
    python3 runpod/launch.py fetch [--wait]       copy results.tgz back and unpack it into runs/gpu/
    python3 runpod/launch.py start --yes          start the pod again after it stopped itself (billing resumes)
    python3 runpod/launch.py down --yes           terminate the pod and delete its volume (billing stops)

API key: RUNPOD_API_KEY, or the file ~/.runpod/api_key. It is sent only in the Authorization header to the RunPod
REST API and is never printed or logged. SSH: your key pair in ~/.ssh (or --ssh-key) is put on the pod as PUBLIC_KEY.
REST API used (https://docs.runpod.io/api-reference/): POST /v1/pods, GET /v1/pods, GET /v1/pods/{id},
POST /v1/pods/{id}/start, DELETE /v1/pods/{id}; base https://rest.runpod.io/v1.
"""
from __future__ import annotations
import argparse, glob, io, json, os, shutil, socket, subprocess, sys, tarfile, tempfile, time
import urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE = os.path.join(HERE, "pod.json")
PLAN = os.path.join(ROOT, "gpu", "plan.json")
API = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")
NAME = "pine-engine"
IMAGE = "runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404"     # CUDA 12.8.1, torch 2.8.0, pushed 2026-09-30
GPU_PREFS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe",
             "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA L40S"]
CUDA_OK = ["12.8", "12.9", "13.0"]        # host drivers able to run a CUDA 12.8 image
REMOTE_DIR = "/workspace/pine-engine"
UA = "pine-engine-launcher/1.0 (python urllib)"   # the default Python-urllib agent is blocked by some CDNs (403 / 1010)
EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv", "venv"}
EXCLUDE_TOP = {os.path.join("runs", "gpu"), os.path.join("runpod", "pod.json")}
NEEDS_YES = 3


class Fail(SystemExit):
    def __init__(self, msg: str, code: int = 1):
        print(f"error: {msg}", file=sys.stderr, flush=True)
        super().__init__(code)


# ----------------------------------------------------------------------------------------------------- secrets
def api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k:
        p = os.path.expanduser("~/.runpod/api_key")
        if os.path.exists(p):
            with open(p) as f:
                k = f.read().strip()
    if not k:
        raise Fail("no RunPod API key: set RUNPOD_API_KEY or write it to ~/.runpod/api_key")
    return k


_SECRETS: list[str] = []


def scrub(text: str) -> str:
    for s in _SECRETS:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    return text


def say(msg: str = "") -> None:
    print(scrub(msg), flush=True)


# ----------------------------------------------------------------------------------------------------- REST
def http(method: str, path: str, body=None, query: dict | None = None, timeout: float = 60.0):
    """One REST call. Returns (status, parsed JSON or None). Raises Fail on network errors."""
    key = api_key()
    if key not in _SECRETS:
        _SECRETS.append(key)
    url = API + path
    if query:
        url += "?" + urllib.parse.urlencode(query, doseq=True)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", UA)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise Fail(f"cannot reach {API} ({e}); check your network")
    text = raw.decode("utf-8", "replace") if raw else ""
    try:
        parsed = json.loads(text) if text.strip() else None
    except ValueError:
        parsed = {"raw": text[:500]}
    return status, parsed


def api_error(status: int, parsed) -> str:
    msg = parsed.get("error") or parsed.get("message") or parsed if isinstance(parsed, dict) else parsed
    hint = {401: " (API key rejected)", 403: " (forbidden: a proxy or CDN may be blocking this machine)",
            404: " (not found)"}.get(status, "")
    return scrub(f"HTTP {status}{hint}: {json.dumps(msg)[:800]}")


def get_pod(pod_id: str) -> dict | None:
    st, p = http("GET", f"/pods/{urllib.parse.quote(pod_id)}")
    if st == 404:
        return None
    if st != 200 or not isinstance(p, dict):
        raise Fail("reading the pod failed: " + api_error(st, p))
    return p


def list_named(name: str) -> list[dict]:
    st, p = http("GET", "/pods", query={"name": name})
    if st != 200:
        raise Fail("listing pods failed: " + api_error(st, p))
    pods = p if isinstance(p, list) else (p or {}).get("pods", []) if isinstance(p, dict) else []
    return [x for x in pods if x.get("name") == name and x.get("desiredStatus") != "TERMINATED"]


# ----------------------------------------------------------------------------------------------------- state
def load_state() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def pod_id_from(a) -> str:
    pid = getattr(a, "pod_id", None) or load_state().get("id")
    if pid:
        return pid
    pods = list_named(NAME)
    if len(pods) == 1:
        return pods[0]["id"]
    if not pods:
        raise Fail(f"no pod recorded in {STATE} and none named {NAME!r} is running")
    raise Fail(f"{len(pods)} pods are named {NAME!r}: pass --pod-id ({', '.join(p['id'] for p in pods)})")


# ----------------------------------------------------------------------------------------------------- plan / cost
def load_plan() -> dict:
    try:
        with open(PLAN) as f:
            return json.load(f)
    except (OSError, ValueError):
        raise Fail(f"{PLAN} is missing or unreadable: run `python3 gpu/make_plan.py` first")


def prices(plan: dict) -> dict:
    return {g["id"]: g for g in plan.get("gpus", [])}


def print_plan(a, plan: dict) -> dict:
    gp = prices(plan)
    prefs = a.gpu or GPU_PREFS
    n = a.gpu_count
    cap = a.max_gpu_hours
    say(f"Pine Engine GPU scale-up, robot {plan.get('robot', {}).get('name', '?')} ({plan.get('robot', {}).get('id', '?')})")
    say(f"  image        {IMAGE}")
    say(f"  pod          {a.cloud} cloud, {n} GPU(s), container disk {a.container_disk} GB, volume {a.volume} GB at /workspace")
    say("  GPU preference (first available is rented):")
    worst = 0.0
    for g in prefs:
        e = gp.get(g, {})
        price = e.get("usd_per_hr")
        hrs = e.get("est_gpu_hours")
        cost = e.get("est_cost_usd")
        worst = max(worst, (price or 0) * n)
        say(f"    {g:<26} " + (f"${price:.2f}/GPU-h  est {hrs:.2f} GPU-h  ~${cost:.2f}" if price else "price unknown"))
    storage = (a.container_disk + a.volume) * 0.10 / 730
    max_cost = cap * (worst / max(1, n)) + cap / max(1, n) * storage
    first = gp.get(prefs[0], {})
    say(f"  estimate     {first.get('est_gpu_hours', float('nan')):.2f} GPU-h, ${first.get('est_cost_usd', float('nan')):.2f} on "
        f"{prefs[0]} (without HiPHI: {first.get('est_gpu_hours_without_hiphi', float('nan')):.2f} GPU-h, "
        f"${first.get('est_cost_usd_without_hiphi', float('nan')):.2f})")
    say(f"  cost guard   PINE_MAX_GPU_HOURS={cap:g}: at most ~${max_cost:.2f} even on the dearest GPU in the list; "
        f"the pod stops itself {a.autostop_min} min after finishing (0 = never)")
    say(f"  HiPHI        {'HF_TOKEN is set: the gated HiPHI set is downloaded if your account has access' if os.environ.get('HF_TOKEN') else 'HF_TOKEN not set: HiPHI (gated) is skipped'}")
    say(f"  prices from  {plan.get('comment', '')[:160]}")
    return dict(max_cost=max_cost, worst_hourly=worst)


def cmd_plan(a) -> int:
    plan = load_plan()
    print_plan(a, plan)
    if a.check_key:
        st, p = http("GET", "/pods")
        say("  API key      " + ("accepted" if st == 200 else api_error(st, p)))
    say(f"\nTo launch: python3 runpod/launch.py up --yes")
    return 0


# ----------------------------------------------------------------------------------------------------- ssh helpers
def ssh_keys(explicit: str | None):
    """(private key path or None, [public key lines]) from --ssh-key or ~/.ssh/*.pub."""
    pubs = []
    if explicit:
        priv = os.path.expanduser(explicit)
        if not os.path.exists(priv):
            raise Fail(f"--ssh-key {explicit}: no such file")
        pub = priv + ".pub"
        if not os.path.exists(pub):
            raise Fail(f"{pub} not found (the public half of --ssh-key)")
        cands = [pub]
    else:
        d = os.path.expanduser("~/.ssh")
        order = ["id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"]
        cands = [os.path.join(d, n) for n in order if os.path.exists(os.path.join(d, n))]
        cands += sorted(p for p in glob.glob(os.path.join(d, "*.pub")) if p not in cands)
    priv = None
    for p in cands:
        try:
            with open(p) as f:
                line = f.read().strip()
        except OSError:
            continue
        if not line.startswith(("ssh-", "ecdsa-", "sk-")):
            continue
        pubs.append(line)
        if priv is None and os.path.exists(p[:-4]):
            priv = p[:-4]
    return priv, pubs


def ssh_base(st: dict, a, tool: str = "ssh") -> list[str]:
    exe = os.environ.get("PINE_SSH" if tool == "ssh" else "PINE_SCP", tool)
    kh = os.path.join(HERE, f"known_hosts_{st['id']}")
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={kh}", "-o", "ServerAliveInterval=30",
            "-o", "ConnectTimeout=20"]
    key = st.get("ssh_key") or getattr(a, "ssh_key", None)
    if key:
        opts += ["-i", os.path.expanduser(key)]
    port_flag = "-p" if tool == "ssh" else "-P"
    return [exe] + opts + [port_flag, str(st["port"])]


def ssh_run(st: dict, a, remote_cmd: str, check: bool = True, capture: bool = False, timeout: float = 600):
    cmd = ssh_base(st, a) + [f"root@{st['ip']}", remote_cmd]
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    except FileNotFoundError:
        raise Fail("ssh is not installed (or not on PATH)")
    except subprocess.TimeoutExpired:
        raise Fail(f"ssh timed out after {timeout:.0f}s")
    if check and r.returncode != 0:
        raise Fail(f"remote command failed (exit {r.returncode}): {remote_cmd[:120]}"
                   + (f"\n{scrub(r.stderr[-800:])}" if capture and r.stderr else ""))
    return r


def scp(st: dict, a, src: str, dst: str, timeout: float = 3600):
    cmd = ssh_base(st, a, "scp") + [src, dst]
    try:
        r = subprocess.run(cmd, timeout=timeout)
    except FileNotFoundError:
        raise Fail("scp is not installed (or not on PATH)")
    except subprocess.TimeoutExpired:
        raise Fail(f"scp timed out after {timeout:.0f}s")
    if r.returncode != 0:
        raise Fail(f"scp failed (exit {r.returncode})")


def ssh_banner(ip: str, port: int, timeout: float = 8.0) -> bool:
    try:
        with socket.create_connection((ip, int(port)), timeout=timeout) as s:
            s.settimeout(timeout)
            return s.recv(64).startswith(b"SSH-")
    except OSError:
        return False


def endpoint(pod: dict):
    ip = pod.get("publicIp") or ""
    pm = pod.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    return (ip, int(port)) if ip and port else (None, None)


def manual_steps(st: dict, reason: str) -> None:
    pid = st.get("id", "<pod id>")
    say(f"\nCould not finish automatically: {reason}")
    say("The pod IS RUNNING AND BILLING. Either finish by hand or stop it:")
    say(f"  python3 runpod/launch.py down --yes          # terminate pod {pid}")
    say("By hand (needs SSH over exposed TCP; see the pod's Connect tab, https://docs.runpod.io/pods/configuration/use-ssh):")
    ip, port = st.get("ip") or "<IP>", st.get("port") or "<PORT>"
    say("  python3 runpod/launch.py bundle --out pine-engine.tgz")
    say(f"  scp -P {port} pine-engine.tgz root@{ip}:/workspace/")
    say(f"  ssh -p {port} root@{ip} 'mkdir -p {REMOTE_DIR} && tar -xzf /workspace/pine-engine.tgz -C {REMOTE_DIR} && "
        f"mkdir -p /workspace/logs && cd {REMOTE_DIR} && setsid nohup bash gpu/run_all.sh > /workspace/logs/run_all.out 2>&1 < /dev/null &'")
    say("Without SSH: open the pod's web terminal in the RunPod console; locally run `runpodctl send pine-engine.tgz`,")
    say("on the pod `cd /workspace && runpodctl receive <code>`, then the tar/run_all.sh line above.")


# ----------------------------------------------------------------------------------------------------- bundle
def _excluded(rel: str) -> bool:
    parts = rel.split(os.sep)
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    if any(rel == e or rel.startswith(e + os.sep) for e in EXCLUDE_TOP):
        return True
    base = parts[-1]
    return (base.startswith("known_hosts_") or base.endswith((".pyc", ".tgz", ".tmp")) or base == "api_key")


def make_bundle(path: str) -> int:
    n = 0
    with tarfile.open(path, "w:gz") as tf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            rel_dir = os.path.relpath(dirpath, ROOT)
            dirnames[:] = sorted(d for d in dirnames if not _excluded(os.path.normpath(os.path.join(rel_dir, d))))
            for fn in sorted(filenames):
                rel = os.path.normpath(os.path.join(rel_dir, fn))
                if _excluded(rel) or os.path.islink(os.path.join(dirpath, fn)):
                    continue
                tf.add(os.path.join(dirpath, fn), arcname=rel.replace(os.sep, "/"), recursive=False)
                n += 1
    return n


def cmd_bundle(a) -> int:
    n = make_bundle(a.out)
    say(f"{a.out}: {n} files, {os.path.getsize(a.out)/1e6:.2f} MB")
    return 0


# ----------------------------------------------------------------------------------------------------- up
def create_body(a, pubkeys: list[str]) -> dict:
    env = {"PINE_ROBOT": a.robot, "PINE_MAX_GPU_HOURS": f"{a.max_gpu_hours:g}", "PINE_AUTOSTOP_MIN": str(a.autostop_min),
           "PINE_WORKSPACE": "/workspace", "PINE_RAW": "/workspace/data/raw"}
    if pubkeys:
        env["PUBLIC_KEY"] = "\n".join(pubkeys)
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    for k in ("PINE_STEPS", "PINE_BATCH", "PINE_COMPILE", "PINE_HIPHI_PARTS", "PINE_SKIP"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    body = dict(name=NAME, imageName=IMAGE, cloudType=a.cloud, computeType="GPU", gpuTypeIds=list(a.gpu or GPU_PREFS),
                gpuTypePriority="custom", gpuCount=a.gpu_count, allowedCudaVersions=CUDA_OK,
                containerDiskInGb=a.container_disk, volumeInGb=a.volume, volumeMountPath="/workspace",
                ports=["22/tcp"], env=env, interruptible=False, minVCPUPerGPU=a.min_vcpu, minRAMPerGPU=a.min_ram)
    if a.cloud == "COMMUNITY":
        body["supportPublicIp"] = True
    return body


def masked(body: dict) -> dict:
    b = json.loads(json.dumps(body))
    for k in ("HF_TOKEN",):
        if k in b.get("env", {}):
            b["env"][k] = "***"
    if "PUBLIC_KEY" in b.get("env", {}):
        b["env"]["PUBLIC_KEY"] = f"<{len(body['env']['PUBLIC_KEY'].splitlines())} public key(s)>"
    return b


def wait_ready(pid: str, timeout_s: float) -> dict:
    t0, last = time.time(), ""
    while time.time() - t0 < timeout_s:
        pod = get_pod(pid)
        if pod is None:
            raise Fail(f"pod {pid} disappeared")
        ip, port = endpoint(pod)
        status = pod.get("desiredStatus")
        line = f"  {time.time()-t0:5.0f}s  status {status}, ip {ip or '-'}, ssh port {port or '-'}"
        if line.split("s  ", 1)[1] != last:
            say(line); last = line.split("s  ", 1)[1]
        if status == "TERMINATED":
            raise Fail(f"pod {pid} was terminated")
        if status == "RUNNING" and ip and port and ssh_banner(ip, port):
            return dict(pod=pod, ip=ip, port=port)
        time.sleep(10)
    raise TimeoutError(f"pod not reachable over SSH after {timeout_s/60:.0f} min")


def cmd_up(a) -> int:
    plan = load_plan()
    cost = print_plan(a, plan)
    priv, pubs = ssh_keys(a.ssh_key)
    if not pubs and not a.no_upload:
        raise Fail("no SSH public key found in ~/.ssh (create one with `ssh-keygen -t ed25519`), or pass --ssh-key; "
                   "use --no-upload to create the pod anyway and follow the manual steps")
    if not a.no_upload and not (shutil.which(os.environ.get("PINE_SSH", "ssh")) and shutil.which(os.environ.get("PINE_SCP", "scp"))):
        raise Fail("ssh and scp are needed for the upload (or pass --no-upload)")
    if not os.path.exists(os.path.join(ROOT, "gpu", "run_all.sh")):
        raise Fail("gpu/run_all.sh is missing")
    body = create_body(a, pubs)
    if a.dry_run:
        say("\nPOST " + API + "/pods (dry run, nothing sent):")
        say(json.dumps(masked(body), indent=1))
        return 0
    if not a.yes:
        say(f"\nThis creates a billed pod (up to ~${cost['worst_hourly']:.2f}/h while it runs, capped by PINE_MAX_GPU_HOURS). "
            "Re-run with --yes to go ahead.")
        return NEEDS_YES
    existing = list_named(NAME)
    if existing and not a.allow_second:
        ids = ", ".join(f"{p['id']} ({p.get('desiredStatus')})" for p in existing)
        raise Fail(f"a pod named {NAME!r} already exists: {ids}. Use `status`, `fetch`, `down --yes`, or --allow-second")
    say(f"\nCreating pod {NAME!r} ...")
    st, p = http("POST", "/pods", body=body, timeout=120)
    if st not in (200, 201) or not isinstance(p, dict) or not p.get("id"):
        raise Fail("pod creation failed: " + api_error(st, p) + "\nNothing is billing. Try --cloud COMMUNITY, another "
                   "--gpu, or lower --min-vcpu / --min-ram.")
    state = dict(id=p["id"], name=NAME, created_unix=time.time(), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 image=p.get("image") or IMAGE, cost_per_hr=p.get("costPerHr"), gpu=(p.get("gpu") or {}).get("displayName"),
                 gpu_count=a.gpu_count, ssh_key=priv, robot=a.robot, max_gpu_hours=a.max_gpu_hours)
    save_state(state)
    say(f"pod {p['id']} created, ${p.get('costPerHr', '?')}/h; state in {STATE}")
    if a.no_upload:
        manual_steps(state, "--no-upload was given")
        return 0
    try:
        ready = wait_ready(p["id"], a.wait_min * 60)
    except TimeoutError as e:
        manual_steps(state, str(e))
        return 1
    state.update(ip=ready["ip"], port=ready["port"], cost_per_hr=ready["pod"].get("costPerHr", state["cost_per_hr"]),
                 gpu=(ready["pod"].get("gpu") or {}).get("displayName") or state.get("gpu"))
    save_state(state)
    try:
        upload_and_start(state, a)
    except Fail as e:
        manual_steps(state, f"upload/start failed ({e})")
        return 1
    say(f"\nRunning on {state.get('gpu') or 'GPU'} at ${state.get('cost_per_hr')}/h. Watch it with:")
    say("  python3 runpod/launch.py status")
    say(f"  ssh -p {state['port']} root@{state['ip']}{' -i ' + priv if priv else ''} 'tail -f /workspace/logs/run_all.log'")
    say("When it says DONE:  python3 runpod/launch.py fetch    then    python3 runpod/launch.py down --yes")
    return 0


def upload_and_start(state: dict, a) -> None:
    with tempfile.TemporaryDirectory() as td:
        bundle = os.path.join(td, "pine-engine.tgz")
        n = make_bundle(bundle)
        say(f"uploading {n} files ({os.path.getsize(bundle)/1e6:.1f} MB) ...")
        ssh_run(state, a, "mkdir -p /workspace/logs " + REMOTE_DIR)
        scp(state, a, bundle, f"root@{state['ip']}:/workspace/pine-engine.tgz")
    t0 = int(state["created_unix"])
    start = (f"tar -xzf /workspace/pine-engine.tgz -C {REMOTE_DIR} && cd {REMOTE_DIR} && "
             f"if [ -e /workspace/RUNNING ]; then echo 'run_all.sh is already running'; else "
             f"PINE_T0={t0} PINE_GPUS={a.gpu_count} PINE_ROBOT={a.robot} PINE_MAX_GPU_HOURS={a.max_gpu_hours:g} "
             f"PINE_AUTOSTOP_MIN={a.autostop_min} setsid nohup bash gpu/run_all.sh > /workspace/logs/run_all.out 2>&1 "
             f"< /dev/null & sleep 3; tail -n 5 /workspace/logs/run_all.out; fi")
    ssh_run(state, a, start)
    say("gpu/run_all.sh started under nohup on the pod")


# ----------------------------------------------------------------------------------------------------- status / fetch / start / down
def describe(pod: dict, state: dict) -> None:
    ip, port = endpoint(pod)
    up_h = (time.time() - state["created_unix"]) / 3600 if state.get("created_unix") else None
    cph = pod.get("costPerHr")
    say(f"pod {pod.get('id')}  {pod.get('name')}  {pod.get('desiredStatus')}")
    say(f"  gpu      {(pod.get('gpu') or {}).get('displayName', '?')} x {(pod.get('gpu') or {}).get('count', '?')}, "
        f"${cph}/h" + (f", up {up_h:.2f} h, ~${up_h * float(cph):.2f} so far" if up_h is not None and cph else ""))
    say(f"  ssh      " + (f"root@{ip} -p {port}" if ip else "not exposed yet"))
    say(f"  changed  {pod.get('lastStatusChange', '')}")


def cmd_status(a) -> int:
    pid = pod_id_from(a)
    pod = get_pod(pid)
    state = load_state()
    if pod is None:
        say(f"pod {pid} not found (terminated?)")
        return 1
    describe(pod, state)
    ip, port = endpoint(pod)
    if pod.get("desiredStatus") == "RUNNING" and ip and not a.no_ssh:
        state.update(id=pid, ip=ip, port=port)
        r = ssh_run(state, a, "cat /workspace/STATUS 2>/dev/null; echo; cat /workspace/DONE 2>/dev/null; "
                              "tail -n 12 /workspace/logs/run_all.log 2>/dev/null", check=False, capture=True, timeout=60)
        say("  on the pod:")
        for line in (r.stdout or "").strip().splitlines():
            say("    " + line)
    elif pod.get("desiredStatus") == "EXITED":
        say("  the pod is stopped (it stops itself after finishing); results are on its volume: "
            "`start --yes` then `fetch`, or `down --yes` to delete it")
    return 0


def safe_extract(tgz: str, dest: str) -> int:
    n = 0
    dest_abs = os.path.abspath(dest)
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf.getmembers():
            target = os.path.abspath(os.path.join(dest, m.name))
            if not (target == dest_abs or target.startswith(dest_abs + os.sep)) or m.issym() or m.islnk() or m.isdev():
                raise Fail(f"refusing unsafe archive member {m.name!r}")
        if sys.version_info >= (3, 12):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)
        n = sum(1 for m in tf.getmembers() if m.isfile())
    return n


def cmd_fetch(a) -> int:
    pid = pod_id_from(a)
    state = load_state()
    state["id"] = pid
    while True:
        pod = get_pod(pid)
        if pod is None:
            raise Fail(f"pod {pid} not found")
        ip, port = endpoint(pod)
        if pod.get("desiredStatus") != "RUNNING" or not ip:
            raise Fail(f"pod is {pod.get('desiredStatus')}; it must be running to fetch (`start --yes` if it stopped itself)")
        state.update(ip=ip, port=port)
        done = ssh_run(state, a, "test -e /workspace/DONE && test -e /workspace/results.tgz", check=False, timeout=60).returncode == 0
        if done or not a.wait:
            break
        say(f"{time.strftime('%H:%M:%S')} not finished yet; checking again in {a.poll_min} min")
        time.sleep(a.poll_min * 60)
    if done:
        remote = "/workspace/results.tgz"
    elif a.partial:
        remote = "/workspace/results_partial.tgz"
        ssh_run(state, a, f"cd {REMOTE_DIR} && bash gpu/run_all.sh --pack-only {remote}")
    else:
        raise Fail("the run has not finished (no /workspace/DONE). Use --partial for what exists so far, or --wait")
    out_dir = os.path.join(ROOT, "runs", "gpu")
    os.makedirs(out_dir, exist_ok=True)
    local = os.path.join(out_dir, os.path.basename(remote))
    scp(state, a, f"root@{state['ip']}:{remote}", local)
    n = safe_extract(local, out_dir)
    runs = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
    say(f"{local}: {n} files unpacked into {out_dir} ({len(runs)} run records)")
    if done:
        ssh_run(state, a, "touch /workspace/FETCHED", check=False, timeout=60)
        say("the pod now stops itself within a few minutes (GPU billing ends; the volume is kept until `down --yes`)")
    return 0


def cmd_start(a) -> int:
    pid = pod_id_from(a)
    if not a.yes:
        say(f"Starting pod {pid} again resumes GPU billing. Re-run with --yes.")
        return NEEDS_YES
    st, p = http("POST", f"/pods/{pid}/start")
    if st not in (200, 201, 204):
        raise Fail("start failed: " + api_error(st, p))
    say(f"pod {pid} starting; `status` shows when SSH is up")
    return 0


def cmd_down(a) -> int:
    pid = pod_id_from(a)
    if not a.yes:
        say(f"This terminates pod {pid} and deletes its volume (results not fetched are lost). Re-run with --yes.")
        return NEEDS_YES
    if not glob.glob(os.path.join(ROOT, "runs", "gpu", "run_*.json")):
        say("warning: no fetched results in runs/gpu/ yet")
    st, p = http("DELETE", f"/pods/{pid}")
    if st not in (200, 202, 204, 404):
        raise Fail("terminate failed: " + api_error(st, p))
    say(f"pod {pid} terminated" + (" (it was already gone)" if st == 404 else ""))
    s = load_state()
    if s.get("id") == pid:
        s["terminated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save_state(s)
        try:
            os.replace(STATE, STATE + f".{pid}.done")
        except OSError:
            pass
    return 0


# ----------------------------------------------------------------------------------------------------- cli
def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def pod_opts(p, billing: bool = False):
        p.add_argument("--gpu", action="append", help="GPU type id (repeat for a preference list); default H100 > A100 > L40S")
        p.add_argument("--gpu-count", type=int, default=int(os.environ.get("PINE_GPUS", "1")))
        p.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
        p.add_argument("--container-disk", type=int, default=60)
        p.add_argument("--volume", type=int, default=400, help="GB at /workspace (HiPHI alone is ~215 GB)")
        p.add_argument("--min-vcpu", type=int, default=8)
        p.add_argument("--min-ram", type=int, default=32)
        p.add_argument("--max-gpu-hours", type=float, default=float(os.environ.get("PINE_MAX_GPU_HOURS", "8")))
        p.add_argument("--autostop-min", type=int, default=int(os.environ.get("PINE_AUTOSTOP_MIN", "120")))
        p.add_argument("--robot", default=os.environ.get("PINE_ROBOT", "h2"), choices=("h2", "g1"))

    p = sub.add_parser("plan", help="print GPU choice, hours and cost (no billable call)")
    pod_opts(p)
    p.add_argument("--check-key", action="store_true", help="also check the API key with a free GET /pods")
    p = sub.add_parser("up", help="create the pod, upload, start gpu/run_all.sh")
    pod_opts(p, True)
    p.add_argument("--yes", action="store_true", help="really create the (billed) pod")
    p.add_argument("--dry-run", action="store_true", help="print the request body and stop")
    p.add_argument("--ssh-key", help="private key to use (its .pub goes on the pod)")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--allow-second", action="store_true")
    p.add_argument("--wait-min", type=float, default=30, help="minutes to wait for SSH")
    for name in ("status", "fetch", "start", "down"):
        p = sub.add_parser(name)
        p.add_argument("--pod-id")
        p.add_argument("--ssh-key")
        if name in ("start", "down"):
            p.add_argument("--yes", action="store_true")
        if name == "status":
            p.add_argument("--no-ssh", action="store_true")
        if name == "fetch":
            p.add_argument("--partial", action="store_true", help="pack and fetch what exists before the run finishes")
            p.add_argument("--wait", action="store_true", help="poll until the run is finished, then fetch")
            p.add_argument("--poll-min", type=float, default=5)
    p = sub.add_parser("bundle", help="write the upload tarball locally (for the manual path)")
    p.add_argument("--out", default="pine-engine.tgz")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    fn = dict(plan=cmd_plan, up=cmd_up, status=cmd_status, fetch=cmd_fetch, start=cmd_start, down=cmd_down,
              bundle=cmd_bundle)[a.cmd]
    try:
        return fn(a)
    except KeyboardInterrupt:
        say("\ninterrupted" + ("; if a pod was created it is still billing: `status` / `down --yes`" if a.cmd == "up" else ""))
        return 130


if __name__ == "__main__":
    sys.exit(main())
