"""Pause a research job (SIGSTOP) while any other process is busy on the shared CPU, resume (SIGCONT) when quiet.

usage: python3 watchdog.py <pid>     (exits when <pid> exits)
A process counts as busy above 40% of one core over two consecutive 2-second samples.
"""
import os, signal, sys, time

target = int(sys.argv[1]); me = os.getpid(); HZ = os.sysconf("SC_CLK_TCK")


def ticks():
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            s = open(f"/proc/{p}/stat").read()
            r = s[s.rfind(")") + 2:].split()
            out[int(p)] = (int(r[11]) + int(r[12]), s[s.find("(") + 1:s.rfind(")")], int(r[1]))
        except (OSError, IndexError, ValueError):
            pass
    return out


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


paused, busy_n, quiet_n, prev, t_prev = False, 0, 0, ticks(), time.time()
log(f"watching for other CPU users while pid {target} runs")
while os.path.exists(f"/proc/{target}"):
    time.sleep(2.0)
    cur, now = ticks(), time.time()
    dt = now - t_prev
    hogs = [(pid, name, (t - prev[pid][0]) / HZ / dt) for pid, (t, name, ppid) in cur.items()
            if pid in prev and pid not in (target, me) and ppid != target and (t - prev[pid][0]) / HZ / dt > 0.4]
    prev, t_prev = cur, now
    if hogs:
        busy_n, quiet_n = busy_n + 1, 0
    else:
        busy_n, quiet_n = 0, quiet_n + 1
    if not paused and busy_n >= 2:
        os.kill(target, signal.SIGSTOP); paused = True
        log("paused: " + ", ".join(f"{n}[{p}] {u:.0%}" for p, n, u in hogs))
    elif paused and quiet_n >= 3:
        os.kill(target, signal.SIGCONT); paused = False
        log("resumed")
log("target exited")
