import os, sys, subprocess
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    p = os.path.join(d, f)
    if os.path.getsize(p) > 1_500_000 or f.endswith((".webp", ".avif", ".gif")):
        out = os.path.splitext(p)[0] + "_prev.jpg"
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-frames:v", "1", "-vf", "scale='min(1400,iw)':-2", out])
        if r.returncode == 0 and os.path.getsize(p) > 1_500_000:
            os.remove(p)
