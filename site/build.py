"""Build the one-pager and the write-up for GitHub Pages (../pine/engine) and for the private preview (../artifact)."""
import shutil
S = ""  # helpers.js and hero.js sit next to this file
src = open("page.src.html").read(); helpers = open(S + "helpers.js").read(); hero = open(S + "hero.js").read()
helpers = helpers.replace('const YAW = -0.62, PITCH = 0.2, cy = Math.cos(YAW), sy = Math.sin(YAW), cp = Math.cos(PITCH), sp = Math.sin(PITCH);',
    'let YAW = -0.62, cy = Math.cos(YAW), sy = Math.sin(YAW);\nconst PITCH = 0.2, cp = Math.cos(PITCH), sp = Math.sin(PITCH);\nconst setYaw = (a) => { YAW = a; cy = Math.cos(a); sy = Math.sin(a); };').replace(
    'return [xr, z * cp - yr * sp];', 'return [xr, z * cp + yr * sp];')
one = src.replace("/*HELPERS*/", helpers).replace("/*HERO*/", hero)
det = open("index.html").read().replace("<h1>Humanoid motion pretraining from public motion capture</h1>", "<h1>High-quality full-body annotated motion data at scale</h1>")
open("onepage.html", "w").write(one)
site = "/home/claude/pine/engine"; open(f"{site}/index.html", "w").write(one); open(f"{site}/details.html", "w").write(det)
for f in ("data.json", "h2.json", "h2.bin"): shutil.copy(f"../demo/{f}", f"{site}/{f}")
import os
os.makedirs(f"{site}/img", exist_ok=True); os.makedirs("../artifact/img", exist_ok=True)
for f in os.listdir("img"):
    shutil.copy(f"img/{f}", f"{site}/img/{f}"); shutil.copy(f"img/{f}", f"../artifact/img/{f}")
def art(t, strip):
    t = t.replace('<link rel="icon" href="../mark-c2.svg" type="image/svg+xml">\n', "")
    if strip:
        for a in ('<!doctype html>\n<html lang="en">\n<head>\n', '</head>\n<body>\n', '</body>\n</html>\n', '<meta charset="utf-8">\n',
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'):
            assert a in t; t = t.replace(a, "")
        t = t.replace("<title>Pine Engine</title>", '<title>Pine Engine</title>\n<meta name="claude:viewport-fit-sides" content="cover">', 1)
        t = t.replace('fetch("h2.bin").then((r) => r.arrayBuffer())', 'fetch("h2.b64.json").then((r) => r.json()).then((j) => Uint8Array.from(atob(j.b64), (c) => c.charCodeAt(0)).buffer)')
    return t.replace('href="./">One page</a>', 'href="index.html">One page</a>')
open("../artifact/index.html", "w").write(art(one, True)); open("../artifact/details.html", "w").write(art(det, False))
for f in ("data.json", "h2.json"): shutil.copy(f"../demo/{f}", f"../artifact/{f}")
print("built", len(one))
