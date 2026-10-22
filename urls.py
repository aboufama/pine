import re, sys, html
from urllib.parse import urljoin
base, path = sys.argv[1], sys.argv[2]
t = open(path, errors="replace").read()
found = set()
for m in re.finditer(r'''(?:src|href|data-src|poster|content|srcset)\s*=\s*["']([^"']+)["']''', t, re.I):
    for part in m.group(1).split(","):
        u = html.unescape(part.strip().split(" ")[0])
        if re.search(r"\.(jpe?g|png|webp|gif|mp4|webm|mov|avif)(\?|$)", u, re.I):
            found.add(urljoin(base, u))
for m in re.finditer(r'''url\(["']?([^"')]+)["']?\)''', t):
    u = m.group(1)
    if re.search(r"\.(jpe?g|png|webp|gif|avif)(\?|$)", u, re.I):
        found.add(urljoin(base, u))
for u in sorted(found):
    print(u)
