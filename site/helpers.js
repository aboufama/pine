const $ = (s) => document.querySelector(s);
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt = (x, d = 0) => Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const INK = "#1d1d1f", QUIET = "#86868b", GHOST = "#c7c7cc", HAIR = "#d2d2d7";

/* film: same behaviour as the Pine site */
const video = $("#video") || document.createElement("video"), film = $("#film") || document.createElement("div");
const syncFilm = () => film.setAttribute("aria-label", video.paused ? "Play film" : "Pause film");
film.addEventListener("click", () => (video.paused ? video.play() : video.pause()));
film.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); video.paused ? video.play() : video.pause(); } });
video.addEventListener("play", syncFilm); video.addEventListener("pause", syncFilm);
if (!REDUCE) video.play().catch(syncFilm);

/* ---------------------------------------------------------------- skeleton drawing */
let EDGES = [], KP = {};
const LEGSET = new Set(["pelvis", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_toe", "r_toe"]);
const TRAIL = ["l_wrist", "r_wrist", "l_ankle", "r_ankle"];
const YAW = -0.62, PITCH = 0.2, cy = Math.cos(YAW), sy = Math.sin(YAW), cp = Math.cos(PITCH), sp = Math.sin(PITCH);

function follow(arr, n) {            // smoothed pelvis track so walkers stay in frame
  const px = new Float32Array(n), py = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let sx = 0, sy2 = 0, c = 0;
    for (let g = Math.max(0, f - 10); g <= Math.min(n - 1, f + 10); g++) { sx += arr[g * 57]; sy2 += arr[g * 57 + 1]; c++; }
    px[f] = sx / c; py[f] = sy2 / c;
  }
  return { px, py };
}
function P(arr, f, k, ox, oy) {      // world cm -> view (x right, y up) relative to a follow centre
  const i = (f * 19 + k) * 3, x = arr[i] - ox, y = arr[i + 1] - oy, z = arr[i + 2];
  const xr = x * cy - y * sy, yr = x * sy + y * cy;
  return [xr, z * cp - yr * sp];
}
class Figure {
  constructor(canvas, opts) {
    this.c = canvas; this.o = Object.assign({ height: 210, ground: 0.86, center: 0.5 }, opts);
    this.ctx = canvas.getContext("2d"); this.size(); this.layers = []; this.n = 1; this.f = 0;
  }
  size() {
    const r = this.c.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
    this.w = Math.max(10, r.width); this.h = Math.max(10, r.height);
    this.c.width = Math.round(this.w * d); this.c.height = Math.round(this.h * d);
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.s = (this.h * this.o.ground * 0.95) / this.o.height;
  }
  set(layers, n, track) { this.layers = layers; this.n = n; this.track = track; this.f = Math.min(this.f, n - 1); }
  xy(v) { return [this.w * this.o.center + v[0] * this.s, this.h * this.o.ground - v[1] * this.s]; }
  draw(f) {
    const g = this.ctx, t = this.track; g.clearRect(0, 0, this.w, this.h);
    const ox = t.px[f], oy = t.py[f];
    // ground: a soft shadow and ticks that move as the body travels
    g.fillStyle = "rgba(0,0,0,0.06)"; g.beginPath();
    g.ellipse(this.w * this.o.center, this.h * this.o.ground + 2, 28 * this.s, 6 * this.s, 0, 0, Math.PI * 2); g.fill();
    g.strokeStyle = HAIR; g.lineWidth = 1;
    for (let k = -6; k <= 6; k++) {
      const gx = Math.round((ox / 50)) * 50 + k * 50;
      const a = P([gx, oy, 0], 0, 0, ox, oy); const q = this.xy(a);
      g.beginPath(); g.moveTo(q[0], q[1]); g.lineTo(q[0] + 4, q[1]); g.stroke();
    }
    for (const L of this.layers) {
      const arr = L.data, ff = Math.min(f, (arr.length / 57 | 0) - 1);
      if (L.trail) {
        for (const name of TRAIL) {
          const k = KP[name]; g.lineWidth = 1.2;
          for (let b = Math.max(0, ff - 15); b < ff; b++) {
            const p0 = this.xy(P(arr, b, k, ox, oy)), p1 = this.xy(P(arr, b + 1, k, ox, oy));
            g.strokeStyle = `rgba(29,29,31,${(0.32 * (b - ff + 16) / 16).toFixed(3)})`;
            g.beginPath(); g.moveTo(p0[0], p0[1]); g.lineTo(p1[0], p1[1]); g.stroke();
          }
        }
      }
      g.lineCap = "round"; g.lineJoin = "round";
      for (const [a, b] of EDGES) {
        if (L.only && !L.only(a, b)) continue;
        const p = this.xy(P(arr, ff, a, ox, oy)), q = this.xy(P(arr, ff, b, ox, oy));
        g.strokeStyle = L.color; g.lineWidth = L.width; g.setLineDash(L.dash || []);
        g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
      }
      g.setLineDash([]);
      if (L.dots) {
        g.fillStyle = L.color;
        for (let k = 0; k < 19; k++) { const p = this.xy(P(arr, ff, k, ox, oy)); g.beginPath(); g.arc(p[0], p[1], L.dots, 0, Math.PI * 2); g.fill(); }
        const h = this.xy(P(arr, ff, KP.head, ox, oy)); g.beginPath(); g.arc(h[0], h[1] - 5 * this.s, 6 * this.s, 0, Math.PI * 2);
        g.strokeStyle = L.color; g.lineWidth = L.width; g.stroke();
      }
    }
    if (this.o.overlay) this.o.overlay(g, f, this);
  }
}

/* one clock for every figure: 20 fps, only while on screen */
const players = new Set(); let last = 0;
function tick(t) {
  if (t - last >= 50) {
    last = t;
    for (const p of players) if (p.visible && !p.paused) { p.fig.f = (p.fig.f + 1) % p.fig.n; p.fig.draw(p.fig.f); }
  }
  requestAnimationFrame(tick);
}
function register(fig, el) {
  const p = { fig, visible: false, paused: REDUCE }; players.add(p);
  new IntersectionObserver((es) => { p.visible = es[0].isIntersecting; }, { rootMargin: "100px" }).observe(el);
  const toggle = () => { p.paused = !p.paused; };
  el.addEventListener("click", toggle);
  el.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
  return p;
}
addEventListener("resize", () => { for (const p of players) { p.fig.size(); p.fig.draw(p.fig.f); } });

/* ---------------------------------------------------------------- charts */
function svgEl(tag, attrs, parent) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e); return e;
}
function chart(el, { W, H, x, y, xlog = true, ylog = false, xticks, yticks, xfmt, yfmt, xlabel, ylabel }) {
  W = W || Math.max(280, Math.round(el.clientWidth)); H = H || Math.round(Math.min(300, Math.max(220, W * 0.62)));
  const m = { l: 52, r: 14, t: 12, b: 40 };
  const s = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, null);
  const sx = (v) => m.l + ((xlog ? Math.log(v) : v) - (xlog ? Math.log(x[0]) : x[0])) / ((xlog ? Math.log(x[1]) : x[1]) - (xlog ? Math.log(x[0]) : x[0])) * (W - m.l - m.r);
  const sy2 = (v) => H - m.b - ((ylog ? Math.log(v) : v) - (ylog ? Math.log(y[0]) : y[0])) / ((ylog ? Math.log(y[1]) : y[1]) - (ylog ? Math.log(y[0]) : y[0])) * (H - m.t - m.b);
  for (const v of yticks) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy2(v), y2: sy2(v), stroke: v === yticks[0] ? HAIR : "#ececf0", "stroke-width": 1 }, s);
    const t = svgEl("text", { x: m.l - 8, y: sy2(v) + 4, "text-anchor": "end" }, s); t.textContent = yfmt(v);
  }
  for (const v of xticks) {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: H - m.b, y2: H - m.b + 4, stroke: HAIR }, s);
    const t = svgEl("text", { x: sx(v), y: H - m.b + 17, "text-anchor": "middle" }, s); t.textContent = xfmt(v);
  }
  const xl = svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "lbl" }, s); xl.textContent = xlabel;
  const yl = svgEl("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "lbl", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, s); yl.textContent = ylabel;
  el.appendChild(s);
  return { s, sx, sy: sy2, m, W, H };
}
const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
function label(c, xv, yv, txt, dx = 8, dy = 4, anchor = "start", cls = "lbl") {
  const t = svgEl("text", { x: c.sx(xv) + dx, y: c.sy(yv) + dy, "text-anchor": anchor, class: cls }, c.s); t.textContent = txt; return t;
}
function bars(el, rows, max) {
  el.innerHTML = "";
  for (const r of rows) {
    const d = document.createElement("div"); d.className = "brow";
    d.innerHTML = `<div>${r.name}</div><div class="t"><span class="${r.light ? "light" : ""}" style="width:${Math.min(100, 100 * r.value / max).toFixed(1)}%"></span></div><div class="v">${r.text}</div>`;
    el.appendChild(d);
  }
}

