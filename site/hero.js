  /* hero: the captured take (left) and Unitree's own H2 model doing it (right) */
  class Robot3D {
    constructor(el, M, bin) {
      this.el = el;
      const r = this.r = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
      r.outputEncoding = THREE.sRGBEncoding; el.appendChild(r.domElement);
      const sc = this.scene = new THREE.Scene();
      this.cam = new THREE.PerspectiveCamera(24, 1, 0.05, 80); this.cam.up.set(0, 0, 1);
      const hemi = new THREE.HemisphereLight(0xffffff, 0xd2d4d9, 0.9); hemi.position.set(0, 0, 1); sc.add(hemi);
      const key = this.key = new THREE.DirectionalLight(0xffffff, 0.95); key.castShadow = true; key.shadow.mapSize.set(1024, 1024);
      Object.assign(key.shadow.camera, { left: -1.4, right: 1.4, top: 1.4, bottom: -1.4, near: 0.1, far: 14 }); key.shadow.bias = -0.0006;
      sc.add(key); sc.add(key.target);
      const fill = new THREE.DirectionalLight(0xffffff, 0.35); fill.position.set(-3, 2, 2); sc.add(fill);
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.ShadowMaterial({ opacity: 0.14 })); floor.receiveShadow = true; sc.add(floor);
      const mats = {}, geos = {};
      const mat = (rgba) => { const k = rgba.join(); if (!mats[k]) { const dark = rgba[0] < 0.5;
        mats[k] = new THREE.MeshStandardMaterial({ color: dark ? 0x404347 : 0xd3d5d8, metalness: 0.25, roughness: dark ? 0.6 : 0.48, side: THREE.DoubleSide, flatShading: true }); } return mats[k]; };
      const geo = (name) => { if (geos[name]) return geos[name]; const m = M.meshes[name], g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bin, m.pos, m.n_verts * 3), 3));
        g.setIndex(new THREE.BufferAttribute(new Uint16Array(bin, m.idx, m.n_tris * 3), 1)); g.computeVertexNormals(); return (geos[name] = g); };
      this.root = new THREE.Group(); sc.add(this.root);
      const nodes = []; this.joints = [];
      for (const b of M.bodies) {
        const g = new THREE.Group(); g.position.fromArray(b.pos); g.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
        const j = new THREE.Group(); g.add(j);
        if (b.joint) this.joints.push({ o: j, i: b.joint.index, a: new THREE.Vector3().fromArray(b.joint.axis).normalize() });
        for (const gm of b.geoms) {
          const mesh = new THREE.Mesh(geo(gm.mesh), mat(gm.rgba)); mesh.position.fromArray(gm.pos);
          mesh.quaternion.set(gm.quat[1], gm.quat[2], gm.quat[3], gm.quat[0]); mesh.castShadow = true; j.add(mesh);
        }
        (b.parent < 0 ? this.root : nodes[b.parent]).add(g); nodes.push(j);
      }
      this.target = new THREE.Vector3(); this.size();
      addEventListener("resize", () => { this.size(); this.render(); });
    }
    size() { const w = Math.max(10, this.el.clientWidth), h = Math.max(10, this.el.clientHeight);
      this.r.setSize(w, h, false); this.r.domElement.style.width = "100%"; this.r.domElement.style.height = "100%";
      this.cam.aspect = w / h; this.cam.updateProjectionMatrix(); }
    set(clip, yaw, track) { this.clip = clip; this.yaw = yaw; this.track = track; }
    pose(f) {
      const c = this.clip, r = c.h2_root, q = c.h2_q, i = f * 7, k = f * 31;
      this.root.position.set(r[i], r[i + 1], r[i + 2]); this.root.quaternion.set(r[i + 4], r[i + 5], r[i + 6], r[i + 3]);
      for (const j of this.joints) j.o.quaternion.setFromAxisAngle(j.a, q[k + j.i]);
      const tx = this.track.px[f] / 100, ty = this.track.py[f] / 100, fx = Math.sin(this.yaw), fy = Math.cos(this.yaw), dist = 5.2;
      this.target.set(tx, ty, 0.88);
      this.cam.position.set(tx - fx * dist, ty - fy * dist, 0.88 + dist * Math.sin(PITCH)); this.cam.lookAt(this.target);
      this.key.position.set(tx + 1.4, ty - 2.2, 4.2); this.key.target.position.set(tx, ty, 0);
    }
    render() { this.r.render(this.scene, this.cam); }
  }
  const robot = new Robot3D($("#st-robot"), MODEL, BIN);
  const left = new Figure($("#cv-human"), { height: 215, ground: 0.9 });
  const TR = [[0, "left hip", INK], [3, "left knee", INK], [6, "right hip", QUIET], [9, "right knee", QUIET]];
  let head = null;
  function traces(c) {
    const el = $("#traces"); el.innerHTML = "";
    const W = Math.max(280, el.clientWidth), H = el.clientHeight || 96, n = c.frames, q = c.h2_q;
    const s = svgEl("svg", { viewBox: `0 0 ${W} ${H}` }, el);
    let lo = 1e9, hi = -1e9;
    for (const [j] of TR) for (let f = 0; f < n; f++) { const v = q[f * 31 + j] * 57.2958; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    lo = Math.floor(lo / 30) * 30; hi = Math.ceil(hi / 30) * 30;
    const X = (f) => 34 + (f / (n - 1)) * (W - 104), Y = (v) => 6 + (1 - (v - lo) / (hi - lo)) * (H - 26);
    for (const v of [lo, 0, hi]) { svgEl("line", { x1: 34, x2: W - 70, y1: Y(v), y2: Y(v), stroke: v === 0 ? HAIR : "#ececf0" }, s);
      const t = svgEl("text", { x: 28, y: Y(v) + 3, "text-anchor": "end" }, s); t.textContent = v + "°"; }
    TR.forEach(([j, name, col], i) => {
      const pts = []; for (let f = 0; f < n; f++) pts.push([X(f), Y(q[f * 31 + j] * 57.2958)]);
      svgEl("path", { d: path(pts), fill: "none", stroke: col, "stroke-width": 1.2, "stroke-dasharray": i % 2 ? "3 2" : "" }, s);
      const t = svgEl("text", { x: W - 64, y: 12 + i * 12 }, s); t.textContent = name; t.setAttribute("fill", col);
      svgEl("line", { x1: W - 70 + 3, x2: W - 70 + 3, y1: 0, y2: 0 }, s);
    });
    if (c.contacts) {   // annotated footfalls, left and right
      [0, 1].forEach((side) => { let f0 = -1;
        for (let f = 0; f <= n; f++) { const on = f < n && c.contacts[f * 2 + side];
          if (on && f0 < 0) f0 = f; if (!on && f0 >= 0) { svgEl("rect", { x: X(f0), y: H - 8 + side * 4, width: Math.max(1, X(f) - X(f0)), height: 3, fill: side ? QUIET : INK, rx: 1 }, s); f0 = -1; } } });
      const t = svgEl("text", { x: W - 64, y: H - 2 }, s); t.textContent = "foot down L / R";
    }
    head = svgEl("line", { x1: 34, x2: 34, y1: 2, y2: H, stroke: INK, "stroke-width": 1 }, s);
    head._X = X;
  }
  const hero = { f: 0, n: 1, size() { left.size(); robot.size(); }, draw(f) { left.draw(f); robot.pose(f); robot.render(); if (head) { const x = head._X(f); head.setAttribute("x1", x); head.setAttribute("x2", x); } } };
  const hp = { fig: hero, visible: false, paused: REDUCE }; players.add(hp);
  new IntersectionObserver((es) => { hp.visible = es[0].isIntersecting; }, { rootMargin: "80px" }).observe($(".hero"));
  for (const id of ["#st-human", "#st-robot"]) {
    const el = $(id); el.addEventListener("click", () => { hp.paused = !hp.paused; });
    el.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); hp.paused = !hp.paused; } });
  }
  const heading = (c) => { let sx = 0, sy3 = 0; for (let f = 0; f < c.frames; f++) { const i = f * 7, w = c.h2_root[i + 3], x = c.h2_root[i + 4], y = c.h2_root[i + 5], z = c.h2_root[i + 6];
    const a = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); sx += Math.cos(a); sy3 += Math.sin(a); } return Math.atan2(sy3, sx); };
  function pickTake(i) {
    const c = D.clips[i], h = heading(c), yaw = Math.atan2(-Math.cos(h + 0.75), -Math.sin(h + 0.75));
    setYaw(yaw);
    left.set([{ data: c.human, color: INK, width: 1.8, dots: 1.8, trail: true }], c.frames, follow(c.human, c.frames));
    robot.set(c, yaw, follow(c.robot, c.frames));
    hero.n = c.frames; hero.f = REDUCE ? (c.frames / 2) | 0 : 0;
    traces(c); hero.draw(hero.f);
    $("#cap-title").textContent = c.desc[0].toUpperCase() + c.desc.slice(1) + (c.activity ? ` · ${c.activity}, grade ${c.grade}` : "");
    $("#cap-src").textContent = (c.source === "cmu" ? "CMU" : "Bandai Namco") + (c.split === "test" ? ", person never trained on" : "");
    document.querySelectorAll("#chips button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  }
  $("#chips").innerHTML = D.clips.map((c) => `<button type="button" aria-pressed="false">${c.desc.split(",")[0]}</button>`).join("");
  document.querySelectorAll("#chips button").forEach((b, i) => b.addEventListener("click", () => pickTake(i)));
  $("#rb-title").textContent = `${R.name}, ${R.dof} joints, ${R.height_m} m`;
  $("#rb-src").textContent = "Unitree's own model and meshes";
  $("#cap-left").innerHTML = `The skeleton solved from optical markers, as recorded; these datasets ship no video. Lines: hip and knee angles a leg suit would measure. Bars: the annotated footfalls.`;
  $("#cap-right").innerHTML = `The same take on Unitree's MJCF with its real link meshes, posed every frame by our inverse kinematics: ${fmt(rt.frames)} frames solved, feet within <b>${fmt(rt.feet_err_cm, 1)} cm</b> of the human's on average. Wrists and head held still. Tap either view to pause.`;
  pickTake(0);
