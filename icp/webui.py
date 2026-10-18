"""Browser dashboard over generate.py (stage 1) and align.py (stage 2): what's on disk, run either stage with
live output, browse results/ with scores, images, downloads and a 3D overlay (three.js from a CDN).

    python icp/webui.py        # http://127.0.0.1:5000, localhost only

--show / --pick open an Open3D window on this machine; the run streams until that window is closed.
"""
import json, subprocess, sys
from pathlib import Path

import ifcopenshell
import numpy as np
import open3d as o3d
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.security import safe_join

from generate import storey

ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
app = Flask(__name__)


def files(kind, exts=(".ply", ".pcd", ".xyz")):
    d = DATA / kind
    return {f.stem: [f.name, f.stat().st_size] for f in sorted(d.iterdir()) if f.suffix.lower() in exts} if d.is_dir() else {}


def points(path):
    """x,y,z float32 buffer (at most ~400k points) for the 3D viewer."""
    if not path or not Path(path).is_file():
        abort(404)
    X = np.asarray(o3d.io.read_point_cloud(str(path)).points, np.float32)
    return Response(X[::max(1, len(X) // 400_000)].tobytes(), mimetype="application/octet-stream")


@app.get("/")
def index():
    return PAGE


@app.get("/api/state")
def state():
    s, b = files("scans"), files("bim")
    runs = []
    for d in sorted((x for x in RESULTS.iterdir() if x.is_dir()), reverse=True) if RESULTS.is_dir() else []:
        rep = d / "report.json"
        runs.append({"name": d.name, "rep": json.loads(rep.read_text()) if rep.is_file() else None,
                     "npy": (d / "transform.npy").is_file()})
    lt = RESULTS / "latest.txt"
    return jsonify(ifc=sorted(f.name for f in (DATA / "ifc").glob("*.ifc")), scans=list(s.values()), bim=list(b.values()),
                   pairs=sorted(set(s) & set(b)), only_scan=sorted(set(s) - set(b)), only_bim=sorted(set(b) - set(s)),
                   latest=Path(lt.read_text().strip()).name if lt.is_file() else None, runs=runs)


@app.get("/api/rooms")
def rooms():
    m = ifcopenshell.open(safe_join(str(ROOT), request.args["ifc"]) or abort(400))
    return jsonify(sorted({((s.Name or "").strip(), (s.LongName or "").strip(), storey(s)) for s in m.by_type("IfcSpace")}))


@app.post("/run/<script>")
def run(script):
    """Form fields named like CLI flags -> argv; streams the script's output, then '[exit N]'."""
    if script not in ("generate", "align"):
        abort(404)
    args = []
    for k, v in request.form.items(multi=True):
        if k.startswith("-") and v.strip():
            args += [k] if v == "on" else [k, *(v.split() if k == "--rooms" else [v.strip()])]
    cmd = [sys.executable, "-u", f"{script}.py", *args]

    def gen():
        yield "$ " + " ".join(cmd[2:]) + "\n"
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        yield from p.stdout
        yield f"\n[exit {p.wait()}]\n"
    return Response(gen(), mimetype="text/plain")


@app.get("/pts/<kind>/<name>")
def pts(kind, name):
    return points(safe_join(str(DATA / kind), name) if kind in ("bim", "scans") else None)


@app.get("/runpts/<run>/<which>")
def runpts(run, which):
    d = safe_join(str(RESULTS), run) or abort(400)
    if which == "source":
        return points(Path(d) / "source_aligned.ply")
    t = Path(json.loads((Path(d) / "report.json").read_text())["run"]["target"]).resolve()
    return points(t if t.is_file() and ROOT in t.parents else DATA / "bim" / t.name)  # old runs: moved project


@app.get("/res/<path:f>")
def res(f):
    return send_from_directory(RESULTS, f)


PAGE = """<!doctype html><meta charset=utf-8><title>SMCODE ICP</title>
<style>
:root{--bg:#0f1216;--panel:#161a20;--panel2:#1c2129;--border:#262c36;--text:#e7ebf0;--dim:#9aa4b2;--accent:#5b9dff;--warn:#e0a63e}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",system-ui,sans-serif;margin:0 auto 260px;max-width:1300px;padding:0 24px}
h1{font-size:22px;margin:24px 0 10px}h2{font-size:16px;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border)}
small{color:var(--dim);font-weight:normal;font-size:12.5px}code{font:12.5px Consolas,monospace;color:var(--dim)}a{color:var(--accent);margin-right:10px}
form{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 18px}
label{display:inline-flex;align-items:center;gap:6px;margin:5px 16px 5px 0;color:var(--dim)}
input,select,button{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 9px}
input[type=text],input[type=number]{width:120px}input[type=checkbox]{accent-color:var(--accent)}
button{cursor:pointer}button:hover{border-color:var(--accent)}
form>button:last-child{background:var(--accent);border:0;color:#fff;font-weight:600;padding:7px 20px;margin-top:10px}
table{border-collapse:collapse;width:100%;margin:6px 0}td,th{border-bottom:1px solid var(--border);padding:6px 10px;text-align:left;vertical-align:middle}
th{color:var(--dim);font-weight:600;font-size:12.5px}tr:hover td{background:#141820}
.warn{background:#3a2b12;color:var(--warn);font-weight:600}
.card{display:inline-block;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 18px;margin:4px 10px 4px 0;color:var(--dim)}
.card b{display:block;font-size:22px;color:var(--text)}
#con{position:fixed;left:0;right:0;bottom:0;height:220px;margin:0;background:#0a0c0f;border-top:1px solid var(--border);color:#cfd6df;
  font:12.5px/1.45 Consolas,monospace;overflow:auto;padding:10px 24px;white-space:pre-wrap}
dialog{width:90vw;height:85vh;padding:0;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:10px}
dialog::backdrop{background:#000a}dialog>div:first-child{padding:8px 12px}#vbox{height:calc(100% - 46px)}
</style>
<h1>SMCODE &mdash; BIM &harr; scan alignment</h1>
<div id=cards></div><p id=latest></p>
<h2>Alignable pairs</h2><table id=pairs></table>
<p>Scans with no BIM cloud: <b id=only_scan></b> &middot; BIM clouds with no scan: <b id=only_bim></b></p>
<h2>Point clouds on disk</h2><table><tr><td style=vertical-align:top><table id=bimt></table><td style=vertical-align:top><table id=scant></table></table>

<h2>1 &middot; Generate BIM cloud <small>(icp/generate.py &rarr; data/bim)</small></h2>
<form id=gen>
<label>IFC <select name=--ifc id=ifc></select></label> <button type=button onclick=listRooms()>List rooms</button>
<table id=roomt></table>
<label>rooms <input type=text name=--rooms id=rooms placeholder="203 207" style=width:200px></label>
<label><input type=checkbox name=--all> all spaces</label><br>
<label>spacing-xy <input type=number step=0.01 name=--spacing-xy placeholder=0.05></label>
<label>spacing-z <input type=number step=0.01 name=--spacing-z placeholder=0.05></label>
<label>ceiling <select name=--ceiling-mode><option value="">(config)<option>auto<option>space_top<option>fixed</select></label>
<label>clear-height <input type=number step=0.01 name=--clear-height placeholder=2.60></label>
<label>name-by <select name=--name-by><option value="">(config)<option>number<option>number_name<option>name</select></label>
<label>opening margin <input type=number step=0.01 name=--opening-margin placeholder=0.06></label><br>
<label><input type=checkbox name=--no-door-openings> don't carve doors</label>
<label><input type=checkbox name=--window-openings> carve windows</label>
<label><input type=checkbox name=--no-floor> no floor</label><label><input type=checkbox name=--no-walls> no walls</label>
<label><input type=checkbox name=--include-ceiling> ceiling</label><label><input type=checkbox name=--binary> binary PLY</label>
<label><input type=checkbox name=-v> verbose</label><br><button>Generate</button></form>

<h2>2 &middot; Align scan <small>(icp/align.py &rarr; results/)</small></h2>
<form id=al>
<label>pair <select name=--pair id=alpair></select></label><label><input type=checkbox name=--all-pairs> all pairs</label>
<label>or source <input type=text name=--source placeholder=data/scans/203.ply style=width:170px></label>
<label>target <input type=text name=--target placeholder=data/bim/203.ply style=width:170px></label><br>
<label>apply saved transform <select name=--apply id=apply></select></label>
<label>method <select name=--method><option value="">(config)<option>yaw<option>fpfh<option>pca<option>manual<option>all</select></label>
<label>tag <input type=text name=--tag></label><label>out dir <input type=text name=--out></label>
<label><input type=checkbox name=--show> Open3D viewer after</label>
<label><input type=checkbox name=--pick> click correspondences (manual)</label><br><button>Align</button></form>

<h2>Results <small>orange footprint = scan and BIM disagree by &gt;100 mm: residual is floored by that, fix geometry first</small></h2>
<table id=runs></table>
<pre id=con></pre>
<dialog id=dlg><div><b id=vt></b> &nbsp;<small>drag orbit &middot; right-drag pan &middot; scroll zoom</small>
<button onclick=dlg.close() style=float:right>Close</button></div><div id=vbox></div></dialog>

<script type=importmap>{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const $ = id => document.getElementById(id), esc = s => String(s ?? "").replace(/[&<>"]/g, c => `&#${c.charCodeAt(0)};`);
const f1 = v => v == null ? "-" : v.toFixed(1), opts = (a, blank) => (blank ? `<option value="">${blank}` : "") + a.map(v => `<option>${esc(v)}`).join("");
const mb = n => (n / 1048576).toFixed(1) + " MB", q = encodeURIComponent;

async function refresh() {
  const d = await (await fetch("/api/state")).json();
  $("cards").innerHTML = [["IFC files", d.ifc.length], ["alignable pairs", d.pairs.length], ["scans missing BIM", d.only_scan.length],
    ["BIM missing scan", d.only_bim.length], ["runs", d.runs.length]].map(([k, v]) => `<span class=card><b>${v}</b> ${k}</span>`).join("");
  $("latest").innerHTML = d.latest ? `Latest run: <code>${esc(d.latest)}</code>` : "No alignment runs yet.";
  $("pairs").innerHTML = d.pairs.map(p => `<tr><td>${esc(p)}<td><button onclick="alpair.value='${esc(p)}';al.scrollIntoView()">Align &rarr;</button>`).join("")
    || "<tr><td>none yet: generate a BIM cloud and/or put a matching scan in data/scans";
  $("only_scan").textContent = d.only_scan.join(", ") || "-"; $("only_bim").textContent = d.only_bim.join(", ") || "-";
  for (const [t, k] of [["bimt", "bim"], ["scant", "scans"]])
    $(t).innerHTML = `<tr><th>data/${k}<th>` + d[k].map(([n, s]) =>
      `<tr><td><code>${esc(n)}</code> ${mb(s)}<td><button onclick='view3d(${JSON.stringify(n)}, [["/pts/${k}/${q(n)}", "#5b9dff"]])'>3D</button>`).join("");
  const pv = alpair.value; $("ifc").innerHTML = d.ifc.map(f => `<option value="data/ifc/${esc(f)}">${esc(f)}`).join("");
  alpair.innerHTML = opts(d.pairs, "(none)"); alpair.value = pv || d.pairs[0] || "";
  $("apply").innerHTML = opts(d.runs.filter(r => r.npy).map(r => `results/${r.name}/transform.npy`), "(none)");
  $("runs").innerHTML = "<tr><th>run<th>best<th>fitness<th>median mm<th>p90 mm<th>yaw&deg;<th>footprint &Delta; mm<th>" + d.runs.map(r => {
    const rep = r.rep || {}, s = (rep.results || {})[rep.best] || {}, fp = (rep.footprint || {}).delta_mm || [0], n = q(r.name);
    const big = Math.max(...fp.map(Math.abs)) > 100;
    return `<tr><td><code>${esc(r.name)}</code><td>${esc(rep.best)}<td>${s.fitness?.toFixed(3) ?? "-"}<td>${f1(s.median_mm)}<td>${f1(s.p90_mm)}
      <td>${s.yaw_deg?.toFixed(2) ?? "-"}<td class=${big ? "warn" : ""}>${esc(fp.join(" / "))}<td>
      <button onclick='view3d(${JSON.stringify(r.name)}, [["/runpts/${n}/target", "#e0a63e"], ["/runpts/${n}/source", "#5b9dff"]])'>3D overlay</button>
      ${["overlay.png", "residual.png", "footprint_debug.png", "transform.txt", "transform.npy", "config_used.yaml", "source_aligned.ply", "report.json"]
        .map(f => `<a href="/res/${n}/${f}" target=_blank>${f}</a>`).join(" ")}`;
  }).join("");
}

async function runForm(e, script) {
  e.preventDefault(); con.textContent = "";
  const r = await fetch("/run/" + script, {method: "POST", body: new FormData(e.target)}), rd = r.body.getReader(), dec = new TextDecoder();
  for (let c; !(c = await rd.read()).done;) { con.textContent += dec.decode(c.value); con.scrollTop = 1e9; }
  refresh();
}
gen.onsubmit = e => runForm(e, "generate"); al.onsubmit = e => runForm(e, "align");

window.listRooms = async () => {
  roomt.innerHTML = "<tr><td>reading IFC...";
  const rows = await (await fetch("/api/rooms?ifc=" + q(ifc.value))).json();
  roomt.innerHTML = "<tr><th><th>number<th>name<th>storey" + rows.map(([n, l, s]) =>
    `<tr><td><input type=checkbox value="${esc(n)}" onchange="rooms.value=[...roomt.querySelectorAll(':checked')].map(c=>c.value).join(' ')">
     <td>${esc(n)}<td>${esc(l)}<td>${esc(s)}`).join("");
};

let R;
window.view3d = async (title, srcs) => {
  vt.textContent = title + " (loading...)"; dlg.showModal();
  const scene = new THREE.Scene(), box = new THREE.Box3();
  scene.background = new THREE.Color(0x101218);
  for (const [url, color] of srcs) {
    const r = await fetch(url);
    if (!r.ok) { vt.textContent = "failed to load " + url; return; }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(await r.arrayBuffer()), 3));
    g.computeBoundingBox(); box.union(g.boundingBox);
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({color, size: 2, sizeAttenuation: false})));
  }
  vt.textContent = title + "  " + srcs.map(s => s[1] + " = " + s[0].split("/").pop()).join(", ");
  const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length() || 1;
  const cam = new THREE.PerspectiveCamera(45, vbox.clientWidth / vbox.clientHeight, s / 1000, s * 20);
  cam.up.set(0, 0, 1); cam.position.copy(c).add(new THREE.Vector3(s * 0.7, s * 0.7, s * 0.5));
  R = new THREE.WebGLRenderer({antialias: true}); R.setSize(vbox.clientWidth, vbox.clientHeight); vbox.replaceChildren(R.domElement);
  const ctl = new OrbitControls(cam, R.domElement); ctl.target.copy(c); ctl.update();
  R.setAnimationLoop(() => R.render(scene, cam));
};
dlg.onclose = () => { R?.setAnimationLoop(null); R?.dispose(); vbox.replaceChildren(); };
refresh();
</script>"""

if __name__ == "__main__":
    print("SMCODE dashboard: http://127.0.0.1:5000")
    app.run(port=5000, threaded=True)
