"""Panorama -> room layout web UI over the vendored HorizonNet / LGT-Net inference scripts.

    python localization360/app.py        # http://127.0.0.1:5001

Pick a model, optionally upload an equirectangular .png/.jpg (empty = the model's bundled demo), Run.
Each run goes to localization360/runs/<id>/. Both repos carry small local patches for current
numpy/OpenCV/PyTorch (torch.load weights_only, np.float aliases, findContours tuple, json floats).
"""
import json, os, shutil, subprocess, sys, uuid
from pathlib import Path

from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
HN, LGT = HERE / "HorizonNet", HERE / "LGT-Net"
DEMO = {"horizonnet": HN / "assets" / "demo.png", "lgtnet": LGT / "src" / "demo" / "demo1.png"}
TIMEOUT = 180
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 << 20


def sh(args, cwd, log, **kw):
    """Run one inference script; append its output to log. True on success."""
    p = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, **kw)
    log.append(f"$ {args[0]}\n{p.stdout}{p.stderr}")
    return p.returncode == 0


def horizonnet(img, run, log):
    pre, inf, a = run / "preprocessed", run / "inferenced", f"{img.stem}_aligned_rgb"
    ok = sh(["preprocess.py", "--img_glob", img, "--output_dir", pre], HN, log) and \
        sh(["inference.py", "--pth", "ckpt/resnet50_rnn__panos2d3d.pth", "--img_glob", pre / f"{a}.png",
            "--output_dir", inf, "--visualize"], HN, log)
    return ok and (inf / f"{a}.raw.png", inf / f"{a}.json")


def lgtnet(img, run, log):
    out = run / "output"  # Agg: the script's plt.show() otherwise blocks forever headless
    ok = sh(["inference.py", "--cfg", "src/config/mp3d.yaml", "--img_glob", img, "--output_dir", out,
             "--post_processing", "manhattan"], LGT, log, env={**os.environ, "MPLBACKEND": "Agg"})
    return ok and (out / f"{img.stem}_pred.png", out / f"{img.stem}_pred.json")


@app.get("/")
def index():
    return PAGE


@app.post("/run")
def run():
    model, up = request.form.get("model"), request.files.get("image")
    if model not in DEMO:
        return {"error": "unknown model"}, 400
    if up and up.filename and Path(up.filename).suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return {"error": "only .png/.jpg"}, 400
    rid = uuid.uuid4().hex
    (RUNS / rid / "input").mkdir(parents=True)
    if up and up.filename:
        img = RUNS / rid / "input" / (secure_filename(up.filename) or "upload.png"); up.save(img)
    else:
        img = Path(shutil.copy(DEMO[model], RUNS / rid / "input"))
    log = []
    try:
        res = (horizonnet if model == "horizonnet" else lgtnet)(img, RUNS / rid, log)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {TIMEOUT}s", "log": "\n".join(log)}, 504
    log = "\n".join(log)[-4000:]
    if not res or not all(f.is_file() for f in res):
        return {"error": "inference failed", "log": log}, 500
    return {"image": f"/runs/{rid}/{res[0].relative_to(RUNS / rid).as_posix()}",
            "json": json.loads(res[1].read_text()), "log": log}


@app.get("/runs/<rid>/<path:f>")
def files(rid, f):
    return send_from_directory(RUNS / secure_filename(rid), f)


PAGE = """<!doctype html><meta charset=utf-8><title>LOCALIZATION-360</title>
<style>
body{background:#12141a;color:#e6e8ef;font:15px/1.5 "Segoe UI",system-ui,sans-serif;max-width:1000px;margin:32px auto;padding:0 24px}
h1{font-size:20px}form{background:#1b1e27;border:1px solid #2c3040;border-radius:10px;padding:16px 20px;line-height:2.2}
label{margin-right:18px}input{font:inherit;color:#8b90a3;accent-color:#5b8cff}
button{font:inherit;background:#5b8cff;border:0;border-radius:6px;padding:7px 22px;color:#fff;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
img{max-width:100%;border-radius:8px;margin:12px 0}summary{cursor:pointer;color:#8b90a3;margin:10px 0}
pre{background:#0b0d12;border:1px solid #2c3040;border-radius:8px;padding:10px;max-height:400px;overflow:auto;font:12.5px Consolas,monospace}
#s{color:#4cc38a}.err{color:#e05a5a!important}
</style>
<h1>LOCALIZATION-360 &mdash; panorama layout</h1>
<form id=f><label><input type=radio name=model value=horizonnet checked> HorizonNet</label>
<label><input type=radio name=model value=lgtnet> LGT-Net</label><br><br>
Panorama (optional, equirectangular .png/.jpg; empty = bundled demo): <input type=file name=image accept=.png,.jpg,.jpeg>
<button id=b>Run</button></form><p id=s></p><img id=i hidden>
<details><summary>Layout JSON</summary><pre id=j></pre></details><details><summary>Run log</summary><pre id=l></pre></details>
<script>
f.onsubmit = async e => {
  e.preventDefault(); b.disabled = true; i.hidden = true; s.className = ""; s.textContent = "Running (up to a minute or two)...";
  try {
    const r = await fetch("/run", {method: "POST", body: new FormData(f)}), d = await r.json();
    l.textContent = d.log || "";
    if (!r.ok) { s.className = "err"; s.textContent = "Failed: " + d.error; return; }
    s.textContent = "Done."; i.src = d.image; i.hidden = false; j.textContent = JSON.stringify(d.json, null, 2);
  } catch (err) { s.className = "err"; s.textContent = err; } finally { b.disabled = false; }
};
</script>"""

if __name__ == "__main__":
    app.run(port=5001, threaded=True)
