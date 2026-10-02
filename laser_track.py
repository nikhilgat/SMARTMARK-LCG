"""Point the galvo laser at objects the camera sees: YOLO box -> distance (D435i depth) -> one basic shape
(4-corner box or circle) around the object in 3D -> galvo mirror values (captures/galvo/galvo_calib.json).

    python calib/laser_track.py                 # GUI, laser on COM5
    python calib/laser_track.py --dry-run       # everything except the laser
    python calib/laser_track.py --selftest      # geometry / target logic, no hardware

Classes: names from the 80 COCO classes (chair, couch, tv, laptop, bottle, dining table, person, ...) run on
YOLO11; any other name (door, whiteboard, ...) switches to open-vocabulary YOLO-World (first use downloads
the model and its text encoder). One object at a time: the laser only draws basic shapes and has no blanking.

Used for: the end application - detect a chosen kind of object live and outline it with the laser.
Works with: galvo_calib.py (Galvo, Camera, to_dac, covered and galvo_calib.json must exist first, which in
          turn needs calibrate.py's intrinsics). Hardware: RealSense D435i (colour + depth), galvo laser on COM5.
Needs: ultralytics (YOLO), torch, pyrealsense2, pyserial, opencv, pillow, tkinter.
"""
import galvo_calib as gc  # first: disables OpenCV OpenCL before cv2 loads
import argparse, threading, time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------- pure logic (tested in --selftest)
def object_depth(depth, box, inner=0.5):
    """How far away the detected object is.
    Median depth (m) of the central `inner` part of the box; None if too few valid pixels.
    (Centre only, so background around the object's edges doesn't skew it.)"""
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    r = depth[int(max(cy - hh, 0)):int(cy + hh) + 1, int(max(cx - hw, 0)):int(cx + hw) + 1]
    r = r[r > 0]
    return float(np.median(r)) if r.size > 20 else None


def shape_3d(box, z, K, dist, kind="box", margin=0.05, n=24):
    """Turn a 2D detection box into the 3D outline the laser should draw.
    Box/circle around the image box, placed in a plane facing the camera at depth z (camera frame, m).
    margin: extra metres around the object; n: points used for a circle. Returns (N, 3) points."""
    x1, y1, x2, y2 = box
    uv = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    xy = cv2.undistortPoints(uv, K, dist).reshape(-1, 2) * z
    c = xy.mean(0)
    hx, hy = np.ptp(xy[:, 0]) / 2, np.ptp(xy[:, 1]) / 2
    if kind == "circle":
        r = np.hypot(hx, hy) + margin  # circumscribes the box
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]
    else:
        hx, hy = hx + margin, hy + margin
        pts = np.array([[c[0] - hx, c[1] - hy], [c[0] + hx, c[1] - hy], [c[0] + hx, c[1] + hy], [c[0] - hx, c[1] + hy]])
    return np.c_[pts, np.full(len(pts), z)]


def pick(dets, state, mode, cycle_s, now):
    """Choose which detection the laser should outline this frame (the laser can only do one at a time).
    Keep the locked track while it is visible; 'cycle' hops left-to-right every cycle_s; else best conf.
    `state` remembers the current target id between calls."""
    if not dets:
        return None
    cur = next((d for d in dets if d["id"] is not None and d["id"] == state.get("id")), None)
    if cur and not (mode == "cycle" and len(dets) > 1 and now - state.get("since", now) >= cycle_s):
        return cur
    if cur:
        order = sorted(dets, key=lambda d: d["box"][0])
        nxt = order[(order.index(cur) + 1) % len(order)]
    else:
        nxt = max(dets, key=lambda d: d["conf"])
    state.update(id=nxt["id"], since=now)
    return nxt


# ---------------------------------------------------------------- tracking worker
class Tracker:
    """Runs everything except the GUI: camera thread + detection/laser loop. The GUI changes `cfg` and
    reads `view` (annotated image) and `status`."""

    def __init__(self, a):
        """Load both calibrations, open the camera (with depth) and the laser (falls back to no laser if the
        port can't be opened), and set the default settings."""
        self.K, self.dist = gc.load_cam()
        self.calib = gc.load_calib()
        self.cam = gc.Camera(depth=True)
        try:
            self.gal, self.laser_msg = gc.Galvo(a.port, dry=a.dry_run), ("dry run: laser disabled" if a.dry_run else f"laser on {a.port}")
        except Exception as e:  # port busy / missing: keep tracking, no laser
            self.gal, self.laser_msg = gc.Galvo(dry=True), f"laser NOT connected ({e})"
        self.cfg = dict(classes="chair", ignore="person", conf=0.35, shape="box", margin=0.05, mode="lock", cycle_s=4.0,
                        smooth=0.5, speed=4000, wait=3000, laser=False, outside=False)
        self.model_name, self.lock, self.running = a.model, threading.Lock(), True
        self.frame, self.view, self.status = None, None, "starting..."
        self.models, self.target_state = {}, {}

    def cam_loop(self):
        """Camera thread: always hold the newest aligned colour + depth frame (older ones are dropped)."""
        while self.running:
            try:
                f = self.cam.align.process(self.cam.pipe.wait_for_frames(2000))
            except RuntimeError:
                continue
            color = np.asanyarray(f.get_color_frame().get_data()).copy()
            depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * self.cam.depth_scale
            with self.lock:
                self.frame = (color, depth)

    def model_for(self, names):
        """Pick (and load once, then cache) the detector for the requested class names.
        (model, class indices or None). COCO names -> YOLO11, anything else -> YOLO-World with those names."""
        from ultralytics import YOLO
        if "base" not in self.models:
            self.models["base"] = YOLO(self.model_name)
        base = self.models["base"]
        coco = {v: k for k, v in base.names.items()}
        if all(n in coco for n in names):
            return base, [coco[n] for n in names]
        key = ("world",) + tuple(names)
        if key not in self.models:
            w = YOLO("yolov8s-worldv2.pt")
            w.set_classes(list(names))
            self.models = {"base": base, key: w}  # keep one world model around
        return self.models[key], None

    def run(self):
        """Main loop, once per camera frame: detect + track objects -> pick one target -> get its depth ->
        build the 3D shape -> convert to mirror values (smoothed) -> send to the laser if allowed.
        The laser is blocked/stopped if the target is outside the calibrated area, out of mirror range,
        or lost for >0.7 s. Shapes are only re-sent when they move noticeably, to avoid flicker."""
        drawing, last_sent, last_seen, smoothed, last_key, t_prev = False, None, 0.0, None, None, time.time()
        while self.running:
            with self.lock:
                fr, self.frame = self.frame, None
            if fr is None:
                time.sleep(0.005)
                continue
            color, depth = fr
            c = dict(self.cfg)
            names = [s.strip() for s in c["classes"].split(",") if s.strip()]
            ignore = {s.strip() for s in c["ignore"].split(",") if s.strip()}
            view, now = color.copy(), time.time()
            x0, x1, y0, y1 = self.calib["coverage"]  # calibrated area (camera-ray directions) -> image rectangle
            cv2.rectangle(view, (int(self.K[0, 0] * x0 + self.K[0, 2]), int(self.K[1, 1] * y0 + self.K[1, 2])),
                          (int(self.K[0, 0] * x1 + self.K[0, 2]), int(self.K[1, 1] * y1 + self.K[1, 2])), (255, 140, 0), 1)
            blocked = ""
            try:
                model, idx = self.model_for(names) if names else (None, None)
            except Exception as e:
                self.status, self.view = f"model error: {e}", view
                time.sleep(0.5)
                continue
            dets = []
            if model is not None:
                r = model.track(color, persist=True, conf=c["conf"], classes=idx, verbose=False)[0]
                b = r.boxes
                for i in range(len(b)):
                    name = r.names[int(b.cls[i])]
                    if name in ignore:
                        continue
                    dets.append(dict(id=None if b.id is None else int(b.id[i]), cls=name, conf=float(b.conf[i]),
                                     box=b.xyxy[i].tolist()))
            for d in dets:
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(view, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(view, f"{d['cls']} {d['conf']:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            tgt, msg = pick(dets, self.target_state, c["mode"], c["cycle_s"], now), "no target"
            if tgt:
                last_seen = now
                x1, y1, x2, y2 = map(int, tgt["box"])
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)
                z = object_depth(depth, tgt["box"])
                if z is None:
                    msg = f"{tgt['cls']}: no depth"
                else:
                    P = shape_3d(tgt["box"], z, self.K, self.dist, c["shape"], c["margin"])
                    uv = cv2.projectPoints(P, np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2).astype(int)
                    cv2.polylines(view, [uv], True, (0, 255, 0), 2)  # where the laser shape should land
                    dac = gc.to_dac(P, self.calib)
                    key = (tgt["id"], c["shape"], len(P))
                    smoothed = dac if smoothed is None or key != last_key else c["smooth"] * smoothed + (1 - c["smooth"]) * dac
                    last_key = key
                    msg = f"{tgt['cls']} #{tgt['id']} {tgt['conf']:.2f} at {z:.2f} m"
                    if not gc.covered(P, self.calib).all() and not c["outside"]:
                        blocked = "target OUTSIDE calibrated area (blue box) - tick 'Draw outside' or recalibrate there"
                    elif (smoothed < 0).any() or (smoothed > 65535).any():
                        blocked = "target out of the laser's mirror range"
                    elif c["laser"]:
                        moved = np.inf if last_sent is None or len(last_sent) != len(smoothed) else np.abs(smoothed - last_sent).max()
                        if moved > 250:  # mirror units; avoids re-sending (and flicker) for sub-cm jitter
                            self.gal.speed = int(c["speed"])
                            self.gal.shape(smoothed, int(c["wait"]))
                            last_sent, drawing = smoothed.copy(), True
                        msg += " | laser ON"
            if c["laser"] and not blocked and not tgt:
                blocked = "no target in view"
            if c["laser"] and blocked:
                msg += " | LASER BLOCKED"
                cv2.putText(view, "LASER BLOCKED: " + blocked, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if drawing and (not c["laser"] or blocked or now - last_seen > 0.7):  # blocked, target gone or laser off
                self.gal.stop()
                drawing, last_sent = False, None
            fps = 1 / max(now - t_prev, 1e-3)
            t_prev = now
            self.status = f"{msg} | {len(dets)} detections | {fps:.0f} fps | {self.laser_msg}"
            self.view = view

    def close(self):
        """Stop the loops, turn the laser off and release the camera."""
        self.running = False
        time.sleep(0.2)
        self.gal.close()
        self.cam.close()


# ---------------------------------------------------------------- GUI
def run_gui(trk):
    """Tkinter window: live annotated video on the left, settings (classes, confidence, shape, margin,
    lock/cycle, smoothing, galvo speed/wait) and the big LASER ON/OFF button on the right.
    SPACE toggles the laser. Settings are copied into trk.cfg every 40 ms."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Laser track")
    video = ttk.Label(root)
    video.grid(row=0, column=0, rowspan=30, padx=6, pady=6)
    ctl = ttk.Frame(root, padding=8)
    ctl.grid(row=0, column=1, sticky="n")

    v = {k: (tk.StringVar if isinstance(val, str) else tk.BooleanVar if isinstance(val, bool) else tk.DoubleVar)(value=val)
         for k, val in trk.cfg.items()}
    row = [0]

    def add(label, widget):
        """Add one labelled control as the next row of the settings panel."""
        ttk.Label(ctl, text=label).grid(row=row[0], column=0, sticky="w", pady=2)
        widget.grid(row=row[0], column=1, sticky="ew", pady=2)
        row[0] += 1

    classes = ttk.Entry(ctl, width=28)
    classes.insert(0, trk.cfg["classes"])
    add("Classes (comma separated)", classes)
    ttk.Button(ctl, text="Apply classes", command=lambda: v["classes"].set(classes.get())).grid(row=row[0], column=1, sticky="ew")
    row[0] += 1
    add("Ignore classes", ttk.Entry(ctl, textvariable=v["ignore"], width=28))
    add("Confidence", tk.Scale(ctl, from_=0.1, to=0.9, resolution=0.05, orient="horizontal", variable=v["conf"]))
    add("Shape", ttk.Combobox(ctl, textvariable=v["shape"], values=["box", "circle"], state="readonly"))
    margin_cm = tk.DoubleVar(value=trk.cfg["margin"] * 100)
    add("Margin (cm)", tk.Scale(ctl, from_=0, to=30, orient="horizontal", variable=margin_cm))
    add("Target", ttk.Combobox(ctl, textvariable=v["mode"], values=["lock", "cycle"], state="readonly"))
    add("Cycle every (s)", tk.Scale(ctl, from_=1, to=15, orient="horizontal", variable=v["cycle_s"]))
    add("Smoothing", tk.Scale(ctl, from_=0, to=0.9, resolution=0.05, orient="horizontal", variable=v["smooth"]))
    add("Galvo speed (F)", tk.Scale(ctl, from_=500, to=20000, resolution=500, orient="horizontal", variable=v["speed"]))
    add("Galvo wait (W)", tk.Scale(ctl, from_=0, to=20000, resolution=500, orient="horizontal", variable=v["wait"]))
    ttk.Checkbutton(ctl, text="Draw outside calibrated area (less accurate)", variable=v["outside"]).grid(
        row=row[0], column=0, columnspan=2, sticky="w")
    row[0] += 1

    btn = tk.Button(ctl, font=("Segoe UI", 16, "bold"), height=2)

    def paint_btn():
        """Update the laser button's text/colour to match the current on/off state."""
        on =v["laser"].get()
        btn.config(text="LASER ON  (click to stop)" if on else "LASER OFF  (click to start)",
                   bg="#2e7d32" if on else "#555", fg="white", activebackground="#1b5e20" if on else "#333")
    btn.config(command=lambda: (v["laser"].set(not v["laser"].get()), paint_btn()))
    paint_btn()
    btn.grid(row=row[0], column=0, columnspan=2, sticky="ew", pady=(14, 6))
    row[0] += 1
    status = ttk.Label(ctl, wraplength=380, justify="left")
    status.grid(row=row[0], column=0, columnspan=2, sticky="w")
    root.bind("<space>", lambda e: btn.invoke())
    ttk.Label(ctl, text="SPACE toggles the laser. Grey = detections, yellow = target, green = laser shape, blue = calibrated area.",
              wraplength=380, foreground="#666").grid(row=row[0] + 1, column=0, columnspan=2, sticky="w", pady=6)

    def tick():
        """Every 40 ms: push GUI settings to the tracker and show its latest image and status."""
        trk.cfg = {k: (var.get() if k != "margin" else margin_cm.get() / 100) for k, var in v.items()}
        if trk.view is not None:
            im = cv2.resize(trk.view, (1024, 576))[:, :, ::-1]
            photo = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(im)))
            video.configure(image=photo)
            video.image = photo
        status.configure(text=trk.status)
        root.after(40, tick)

    def on_close():
        """Window closed: shut the tracker down (laser off) before exiting."""
        trk.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    tick()
    root.mainloop()


def selftest():
    """No-hardware check of the pure logic: object_depth, shape_3d (corners land on the box, margin, circle),
    pick (lock/cycle behaviour) and the serial message format."""
    K = np.array([[892.0, 0, 645], [0, 890, 385], [0, 0, 1]]); dist = np.array([0.11, -0.26, 0.003, -0.003, 0.05])
    depth = np.zeros((720, 1280), np.float32); depth[200:400, 300:600] = 2.0; depth[250:260, 400:410] = 9.0
    assert object_depth(depth, (300, 200, 600, 400)) == 2.0 and object_depth(np.zeros((720, 1280)), (0, 0, 50, 50)) is None
    box = (400.0, 200.0, 700.0, 500.0)
    P = shape_3d(box, 2.0, K, dist, "box", margin=0.0)
    uv = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    assert np.abs(uv[0] - box[:2]).max() < 3 and np.abs(uv[2] - box[2:]).max() < 3, uv   # corners land on the box
    Pm = shape_3d(box, 2.0, K, dist, "box", margin=0.1)
    assert np.isclose(np.ptp(Pm[:, 0]) - np.ptp(P[:, 0]), 0.2) and (Pm[:, 2] == 2.0).all()
    C = shape_3d(box, 2.0, K, dist, "circle", margin=0.0, n=24)
    r = np.linalg.norm(C[:, :2] - C[:, :2].mean(0), axis=1)
    assert len(C) == 24 and np.allclose(r, r[0]) and r[0] >= np.hypot(np.ptp(P[:, 0]), np.ptp(P[:, 1])) / 2 - 1e-9
    d = [dict(id=1, conf=0.5, box=[100, 0, 1, 1]), dict(id=2, conf=0.9, box=[500, 0, 1, 1]), dict(id=3, conf=0.6, box=[300, 0, 1, 1])]
    st = {}
    assert pick(d, st, "lock", 4, 0)["id"] == 2                     # best confidence first
    d[0]["conf"] = 0.99
    assert pick(d, st, "lock", 4, 10)["id"] == 2                    # stays locked while visible
    assert pick(d[:1] + d[2:], st, "lock", 4, 11)["id"] == 1        # lost -> best remaining
    st = {}
    pick(d, st, "cycle", 4, 0)
    assert pick(d, st, "cycle", 4, 1)["id"] == st["id"] and pick(d, st, "cycle", 4, 5)["id"] == 3 and pick(d, st, "cycle", 4, 9.5)["id"] == 2
    assert pick([], {}, "lock", 4, 0) is None
    m = gc.shape_message([(-5, 70000), (100.4, 200.6)], 4000, 3000)
    assert m == "SC\nX0Y65535F4000W3000E\nX100Y201F4000W3000E\nM11eR", m
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO model for COCO classes (downloads on first use)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        import torch  # noqa: F401  main-thread imports before threads start (concurrent pybind11/torch imports can hang)
        import ultralytics  # noqa: F401
        trk = Tracker(a)
        threading.Thread(target=trk.cam_loop, daemon=True).start()
        threading.Thread(target=trk.run, daemon=True).start()
        run_gui(trk)
