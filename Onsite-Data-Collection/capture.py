"""Site capture: RealSense (colour + depth) + Insta360 X4 (360) + Unitree L2, live view, one button grabs all.

Prereq: LiDAR bridge running (..\\bridge\\start_lidar.ps1), RealSense on USB3, X4 in USB webcam mode.

    python capture.py                  # saves to Onsite-Data-Collection/captures/

Each capture -> captures/<name>_<YYYYmmdd-HHMMSS>/
    <name>_realsense_color.png    colour frame taken inside the LiDAR window
    <name>_realsense_depth.png    uint16 depth aligned to the colour frame (same pixel grid + intrinsics);
                                  metres = value * depth_scale (meta.json), 0 = no reading
    <name>_realsense.ply          coloured point cloud from depth, RealSense colour-camera frame (x right, y down, z fwd), m
    <name>_insta360.jpg           X4 stitched equirectangular frame (2880x1440)
    <name>_lidar.npz              xyz, intensity, frame_sizes, frame_stamps, frame_host_times
    <name>_lidar.ply              same points, intensity as grey
    <name>_meta.json              name, notes, timestamps per sensor, RealSense intrinsics + depth_scale,
                                  IMU gravity (imu_accel_mean, tilt_deg), missing sensors
Plus captures/index.csv: one row per capture (session overview).
A sensor that delivers nothing in the window is listed under "missing" and the rest is still saved.
"""
import argparse, collections, csv, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calib"))
import capture_gui as cg  # noqa: E402  first: disables OpenCV OpenCL before cv2 loads
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import re, subprocess  # noqa: E402

INSTA_NAME, INSTA_W, INSTA_H = "Insta360 X4", 2880, 1440  # X4 in USB webcam mode: in-camera stitched equirect


def insta_index(name=INSTA_NAME):
    """DirectShow index of the X4 (ffmpeg's dshow device order = OpenCV DSHOW index), or None."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                         capture_output=True, text=True).stderr
    videos = re.findall(r'"([^"]+)" \(video\)', out)
    return next((i for i, v in enumerate(videos) if name.lower() in v.lower()), None)


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s.strip()) or "capture"


def depth_to_points(depth, color, K, scale):
    """Aligned depth + colour -> (xyz (N,3) m, rgb (N,3) 0-1), pixels without a reading dropped.
    ponytail: pinhole only, ignores the colour lens distortion (D435i colour coeffs are ~0); undistort if that changes."""
    v, u = np.nonzero(depth)
    z = depth[v, u] * scale
    xyz = np.c_[(u - K[0][2]) * z / K[0][0], (v - K[1][2]) * z / K[1][1], z]
    return xyz, color[v, u, ::-1] / 255.0


def tilt_deg(accel):
    """Angle between the camera's 'up' (-y, RealSense frame) and measured gravity reaction; 0 = camera level."""
    return float(np.degrees(np.arccos(np.clip(-accel[1] / np.linalg.norm(accel), -1, 1))))


EXTRINSICS = HERE.parent / "captures" / "calibration" / "extrinsics.json"


def level_view(accel=None):
    """Start view for the LiDAR cloud: (forward, up) in the LiDAR frame. Forward = LiDAR +z (into the screen, same
    as the RealSense viewing axis on this rig), flattened to horizontal; up = gravity from the RealSense IMU rotated
    in with the calib extrinsics. No IMU -> the camera's -y as up (level only if the rig is). No extrinsics -> identity."""
    R = np.array(json.loads(EXTRINSICS.read_text())["T_lidar_cam"])[:3, :3] if EXTRINSICS.exists() else np.eye(3)
    up = R @ (np.asarray(accel, float) if accel is not None else np.array([0.0, -1.0, 0.0]))  # at rest accel points up
    up /= np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
    return fwd / np.linalg.norm(fwd), up


def fit_view(xyz, accel=None, aspect=1.0, fov_deg=60.0):
    """(center, eye, up): look along level_view's forward (LiDAR +z into the screen), centred on the cloud, backed
    off just far enough that it fills the view. aspect = view width / height; fov_deg is vertical (Open3D).
    ponytail: fits 99.5% of points so a few stray far returns don't shrink everything else."""
    fwd, up = level_view(accel)
    axes = np.c_[np.cross(fwd, up), up, fwd]          # screen right, screen up, depth
    p = xyz @ axes
    lo, hi = np.percentile(p, [0.5, 99.5], axis=0)
    mid = (lo + hi) / 2
    a, b, c = (p - mid).T
    tv = np.tan(np.radians(fov_deg / 2))
    need = np.maximum(np.abs(a) / (tv * aspect), np.abs(b) / tv) - c  # eye distance at which each point is in view
    dist = max(np.percentile(need, 99.5) * 1.05, 0.5)
    center = axes @ mid
    return center, center - fwd * dist, up


INDEX_FIELDS = ["folder", "name", "timestamp", "seconds", "lidar_points", "tilt_deg", "missing", "notes"]


class SiteRig(cg.Rig):
    """capture_gui.Rig's LiDAR thread, plus RealSense colour+depth and X4 threads, and a per-capture folder."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.depth, self.depth_scale = None, None
        self.pano, self.cap_pano, self.insta_status = None, None, "starting..."
        self.accel, self.imu_status = collections.deque(maxlen=2000), "starting..."  # (t, ax, ay, az), ~20 s at 100 Hz

    def imu_loop(self):
        """D435i accelerometer on its own pipeline (mixing motion + video in one pipeline is flaky). Keeps recent samples."""
        import pyrealsense2 as rs
        while self.running:
            pipe = rs.pipeline()
            try:
                cfg = rs.config(); cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)  # fw 5.17: 100/200/400 Hz
                pipe.start(cfg)
                self.imu_status = "streaming"
                while self.running:
                    a = pipe.wait_for_frames(2000).first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
                    self.accel.append((time.time(), a.x, a.y, a.z))
            except RuntimeError as e:
                self.imu_status = f"not available ({e})"
                time.sleep(2.0)
            finally:
                try:
                    pipe.stop()
                except RuntimeError:
                    pass

    def camera_loop(self, width, height):
        """RealSense colour + depth. Newest -> cam_img/depth (raw, preview), first pair in a capture -> cap_img (depth aligned)."""
        import pyrealsense2 as rs
        while self.running:
            if not len(rs.context().query_devices()):  # pipe.start() with no device blocks ~15 s holding the GIL
                self.cam_status = "no RealSense found - plug it in (USB3)"
                time.sleep(1.0)
                continue
            pipe, align = rs.pipeline(), rs.align(rs.stream.color)
            try:
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
                prof = pipe.start(cfg)
                self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
                i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.intr = {"K": [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], "dist": list(i.coeffs),
                             "model": str(i.model), "width": i.width, "height": i.height, "source": "realsense factory"}
                self.cam_status = f"streaming colour + depth {i.width}x{i.height}"
                while self.running:
                    fs = pipe.wait_for_frames(2000)
                    img = np.asanyarray(fs.get_color_frame().get_data()).copy()
                    depth = np.asanyarray(fs.get_depth_frame().get_data()).copy()  # unaligned: live preview only
                    t = time.time()
                    with self.lock:
                        self.cam_img, self.depth, self.cam_t = img, depth, t
                        want = self.cap_on and self.cap_img is None and t >= self.cap_start
                    if want:  # align (CPU-heavy) only the one frame that gets saved
                        aligned = np.asanyarray(align.process(fs).get_depth_frame().get_data()).copy()
                        with self.lock:
                            self.cap_img = (t, img, aligned)
            except RuntimeError as e:
                self.cam_status = f"not available ({e}) - retrying"
                time.sleep(2.0)
            finally:
                try:
                    pipe.stop()
                except RuntimeError:
                    pass

    def insta_loop(self):
        """Keep the X4 stream open; newest frame -> self.pano, first frame inside a capture window -> cap_pano."""
        while self.running:
            try:
                i = insta_index()
                if i is None:
                    raise RuntimeError(f"{INSTA_NAME} not found (webcam mode?)")
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, INSTA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INSTA_H)
                while self.running:
                    ok, img = cap.read()
                    if not ok:
                        raise RuntimeError("no frame")
                    t = time.time()
                    self.insta_status = f"streaming {img.shape[1]}x{img.shape[0]}"
                    with self.lock:
                        self.pano = img
                        if self.cap_on and self.cap_pano is None and t >= self.cap_start:
                            self.cap_pano = (t, img)
                cap.release()
            except RuntimeError as e:
                self.insta_status = f"not available ({e})"
                time.sleep(2.0)

    def capture(self, seconds, name, notes=""):
        """Record `seconds` of LiDAR plus one RealSense pair and one 360 frame from the same window, save them in
        captures/<name>_<timestamp>/ and append a row to captures/index.csv. Returns (folder or None, message)."""
        with self.lock:
            self.cap_frames, self.cap_img, self.cap_pano, self.cap_start, self.cap_on = [], None, None, time.time(), True
        time.sleep(seconds)
        deadline = time.time() + 1.0  # camera frames may land just after the window opened
        while (self.cap_img is None or self.cap_pano is None) and time.time() < deadline:
            time.sleep(0.02)
        with self.lock:
            self.cap_on = False
            frames, cam, pano, t0 = [f for f in self.cap_frames if len(f[2])], self.cap_img, self.cap_pano, self.cap_start
        imu = np.array([s[1:] for s in list(self.accel) if t0 <= s[0] <= t0 + seconds])
        missing = [s for s, v in (("lidar", frames), ("realsense", cam), ("insta360", pano), ("imu", len(imu))) if not v]
        if len(missing) == 4:
            return None, "NOT saved: no sensor delivered data"

        name = safe_name(name)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(t0))
        d = self.out / f"{name}_{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        f = lambda sensor: d / f"{name}_{sensor}"
        meta = {"name": name, "notes": notes, "timestamp": stamp, "seconds": seconds, "t_window_start": t0,
                "t_window_end": t0 + seconds, "missing": missing}
        if len(imu):  # tripod still -> mean accel = gravity reaction in the RealSense frame
            g = imu.mean(0)
            meta.update(imu_accel_mean=g.tolist(), imu_accel_std=imu.std(0).tolist(), imu_samples=len(imu),
                        tilt_deg=tilt_deg(g), imu_frame="realsense accel (x right, y down, z forward), m/s^2")
        if frames:
            xyz = np.vstack([fr[2] for fr in frames]); inten = np.concatenate([fr[3] for fr in frames])
            np.savez_compressed(f("lidar.npz"), xyz=xyz, intensity=inten,
                                frame_sizes=np.array([len(fr[2]) for fr in frames]),
                                frame_stamps=np.array([fr[1] for fr in frames]),
                                frame_host_times=np.array([fr[0] for fr in frames]))
            cg.write_ply(f("lidar.ply"), xyz, inten)
            meta.update(lidar_frames=len(frames), lidar_points=len(xyz))
        if cam:
            cv2.imwrite(str(f("realsense_color.png")), cam[1])
            cv2.imwrite(str(f("realsense_depth.png")), cam[2])  # uint16 PNG, lossless
            import open3d as o3d
            xyz, rgb = depth_to_points(cam[2], cam[1], self.intr["K"], self.depth_scale)
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            pc.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud(str(f("realsense.ply")), pc)
            meta.update(t_realsense=cam[0], realsense_intrinsics=self.intr, depth_scale=self.depth_scale,
                        depth_aligned_to="color")
        if pano:
            cv2.imwrite(str(f("insta360.jpg")), pano[1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta.update(t_insta360=pano[0], insta360_size=[pano[1].shape[1], pano[1].shape[0]])
        meta["files"] = sorted(p.name for p in d.iterdir()) + [f"{name}_meta.json"]
        f("meta.json").write_text(json.dumps(meta, indent=2))
        index = self.out / "index.csv"
        new = not index.exists()
        with open(index, "a", newline="") as fh:
            w = csv.DictWriter(fh, INDEX_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({**meta, "folder": d.name, "missing": " ".join(missing),
                        "tilt_deg": f"{meta['tilt_deg']:.2f}" if "tilt_deg" in meta else ""})
        msg = f"saved {d.name}"
        if frames:
            msg += f": {len(frames)} LiDAR frames, {meta['lidar_points']} pts"
        if "tilt_deg" in meta:
            msg += f", tilt {meta['tilt_deg']:.1f} deg"
        return d, msg + (f" | MISSING: {', '.join(missing)}" if missing else " | all sensors")


def run_gui(rig, seconds, start_sensors):
    """Left: RealSense colour + depth, 360 preview, status, seconds slider, name, CAPTURE. Right: live LiDAR cloud."""
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    w = app.create_window("Site capture: RealSense + Insta360 X4 + Unitree L2", 1700, 980)
    em = w.theme.font_size
    pw = 640  # preview width

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([0.06, 0.06, 0.08, 1.0])
    scene.scene.show_axes(True)
    mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.0

    panel = gui.Vert(0.4 * em, gui.Margins(em, em, em, em))
    blank = lambda wd, ht: o3d.geometry.Image(np.zeros((ht, wd, 3), np.uint8))
    rs_img = gui.ImageWidget(blank(pw, pw * 9 // 32))  # colour | depth side by side
    pano_img = gui.ImageWidget(blank(pw, pw // 2))
    rs_lbl, pano_lbl, lidar_lbl = gui.Label("realsense: -"), gui.Label("insta360: -"), gui.Label("lidar: -")
    imu_lbl = gui.Label("tilt: -")
    name = gui.TextEdit(); name.placeholder_text = "capture name, e.g. floor2_roomA_corner1"
    notes = gui.TextEdit(); notes.placeholder_text = "optional, e.g. door open, scaffold in view"
    secs = gui.Slider(gui.Slider.DOUBLE); secs.set_limits(0.5, 10.0); secs.double_value = seconds
    btn = gui.Button("CAPTURE  (RealSense + 360 + LiDAR)")
    status = gui.Label("Keep the tripod still during capture.")

    for wd in (rs_img, rs_lbl, pano_img, pano_lbl, lidar_lbl, imu_lbl):
        panel.add_child(wd)
    for label, wd in (("Name", name), ("Notes", notes), ("LiDAR seconds", secs)):
        row = gui.Horiz(0.5 * em); row.add_child(gui.Label(label)); row.add_child(wd); panel.add_child(row)
    reset_btn = gui.Button("Reset LiDAR view (whole cloud, z into screen, level)")
    panel.add_child(btn); panel.add_child(status); panel.add_child(reset_btn)
    w.add_child(scene); w.add_child(panel)

    def on_layout(ctx):
        r = w.content_rect
        x = pw + 2 * int(em)
        panel.frame = gui.Rect(r.x, r.y, x, r.height)
        scene.frame = gui.Rect(r.x + x, r.y, r.width - x, r.height)
    w.set_on_layout(on_layout)

    def on_capture():
        btn.enabled = False
        status.text = f"CAPTURING {secs.double_value:.1f}s - keep the tripod still..."

        def work():
            _, msg = rig.capture(secs.double_value, name.text_value, notes.text_value)

            def done():
                status.text = msg
                btn.enabled = True
            app.post_to_main_thread(w, done)
        threading.Thread(target=work, daemon=True).start()
    btn.set_on_clicked(on_capture)

    view, busy = {"reset": True, "levelled": False}, threading.Event()
    reset_btn.set_on_clicked(lambda: view.update(reset=True))

    def show(widget, img, width):
        img = cv2.resize(img, (width, width * img.shape[0] // img.shape[1]), interpolation=cv2.INTER_AREA)
        widget.update_image(o3d.geometry.Image(np.ascontiguousarray(img[:, :, ::-1])))

    def refresh():
        try:
            rs_lbl.text = f"realsense: {rig.cam_status}"
            pano_lbl.text = f"insta360: {rig.insta_status}"
            lidar_lbl.text = f"lidar: {rig.lidar_status}"
            recent = list(rig.accel)[-30:]
            imu_lbl.text = (f"tilt from level: {tilt_deg(np.mean([s[1:] for s in recent], 0)):.1f} deg  (RealSense IMU)"
                            if recent else f"imu: {rig.imu_status}")
            with rig.lock:
                img, depth = rig.cam_img, rig.depth
            if img is not None and depth is not None:  # depth preview: 0-6 m mapped to colour, no reading = black
                img, depth = img[::4, ::4], depth[::4, ::4]  # shrink before colouring: full-res took ~32 ms per redraw
                vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / (6.0 / (rig.depth_scale or 0.001))),
                                        cv2.COLORMAP_TURBO)
                vis[depth == 0] = 0
                show(rs_img, np.hstack([img, vis]), pw)  # one widget: two same-size Open3D ImageWidgets showed the same texture
            if rig.pano is not None:
                show(pano_img, rig.pano, pw)
            xyz, inten = rig.live_cloud()
            if xyz is not None:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
                col = cv2.applyColorMap(np.clip(inten, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
                pcd.colors = o3d.utility.Vector3dVector(col[:, 0, ::-1] / 255.0)
                scene.scene.clear_geometry()
                scene.scene.add_geometry("cloud", pcd, mat)
                # (re)set the view on the first cloud, on Reset view, and once more when the IMU first has data
                if view["reset"] or (recent and not view["levelled"]):
                    fr = scene.frame
                    center, eye, up = fit_view(xyz, np.mean([s[1:] for s in recent], 0) if recent else None,
                                               fr.width / max(fr.height, 1))
                    scene.setup_camera(60.0, pcd.get_axis_aligned_bounding_box(), center.tolist())
                    scene.look_at(center.tolist(), eye.tolist(), up.tolist())
                    view.update(reset=False, levelled=bool(recent))
        finally:
            busy.clear()

    def ticker():
        while rig.running:
            if not busy.is_set():  # never queue a redraw behind an unfinished one
                busy.set()
                app.post_to_main_thread(w, refresh)
            time.sleep(1 / 12)

    def on_close():
        rig.running = False
        return True
    w.set_on_close(on_close)

    start_sensors()
    threading.Thread(target=ticker, daemon=True).start()
    app.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "captures"))
    ap.add_argument("--seconds", type=float, default=2.0, help="initial LiDAR capture length")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    # import both pybind11 extensions on the main thread first: importing them concurrently deadlocks
    import open3d.visualization.gui  # noqa: F401
    import pyrealsense2  # noqa: F401
    rig = SiteRig(a.out)

    def start_sensors():
        threading.Thread(target=rig.lidar_loop, daemon=True).start()
        threading.Thread(target=rig.camera_loop, args=(a.width, a.height), daemon=True).start()
        threading.Thread(target=rig.insta_loop, daemon=True).start()
        threading.Thread(target=rig.imu_loop, daemon=True).start()
    run_gui(rig, a.seconds, start_sensors)
    os._exit(0)  # camera driver teardown in the daemon threads hangs normal interpreter exit; captures are already on disk
