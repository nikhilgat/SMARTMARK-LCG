"""Live RealSense + Unitree L2 view with one Capture button that grabs both at once.

Prereq: the native LiDAR bridge is running (bridge/start_lidar.ps1; stop with -Stop),
which serves parsed frames on 127.0.0.1:9899. The RealSense is plugged in over USB3.

    python calib/capture_gui.py                    # saves to UNITREE-INTEL/captures/
    python calib/capture_gui.py --selftest         # fake LiDAR server + fake camera, no hardware

Each capture -> captures/<YYYYmmdd-HHMMSS>_<tag>/
    image.png     color frame taken inside the LiDAR window
    lidar.npz     xyz (N,3) f32, intensity (N,) f32, frame_sizes, frame_stamps (sensor), frame_host_times
                  -> calib_target.detect.extract_target(d["xyz"], d["intensity"], BoardSpec(), Config())
    lidar.ply     same points, intensity as grey (CloudCompare / VS Code)
    meta.json     tag, host timestamps, camera intrinsics, board-in-image flag

Used for: collecting the paired camera + LiDAR snapshots of the checkerboard that calibrate.py needs.
Works with: bridge/start_lidar.ps1 + udp_relay.py (LiDAR data source), calibrate.py (consumes the captures).
          Its Rig class (sensor threads + live cloud) is reused by fusion_gui.py.
Needs: pyrealsense2, open3d, opencv, numpy.
"""
import argparse, collections, json, os, socket, struct, threading, time
from pathlib import Path

# OpenCL on the Intel iGPU (shared with the Open3D renderer) crashed with CL_OUT_OF_HOST_MEMORY. Must be set
# before cv2 is imported: cv2.ocl.setUseOpenCL(False) is per-thread and misses the camera/capture threads.
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HEADER = struct.Struct("<4sdI")  # lidar_bridge wire format (see native_bridge/lidar_client.py)
BOARD = (7, 5)                   # inner corners of the 8x6-square A1 board


def recv_exact(sock, n):
    """Read exactly n bytes from a TCP socket (recv can return less), or raise if the bridge disconnects."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("lidar_bridge closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock):
    """Read one LiDAR frame from lidar_bridge: header "PCLD" + sensor timestamp + point count, then
    N x (x, y, z, intensity) float32. Returns (stamp, xyz (N,3), intensity (N,))."""
    magic, stamp, n = HEADER.unpack(recv_exact(sock, HEADER.size))
    if magic != b"PCLD":
        raise ValueError(f"bad frame magic {magic!r}")
    p = np.frombuffer(recv_exact(sock, n * 16), "<f4").reshape(-1, 4)
    return stamp, p[:, :3].copy(), p[:, 3].copy()


class Rig:
    """Sensor threads + capture. No GUI, so --selftest can drive it.
    One thread keeps reading LiDAR frames, another keeps reading camera frames; the GUI just
    looks at the latest data. Also used by fusion_gui.py for its live view."""

    def __init__(self, out, host="127.0.0.1", port=9899, decay=0.8):
        """out: folder for captures. host/port: lidar_bridge address. decay: seconds of LiDAR
        frames stacked together in the live view (the L2 is sparse per frame)."""
        self.out, self.addr, self.decay = Path(out), (host, port), decay
        self.lock, self.running = threading.Lock(), True
        self.live = collections.deque()          # (host_t, xyz, inten) for the live view
        self.cap_frames, self.cap_on, self.cap_start = [], False, 0.0
        self.cam_img, self.cam_t, self.cap_img, self.intr = None, 0.0, None, None
        self.lidar_status, self.cam_status = "connecting...", "starting..."
        self.board_uv = None

    # ---- LiDAR
    def lidar_loop(self):
        """Background thread: connect to lidar_bridge, read frames forever, keep the last `decay` seconds for
        the live view, and also store frames while a capture is running. Reconnects every 1 s if it drops."""
        while self.running:
            try:
                sock = socket.create_connection(self.addr, timeout=2.0)
                sock.settimeout(2.0)
                self.lidar_status = "connected"
                while self.running:
                    stamp, xyz, inten = read_frame(sock)
                    t = time.time()
                    with self.lock:
                        if len(xyz):
                            self.live.append((t, xyz, inten))
                        while self.live and self.live[0][0] < t - self.decay:
                            self.live.popleft()
                        if self.cap_on and t >= self.cap_start:
                            self.cap_frames.append((t, stamp, xyz, inten))
            except (OSError, ConnectionError, ValueError) as e:
                self.lidar_status = f"not connected ({e.__class__.__name__}) - is start_lidar.ps1 running?"
                time.sleep(1.0)

    def live_cloud(self):
        """All recent LiDAR frames merged into one cloud: (xyz, intensity), or (None, None) if no data yet."""
        with self.lock:
            if not self.live:
                return None, None
            return np.vstack([f[1] for f in self.live]), np.concatenate([f[2] for f in self.live])

    # ---- camera
    def on_camera_frame(self, img, t):
        """Store the newest camera image; during a capture, keep the first image taken inside the LiDAR window."""
        with self.lock:
            self.cam_img, self.cam_t = img, t
            if self.cap_on and self.cap_img is None and t >= self.cap_start:
                self.cap_img = (t, img)

    def camera_loop(self, width, height):
        """Background thread: start the RealSense colour stream, record its factory intrinsics, feed every frame
        to on_camera_frame, and ~2x a second check whether the checkerboard is visible (for the preview only).
        Waits/retries if the camera is unplugged."""
        import pyrealsense2 as rs
        last_det = 0.0
        while self.running:
            if not len(rs.context().query_devices()):  # pipe.start() with no device blocks ~15 s holding the GIL
                self.cam_status = "no RealSense found - plug it in (USB3)"
                time.sleep(1.0)
                continue
            pipe = rs.pipeline()
            try:
                cfg = rs.config(); cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
                i = pipe.start(cfg).get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.intr = {"K": [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], "dist": list(i.coeffs),
                             "model": str(i.model), "width": i.width, "height": i.height, "source": "realsense factory"}
                self.cam_status = f"streaming {i.width}x{i.height}"
                while self.running:
                    img = np.asanyarray(pipe.wait_for_frames(2000).get_color_frame().get_data()).copy()
                    t = time.time()
                    self.on_camera_frame(img, t)
                    if t - last_det > 0.4:  # throttled preview-only board check on a half-size image
                        g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
                        ok, c = cv2.findChessboardCornersSB(g, BOARD)
                        self.board_uv, last_det = (c.reshape(-1, 2) * 2 if ok else None), time.time()
            except RuntimeError as e:
                self.cam_status = f"not available ({e}) - retrying"
                time.sleep(2.0)
            finally:
                try:
                    pipe.stop()
                except RuntimeError:
                    pass

    # ---- capture
    def capture(self, seconds, tag):
        """Grab `seconds` of LiDAR frames plus one camera image from the same window and save them as a
        capture folder (image.png, lidar.npz, lidar.ply, meta.json). The rig must stay still.
        Blocks for `seconds`; returns (folder, message). Folder is None on failure."""
        with self.lock:
            self.cap_frames, self.cap_img, self.cap_start, self.cap_on = [], None, time.time(), True
        time.sleep(seconds)
        deadline = time.time() + 1.0
        while self.cap_img is None and time.time() < deadline:
            time.sleep(0.02)
        with self.lock:
            self.cap_on = False
            frames, cam, t0 = self.cap_frames, self.cap_img, self.cap_start
        frames = [f for f in frames if len(f[2])]
        if not frames:
            return None, "NOT saved: no LiDAR frames in the window"
        if cam is None:
            return None, "NOT saved: no camera frame in the window"
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag.strip()) or "capture"
        d = self.out / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(t0))}_{safe}"
        d.mkdir(parents=True, exist_ok=True)
        xyz = np.vstack([f[2] for f in frames]); inten = np.concatenate([f[3] for f in frames])
        cv2.imwrite(str(d / "image.png"), cam[1])
        np.savez_compressed(d / "lidar.npz", xyz=xyz, intensity=inten,
                            frame_sizes=np.array([len(f[2]) for f in frames]),
                            frame_stamps=np.array([f[1] for f in frames]),
                            frame_host_times=np.array([f[0] for f in frames]))
        write_ply(d / "lidar.ply", xyz, inten)
        ok, _ = cv2.findChessboardCornersSB(cv2.cvtColor(cam[1], cv2.COLOR_BGR2GRAY), BOARD)
        (d / "meta.json").write_text(json.dumps({
            "tag": tag, "seconds": seconds, "t_window_start": t0, "t_window_end": t0 + seconds,
            "t_image": cam[0], "lidar_frames": len(frames), "lidar_points": len(xyz),
            "board_in_image": bool(ok), "board_inner_corners": BOARD, "camera_intrinsics": self.intr}, indent=2))
        return d, f"saved {d.name}: {len(frames)} frames, {len(xyz)} pts, board in image: {'yes' if ok else 'NO'}"


def write_ply(path, xyz, inten):
    """Save a point cloud as a binary .ply file (intensity stored as grey colour) so it opens in
    CloudCompare, MeshLab, etc."""
    g =np.clip(inten, 0, 255).astype(np.uint8)
    a = np.zeros(len(xyz), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a["x"], a["y"], a["z"] = xyz.T
    a["r"] = a["g"] = a["b"] = g
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(xyz)).encode())
        f.write(a.tobytes())


def run_gui(rig, a, start_sensors):
    """Open the Open3D window: left panel = camera preview (board corners drawn when found), status labels,
    capture length slider, tag box and the CAPTURE button; right = live 3D LiDAR cloud coloured by intensity.
    start_sensors() is called once the window exists; the view refreshes ~12 times a second."""
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    w = app.create_window("RealSense + Unitree L2 capture", 1760, 940)
    em = w.theme.font_size

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([0.06, 0.06, 0.08, 1.0])
    scene.scene.show_axes(True)
    mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.0

    panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
    preview_w = 800
    blank = np.zeros((preview_w * 9 // 16, preview_w, 3), np.uint8)
    image = gui.ImageWidget(o3d.geometry.Image(blank))
    cam_lbl, lidar_lbl, board_lbl = gui.Label("camera: -"), gui.Label("lidar: -"), gui.Label("board: -")
    tag = gui.TextEdit(); tag.placeholder_text = "location tag, e.g. corridor_A"
    secs = gui.Slider(gui.Slider.DOUBLE); secs.set_limits(1.0, 2.0); secs.double_value = a.seconds
    btn = gui.Button("CAPTURE  (camera + LiDAR)")
    status = gui.Label("Rig must be still during capture.")
    n_done = [len([p for p in rig.out.glob("*") if p.is_dir()]) if rig.out.exists() else 0]
    count_lbl = gui.Label(f"captures in {rig.out}: {n_done[0]}")

    panel.add_child(image)
    for wd in (cam_lbl, lidar_lbl, board_lbl):
        panel.add_child(wd)
    row = gui.Horiz(0.5 * em); row.add_child(gui.Label("LiDAR seconds")); row.add_child(secs)
    panel.add_child(row)
    row = gui.Horiz(0.5 * em); row.add_child(gui.Label("Tag")); row.add_child(tag)
    panel.add_child(row)
    panel.add_child(btn); panel.add_child(status); panel.add_child(count_lbl)
    w.add_child(scene); w.add_child(panel)

    def on_layout(ctx):
        """Fixed-width control panel on the left, 3D view fills the rest."""
        r = w.content_rect
        pw = preview_w + 2 * int(em)
        panel.frame = gui.Rect(r.x, r.y, pw, r.height)
        scene.frame = gui.Rect(r.x + pw, r.y, r.width - pw, r.height)
    w.set_on_layout(on_layout)

    def on_capture():
        """CAPTURE button: run rig.capture in a background thread (so the GUI keeps drawing), then show the result."""
        btn.enabled = False
        status.text = f"CAPTURING {secs.double_value:.1f}s - keep the rig still..."

        def work():
            d, msg = rig.capture(secs.double_value, tag.text_value)

            def done():
                if d is not None:
                    n_done[0] += 1
                    count_lbl.text = f"captures in {rig.out}: {n_done[0]}"
                status.text = msg
                btn.enabled = True
            app.post_to_main_thread(w, done)
        threading.Thread(target=work, daemon=True).start()
    btn.set_on_clicked(on_capture)

    first, busy = [True], threading.Event()

    def refresh():
        """Redraw once, then mark the GUI as free for the next redraw."""
        try:
            _refresh()
        finally:
            busy.clear()

    def _refresh():
        """Update status labels, the camera preview and the 3D cloud from the rig's latest data."""
        cam_lbl.text = f"camera: {rig.cam_status}"
        lidar_lbl.text = f"lidar: {rig.lidar_status}"
        board_lbl.text = "board in image: " + ("YES" if rig.board_uv is not None else "no")
        img = rig.cam_img
        if img is not None:
            view = img.copy()
            if rig.board_uv is not None:
                cv2.drawChessboardCorners(view, BOARD, rig.board_uv.reshape(-1, 1, 2).astype(np.float32), True)
            view = cv2.resize(view, (preview_w, preview_w * img.shape[0] // img.shape[1]))
            image.update_image(o3d.geometry.Image(np.ascontiguousarray(view[:, :, ::-1])))
        xyz, inten = rig.live_cloud()
        if xyz is not None:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
            col = cv2.applyColorMap(np.clip(inten, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
            pcd.colors = o3d.utility.Vector3dVector(col[:, 0, ::-1] / 255.0)
            scene.scene.clear_geometry()
            scene.scene.add_geometry("cloud", pcd, mat)
            if first[0]:  # sensor at origin looking along +x, z up
                scene.setup_camera(60.0, pcd.get_axis_aligned_bounding_box(), [0, 0, 0])
                scene.look_at([2, 0, 0], [-2.5, 0, 1.2], [0, 0, 1])
                first[0] = False

    def ticker():
        """Background thread asking the GUI thread to redraw ~12x a second."""
        while rig.running:
            if not busy.is_set():  # never queue a redraw behind an unfinished one (GUI would lag, then freeze)
                busy.set()
                app.post_to_main_thread(w, refresh)
            time.sleep(1 / 12)

    def on_close():
        """Window closed: tell the sensor threads to stop."""
        rig.running = False
        return True
    w.set_on_close(on_close)

    start_sensors()  # only after the window exists
    threading.Thread(target=ticker, daemon=True).start()
    app.run()


def selftest():
    """Fake lidar_bridge (PCLD frames at 10 Hz) + fake camera; one 1 s capture must save a full folder."""
    import tempfile
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]

    def fake_bridge():
        """Pretend lidar_bridge: sends a 500-point frame every 0.1 s."""
        c, _ = srv.accept()
        k = 0
        while True:
            p = np.c_[np.full((500, 3), k, np.float32), np.full(500, 100, np.float32)]
            try:
                c.sendall(HEADER.pack(b"PCLD", 1000.0 + k * 0.1, 500) + p.astype("<f4").tobytes())
            except OSError:
                return
            k += 1
            time.sleep(0.1)
    threading.Thread(target=fake_bridge, daemon=True).start()

    out = Path(tempfile.mkdtemp())
    rig = Rig(out, port=port)
    threading.Thread(target=rig.lidar_loop, daemon=True).start()

    def fake_cam():
        """Pretend camera: a plain grey 1280x720 image at 30 fps."""
        while rig.running:
            rig.on_camera_frame(np.full((720, 1280, 3), 128, np.uint8), time.time())
            time.sleep(1 / 30)
    threading.Thread(target=fake_cam, daemon=True).start()
    time.sleep(0.5)
    assert rig.live_cloud()[0] is not None, rig.lidar_status
    d, msg = rig.capture(1.0, "self test/1")
    rig.running = False
    print(msg)
    z = np.load(d / "lidar.npz"); meta = json.loads((d / "meta.json").read_text())
    assert d.name.endswith("_self_test_1")
    assert 8 <= len(z["frame_sizes"]) <= 12 and len(z["xyz"]) == z["frame_sizes"].sum() == meta["lidar_points"]
    assert (z["intensity"] == 100).all() and np.all(np.diff(z["frame_stamps"]) > 0)
    assert meta["t_window_start"] <= meta["t_image"] <= meta["t_window_end"]
    assert cv2.imread(str(d / "image.png")).shape == (720, 1280, 3) and (d / "lidar.ply").stat().st_size > 0
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "captures"))
    ap.add_argument("--seconds", type=float, default=1.5, help="initial LiDAR capture length (1-2 s)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9899)
    ap.add_argument("--decay", type=float, default=0.8, help="seconds of frames overlaid in the live view")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        # import both pybind11 extensions on the main thread first: importing them concurrently deadlocks
        import open3d.visualization.gui  # noqa: F401
        import pyrealsense2  # noqa: F401
        rig = Rig(a.out, a.host, a.port, a.decay)

        def start_sensors():
            """Start the LiDAR and camera background threads."""
            threading.Thread(target=rig.lidar_loop, daemon=True).start()
            threading.Thread(target=rig.camera_loop, args=(a.width, a.height), daemon=True).start()
        run_gui(rig, a, start_sensors)
