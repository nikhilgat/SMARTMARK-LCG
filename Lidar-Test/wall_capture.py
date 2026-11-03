"""L2 range ground truth against a wall: preview the cloud, capture, type your measured distance.

    .\\bridge\\start_lidar.ps1
    python Lidar-Test/wall_capture.py [--label corridor]    # v = preview, Enter = capture, q = quit
    python Lidar-Test/wall_capture.py --fit                 # verdict over Lidar-Test/readings.csv
    python Lidar-Test/wall_capture.py --selftest

Per capture: Lidar-Test/<time>_<meter>m/{lidar.npz, lidar.ply} (raw cloud, re-analysable later) and one row in
readings.csv. LiDAR distance = perpendicular distance to the wall plane found in front of the camera, raw and
range-corrected (verify_lidar.wall_distance). RealSense distance = plane fit on the central depth patch, from the
D435i depth origin (a different point than the LiDAR's, so compare slopes, not offsets). It also reports how far
the rig is from square to the wall: keep that under ~3 deg.

Measure with a laser meter from one fixed mark on the rig, perpendicular to the wall (shortest reading).
3 captures per spot at about 1, 2, 3, 4, 6, 8 m. Let the L2 warm up 15-20 min first.

Board instead of a wall (--gate 0.4 --label whiteboard): matte face (gloss shifts the L2 range and hotspots the
RealSense), >= 1 m free behind it. You type the distance before the capture; only points within the gate of it
count, otherwise at 4 m+ the bigger wall behind the board wins the plane fit.
"""
import argparse, csv, socket, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calib"))
import verify_lidar as vl  # first: pulls in capture_gui, which disables OpenCV OpenCL before cv2 loads
import capture_gui as cg
import numpy as np
from calibrate import fit_plane

CSV = HERE / "readings.csv"
COLS = ["time", "label", "folder", "meter_m", "lidar_raw_m", "lidar_corrected_m", "scatter_cm", "points",
        "realsense_m", "realsense_tilt_deg", "calib"]


def grab(host, port, seconds):
    """seconds of LiDAR frames from lidar_bridge as (xyz (N,3), intensity (N,))."""
    sock = socket.create_connection((host, port), timeout=3.0)
    xyz, inten, t0 = [], [], time.time()
    try:
        while time.time() - t0 < seconds:
            _, p, i = cg.read_frame(sock)
            xyz.append(p); inten.append(i)
    finally:
        sock.close()
    return np.vstack(xyz), np.concatenate(inten)


def plane_robust(p, tol=0.02, iters=200, rng=np.random.default_rng(0)):
    """RANSAC (3-point planes, tol inliers), then fit_plane on the inliers. A plain fit gets dragged off the wall by
    clutter in the patch (a chair, a frame edge)."""
    best = None
    for _ in range(iters):
        s = p[rng.choice(len(p), 3, replace=False)]
        m = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(m) < 1e-9:
            continue
        m /= np.linalg.norm(m)
        inl = np.abs((p - s[0]) @ m) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return fit_plane(p[best])


def realsense_distance(frames=15, half=0.15):
    """(distance m, tilt deg) of the wall in the central patch (2*half of each image side) of the D435i depth,
    averaged over frames. Tilt = angle between camera axis and wall normal, 0 = square on. None without camera."""
    try:
        import pyrealsense2 as rs
        if not len(rs.context().query_devices()):
            return None
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)  # 848x480 = D435 best depth accuracy
        pipe.start(cfg)
    except Exception as e:
        print(f"RealSense unavailable: {e}")
        return None
    pts, pc, h, w = [], rs.pointcloud(), int(480 * half), int(848 * half)
    try:
        for _ in range(30):  # let auto-exposure settle
            pipe.wait_for_frames()
        for _ in range(frames):
            v = np.asanyarray(pc.calculate(pipe.wait_for_frames().get_depth_frame()).get_vertices())
            pts.append(v.view(np.float32).reshape(480, 848, 3)[240 - h:240 + h, 424 - w:424 + w].reshape(-1, 3))
    finally:
        pipe.stop()
    p = np.vstack(pts)
    p = p[p[:, 2] > 0.1]  # 0 = no depth
    if len(p) < 100:
        return None
    n, d = plane_robust(p)
    return float(d), float(np.degrees(np.arccos(abs(n[2]))))


def show(xyz, inten, axis, cone):
    """Blocking top-down + side view, grey = all points, red = the cone searched for the wall, arrow = camera axis.
    Matplotlib, not Open3D's legacy viewer: that one fails to get an OpenGL context here (GLFW WGL error)."""
    import matplotlib.pyplot as plt
    r = np.linalg.norm(xyz, axis=1)
    inc = (xyz @ axis > r * np.cos(np.radians(cone))) & (r > 0.3)
    _, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (i, j), name in zip(axs, [(0, 1), (0, 2)], ["top (x-y)", "side (x-z)"]):
        ax.scatter(xyz[~inc, i], xyz[~inc, j], s=0.3, c="0.6")
        ax.scatter(xyz[inc, i], xyz[inc, j], s=0.8, c="r")
        ax.arrow(0, 0, axis[i], axis[j], width=0.02, color="b")
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(f"L2 {name}, m - red: searched for the wall")
    print(f"{inc.sum()} points in the cone. Close the plot window to continue.")
    plt.show()


def ask_meter():
    """Measured distance in m, or None to discard."""
    while True:
        s = input("measured distance in m (empty = discard): ").strip().replace(",", ".")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            print("a number like 2.345")


def run(a):
    kappa, delta, Tlc = vl.load_model(a.calib)
    axis = Tlc[:3, 2] / np.linalg.norm(Tlc[:3, 2])  # camera viewing direction in the LiDAR frame
    print(f"range model from {Path(a.calib).name}: r_meas = {kappa:.4f} * r_true + {delta * 100:.1f} cm")
    print("Face the rig square to a flat, matte wall (no glass, nothing in front).")
    while True:
        cmd = input("\nv = preview, Enter = capture, q = quit: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "v":
            show(*grab(a.host, a.port, 1.0), axis, a.cone)
            continue
        meter = ask_meter() if a.gate else None  # board: distance first, it gates out the wall behind
        if a.gate and meter is None:
            continue
        xyz, inten = grab(a.host, a.port, a.seconds)
        near = np.abs(xyz @ axis - (kappa * meter + delta)) < a.gate if a.gate else slice(None)
        res = vl.wall_distance(xyz[near], axis, a.cone, kappa, delta)
        rsd = None if a.no_realsense else realsense_distance()
        if res:
            print(f"LiDAR:     raw {res[0]:.3f} m, corrected {res[1]:.3f} m, plane scatter {res[2] * 100:.1f} cm, {res[3]} points")
        else:
            print("LiDAR:     no wall found in front - closer, squarer, or a wider --cone (cloud still saved)")
        if rsd:
            print(f"RealSense: {rsd[0]:.3f} m, rig {rsd[1]:.1f} deg off square{'  <- square up the rig' if rsd[1] > 3 else ''}")
        if not a.gate:
            meter = ask_meter()
        elif input("save? Enter = yes, n = discard: ").strip().lower() == "n":
            meter = None
        if meter is None:
            continue
        d = HERE / f"{time.strftime('%Y%m%d-%H%M%S')}_{meter:.3f}m"
        d.mkdir()
        np.savez_compressed(d / "lidar.npz", xyz=xyz.astype(np.float32), intensity=inten.astype(np.float32))
        cg.write_ply(d / "lidar.ply", xyz, inten)
        if not res:  # readings.csv only holds rows wall-fit can use
            print(f"cloud saved to {d.name}, no CSV row (no LiDAR wall)")
            continue
        new = not CSV.exists()
        with open(CSV, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(COLS)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), a.label, d.name, meter, round(res[0], 4), round(res[1], 4),
                        round(res[2] * 100, 2), res[3], round(rsd[0], 4) if rsd else "", round(rsd[1], 2) if rsd else "",
                        Path(a.calib).name])
        print(f"saved {d.name}: LiDAR raw {(res[0] - meter) * 100:+.1f} cm, corrected {(res[1] - meter) * 100:+.1f} cm"
              + (f", RealSense {(rsd[0] - meter) * 100:+.1f} cm" if rsd else "") + " vs your measurement (incl. mark offset)")


def fit(a, path=CSV):
    """LiDAR verdict (verify_lidar's wall-fit on this CSV), then the RealSense line for comparison."""
    vl.CSV = path
    vl.wall_fit(a)
    rows = [(float(r["meter_m"]), float(r["realsense_m"])) for r in csv.DictReader(open(path)) if r["realsense_m"]]
    if len(rows) >= 4:
        m, d = np.array(rows).T
        slope, off = np.polyfit(m, d, 1)
        print(f"\nRealSense = {slope:.4f} * meter + {off * 100:.1f} cm, residual std {np.std(d - slope * m - off) * 100:.1f} cm "
              f"(stereo error grows with distance squared; trust it to ~3 m)")


def selftest():
    """plane_robust ignores clutter in the patch; a CSV written with COLS goes through fit (wall-fit + RealSense)."""
    rng = np.random.default_rng(0)
    wall = np.c_[rng.uniform(-.3, .3, (2000, 2)), np.full(2000, 2.5) + rng.normal(0, .003, 2000)]
    clutter = np.c_[rng.uniform(-.3, .3, (300, 2)), rng.uniform(1.0, 2.3, 300)]
    n, d = plane_robust(np.vstack([wall, clutter]))
    assert abs(d - 2.5) < 0.005 and abs(n[2]) > 0.999, (n, d)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "r.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(COLS)
            for m in [1, 1, 2, 2, 4, 4]:
                w.writerow(["", "", "", m, m * 1.0 + 0.19, m + 0.02, 1, 1000, m + 0.004 * m * m, 1, ""])
        fit(None, p)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="verdict over readings.csv instead of capturing")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calib", default=str(vl.CAPS / "calibration" / "extrinsics_joint.json"),
                    help="range model + LiDAR-camera transform (default: the offset model, best supported)")
    ap.add_argument("--seconds", type=float, default=5.0, help="LiDAR seconds per capture")
    ap.add_argument("--cone", type=float, default=15.0, help="deg around the camera view direction searched for the wall")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="board/whiteboard: m kept around the expected distance (typed before capture), e.g. 0.4; "
                         "drops the wall behind. Needs > gate of free space behind the board")
    ap.add_argument("--label", default="", help="note stored with each row, e.g. corridor")
    ap.add_argument("--no-realsense", action="store_true")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9899)
    a = ap.parse_args()
    selftest() if a.selftest else fit(a) if a.fit else run(a)
