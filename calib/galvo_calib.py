"""Galvo laser <-> RealSense camera calibration. The galvo is treated as an inverse camera: its mirror
values (0..65535) play the role of pixels, so OpenCV's camera calibration gives its intrinsics, distortion
(the pincushion/keystone of the two-mirror optics) and its pose relative to the camera.

    python calib/galvo_calib.py capture     # per board pose press SPACE: the laser dot grid is swept automatically
    python calib/galvo_calib.py fit         # -> captures/galvo/galvo_calib.json
    python calib/galvo_calib.py aim         # click in the camera image: the laser points there (D435i depth)
    python calib/galvo_calib.py selftest    # synthetic galvo, no hardware

Capture: put the A1 checkerboard where the laser can reach it, 1-3 m away. Per view the script takes a
laser-off frame (board plane from the checkerboard), sweeps a coarse 5x5 dot grid to learn roughly where
the mirrors point, then a dense grid over the board. Dots are found as the green-channel difference to the
laser-off frame and turned into 3D by intersecting the camera ray with the board plane. Take 5+ views with
the board at different distances and tilts. Nobody in front of the rig while it sweeps.

Serial protocol as in SMART-MARK/Unitree4DLidarL2/galvosensor (draw_circle.py): "SC\\nX<x>Y<y>F<speed>E\\nM11eR"
runs a point list with the laser on, "S" stops.

Used for: learning the mapping "3D point seen by the camera -> galvo mirror values", so the laser can be
          aimed at anything the camera sees.
Works with: calibrate.py (needs intrinsics_calibrated.json; reuses its board detector), laser_track.py
          (uses the Galvo/Camera classes and galvo_calib.json made here). Hardware: RealSense D435i, galvo
          laser on a serial port (COM5), the A1 checkerboard. Needs pyrealsense2, pyserial, opencv, numpy.
"""
import os

os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"  # iGPU OpenCL crashed OpenCV earlier (see capture_gui.py)
import argparse, json, time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from calibrate import board_obj, detect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "captures" / "galvo"
CAM_CALIB = ROOT / "captures" / "calibration" / "intrinsics_calibrated.json"
S = 1024 / 65536          # mirror units -> "pixels" for OpenCV (keeps the numbers well conditioned)
BOARD, SQ = (7, 5), 0.095  # inner corners, square size (m) of the A1 board
# the printed pattern spans -1..7 x -1..5 squares in the inner-corner frame; the sheet adds 40 / 12 mm
BOARD_X = (-SQ - 0.03, 7 * SQ + 0.03)
BOARD_Y = (-SQ - 0.005, 5 * SQ + 0.005)


# ---------------------------------------------------------------- hardware
def shape_message(pts, speed=4000, wait=3000):
    """Build the serial text command that makes the galvo trace a list of (x, y) mirror points with the laser on.
    Point list in the galvosensor protocol (draw_circle.py): clamped to the 0..65535 mirror grid.
    speed = F (move speed), wait = W (dwell at each point)."""
    w = f"W{wait}" if wait else ""
    xy = [(int(np.clip(round(x), 0, 65535)), int(np.clip(round(y), 0, 65535))) for x, y in pts]
    return "SC\n" + "\n".join(f"X{x}Y{y}F{speed}{w}E" for x, y in xy) + "\nM11eR"


class Galvo:
    """Talks to the galvo laser over serial. dry=True opens no port and sends nothing (laser stays off)."""

    def __init__(self, port="COM5", speed=4000, dry=False):
        self.speed, self.ser = speed, None
        if not dry:
            import serial
            self.ser = serial.Serial(port, baudrate=1000000)

    def _w(self, s):
        """Send a raw command string (no-op in dry-run)."""
        if self.ser:
            self.ser.write(s.encode("utf-8"))

    def dot(self, x, y):
        """Stop whatever is running and hold the laser on a single mirror position (x, y)."""
        x, y = (int(np.clip(round(v), 0, 65535)) for v in (x, y))
        self._w("S")
        self._w(f"SC\nX{x}Y{y}F{self.speed}E\nM11eR")

    def shape(self, pts, wait=3000):
        """Run a closed point list (a basic shape: 4 corners, a circle) with the laser on."""
        self._w("S")
        self._w(shape_message(pts, self.speed, wait))

    def stop(self):
        """Stop drawing (laser off)."""
        self._w("S")

    def close(self):
        """Stop the laser and close the serial port."""
        self.stop()
        if self.ser:
            self.ser.close()


class Camera:
    """RealSense colour stream, optionally with depth aligned to the colour image (needed for aiming)."""

    def __init__(self, depth=False, width=1280, height=720):
        import pyrealsense2 as rs
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        if depth:
            cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color) if depth else None
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale() if depth else None
        for _ in range(30):  # auto-exposure settle
            self.pipe.wait_for_frames()

    def frame(self):
        """(color, depth_m or None) from a frame exposed after this call: queued frames are dropped first."""
        while self.pipe.poll_for_frames():
            pass
        self.pipe.wait_for_frames()
        f = self.pipe.wait_for_frames()
        if self.align:
            f = self.align.process(f)
            d = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        else:
            d = None
        return np.asanyarray(f.get_color_frame().get_data()).copy(), d

    def close(self):
        """Stop the camera stream."""
        self.pipe.stop()


def load_cam():
    """Camera matrix K and distortion from calibrate.py's intrinsics_calibrated.json."""
    e =json.loads(CAM_CALIB.read_text())
    return np.array(e["K"]), np.array(e["dist"])


# ---------------------------------------------------------------- geometry
def find_dot(on, off, min_diff=40.0, max_area=800):
    """Find where the laser dot is in the image (pixel x, y), or None if no clear dot.
    Sub-pixel laser dot from the green-channel increase between a laser-on and a laser-off frame.
    (Green channel, not green-excess: a saturated dot turns white in the image.)"""
    d = cv2.GaussianBlur(on[:, :, 1].astype(np.float32) - off[:, :, 1].astype(np.float32), (5, 5), 0)
    _, mx, _, (x0, y0) = cv2.minMaxLoc(d)
    if mx < min_diff:
        return None
    _, lab, st, _ = cv2.connectedComponentsWithStats((d > 0.5 * mx).astype(np.uint8))
    k = lab[y0, x0]
    if st[k, cv2.CC_STAT_AREA] > max_area:  # a big bright patch is an exposure change, not the dot
        return None
    ys, xs = np.nonzero(lab == k)
    w = d[ys, xs]
    return np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])


def board_pose(img, K, dist):
    """Checkerboard position/orientation in the camera frame as (R, t), or None if the board isn't visible."""
    uv =detect(img, *BOARD)
    if uv is None:
        return None
    _, rvec, tvec = cv2.solvePnP(board_obj(*BOARD, SQ), uv, K, dist)
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def ray_plane(px, K, dist, R, t):
    """Turn a pixel into a 3D point by shooting its camera ray onto the board plane.
    Pixel -> 3D point (camera frame) on the board plane (normal = board z axis)."""
    x, y = cv2.undistortPoints(np.array([[px]], np.float64), K, dist).ravel()
    ray = np.array([x, y, 1.0])
    n = R[:, 2]
    return ray * (n @ t) / (n @ ray)


def on_board(p, R, t):
    """True if a 3D point (camera frame) lies on the printed sheet; dots off the sheet aren't on the plane."""
    b =R.T @ (p - t)
    return bool(BOARD_X[0] <= b[0] <= BOARD_X[1] and BOARD_Y[0] <= b[1] <= BOARD_Y[1])  # plain bool: goes into JSON


def grid(x0, x1, y0, y1, n):
    """n x n evenly spaced (x, y) points over a rectangle, row by row (used for the coarse mirror sweep)."""
    return [(x, y) for y in np.linspace(y0, y1, n) for x in np.linspace(x0, x1, n)]


def quad_grid(quad, n):
    """n x n grid bilinearly spread over a quad (4 corners in order). Used for the dense sweep aimed at the board."""
    a, b, c, d = np.asarray(quad, float)
    return [tuple((1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * d + u * c))
            for v in np.linspace(0, 1, n) for u in np.linspace(0, 1, n)]


# ---------------------------------------------------------------- fit
def inliers(v, thresh_px=8.0):
    """Dots of one view all lie on the board plane, so mirror -> pixel is (nearly) one homography.
    False detections (frame grabbed before the mirrors settled, reflections) break it: keep the RANSAC inliers."""
    D = np.array([p["dac"] for p in v["points"]], float)
    px = np.array([p["px"] for p in v["points"]], float)
    if len(D) < 8:
        return np.ones(len(D), bool)
    _, m = cv2.findHomography(D, px, cv2.RANSAC, thresh_px)
    return m.ravel().astype(bool) if m is not None else np.ones(len(D), bool)


def fit(views, refit=True):
    """Solve the galvo model from all captured views: treat the galvo as a camera whose "pixels" are mirror values,
    run OpenCV camera calibration per view to start, then one joint fit over every dot in the camera frame
    (that pose = where the galvo sits relative to the camera). Optionally drops outlier dots and refits.
    views: [{"R", "t" (board pose in camera), "points": [{"dac", "p_cam", "px"?}]}] -> calibration dict
    (K, dist, rvec, tvec + accuracy stats in mm and the covered area)."""
    obj_b, img_b, P_all, D_all = [], [], [], []
    n_drop = 0
    for v in views:
        R, t = np.array(v["R"]), np.array(v["t"])
        keep = inliers(v) if all("px" in p for p in v["points"]) else np.ones(len(v["points"]), bool)
        n_drop += int((~keep).sum())
        if keep.sum() < 6:
            continue
        P = np.array([p["p_cam"] for p in v["points"]], float)[keep]
        D = np.array([p["dac"] for p in v["points"]], float)[keep] * S
        pb = (P - t) @ R  # board-local coordinates (planar per view -> Zhang initialisation)
        pb[:, 2] = 0
        obj_b.append(pb.astype(np.float32)); img_b.append(D.astype(np.float32).reshape(-1, 1, 2))
        P_all.append(P); D_all.append(D)
    size = (1024, 1024)
    _, K, dist, _, _ = cv2.calibrateCamera(obj_b, img_b, size, None, None)
    P, D = np.vstack(P_all), np.vstack(D_all)

    def joint(P, D, K, dist):
        """Fit K, dist and camera->galvo pose to all dots at once; returns (calib, per-dot error in mirror units)."""
        # refine on every dot in the camera frame: the single "view" pose IS the camera -> galvo transform
        _, K, dist, rv, tv = cv2.calibrateCamera([P.astype(np.float32)], [D.astype(np.float32).reshape(-1, 1, 2)],
                                                 size, K, dist, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        c = dict(K=K, dist=dist.ravel(), rvec=rv[0].ravel(), tvec=tv[0].ravel())
        return c, np.linalg.norm(to_dac(P, c) - D / S, axis=1)

    c, err_dac = joint(P, D, K, dist)
    if refit:  # one more pass without dots the model disagrees with strongly (missed by the per-view check)
        good = err_dac <= max(4 * np.median(err_dac), 100)
        n_drop += int((~good).sum())
        P, D = P[good], D[good]
        c, err_dac = joint(P, D, c["K"], c["dist"])
    K = c["K"]
    R = cv2.Rodrigues(c["rvec"])[0]
    rng = np.linalg.norm(P @ R.T + c["tvec"], axis=1)          # distance galvo -> dot
    err_mm = err_dac * S / K[0, 0] * rng * 1000                  # small-angle: pixels / f = radians
    c.update(rms_dac=float(np.sqrt(np.mean(err_dac ** 2))), err_mm_median=float(np.median(err_mm)),
             err_mm_p95=float(np.percentile(err_mm, 95)), n_views=len(obj_b), n_points=len(P), n_dropped=n_drop,
             # camera-ray directions (x/z, y/z) the dots covered: the distortion model is not valid outside
             coverage=[*np.percentile(P[:, 0] / P[:, 2], [0, 100]), *np.percentile(P[:, 1] / P[:, 2], [0, 100])],
             galvo_in_cam_m=(-R.T @ c["tvec"]).tolist())
    return c


def to_dac(P, c):
    """The main use of the calibration: 3D points in the camera frame (m) -> galvo mirror values (x, y) to aim at them."""
    uv =cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 3), c["rvec"], c["tvec"], c["K"], c["dist"])[0]
    return uv.reshape(-1, 2) / S


def save_calib(c, path):
    """Write the calibration dict to JSON (numpy arrays -> lists), plus a note on how to apply it."""
    Path(path).write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
                                     | {"scale_dac_to_px": S, "apply": "dac = projectPoints(p_cam, rvec, tvec, K, dist) / scale"},
                                     indent=2))


def load_calib(path=OUT / "galvo_calib.json"):
    """Read galvo_calib.json back into the arrays to_dac() and covered() need."""
    e =json.loads(Path(path).read_text())
    return {k: np.array(e[k]) for k in ("K", "dist", "rvec", "tvec", "coverage")}


def covered(P, c, margin=0.02):
    """True where the camera-ray direction lies inside the area the calibration dots covered.
    Outside it the fitted model is extrapolating, so aiming there is less accurate."""
    P = np.asarray(P, float).reshape(-1, 3)
    x, y = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    x0, x1, y0, y1 = c["coverage"]
    return (x >= x0 - margin) & (x <= x1 + margin) & (y >= y0 - margin) & (y <= y1 + margin)


# ---------------------------------------------------------------- modes
def show(img, text, dots=(), scale=0.75):
    """Show the camera image with a status line and circles at `dots` [(pixel, ok)] (green ok / red not);
    returns the key pressed (0xFF if none)."""
    v =img.copy()
    for p, ok in dots:
        cv2.circle(v, tuple(int(round(q)) for q in p), 6, (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.putText(v, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("galvo calib", cv2.resize(v, None, fx=scale, fy=scale))
    return cv2.waitKey(1) & 0xFF


def capture(a):
    """`capture` mode: live camera view; each SPACE press records one board view. Takes a laser-off frame,
    sweeps a coarse 5x5 dot grid, uses it to aim a dense grid at the board, and saves every dot that landed
    on the board (mirror value + 3D position) to captures/galvo/view_NN/dots.json. q quits."""
    K, dist = load_cam()
    OUT.mkdir(parents=True, exist_ok=True)
    cam, gal = Camera(), Galvo(a.port, a.speed, a.dry_run)
    n_view = len(list(OUT.glob("view_*/dots.json")))
    try:
        while True:
            img, _ = cam.frame()
            pose = board_pose(img, K, dist)
            k = show(img, f"views {n_view} | board {'OK' if pose else 'NOT found'} | SPACE: sweep  q: quit")
            if k == ord("q"):
                break
            if k != ord(" ") or pose is None:
                continue
            gal.stop(); time.sleep(a.settle)
            off, _ = cam.frame()
            pose = board_pose(off, K, dist)
            if pose is None:
                print("board lost, move nothing and retry"); continue
            R, t = pose
            pts, marks = [], []

            def sweep(dacs, label):
                """Put the dot at each mirror position, find it in the image, and record it with its 3D point."""
                for i, (x, y) in enumerate(dacs):
                    gal.dot(x, y); time.sleep(a.settle)
                    on, _ = cam.frame()
                    px = find_dot(on, off)
                    if px is None:
                        continue
                    p = ray_plane(px, K, dist, R, t)
                    ok = on_board(p, R, t)
                    marks.append((px, ok))
                    pts.append(dict(dac=[float(x), float(y)], px=px.tolist(), p_cam=p.tolist(), on_board=ok))
                    if show(on, f"{label} {i + 1}/{len(dacs)}: {sum(m[1] for m in marks)} dots on board", marks) == ord("q"):
                        raise KeyboardInterrupt
                gal.stop()

            sweep(grid(a.x_min, a.x_max, a.y_min, a.y_max, 5), "coarse")
            seen = [p for p in pts]
            if len(seen) >= 4:  # rough mirror->pixel homography, then aim a dense grid at the board
                H, _ = cv2.findHomography(np.array([p["dac"] for p in seen]), np.array([p["px"] for p in seen]), cv2.RANSAC, 20)
                o = np.array([[-SQ, -SQ], [7 * SQ, -SQ], [7 * SQ, 5 * SQ], [-SQ, 5 * SQ]])
                o = (o - o.mean(0)) * a.shrink + o.mean(0)
                corners_px = cv2.projectPoints(np.c_[o, np.zeros(4)], cv2.Rodrigues(R)[0], t, K, dist)[0]
                quad = cv2.perspectiveTransform(corners_px.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
                if np.isfinite(quad).all():
                    sweep([tuple(np.clip(q, 0, 65535)) for q in quad_grid(quad, a.dense)], "dense")
            good = [p for p in pts if p["on_board"]]
            if len(good) < 6:
                print(f"only {len(good)} dots on the board - not saved (widen --x-min/--x-max/... or move the board)")
                continue
            d = OUT / f"view_{n_view:02d}"; d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / "off.png"), off)
            (d / "dots.json").write_text(json.dumps(dict(R=R.tolist(), t=t.tolist(), points=good), indent=1))
            print(f"view_{n_view:02d}: {len(good)} dots on board ({len(pts) - len(good)} elsewhere)")
            n_view += 1
    except KeyboardInterrupt:
        print("sweep aborted")
    finally:
        gal.close(); cam.close(); cv2.destroyAllWindows()


def fit_mode(a):
    """`fit` mode: load every saved view, run fit(), save galvo_calib.json and print the accuracy."""
    views =[json.loads(f.read_text()) for f in sorted(OUT.glob("view_*/dots.json"))]
    assert len(views) >= 3, f"need >= 3 views in {OUT}, have {len(views)}"
    c = fit(views)
    save_calib(c, OUT / "galvo_calib.json")
    print(f"{c['n_views']} views, {c['n_points']} dots ({c['n_dropped']} bad dots dropped) | reprojection {c['rms_dac']:.0f} mirror units RMS | "
          f"on target: median {c['err_mm_median']:.1f} mm, 95% {c['err_mm_p95']:.1f} mm")
    print(f"galvo position in camera frame: {np.round(c['galvo_in_cam_m'], 3)} m -> {OUT / 'galvo_calib.json'}")


def aim(a):
    """`aim` mode (test the calibration): click a pixel -> read its depth -> 3D point -> mirror values -> fire
    the dot there, then measure how many pixels the real dot landed from the click. s = laser off, q = quit."""
    K, dist = load_cam()
    c = load_calib()
    cam, gal = Camera(depth=True), Galvo(a.port, a.speed, a.dry_run)
    click, state = [None], {"msg": "click a point in the image (s: laser off, q: quit)", "marks": []}
    cv2.namedWindow("galvo calib")
    cv2.setMouseCallback("galvo calib", lambda ev, x, y, *_: click.__setitem__(0, (x / 0.75, y / 0.75)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    try:
        while True:
            img, depth = cam.frame()
            k = show(img, state["msg"], state["marks"])
            if k == ord("q"):
                break
            if k == ord("s"):
                gal.stop(); state["marks"] = []
            if click[0] is None:
                continue
            u, v = (int(round(q)) for q in click[0]); click[0] = None
            z = depth[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
            z = float(np.median(z[z > 0])) if (z > 0).any() else 0.0
            if not z:
                state["msg"] = "no depth at that pixel - click elsewhere"; continue
            x, y = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist).ravel()
            dac = to_dac([x * z, y * z, z], c)[0]
            warn = "" if covered([x * z, y * z, z], c)[0] else " | OUTSIDE calibrated area: capture views there"
            gal.stop(); time.sleep(a.settle); off, _ = cam.frame()
            gal.dot(*dac); time.sleep(a.settle); on, _ = cam.frame()
            px = find_dot(on, off)
            err = "dot not seen" if px is None else f"error {np.linalg.norm(px - (u, v)):.1f} px"
            state["marks"] = [((u, v), True)] + ([(px, False)] if px is not None else [])
            state["msg"] = f"target {z:.2f} m -> mirror ({dac[0]:.0f}, {dac[1]:.0f}) | {err}{warn}  (green = click, red = dot)"
            print(state["msg"])
    finally:
        gal.close(); cam.close(); cv2.destroyAllWindows()


def selftest():
    """No-hardware check: simulate a galvo with known parameters, check fit() recovers it well enough to hit
    new points within a few mm, and check find_dot/ray_plane/on_board on synthetic data."""
    rng = np.random.default_rng(3)
    K_true = np.array([[1400.0, 0, 520], [0, 1380, 500], [0, 0, 1]]); d_true = np.array([0.08, -0.05, 0.001, -0.002, 0])
    r_true, t_true = np.radians([2.0, -3.0, 1.0]), np.array([0.09, -0.04, 0.02])  # camera -> galvo
    truth = dict(K=K_true, dist=d_true, rvec=r_true, tvec=t_true)
    views = []
    for _ in range(6):
        R = cv2.Rodrigues(np.radians(rng.uniform(-30, 30, 3)) * [1, 1, 0.3])[0]
        t = np.r_[rng.uniform(-0.3, 0.1, 2), rng.uniform(1.2, 2.8)]
        pb = np.c_[rng.uniform(*BOARD_X, 45), rng.uniform(*BOARD_Y, 45), np.zeros(45)]
        P = pb @ R.T + t
        D = to_dac(P, truth) + rng.normal(0, 30, (45, 2))  # ~30 mirror units dot noise
        views.append(dict(R=R.tolist(), t=t.tolist(), points=[dict(dac=d.tolist(), p_cam=p.tolist()) for d, p in zip(D, P)]))
    c = fit(views)
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(c["rvec"])[0] @ cv2.Rodrigues(r_true)[0].T)[0]))
    dt = np.linalg.norm(c["tvec"] - t_true) * 1000
    # what matters is hitting new points, incl. depths outside the calibration range (1.2-2.8 m): rotation alone
    # trades off against the principal point, so it is reported, not asserted
    x0, x1, y0, y1 = c["coverage"]  # held-out directions inside the covered area, depths outside the trained range
    Q = np.c_[rng.uniform(x0, x1, 200), rng.uniform(y0, y1, 200), np.ones(200)] * rng.choice([1.0, 3.5], 200)[:, None]
    assert covered(Q, c).all() and not covered([[5.0, 0, 1.0]], c)[0]
    miss = np.linalg.norm(to_dac(Q, c) - to_dac(Q, truth), axis=1) * S / K_true[0, 0] * np.linalg.norm(Q, axis=1) * 1000
    print(f"selftest fit: rot err {ang:.3f} deg, trans err {dt:.1f} mm, on-target median {c['err_mm_median']:.1f} mm, "
          f"held-out at 1.0/3.5 m: median {np.median(miss):.1f} mm, max {miss.max():.1f} mm")
    assert dt < 15 and c["err_mm_median"] < 3 and np.median(miss) < 3 and miss.max() < 10
    # dot finder on a synthetic laser spot (saturated centre) + ray/plane round trip
    off = np.full((720, 1280, 3), 60, np.uint8); on = off.copy()
    cv2.circle(on, (700, 300), 5, (255, 255, 255), -1); cv2.circle(on, (700, 300), 9, (40, 220, 40), 2)
    assert np.allclose(find_dot(on, off), (700, 300), atol=0.5) and find_dot(off, off) is None
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]]); R = cv2.Rodrigues(np.radians([10.0, -20, 5]))[0]; t = np.array([0.1, 0, 2.0])
    p = R @ np.array([0.3, 0.2, 0]) + t
    px = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), K, np.zeros(5))[0].ravel()
    assert np.allclose(ray_plane(px, K, np.zeros(5), R, t), p, atol=1e-6) and on_board(p, R, t)
    assert not on_board(R @ np.array([1.5, 0.2, 0]) + t, R, t)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "fit", "aim", "selftest"])
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--speed", type=int, default=4000)
    ap.add_argument("--settle", type=float, default=0.25, help="s to wait after moving the dot")
    ap.add_argument("--x-min", type=float, default=8000); ap.add_argument("--x-max", type=float, default=57535)
    ap.add_argument("--y-min", type=float, default=8000); ap.add_argument("--y-max", type=float, default=57535)
    ap.add_argument("--dense", type=int, default=7, help="dense grid is dense x dense dots over the board")
    ap.add_argument("--shrink", type=float, default=0.85, help="dense grid covers this fraction of the board")
    ap.add_argument("--dry-run", action="store_true", help="no serial output (laser stays off)")
    a = ap.parse_args()
    {"capture": capture, "fit": fit_mode, "aim": aim, "selftest": lambda _: selftest()}[a.mode](a)
