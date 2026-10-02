"""Solve T_cam_lidar from capture_gui.py captures (captures/<name>/image.png + lidar.npz).

    python calib/calibrate.py            # defaults: A1 board, 7x5 inner corners, 0.095 m squares
    python calib/calibrate.py --selftest

Camera board plane: PnP on the checkerboard corners.
LiDAR board points: calib_target.extract_target (SMART-MARK/Unitree4DLidarL2), cached per capture.
L2 range error: the LiDAR sees the board `scale` times too big because ranges read long; with
r_meas = kappa * r_true + delta, scale = kappa + delta * scale / r_meas is fitted over all captures
and removed along each ray before the solve (--range-model none to disable).
Writes captures/calibration/{extrinsics.json, intrinsics_calibrated.json, overlays/}.

Used for: finding exactly where the RealSense camera sits relative to the L2 LiDAR (rotation + translation,
          "T_cam_lidar"), refining the camera's lens parameters, and correcting the L2's range error. Every
          other script that combines camera and LiDAR depends on its output.
How: the same checkerboard is seen by both sensors in many poses. The camera gives the board plane from its
     corners; the LiDAR gives points on that plane. The transform that puts the LiDAR points onto the camera
     planes is the answer.
Works with: capture_gui.py (makes the captures), fusion_gui.py and board_size.py (use extrinsics.json),
          galvo_calib.py (uses intrinsics_calibrated.json and the detect/board_obj helpers here).
Needs: opencv, numpy, scipy, and the SMART-MARK calib_target package (path in CALIB_TARGET).
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def fit_plane(pts):
    """Best-fit plane through a set of 3D points, as n·p = d with |n|=1 and d>0
    (normal points away from the sensor at the origin; d = distance from the sensor to the plane)."""
    c = pts.mean(0)
    n = np.linalg.svd(pts - c)[2][2]
    d = n @ c
    return (-n, -d) if d < 0 else (n, d)


def board_obj(cols, rows, square):
    """3D positions (m) of the checkerboard's inner corners in the board's own frame (flat, z=0),
    in the same order OpenCV's corner detector returns them. Used for PnP and camera calibration."""
    obj =np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square
    return obj


def detect(img, cols, rows):
    """Find the checkerboard's inner corners in a camera image; returns their pixel positions or None.
    SB detector, falling back to the classic one (+ sub-pixel refine): SB misses ~half of these images."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, uv = cv2.findChessboardCornersSB(g, (cols, rows))
    if ok:
        return uv
    ok, uv = cv2.findChessboardCorners(g, (cols, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(g, uv, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))


def calibrate_intrinsics(corner_sets, obj, size, K0):
    """Re-estimate the camera's lens parameters (focal length, centre, distortion) from many board images.
    Zhang calibration seeded with the factory K. Returns reprojection RMS (px), K, dist."""
    rms, K, dist, _, _ = cv2.calibrateCamera([obj] * len(corner_sets), corner_sets, size, K0.copy(), None,
                                             flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    return rms, K, dist.ravel()


def cam_plane(uv, K, dist, cols, rows, square):
    """Board plane (n, d) in the camera frame from detected corners: PnP gives the board's pose,
    then a plane is fitted through its corners placed in 3D."""
    obj = board_obj(cols, rows, square)
    _, rvec, tvec = cv2.solvePnP(obj, uv, K, dist)
    return fit_plane((cv2.Rodrigues(rvec)[0] @ obj.T + tvec).T)


CALIB_TARGET = r"C:\Users\nikhi\Documents\SMART-MARK\Unitree4DLidarL2"
COARSE = dict(voxel=0.02, min_seg_points=100, px=0.015, min_coverage=0.5)  # sparse 1-2 s scans of far boards


def lidar_board(cap, root=CALIB_TARGET):
    """Find the checkerboard in one capture's LiDAR scan (using the SMART-MARK calib_target detector).
    Accepted board detection in cap/lidar.npz -> (walk-corrected board points, scale, distance), else a reason.
    scale = how much bigger the LiDAR sees the board than it really is (feeds the range model).
    Tries a coarse config for sparse/far scans first, then the default one.
    Cached in cap/lidar_board.npz (delete it to re-detect)."""
    cache = cap / "lidar_board.npz"
    if not cache.exists() or "sheet" not in np.load(cache).files:
        sys.path.insert(0, root)
        from calib_target.board import BoardSpec
        from calib_target.detect import Config, extract_target
        z = np.load(cap / "lidar.npz")
        for cfg in (Config(**COARSE), Config()):
            det = extract_target(z["xyz"], z["intensity"], BoardSpec(), cfg)
            if det is not None and det.accepted:
                break
        ok = det is not None and det.accepted
        np.savez(cache, ok=ok, points=det.board_points if ok else np.zeros((0, 3)),
                 scale=det.scale if ok else 0.0, centre=det.T_lidar_target[:3, 3] if ok else np.zeros(3),
                 sheet=np.array(det.sheet_dims) if ok else np.zeros(2),  # (long, short) flat-region extent, m
                 ncc=det.ncc if det is not None else 0.0, note="" if ok else (det.note if det is not None else "no board"))
    z = np.load(cache)
    return (z["points"], float(z["scale"]), float(np.linalg.norm(z["centre"]))) if z["ok"] else str(z["note"])


def fit_range_model(scales, dists):
    """Fit the L2's range error r_meas = kappa * r_true + delta from how oversized the board looked
    in each capture at each distance.
    scale_i = kappa + delta * scale_i / r_i  (least squares) -> kappa, delta [m]."""
    s, r = np.asarray(scales), np.asarray(dists)
    kappa, delta = np.linalg.lstsq(np.c_[np.ones_like(s), s / r], s, rcond=None)[0]
    return kappa, delta


def correct_range(p, kappa, delta):
    """Undo the LiDAR range error: move each point along its ray from the sensor to its true distance
    r_true = (r_meas - delta) / kappa. Direction is unchanged. Also used by fusion_gui.py."""
    r =np.linalg.norm(p, axis=1, keepdims=True)
    return p / r * ((r - delta) / kappa)


def kabsch(A, B):
    """Best rotation that turns the vectors in A into the vectors in B (Kabsch algorithm, via SVD).
    R minimising |R a_i - b_i| over rows (A, B already centred or direction vectors)."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T


def solve(cam_planes, lidar_pts, max_pts=1000, seed=0):
    """The core LiDAR -> camera solve: find R, t so every LiDAR board point lands on the matching camera board plane.
    cam_planes [(n_c, d_c)], lidar_pts [Nx3 board points]. Returns R, t, per-pose RMS (m).
    Step 1: quick closed-form guess from the plane normals. Step 2: refine with robust least squares.
    Needs varied board tilts: parallel boards leave the rotation about their common normal free."""
    nc = np.array([n for n, _ in cam_planes]); dc = np.array([d for _, d in cam_planes])
    lp = [fit_plane(p) for p in lidar_pts]
    nl = np.array([n for n, _ in lp]); dl = np.array([d for _, d in lp])
    # closed-form init: rotate lidar normals onto camera normals (Kabsch), then linear t
    R = kabsch(nl, nc)
    t = np.linalg.lstsq(nc, dc - np.einsum("ij,ij->i", nc, (nl * dl[:, None]) @ R.T), rcond=None)[0]
    # refine: point-to-plane on <=max_pts per pose, Huber against stray points
    rng = np.random.default_rng(seed)
    sub = [p[rng.choice(len(p), min(len(p), max_pts), replace=False)] for p in lidar_pts]
    P = np.vstack(sub)
    N = np.repeat(nc, [len(s) for s in sub], 0); D = np.repeat(dc, [len(s) for s in sub])

    def res(x):
        """Signed distance of every LiDAR point from its camera plane for pose x = (rotvec, t)."""
        return np.einsum("ij,ij->i", P @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:], N) - D

    x = least_squares(res, np.r_[Rotation.from_matrix(R).as_rotvec(), t], loss="huber", f_scale=0.02).x
    R, t = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:]
    rms = [np.sqrt(np.mean((p @ R.T @ n + t @ n - d) ** 2)) for p, (n, d) in zip(lidar_pts, cam_planes)]
    return R, t, rms


def tilt_spread(normals):
    """How varied the board orientations are across captures (a quality check before solving).
    Smallest/largest singular value of the stacked board normals: 0 = all parallel, ~0.5+ = well spread."""
    sv = np.linalg.svd(np.asarray(normals), compute_uv=False)
    return sv[-1] / sv[0]


def selftest():
    """No-hardware check (--selftest): builds fake boards/images with a known answer and asserts that
    solve(), calibrate_intrinsics() and the range model recover it."""
    rng = np.random.default_rng(1)
    R_true = Rotation.from_euler("xyz", [-92, 3, -88], degrees=True).as_matrix()
    t_true = np.array([0.05, -0.12, 0.03])
    planes, pts = [], []
    for _ in range(12):
        n = Rotation.from_euler("xy", rng.uniform(-40, 40, 2), degrees=True).apply([0, 0, 1])
        c = np.r_[rng.uniform(-1, 1, 2), rng.uniform(1.5, 4)]
        a = np.cross(n, [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
        pc = c + rng.uniform(-0.4, 0.4, (400, 1)) * a + rng.uniform(-0.3, 0.3, (400, 1)) * b
        pc += rng.normal(0, 0.02, (400, 1)) * n  # ~2 cm lidar range noise
        planes.append((n, n @ c) if n @ c > 0 else (-n, -n @ c))
        pts.append((pc - t_true) @ R_true)  # camera -> lidar frame
    R, t, rms = solve(planes, pts)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ R_true).as_rotvec()))
    print(f"selftest: rot err {ang:.3f} deg, trans err {np.linalg.norm(t - t_true) * 1000:.1f} mm, rms {np.mean(rms) * 100:.1f} cm")
    assert ang < 0.3 and np.linalg.norm(t - t_true) < 0.01
    assert tilt_spread([n for n, _ in planes]) > 0.15
    assert tilt_spread([[0, 0, 1], [0.01, 0, 1], [0, 0.02, 1]]) < 0.05
    # intrinsics: 15 synthetic views of the A1 board, 0.2 px corner noise
    K_true = np.array([[640.0, 0, 645], [0, 638, 362], [0, 0, 1]]); d_true = np.array([0.05, -0.1, 0, 0, 0.02])
    obj = board_obj(7, 5, 0.095); sets = []
    for _ in range(15):
        rv = rng.uniform(-0.5, 0.5, 3); tv = np.r_[rng.uniform(-0.5, 0.2, 2), rng.uniform(1.0, 2.5)]
        sets.append((cv2.projectPoints(obj, rv, tv, K_true, d_true)[0] + rng.normal(0, 0.2, (35, 1, 2))).astype(np.float32))
    K0 = K_true * [[1.03], [1.03], [1]]  # factory guess 3% off
    rms_px, K, _ = calibrate_intrinsics(sets, obj, (1280, 720), K0)
    print(f"selftest: intrinsics fx err {abs(K[0, 0] - 640):.2f} px, reproj {rms_px:.2f} px")
    assert abs(K[0, 0] - 640) < 3 and abs(K[0, 2] - 645) < 3
    # range model: true distances 1-2.5 m, r_meas = 1.05 r + 0.2  ->  apparent scale = r_meas / r_true
    r_true = rng.uniform(1.0, 2.5, 20); r_meas = 1.05 * r_true + 0.2
    kappa, delta = fit_range_model(r_meas / r_true, r_meas)
    p = rng.normal(size=(5, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
    back = correct_range(p * (1.05 * 2.0 + 0.2), kappa, delta)
    print(f"selftest: range model kappa {kappa:.3f} delta {delta:.3f} m")
    assert abs(kappa - 1.05) < 1e-6 and abs(delta - 0.2) < 1e-6 and np.allclose(np.linalg.norm(back, axis=1), 2.0)


def main():
    """Full pipeline: load captures -> find the board in each image (and optionally recalibrate the camera)
    -> find the board in each LiDAR scan -> check tilt variety -> fit the range model -> solve the
    transform -> write extrinsics.json and overlay images (LiDAR points drawn on each photo to eyeball the result)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=str(Path(__file__).resolve().parent.parent / "captures"))
    ap.add_argument("--cols", type=int, default=7); ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=0.095, help="square size in metres (measure the print!)")
    ap.add_argument("--intrinsics", choices=["calibrated", "factory"], default="calibrated")
    ap.add_argument("--range-model", choices=["board", "none"], default="board",
                    help="board: fit the L2 range error from the apparent board size and remove it")
    ap.add_argument("--calib-target", default=CALIB_TARGET, help="folder containing the calib_target package")
    ap.add_argument("--lidar-frame", default="unilidar_lidar")
    ap.add_argument("--cam-frame", default="camera_color_optical_frame")
    ap.add_argument("--force", action="store_true", help="solve even if the board tilts are too similar")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    caps = sorted(d for d in Path(a.captures).iterdir() if (d / "lidar.npz").exists() and (d / "image.png").exists())
    out_dir = Path(a.captures) / "calibration"; (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    intr = json.loads((caps[0] / "meta.json").read_text())["camera_intrinsics"]
    K, dist = np.array(intr["K"]), np.array(intr["dist"])
    imgs = {d.name: cv2.imread(str(d / "image.png")) for d in caps}
    corners = {n: detect(im, a.cols, a.rows) for n, im in imgs.items()}
    used = "factory"
    sets = [c for c in corners.values() if c is not None]
    if a.intrinsics == "calibrated" and len(sets) >= 10:
        used = "calibrated"
        rms_px, K, dist = calibrate_intrinsics(sets, board_obj(a.cols, a.rows, a.square), (intr["width"], intr["height"]), K)
        f = np.array(intr["K"])
        print(f"intrinsics from {len(sets)} images: reproj {rms_px:.3f} px | fx {K[0, 0]:.1f} (factory {f[0, 0]:.1f}) "
              f"fy {K[1, 1]:.1f} ({f[1, 1]:.1f}) cx {K[0, 2]:.1f} ({f[0, 2]:.1f}) cy {K[1, 2]:.1f} ({f[1, 2]:.1f})")
        if rms_px > 0.5:
            print("WARNING: reprojection > 0.5 px - blurry images or a non-flat board; or use --intrinsics factory")
        (out_dir / "intrinsics_calibrated.json").write_text(json.dumps(
            {**intr, "K": K.tolist(), "dist": dist.tolist(), "reproj_rms_px": rms_px, "n_images": len(sets)}, indent=2))
    elif a.intrinsics == "calibrated":
        print(f"only {len(sets)} board images, need >= 10 for intrinsics -> using factory")

    names, planes, pts, scales, dists = [], [], [], [], []
    for i, d in enumerate(caps):
        print(f"[{i + 1}/{len(caps)}] {d.name}: ", end="", flush=True)
        if corners[d.name] is None:
            print("board not found in camera, skipped"); continue
        cp = cam_plane(corners[d.name], K, dist, a.cols, a.rows, a.square)
        lb = lidar_board(d, a.calib_target)
        if isinstance(lb, str):
            print(f"board not found in lidar ({lb}), skipped"); continue
        p, sc, r = lb
        print(f"cam plane {cp[1]:.2f} m | lidar {len(p)} pts, {r:.2f} m, scale {sc:.3f}")
        names.append(d.name); planes.append(cp); pts.append(p); scales.append(sc); dists.append(r)
    assert len(planes) >= 3, "need >= 3 captures with the board found in both sensors"
    spread = tilt_spread([n for n, _ in planes])
    print(f"board tilt spread {spread:.2f} (need > 0.15, good > 0.3)")
    if spread < 0.15 and not a.force:
        sys.exit("STOP: the boards all face nearly the same way, so the rotation about that direction is not "
                 "observable and any extrinsic would be wrong.\nRecapture with the board tilted +-30-45 deg in yaw AND "
                 "pitch (lean it back/forward, turn it left/right) at 1-2.5 m, then re-run. (--force to solve anyway)")

    kappa, delta = 1.0, 0.0
    R0, t0, rms0 = solve(planes, pts)
    if a.range_model == "board":
        kappa, delta = fit_range_model(scales, dists)
        print(f"range model from {len(scales)} boards: r_meas = {kappa:.4f} * r_true + {delta:.3f} m "
              f"(a true 1.5 m reads {kappa * 1.5 + delta:.3f} m)")
        pts = [correct_range(p, kappa, delta) for p in pts]
    R, t, rms = solve(planes, pts)
    print(f"plane RMS: no range model {np.mean(rms0) * 100:.2f} cm | used {np.mean(rms) * 100:.2f} cm; "
          f"translation difference {np.linalg.norm(t - t0) * 100:.1f} cm")
    med = np.median(rms)
    for nm, r in zip(names, rms):
        print(f"  {nm}: plane rms {r * 100:.1f} cm" + ("   <-- outlier? delete the capture & re-run" if r > max(3 * med, 0.03) else ""))
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
    Ti = np.linalg.inv(T)
    q = Rotation.from_matrix(Ti[:3, :3]).as_quat()
    out = {
        "T_cam_lidar": T.tolist(), "T_lidar_cam": Ti.tolist(),
        "K": K.tolist(), "dist": np.ravel(dist).tolist(), "intrinsics": used,
        "range_model": {"kappa": kappa, "delta_m": delta,
                        "apply": "r_true = (r_meas - delta) / kappa along each ray, before T_cam_lidar"},
        "cam_in_lidar_xyz_m": Ti[:3, 3].tolist(),
        "cam_in_lidar_rpy_deg": Rotation.from_matrix(Ti[:3, :3]).as_euler("xyz", degrees=True).tolist(),
        "tilt_spread": spread,
        "plane_rms_cm": {nm: round(r * 100, 2) for nm, r in zip(names, rms)},
        "ros2_static_tf": "ros2 run tf2_ros static_transform_publisher "
                          + " ".join(f"{v:.6f}" for v in [*Ti[:3, 3], *q]) + f" {a.lidar_frame} {a.cam_frame}",
    }
    (out_dir / "extrinsics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("cam_in_lidar_xyz_m", "cam_in_lidar_rpy_deg", "ros2_static_tf")}, indent=2))

    for nm in names:  # every point range-coloured, the detected board points on top in magenta
        img = imgs[nm].copy()
        z = np.load(Path(a.captures) / nm / "lidar.npz")
        board = correct_range(np.load(Path(a.captures) / nm / "lidar_board.npz")["points"], kappa, delta)
        for P, fixed in ((correct_range(z["xyz"].astype(np.float64), kappa, delta), None), (board, (255, 0, 255))):
            p = P @ R.T + t
            p = p[np.isfinite(p).all(1) & (p[:, 2] > 0.2)]
            uv = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2).astype(int)
            col = cv2.applyColorMap(np.clip(p[:, 2] / 6 * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
            keep = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
            for (u, v), c in zip(uv[keep], col[keep, 0]):
                cv2.circle(img, (int(u), int(v)), 1, fixed or c.tolist(), -1)
        cv2.imwrite(str(out_dir / "overlays" / f"{nm}.png"), img)
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
