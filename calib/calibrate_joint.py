"""Joint LiDAR-camera calibration (test): solve T_cam_lidar AND the L2 range error together, on interior board points.

    python calib/calibrate_joint.py                    # both range models, reproducibility report, writes the --model one
    python calib/calibrate_joint.py --model linear
    python calib/calibrate_joint.py --selftest
    python calib/fusion_gui.py --calib captures/calibration/extrinsics_joint.json   # look at it live

Why: calibrate.py fits the range error from how big the board looks, but the L2's beam footprint also makes every
edge ~10 cm too wide (A1 sheet reads ~1040 x 790 mm), so board size mixes edge bloom with range error.
Here (Mirzaei et al. 2012, planar-target LiDAR intrinsic + extrinsic calibration):
  1. drop detections where the board merged with its surroundings (sheet short side > --max-sheet-short)
  2. erode each board by --erode m from its outline: only interior points, no mixed edge returns
  3. least squares over (R, t, delta[, kappa]): range-corrected LiDAR points must lie on the camera's board planes.
     A range offset moves points along their rays, which fan out over the view, so it is separable from t.
Reproducibility: leave-one-capture-out spread and a first-half -> second-half held-out test, per model.
Ground truth it cannot replace: the slope (kappa) is only seen over the capture distances, so check far ranges
with a laser distance meter against a wall.

Works with: calibrate.py (helpers + cached lidar_board.npz, intrinsics_calibrated.json), fusion_gui.py (--calib).
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from calibrate import CALIB_TARGET, cam_plane, correct_range, detect, lidar_board, solve, tilt_spread

ROOT = Path(__file__).resolve().parent.parent
RANGES = (1.5, 4.0, 8.0)  # m, true ranges at which the fitted model's reading is reported


def erode(P, margin):
    """Keep board points at least `margin` m inside the outline of the detected board (in its own plane)."""
    c = P.mean(0)
    uv = (P - c) @ np.linalg.svd(P - c, full_matrices=False)[2][:2].T
    eq = ConvexHull(uv).equations  # unit normals: n.x + b <= 0 inside, -(n.x + b) = distance to that edge
    return P[-(uv @ eq[:, :2].T + eq[:, 2]).max(1) >= margin]


def solve_joint(planes, pts, model, max_pts=500, seed=0):
    """R, t, kappa, delta so that correct_range(lidar points) @ R.T + t lie on the camera planes.
    model 'offset': r_meas = r_true + delta; 'linear': r_meas = kappa * r_true + delta."""
    R0, t0, _ = solve(planes, pts, max_pts, seed)
    rng = np.random.default_rng(seed)
    sub = [p[rng.choice(len(p), min(len(p), max_pts), replace=False)] for p in pts]
    P = np.vstack(sub)
    N = np.repeat([n for n, _ in planes], [len(s) for s in sub], 0)
    D = np.repeat([d for _, d in planes], [len(s) for s in sub])
    lin = model == "linear"

    def unpack(x):
        return Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6], x[7] if lin else 1.0, x[6]

    def res(x):
        R, t, k, d = unpack(x)
        return np.einsum("ij,ij->i", correct_range(P, k, d) @ R.T + t, N) - D

    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), t0, 0.0, [1.0] if lin else []]
    return unpack(least_squares(res, x0, loss="huber", f_scale=0.02).x)


def plane_rms(planes, pts, R, t, k, d):
    """Per-board RMS distance (m) of the range-corrected, transformed LiDAR points from the camera plane."""
    return np.array([np.sqrt(np.mean((correct_range(p, k, d) @ R.T @ n + t @ n - dd) ** 2)) for p, (n, dd) in zip(pts, planes)])


def reads(k, d):
    """What the L2 reads at each true range in RANGES under this model."""
    return [k * r + d for r in RANGES]


def report(planes, pts, model):
    """Fit on everything, then leave-one-capture-out and first-half -> second-half; returns the full fit."""
    R, t, k, d = solve_joint(planes, pts, model)
    rms = plane_rms(planes, pts, R, t, k, d)
    jk = []
    for i in range(len(pts)):
        Ri, ti, ki, di = solve_joint(planes[:i] + planes[i + 1:], pts[:i] + pts[i + 1:], model)
        jk.append([*ti, ki, di, *reads(ki, di)])
    jk = np.array(jk)
    h = len(pts) // 2
    Ra, ta, ka, da = solve_joint(planes[:h], pts[:h], model)
    Rb, tb, kb, db = solve_joint(planes[h:], pts[h:], model)
    held = plane_rms(planes[h:], pts[h:], Ra, ta, ka, da)
    own = plane_rms(planes[h:], pts[h:], Rb, tb, kb, db)
    print(f"\n[{model}] r_meas = {k:.4f} * r_true + {d * 100:.1f} cm | plane RMS {rms.mean() * 100:.2f} cm "
          f"| t = {np.round(t * 100, 1)} cm")
    print(f"  leave-one-out std: delta {jk[:, 4].std() * 100:.2f} cm, kappa {jk[:, 3].std():.4f}, "
          f"t {np.round(jk[:, :3].std(0) * 100, 2)} cm")
    for j, r in enumerate(RANGES):
        v = jk[:, 5 + j]
        print(f"  true {r:.1f} m reads {k * r + d:.3f} m  (leave-one-out range {v.min():.3f} .. {v.max():.3f})")
    print(f"  first half -> second half: delta {da * 100:.1f} vs {db * 100:.1f} cm, kappa {ka:.4f} vs {kb:.4f}, "
          f"held-out plane RMS {held.mean() * 100:.2f} cm (fitted on itself {own.mean() * 100:.2f}), "
          f"4 m reads {ka * 4 + da:.3f} vs {kb * 4 + db:.3f}")
    return R, t, k, d, rms


def synth(kappa, delta, seed=2):
    """25 synthetic boards seen by a rig with a known transform and range error r_meas = kappa * r + delta."""
    rng = np.random.default_rng(seed)
    R_true = Rotation.from_euler("xyz", [1, -0.5, -114], degrees=True).as_matrix()
    t_true = np.array([0.015, 0.065, -0.026])
    planes, pts = [], []
    for _ in range(25):
        n = Rotation.from_euler("xy", rng.uniform(-40, 40, 2), degrees=True).apply([0, 0, 1])
        c = np.r_[rng.uniform(-0.8, 0.8, 2), rng.uniform(1.0, 4.0)]
        a = np.cross(n, [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
        pc = c + rng.uniform(-0.3, 0.3, (600, 1)) * a + rng.uniform(-0.2, 0.2, (600, 1)) * b
        planes.append((n, n @ c) if n @ c > 0 else (-n, -n @ c))
        pl = (pc - t_true) @ R_true                      # camera -> lidar frame
        r = np.linalg.norm(pl, axis=1, keepdims=True)
        pts.append(pl / r * (kappa * r + delta + rng.normal(0, 0.015, r.shape)))
    return planes, pts, t_true


def selftest():
    """Each range model must recover its own synthetic truth (delta, kappa, t); erode must trim a square board."""
    for model, kappa in (("linear", 1.03), ("offset", 1.0)):
        planes, pts, t_true = synth(kappa, 0.15)
        R, t, k, d = solve_joint(planes, pts, model)
        print(f"selftest {model}: kappa {k:.4f} delta {d * 100:.1f} cm, t err {np.linalg.norm(t - t_true) * 1000:.1f} mm")
        assert abs(d - 0.15) < 0.01 and abs(k - kappa) < 0.005 and np.linalg.norm(t - t_true) < 0.01
    sq = np.c_[np.random.default_rng(0).uniform(0, 1, (4000, 2)), np.zeros(4000)]
    e = erode(sq, 0.1)
    assert e[:, :2].min() > 0.09 and e[:, :2].max() < 0.91 and len(e) > 0.5 * 4000 * 0.64
    print("selftest ok")


def main():
    """Load captures -> camera planes + filtered, eroded LiDAR boards -> both range models with reproducibility
    numbers -> extrinsics_joint.json (same keys as extrinsics.json) for the chosen model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=str(ROOT / "captures"))
    ap.add_argument("--cols", type=int, default=7); ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=0.095)
    ap.add_argument("--erode", type=float, default=0.10, help="m trimmed off every board edge (edge bloom ~0.1 m)")
    ap.add_argument("--max-sheet-short", type=float, default=0.95,
                    help="m; a detected sheet wider than this merged with its surroundings (real 0.594, bloomed ~0.79)")
    ap.add_argument("--min-points", type=int, default=200, help="interior points a board needs after eroding")
    ap.add_argument("--model", choices=["offset", "linear"], default="offset",
                    help="written to the json. ponytail: offset by default, kappa needs captures spread over 1-5 m")
    ap.add_argument("--calib-target", default=CALIB_TARGET)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import cv2
    caps_dir = Path(a.captures)
    out = caps_dir / "calibration"
    intr = json.loads((out / "intrinsics_calibrated.json").read_text())
    K, dist = np.array(intr["K"]), np.array(intr["dist"])
    names, planes, pts, skipped = [], [], [], {}
    for d in sorted(p for p in caps_dir.iterdir() if (p / "lidar.npz").exists() and (p / "image.png").exists()):
        uv = detect(cv2.imread(str(d / "image.png")), a.cols, a.rows)
        lb = lidar_board(d, a.calib_target)
        if uv is None or isinstance(lb, str):
            skipped.setdefault("board not in both sensors", []).append(d.name); continue
        sheet = np.load(d / "lidar_board.npz")["sheet"]
        if sheet[1] > a.max_sheet_short:
            skipped.setdefault("board merged with surroundings", []).append(f"{d.name} ({sheet[0]:.2f} x {sheet[1]:.2f} m)"); continue
        p = erode(lb[0], a.erode)
        if len(p) < a.min_points:
            skipped.setdefault("too few interior points", []).append(f"{d.name} ({len(p)})"); continue
        names.append(d.name); planes.append(cam_plane(uv, K, dist, a.cols, a.rows, a.square)); pts.append(p)
    for why, lst in skipped.items():
        print(f"skipped ({why}): {len(lst)}" + "".join(f"\n  {s}" for s in lst if why != "board not in both sensors"))
    assert len(planes) >= 6, "need >= 6 usable boards"
    dists = [np.linalg.norm(p.mean(0)) for p in pts]
    print(f"using {len(planes)} boards, eroded {a.erode * 100:.0f} cm | LiDAR distance {min(dists):.2f}-{max(dists):.2f} m "
          f"| tilt spread {tilt_spread([n for n, _ in planes]):.2f}")

    old = json.loads((out / "extrinsics.json").read_text()) if (out / "extrinsics.json").exists() else None
    if old:
        T = np.array(old["T_cam_lidar"]); rm = old["range_model"]
        r = plane_rms(planes, pts, T[:3, :3], T[:3, 3], rm["kappa"], rm["delta_m"])
        print(f"\n[current extrinsics.json] r_meas = {rm['kappa']:.4f} * r_true + {rm['delta_m'] * 100:.1f} cm "
              f"| plane RMS on these interior points {r.mean() * 100:.2f} cm")
    fits = {m: report(planes, pts, m) for m in ("offset", "linear")}

    R, t, k, d, rms = fits[a.model]
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
    Ti = np.linalg.inv(T)
    res = dict(T_cam_lidar=T.tolist(), T_lidar_cam=Ti.tolist(), K=K.tolist(), dist=dist.tolist(), intrinsics="calibrated",
               range_model={"kappa": k, "delta_m": d, "model": a.model, "method": "joint plane fit, eroded boards",
                            "apply": "r_true = (r_meas - delta) / kappa along each ray, before T_cam_lidar"},
               cam_in_lidar_xyz_m=Ti[:3, 3].tolist(),
               cam_in_lidar_rpy_deg=Rotation.from_matrix(Ti[:3, :3]).as_euler("xyz", degrees=True).tolist(),
               plane_rms_cm={n: round(r * 100, 2) for n, r in zip(names, rms)})
    (out / "extrinsics_joint.json").write_text(json.dumps(res, indent=2))
    print(f"\n[{a.model}] written -> {out / 'extrinsics_joint.json'} | camera in LiDAR "
          f"{np.round(Ti[:3, 3] * 100, 1)} cm ({np.linalg.norm(Ti[:3, 3]) * 100:.1f} cm away)")


if __name__ == "__main__":
    main()
