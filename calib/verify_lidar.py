"""Check the L2 accuracy claims yourself: board evidence from the existing captures, and a wall test for far range.

    python calib/verify_lidar.py boards               # per-capture table from captures/ (no hardware)
    python calib/verify_lidar.py points               # L2 point accuracy + precision on the board (no hardware, ~5 min)
    python calib/verify_lidar.py scans                # room scans vs BIM walls (no hardware) -> captures/verify/scans_*.png
    .\\bridge\\start_lidar.ps1
    python calib/verify_lidar.py wall                 # live: face a flat wall, type the laser-meter distance
    python calib/verify_lidar.py wall-fit             # verdict over all wall readings (captures/verify/wall.csv)
    python calib/verify_lidar.py selftest

boards  - for every capture: LiDAR board distance vs the camera's (checkerboard PnP, no board-size model involved),
          detected sheet size vs the real 841 x 594 mm print. Then: is the bias the same in both halves of the data?
points  - RAW L2 points (lidar.npz, not the detector's walk-flattened ones) on the board interior, against the
          camera's checkerboard plane. Precision: point spread about the board plane (white / black squares
          separately), and how much the board distance jitters frame to frame. Accuracy: range error along each beam
          = measured range - where the beam meets the camera's board plane, raw and range-corrected. T and the range
          offset for each board are fitted on all OTHER boards (leave-one-out), so the error is out-of-sample.
scans   - icp/data/scans/<room>*.ply placed on icp/data/bim/<room>.ply with the newest icp/results/*_<room>_* transform.
          Per scan wall point: how far outside the nearest BIM wall it lies. Opposite walls are added up, so the
          ICP translation cancels: what is left is how much wider the room reads than the BIM, per axis. A range
          error r_meas = kappa * r + delta pushes both walls out wherever the sensor stood: excess ~ 2 * delta +
          (kappa - 1) * width. 5-10 m rooms = the distances the boards don't reach. Also prints wall thickness
          (LiDAR noise at room distances) and, with two scans of a room, how well they repeat.
wall    - grabs --seconds of LiDAR, keeps points within --cone deg of the camera's viewing direction, finds the
          wall plane (RANSAC) and its perpendicular distance, raw and range-corrected (--calib model). You type the
          laser-meter distance; one CSV row per reading. Measure from one fixed mark on the rig, perpendicular to
          the wall (shortest reading). Take 3 readings per spot at about 1, 2, 4, 6, 8 m.
wall-fit- fits lidar = slope * meter + offset. The mark-to-LiDAR-origin distance only changes the offset, so
          the slope is clean: slope = the L2's range scale. "corrected - meter" must be flat over distance (the
          constant is the mark offset); if it drifts with distance the range model is wrong out there.
"""
import argparse, csv, json, socket, sys, time
from pathlib import Path

import capture_gui as cg  # first: disables OpenCV OpenCL before cv2 loads
import cv2
import numpy as np
from calibrate import board_obj, correct_range, detect, fit_plane, fit_range_model

ROOT = Path(__file__).resolve().parent.parent
CAPS = ROOT / "captures"
CSV = CAPS / "verify" / "wall.csv"
SQ, SHEET = 0.095, (841.0, 594.0)


def load_model(path):
    """(kappa, delta, T_lidar_cam) from an extrinsics json."""
    e = json.loads(Path(path).read_text())
    rm = e.get("range_model", {"kappa": 1.0, "delta_m": 0.0})
    return rm["kappa"], rm["delta_m"], np.array(e["T_lidar_cam"])


# ---------------------------------------------------------------- boards
def boards(a):
    """Per-capture table + split-half check, from cached LiDAR board detections and the camera images."""
    e = json.loads(Path(a.calib).read_text())
    K, dist, Tlc = np.array(e["K"]), np.array(e["dist"]), np.array(e["T_lidar_cam"])
    rows = []
    for d in sorted(p for p in CAPS.iterdir() if (p / "lidar_board.npz").exists()):
        z = np.load(d / "lidar_board.npz")
        if not z["ok"]:
            continue
        uv = detect(cv2.imread(str(d / "image.png")), 7, 5)
        if uv is None:
            continue
        _, rv, tv = cv2.solvePnP(board_obj(7, 5, SQ), uv, K, dist)
        Rb = cv2.Rodrigues(rv)[0]
        cc = Rb @ np.array([3 * SQ, 2 * SQ, 0]) + tv.ravel()          # board centre, camera frame
        r_cam = np.linalg.norm(Tlc[:3, :3] @ cc + Tlc[:3, 3])          # same point, from the LiDAR origin
        rows.append(dict(name=d.name, r_lid=float(np.linalg.norm(z["centre"])), r_cam=float(r_cam),
                         scale=float(z["scale"]), sheet=z["sheet"] * 1000,
                         tilt=float(np.degrees(np.arccos(abs(Rb[2, 2]))))))
    assert rows, f"no captures with a LiDAR board in {CAPS} (run calibrate.py first)"
    print(f"{'capture':26s} {'LiDAR m':>8s} {'camera m':>9s} {'reads long':>11s} {'tilt':>5s} {'sheet mm':>12s}")
    for r in rows:
        print(f"{r['name']:26s} {r['r_lid']:8.3f} {r['r_cam']:9.3f} {(r['r_lid'] - r['r_cam']) * 100:9.1f} cm "
              f"{r['tilt']:4.0f}d {r['sheet'][0]:5.0f}x{r['sheet'][1]:<5.0f}")
    diff = np.array([r["r_lid"] - r["r_cam"] for r in rows]); rc = np.array([r["r_cam"] for r in rows])
    S, R = np.array([r["scale"] for r in rows]), np.array([r["r_lid"] for r in rows])
    sh = np.array([r["sheet"] for r in rows]); h = len(rows) // 2
    print(f"\n{len(rows)} boards, camera distance {rc.min():.2f}-{rc.max():.2f} m")
    print(f"CLAIM 'LiDAR reads long': {np.sum(diff > 0)}/{len(diff)} boards read long, mean {diff.mean() * 100:.1f} cm, "
          f"std {diff.std() * 100:.1f} cm (single-board scatter)")
    print(f"CLAIM 'repeatable': first half {diff[:h].mean() * 100:.1f} cm vs second half {diff[h:].mean() * 100:.1f} cm")
    tilt = np.array([r["tilt"] for r in rows])
    for lo, hi in ((0, 25), (25, 40), (40, 90)):  # the halves differ mostly by how tilted their boards were
        m = (tilt >= lo) & (tilt < hi)
        if m.any():
            print(f"  board tilt {lo:2d}-{hi:2d} deg: n={m.sum():2d}, reads long {diff[m].mean() * 100:5.1f} cm (std {diff[m].std() * 100:.1f})")
    for i, (k, d) in enumerate((fit_range_model(S[:h], R[:h]), fit_range_model(S[h:], R[h:]))):
        print(f"  board-size model, half {i + 1}: true 1.5 m reads {k * 1.5 + d:.3f} m, true 4 m reads {k * 4 + d:.3f} m")
    print(f"CLAIM 'edge bloom': sheet reads {sh[:, 0].mean():.0f} x {sh[:, 1].mean():.0f} mm vs real {SHEET[0]:.0f} x {SHEET[1]:.0f} "
          f"(+{sh[:, 0].mean() - SHEET[0]:.0f} / +{sh[:, 1].mean() - SHEET[1]:.0f} mm)")
    print(f"NOT COVERED: anything beyond {rc.max():.1f} m -> run the wall test")


# ---------------------------------------------------------------- points
WHITE, DARK = 248, 160  # intensity: saturated white squares / black squares (calib_target walk curve spans 100-256)


def board_raw(cap, erode_m):
    """Raw (not walk-corrected) L2 points on the board interior of one capture, with intensity and frame index.
    Region = the cached detection's plane +-15 cm (range walk is up to ~14 cm) and its outline shrunk by erode_m."""
    from scipy.spatial import ConvexHull
    z, b = np.load(cap / "lidar.npz"), np.load(cap / "lidar_board.npz")["points"]
    n, d = fit_plane(b)
    c = b.mean(0); ax = np.linalg.svd(b - c, full_matrices=False)[2][:2]
    eq = ConvexHull((b - c) @ ax.T).equations
    P, I = z["xyz"].astype(np.float64), z["intensity"]
    F = np.repeat(np.arange(len(z["frame_sizes"])), z["frame_sizes"])
    keep = np.isfinite(P).all(1) & (np.abs(P @ n - d) < 0.15)
    uv = (P[keep] - c) @ ax.T
    inner = -(uv @ eq[:, :2].T + eq[:, 2]).max(1) >= erode_m
    idx = np.where(keep)[0][inner]
    return P[idx], I[idx], F[idx]


def point_stats(P, I, F, n_l, d_l, kappa, delta):
    """Accuracy/precision of raw board points against the camera's board plane (n_l . p = d_l, LiDAR frame).
    Range error per point = measured range - range where its ray meets the camera plane."""
    r = np.linalg.norm(P, axis=1); u = P / r[:, None]
    err = r - d_l / (u @ n_l)                       # raw range error (m), along the beam
    err_c = (r - delta) / kappa - d_l / (u @ n_l)    # after the range correction
    n, d = fit_plane(P)
    w, k = I >= WHITE, I < DARK
    noise = lambda m: float(np.std(P[m] @ n - d)) if m.sum() > 20 else np.nan  # spread about the board plane
    frames = [np.mean(P[(F == f) & w] @ n - d) for f in np.unique(F) if ((F == f) & w).sum() >= 10]
    return dict(n=len(P), raw=float(np.median(err)), cor=float(np.median(err_c)),
                cor_white=float(np.median(err_c[w])) if w.any() else np.nan,
                cor_dark=float(np.median(err_c[k])) if k.any() else np.nan,
                noise_white=noise(w), noise_dark=noise(k), noise_all=float(np.std(P @ n - d)),
                frame_jitter=float(np.std(frames)) if len(frames) >= 3 else np.nan, frames=len(frames),
                incidence=float(np.degrees(np.arccos(min(abs(n_l @ u.mean(0) / np.linalg.norm(u.mean(0))), 1.0)))))


def points(a):
    """Per capture: precision (point noise, frame-to-frame jitter) and accuracy (range error vs the camera's board
    plane, raw and corrected), with T and the range offset fitted on the OTHER captures (leave-one-out)."""
    from calibrate import cam_plane, lidar_board
    from calibrate_joint import erode, solve_joint
    intr = json.loads((CAPS / "calibration" / "intrinsics_calibrated.json").read_text())
    K, dist = np.array(intr["K"]), np.array(intr["dist"])
    caps, planes, pts = [], [], []
    for d in sorted(p for p in CAPS.iterdir() if (p / "lidar_board.npz").exists()):
        uv = detect(cv2.imread(str(d / "image.png")), 7, 5)
        lb = lidar_board(d)
        if uv is None or isinstance(lb, str) or np.load(d / "lidar_board.npz")["sheet"][1] > 0.95:
            continue
        caps.append(d); planes.append(cam_plane(uv, K, dist, 7, 5, SQ)); pts.append(erode(lb[0], a.erode))
    print(f"{len(caps)} boards | leave-one-out: T + range offset refitted without each board (~10 s per board)")
    rows = []
    for i, cap in enumerate(caps):
        R, t, k, dl = solve_joint(planes[:i] + planes[i + 1:], pts[:i] + pts[i + 1:], "offset")
        n_c, d_c = planes[i]
        s = point_stats(*board_raw(cap, a.erode), R.T @ n_c, d_c - n_c @ t, k, dl)
        s.update(name=cap.name, dist=float(d_c), delta=dl)
        rows.append(s)
        print(f"  [{i + 1:2d}/{len(caps)}] {cap.name}: {s['dist']:.2f} m, incidence {s['incidence']:2.0f} deg, {s['n']:5d} pts | "
              f"noise {s['noise_white'] * 1000:4.1f} mm (white) {s['noise_dark'] * 1000:4.1f} mm (dark), frame jitter "
              f"{s['frame_jitter'] * 1000:4.1f} mm | range error raw {s['raw'] * 100:+5.1f} cm, corrected {s['cor'] * 100:+5.1f} cm "
              f"(white {s['cor_white'] * 100:+5.1f}, dark {s['cor_dark'] * 100:+5.1f})")
    g = lambda key: np.array([r[key] for r in rows])
    cor, inc = g("cor"), g("incidence")
    print(f"\nPRECISION (random, per point): white squares {np.nanmedian(g('noise_white')) * 1000:.1f} mm, dark squares "
          f"{np.nanmedian(g('noise_dark')) * 1000:.1f} mm, all points incl. range walk {np.median(g('noise_all')) * 1000:.1f} mm (1 sigma)")
    print(f"PRECISION (one frame, ~{np.median(g('frames')):.0f} frames per capture): board distance jitters "
          f"{np.nanmedian(g('frame_jitter')) * 1000:.1f} mm frame to frame")
    print(f"REFLECTIVITY: dark squares read {np.nanmedian(g('cor_dark') - g('cor_white')) * 100:+.1f} cm vs white "
          f"(range walk; + = farther)")
    print(f"ACCURACY raw:       L2 reads {np.median(g('raw')) * 100:+.1f} cm (median over boards, along the beam)")
    print(f"ACCURACY corrected: {np.mean(cor) * 100:+.1f} cm mean, {np.std(cor) * 100:.1f} cm std over boards, "
          f"worst {np.abs(cor).max() * 100:.1f} cm  (out-of-sample: each board's own data not in its calibration)")
    for lo, hi in ((0, 25), (25, 40), (40, 90)):
        m = (inc >= lo) & (inc < hi)
        if m.any():
            print(f"  incidence {lo:2d}-{hi:2d} deg: n={m.sum():2d}, corrected error {cor[m].mean() * 100:+5.1f} cm (std {cor[m].std() * 100:.1f})")
    for lo, hi in ((0, 1.5), (1.5, 9)):
        m = (g("dist") >= lo) & (g("dist") < hi)
        if m.any():
            print(f"  distance {lo:.1f}-{hi:.1f} m: n={m.sum():2d}, corrected error {cor[m].mean() * 100:+5.1f} cm, "
                  f"noise {np.nanmedian(g('noise_white')[m]) * 1000:.1f} mm")
    out = CAPS / "verify"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"per-board numbers -> {out / 'points.csv'}")


# ---------------------------------------------------------------- scans
SIDES = (("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("-y", 1, -1))


def band_z(p, lo, hi):
    """Points between lo and hi m above the cloud's floor (0.5th percentile of z)."""
    z0 = np.percentile(p[:, 2], 0.5)
    return p[(p[:, 2] > z0 + lo) & (p[:, 2] < z0 + hi)]


def wall_offsets(B, S, max_d=0.8):
    """B: BIM wall points, S: scan wall points (same frame, BIM walls along x/y). For each side (+x = wall whose
    outward normal is +x, ...): offsets (m) of scan points along the outward normal of their nearest BIM wall point;
    the side's offset = peak of that histogram (median within 6 cm of it). Returns {side: (offset, samples)}."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.2))
    N = np.asarray(pc.normals)[:, :2]
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    N[np.einsum("ij,ij->i", B[:, :2] - B[:, :2].mean(0), N) < 0] *= -1  # outward = away from the room centre
    d, i = cKDTree(B[:, :2]).query(S[:, :2], distance_upper_bound=max_d)
    ok = np.isfinite(d)
    off, n = np.einsum("ij,ij->i", S[ok, :2] - B[i[ok], :2], N[i[ok]]), N[i[ok]]
    out = {}
    for name, ax, sgn in SIDES:
        g = off[(n[:, ax] * sgn > 0.9) & (off > -0.4) & (off < max_d)]
        if len(g) < 100:
            out[name] = (np.nan, g); continue
        h, e = np.histogram(g, bins=np.arange(-0.4, max_d, 0.02))
        # tallest peak, or an outer one nearly as tall: cabinets stand in front of a wall, but a recess/window niche
        # also sits outside it. ponytail: fixed 0.8 ratio, check the per-wall plots when a wall has two surfaces
        strong = [j for j in range(len(h)) if h[j] >= 0.8 * h.max() and h[j] >= h[max(j - 1, 0)] and h[j] >= h[min(j + 1, len(h) - 1)]]
        pk = e[strong[-1]] + 0.01
        out[name] = (float(np.median(g[np.abs(g - pk) < 0.06])), g)
    return out


def scans(a):
    """Room scans vs BIM: width excess per axis, wall thickness, repeat between scans, fit over all rooms, plots."""
    import glob
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    icp = ROOT / "icp"
    out_dir = CAPS / "verify"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bim in sorted((icp / "data" / "bim").glob("*.ply")):
        room = bim.stem
        runs = sorted(glob.glob(str(icp / "results" / f"*_{room}_*" / "transform.npy")))
        files = sorted((icp / "data" / "scans").glob(f"{room}*.ply"))
        if not runs or not files:
            print(f"{room}: no scan or no ICP transform (run icp/align.py --pair {room}), skipped"); continue
        M = np.load(runs[-1])  # same station: every scan of the room shares the frame of <room>.ply
        B = band_z(np.asarray(o3d.io.read_point_cloud(str(bim)).points), *a.band)
        W = np.ptp(B[:, 0]), np.ptp(B[:, 1])
        fig, axs = plt.subplots(len(files), 4, figsize=(16, 3.2 * len(files)), squeeze=False)
        for r, f in enumerate(files):
            S = np.asarray(o3d.io.read_point_cloud(str(f)).points) @ M[:3, :3].T + M[:3, 3]
            o = wall_offsets(B, band_z(S, *a.band))
            ex = (o["+x"][0] + o["-x"][0], o["+y"][0] + o["-y"][0])
            thick = [np.std(g[np.abs(g - v) < 0.06]) for v, g in o.values() if np.isfinite(v)]
            print(f"{f.name:28s} walls +x {o['+x'][0] * 100:5.1f} -x {o['-x'][0] * 100:5.1f} +y {o['+y'][0] * 100:5.1f} "
                  f"-y {o['-y'][0] * 100:5.1f} cm | reads wider: x {ex[0] * 100:5.1f} cm over {W[0]:.2f} m, "
                  f"y {ex[1] * 100:5.1f} cm over {W[1]:.2f} m | wall thickness {np.median(thick) * 100:.1f} cm")
            rows += [(W[0], ex[0]), (W[1], ex[1])]
            for c, (name, _, _) in enumerate(SIDES):
                v, g = o[name]
                ax = axs[r, c]; ax.hist(g * 100, bins=np.arange(-40, 80, 2)); ax.axvline(0, color="k", lw=1)
                if np.isfinite(v):
                    ax.axvline(v * 100, color="r", lw=1)
                ax.set_title(f"{f.stem} wall {name}: {v * 100:.1f} cm", fontsize=9); ax.set_xlabel("cm outside BIM wall")
        fig.tight_layout(); fig.savefig(out_dir / f"scans_{room}.png", dpi=90); plt.close(fig)
    assert rows, "no scan/BIM pairs with an ICP transform"
    Wv, E = (np.array(c) for c in zip(*rows))
    ok = np.isfinite(E); Wv, E = Wv[ok], E[ok]
    print(f"\n{len(E)} room widths {Wv.min():.1f}-{Wv.max():.1f} m: scan reads wider by median {np.median(E) * 100:.1f} cm "
          f"(min {E.min() * 100:.1f}, max {E.max() * 100:.1f})")
    slope, c0 = np.polyfit(Wv, E, 1)
    print(f"fit: excess = {c0 * 100:.1f} cm + {slope * 100:.2f}% * width  ->  delta ~ {c0 * 50:.1f} cm per wall, kappa ~ {1 + slope:.4f}")
    for name, k, d in (("offset     (calibrate_joint)", 1.0, 0.194), ("linear     (calibrate_joint)", 1.0336, 0.192),
                       ("board-size (calibrate.py)  ", 1.0291, 0.125)):
        pred = 2 * d + (k - 1) * Wv
        print(f"  model {name}: predicts {pred.min() * 100:.0f}-{pred.max() * 100:.0f} cm, median miss {np.median(np.abs(pred - E)) * 100:.1f} cm")
    print(f"plots, one histogram per wall (black = BIM wall, red = scan wall): {out_dir / 'scans_<room>.png'}")


# ---------------------------------------------------------------- wall
def wall_distance(xyz, axis, cone_deg, kappa, delta, rng=np.random.default_rng(0)):
    """Perpendicular distance (m) from the LiDAR origin to the wall in front, raw and range-corrected, plus the
    plane fit scatter (m) and point count. Wall = largest plane among points within cone_deg of `axis` whose
    normal faces the sensor (within 30 deg), so a floor or ceiling inside the cone is skipped."""
    xyz = xyz[np.isfinite(xyz).all(1)]
    r = np.linalg.norm(xyz, axis=1)
    p = xyz[(r > 0.3) & (xyz @ axis > r * np.cos(np.radians(cone_deg)))]
    for _ in range(4):
        if len(p) < 50:
            return None
        n, best = None, None
        for _ in range(300):  # RANSAC, 3-point planes, 2 cm inliers
            s = p[rng.choice(len(p), 3, replace=False)]
            m = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(m) < 1e-9:
                continue
            m /= np.linalg.norm(m)
            inl = np.abs((p - s[0]) @ m) < 0.02
            if best is None or inl.sum() > best.sum():
                n, best = m, inl
        if abs(n @ axis) > np.cos(np.radians(30)):
            w = p[best]
            (_, d_raw), (_, d_cor) = fit_plane(w), fit_plane(correct_range(w, kappa, delta))
            nn, dd = fit_plane(w)
            return d_raw, d_cor, float(np.std(w @ nn - dd)), len(w)
        p = p[~best]  # floor/ceiling: drop it and look again
    return None


def grab(host, port, seconds):
    """seconds of LiDAR frames from lidar_bridge as one (N,3) cloud."""
    sock = socket.create_connection((host, port), timeout=3.0)
    out, t0 = [], time.time()
    try:
        while time.time() - t0 < seconds:
            out.append(cg.read_frame(sock)[1])
    finally:
        sock.close()
    return np.vstack(out)


def wall(a):
    """Live loop: Enter = take a reading at the current spot, type the laser-meter distance, q = quit."""
    kappa, delta, Tlc = load_model(a.calib)
    axis = Tlc[:3, 2] / np.linalg.norm(Tlc[:3, 2])  # camera viewing direction in the LiDAR frame
    CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV.exists()
    print(f"range model from {a.calib}: r_meas = {kappa:.4f} * r_true + {delta * 100:.1f} cm")
    print("Face the rig square to a flat, matte wall (no glass, nothing in front). 3 readings per spot.")
    while True:
        cmd = input("\nEnter = read, q = quit: ").strip().lower()
        if cmd == "q":
            break
        res = wall_distance(grab(a.host, a.port, a.seconds), axis, a.cone, kappa, delta)
        if res is None:
            print("no wall found in front - closer, squarer, or a wider --cone"); continue
        d_raw, d_cor, sd, n = res
        print(f"LiDAR: raw {d_raw:.3f} m, corrected {d_cor:.3f} m, plane scatter {sd * 100:.1f} cm, {n} points")
        ref = input("laser-meter distance in m (empty = discard): ").strip()
        if not ref:
            continue
        with open(CSV, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "label", "meter_m", "lidar_raw_m", "lidar_corrected_m", "scatter_cm", "points", "calib"]); new = False
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), a.label, float(ref), round(d_raw, 4), round(d_cor, 4),
                        round(sd * 100, 2), n, Path(a.calib).name])
        print(f"saved: raw reads {(d_raw - float(ref)) * 100:+.1f} cm, corrected {(d_cor - float(ref)) * 100:+.1f} cm vs meter "
              f"(includes the fixed mark-to-LiDAR offset)")


def wall_fit_rows(rows):
    """rows of (meter, raw, corrected) -> slope, offset of raw = slope * meter + offset, and per-spot stats
    [(meter, n, corrected - meter mean, repeat std)] grouped by meter reading to 5 cm."""
    m, raw, cor = (np.array(c, float) for c in zip(*rows))
    slope, off = np.polyfit(m, raw, 1)
    spots = []
    for key in np.unique(np.round(m / 0.05)):
        s = np.round(m / 0.05) == key
        spots.append((m[s].mean(), int(s.sum()), (cor[s] - m[s]).mean(), raw[s].std()))
    return slope, off, spots


def wall_fit(a):
    """Verdict over every wall reading in wall.csv."""
    rows = [(float(r["meter_m"]), float(r["lidar_raw_m"]), float(r["lidar_corrected_m"])) for r in csv.DictReader(open(CSV))]
    assert len(rows) >= 4, f"need >= 4 wall readings in {CSV}"
    slope, off, spots = wall_fit_rows(rows)
    print(f"{len(rows)} readings, {len(spots)} spots, {spots[0][0]:.2f}-{spots[-1][0]:.2f} m")
    print(f"raw LiDAR = {slope:.4f} * meter + {off * 100:.1f} cm   (slope = the L2's range scale; offset includes your mark)")
    print(f"{'meter m':>8s} {'n':>3s} {'corrected - meter':>18s} {'repeat std':>11s}")
    for mm, n, e, sd in spots:
        print(f"{mm:8.2f} {n:3d} {e * 100:15.1f} cm {sd * 100:8.1f} cm")
    errs = np.array([s[2] for s in spots])
    drift = errs.max() - errs.min()
    print(f"\ncorrected error drifts {drift * 100:.1f} cm over the range ({'PASS: correction holds' if drift < 0.03 else 'FAIL: correction does not hold'}"
          f" at < 3 cm). The constant part ({errs.mean() * 100:+.1f} cm) is the mark offset, not LiDAR error.")


def selftest():
    """Synthetic wall + floor seen with r_meas = 1.03 r + 0.19: wall_distance finds the wall (not the floor) and the
    corrected distance is right; wall_fit_rows recovers the slope and a flat corrected error."""
    rng = np.random.default_rng(0)
    axis = np.array([1.0, 0, 0])
    for D in (1.0, 4.0, 8.0):
        wall_pts = np.c_[np.full(4000, D), rng.uniform(-2, 2, 4000), rng.uniform(-0.9, 1.5, 4000)]
        floor = np.c_[rng.uniform(0.5, D, 6000), rng.uniform(-2, 2, 6000), np.full(6000, -1.0)]
        P = np.vstack([wall_pts, floor]); r = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / r * (1.03 * r + 0.19 + rng.normal(0, 0.01, r.shape))
        d_raw, d_cor, sd, n = wall_distance(P, axis, 25, 1.03, 0.19)
        print(f"selftest wall at {D} m: raw {d_raw:.3f}, corrected {d_cor:.3f}, scatter {sd * 100:.1f} cm")
        assert abs(d_cor - D) < 0.01 and d_raw > D + 0.15
    rows = [(m, 1.03 * (m + 0.05) + 0.19, m + 0.05) for m in (1, 1, 2, 4, 6, 8)]  # mark 5 cm in front of origin
    slope, off, spots = wall_fit_rows(rows)
    assert abs(slope - 1.03) < 1e-9 and all(abs(s[2] - 0.05) < 1e-9 for s in spots) and spots[0][1] == 2
    # points: board 2 m ahead, L2 reads 19 cm long, dark squares 2 cm farther, 5 mm noise, 10 frames
    q = np.c_[rng.uniform(-0.3, 0.3, (4000, 2)), np.full(4000, 2.0)]
    dark = rng.random(4000) < 0.5
    rr = np.linalg.norm(q, axis=1) + 0.19 + 0.02 * dark + rng.normal(0, 0.005, 4000)
    s = point_stats(q / np.linalg.norm(q, axis=1, keepdims=True) * rr[:, None], np.where(dark, 130, 255),
                    np.arange(4000) % 10, np.array([0, 0, 1.0]), 2.0, 1.0, 0.19)
    assert abs(s["raw"] - 0.20) < 0.005 and abs(s["cor_white"]) < 0.002 and abs(s["cor_dark"] - 0.02) < 0.002
    assert abs(s["noise_white"] - 0.005) < 0.001 and s["frame_jitter"] < 0.002 and s["incidence"] < 1
    # scans: 6 x 9 m box room, scan 15 cm out on every wall and shifted 7 cm in x by ICP -> reads 30 cm wider per axis
    g = np.linspace(-0.5, 0.5, 200)

    def box(w, h, o):
        """Wall points of a w x h room, every wall pushed out by o, stacked over 1 m of height."""
        ring = np.vstack([np.c_[np.full_like(g, -w / 2 - o), g * h], np.c_[np.full_like(g, w / 2 + o), g * h],
                          np.c_[g * w, np.full_like(g, -h / 2 - o)], np.c_[g * w, np.full_like(g, h / 2 + o)]])
        return np.vstack([np.c_[ring, np.full(len(ring), z)] for z in np.arange(0, 1, 0.05)])
    cab = np.c_[np.full(3000, 3.05), np.linspace(-1, 1, 3000), np.linspace(0, 1, 3000)]  # cabinet 5 cm in front of +x
    o = wall_offsets(box(6, 9, 0), np.vstack([box(6, 9, 0.15) + [0.07, 0, 0], cab]))
    assert abs(o["+x"][0] + o["-x"][0] - 0.30) < 0.01 and abs(o["+y"][0] + o["-y"][0] - 0.30) < 0.01, o
    assert abs(o["+x"][0] - 0.22) < 0.01  # translation shows up per wall, cancels per axis
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["boards", "points", "scans", "wall", "wall-fit", "selftest"])
    ap.add_argument("--erode", type=float, default=0.10, help="points: m trimmed off the board edges (edge bloom)")
    ap.add_argument("--band", type=float, nargs=2, default=(0.6, 1.9), help="scans: wall band, m above the floor")
    ap.add_argument("--calib", default=str(CAPS / "calibration" / "extrinsics.json"),
                    help="range model + transform to use (e.g. captures/calibration/extrinsics_joint.json)")
    ap.add_argument("--seconds", type=float, default=3.0, help="LiDAR seconds per wall reading")
    ap.add_argument("--cone", type=float, default=15.0, help="deg around the camera view direction searched for the wall")
    ap.add_argument("--label", default="", help="note stored with each wall reading, e.g. corridor")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9899)
    a = ap.parse_args()
    {"boards": boards, "points": points, "scans": scans, "wall": wall, "wall-fit": wall_fit, "selftest": lambda _: selftest()}[a.mode](a)
