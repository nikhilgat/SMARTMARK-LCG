"""Board (whiteboard) width x height as the L2 sees it, from the clouds wall_capture.py saved.

    python Lidar-Test/board_size.py [--real 1.20 0.90]
    python Lidar-Test/board_size.py --selftest

Per capture: points within --gate of the board depth (folder name = metered distance) and --cone deg of the camera
axis -> RANSAC plane -> 2D grid of the inliers in that plane -> the connected patch the camera axis hits (drops the
stand, the wall behind, anything coplanar beside it) -> minimum-area rectangle (board rotation doesn't matter).
Prints the full rectangle and a 1-99 % trimmed one: the L2 beam footprint smears edges outward (~9 cm per edge on
the checkerboard at 1-2 m), so expect the board to read larger, more so farther away as the beam widens.
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calib"))
import verify_lidar as vl  # first: pulls in capture_gui, which disables OpenCV OpenCL before cv2 loads
import cv2
import numpy as np
from scipy import ndimage
from wall_capture import plane_robust


def board_rect(P, axis, dist, gate=0.4, cone=40.0, cell=0.03):
    """(width, height, trimmed width, trimmed height, points) in m of the board hit by `axis` at about `dist`
    (raw L2 range along the axis), or None."""
    r = np.linalg.norm(P, axis=1)
    p = P[(np.abs(P @ axis - dist) < gate) & (P @ axis > r * np.cos(np.radians(cone)))]
    if len(p) < 100:
        return None
    n, d = plane_robust(p)
    p = p[np.abs(p @ n - d) < 0.02]
    u = np.cross(n, axis if abs(n @ axis) < 0.9 else [0, 0, 1.0])  # any in-plane basis; the rectangle fit rotates
    u = np.cross(n, u); u /= np.linalg.norm(u); v = np.cross(n, u)
    q = np.c_[p @ u, p @ v]
    lo = q.min(0)
    ij = ((q - lo) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 1, bool); grid[ij[:, 0], ij[:, 1]] = True
    lab, _ = ndimage.label(ndimage.binary_closing(grid, iterations=2) | grid)
    hit = (d / (n @ axis)) * axis  # where the camera axis meets the plane
    c = ((np.array([hit @ u, hit @ v]) - lo) / cell).astype(int)
    if not (0 <= c[0] < grid.shape[0] and 0 <= c[1] < grid.shape[1]) or lab[c[0], c[1]] == 0:
        return None
    q = q[lab[ij[:, 0], ij[:, 1]] == lab[c[0], c[1]]].astype(np.float32)
    (_, _), (w, h), ang = cv2.minAreaRect(q)
    a = np.radians(ang)
    rq = q @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    tw, th = np.ptp(np.percentile(rq, [1, 99], axis=0), axis=0)
    (w, h), (tw, th) = sorted((w, h), reverse=True), sorted((tw, th), reverse=True)
    return w, h, tw, th, len(q)


def selftest():
    """1.2 x 0.9 m board, rotated 20 deg in its plane and tilted, at 4 m, wall 1.5 m behind and a stand below
    touching nothing: board_rect returns 1.2 x 0.9."""
    rng = np.random.default_rng(0)
    b = np.c_[rng.uniform(-.6, .6, 20000), rng.uniform(-.45, .45, 20000)]
    rot = np.array([[np.cos(.35), -np.sin(.35)], [np.sin(.35), np.cos(.35)]])
    b = b @ rot.T
    t = np.radians(5)  # tilt about the vertical
    board = np.c_[4 + b[:, 0] * np.sin(t), b[:, 0] * np.cos(t), b[:, 1]]
    wall = np.c_[np.full(30000, 5.5), rng.uniform(-3, 3, 30000), rng.uniform(-1.5, 2, 30000)]
    stand = np.c_[np.full(500, 4.0), rng.uniform(-.02, .02, 500), rng.uniform(-1.4, -.8, 500)]
    w, h, *_ = board_rect(np.vstack([board, wall, stand]), np.array([1.0, 0, 0]), 4.0)
    print(f"selftest: {w:.3f} x {h:.3f} m (true 1.200 x 0.900)")
    assert abs(w - 1.2) < 0.03 and abs(h - 0.9) < 0.03
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=float, nargs=2, metavar=("W", "H"), help="tape-measured board size, m")
    ap.add_argument("--gate", type=float, default=0.4); ap.add_argument("--cone", type=float, default=40.0)
    ap.add_argument("--calib", default=str(vl.CAPS / "calibration" / "extrinsics_joint.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit()
    kappa, delta, Tlc = vl.load_model(a.calib)
    axis = Tlc[:3, 2] / np.linalg.norm(Tlc[:3, 2])
    print(f"{'capture':26s} {'full W x H':>15s} {'trimmed W x H':>15s} {'points':>7s}" + ("   excess (trimmed)" if a.real else ""))
    for d in sorted(HERE.glob("*_*m")):
        meter = float(d.name.split("_")[1][:-1])
        res = board_rect(np.load(d / "lidar.npz")["xyz"].astype(float), axis, kappa * meter + delta, a.gate, a.cone)
        if res is None:
            print(f"{d.name:26s} board not found"); continue
        w, h, tw, th, n = res
        ex = f"   {(tw - a.real[0]) * 100:+5.1f} x {(th - a.real[1]) * 100:+5.1f} cm" if a.real else ""
        print(f"{d.name:26s} {w:6.3f} x {h:5.3f} m {tw:6.3f} x {th:5.3f} m {n:7d}{ex}")
