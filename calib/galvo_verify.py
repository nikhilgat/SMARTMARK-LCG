"""Check galvo_calib.json against the checkerboard: aim the laser at every inner corner of the A1 board (7x5 = 35),
find each dot in the camera image and record how far it landed from the corner.

    python calib/galvo_verify.py --label door_near   # per board pose SPACE: dot visits corner by corner, q quits
    python calib/galvo_verify.py selftest   # no hardware

Corner 3D positions come from the board pose in a laser-off frame, so this checks the whole chain the laser
uses: camera intrinsics + board pose + galvo calibration. Error is measured on the board plane: dot pixel ->
camera ray -> board plane, minus the corner, as (dx, dy) mm along the board x/y; "bias" is the mean offset in
the camera frame (right, down), "spread" the error left after removing it: a large bias with a small spread is
a systematic shift, not noise. Also try poses/distances that were NOT in the calibration views.

Saved per run in captures/galvo/verify/<date_time>[_label]/, per pass: pass_NN_board.png (laser off),
pass_NN_dots.png (all real dots in one picture), pass_NN_annotated.png (green corner, red dot, yellow arrow =
miss x5), pass_NN.json (every corner). Every pass of every run is also appended to captures/galvo/verify/
summary.csv (distance, board position, tilt, errors, bias, spread) to compare spots and distances.
"""
from galvo_calib import (BOARD, OUT, SQ, Camera, Galvo, board_pose, covered, find_dot, load_cam, load_calib,  # first:
                         ray_plane, show, to_dac)  # sets the OpenCL env var before cv2 loads
import argparse, csv, json, time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from calibrate import board_obj, detect  # noqa: E402

VOUT = OUT / "verify"


def board_err(px, K, dist, R, t, corner):
    """Where a dot seen at pixel px landed relative to its target corner: (dx, dy) in mm in the board frame."""
    return (R.T @ (ray_plane(px, K, dist, R, t) - corner))[:2] * 1000


def annotate(img, hit, s, gain=5):
    """Corners (green), real dots (red) and each miss as a yellow arrow drawn `gain` x longer so mm misses show."""
    v = img.copy()
    for r in hit:
        u, d = np.array(r["corner_px"]), np.array(r["dot_px"])
        cv2.arrowedLine(v, tuple(np.round(u).astype(int)), tuple(np.round(u + gain * (d - u)).astype(int)), (0, 255, 255), 2, tipLength=0.2)
        cv2.circle(v, tuple(np.round(u).astype(int)), 5, (0, 255, 0), 1); cv2.circle(v, tuple(np.round(d).astype(int)), 5, (0, 0, 255), 1)
    for j, line in enumerate([f"{s['dist_m']:.2f} m, tilt {s['tilt_deg']:.0f} deg | {s['seen']}/{s['corners']} dots | arrows x{gain}",
                              f"error median {s['median_mm']:.1f} mm, 95% {s['p95_mm']:.1f}, max {s['max_mm']:.1f} | "
                              f"bias right {s['bias_right_mm']:+.1f} down {s['bias_down_mm']:+.1f} mm"]):
        cv2.putText(v, line, (20, 40 + 35 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return v


def run(a):
    """Live view; SPACE runs one pass over all corners for the current board pose and saves it. q quits.
    Each run gets its own folder verify/<date_time>[_label]/; every pass is also appended to verify/summary.csv."""
    K, dist = load_cam()
    c = load_calib()
    sess = VOUT / (time.strftime("%Y%m%d_%H%M%S") + (f"_{a.label}" if a.label else ""))
    sess.mkdir(parents=True, exist_ok=True)
    cam, gal = Camera(), Galvo(a.port, a.speed, a.dry_run)
    n = 0
    try:
        while True:
            img, _ = cam.frame()
            k = show(img, f"passes {n} | board {'OK' if board_pose(img, K, dist) else 'NOT found'} | SPACE: hit corners  q: quit")
            if k == ord("q"):
                break
            if k != ord(" "):
                continue
            gal.stop(); time.sleep(a.settle)
            off, _ = cam.frame()
            pose, uv = board_pose(off, K, dist), detect(off, *BOARD)
            if pose is None or uv is None:
                print("board lost, move nothing and retry"); continue
            R, t = pose
            P = board_obj(*BOARD, SQ) @ R.T + t  # corners in the camera frame, same order as uv
            rows, marks, dots_img = [], [], off.copy()
            for i, (p, d, u) in enumerate(zip(P, to_dac(P, c), uv.reshape(-1, 2))):
                gal.dot(*d); time.sleep(a.settle)
                on, _ = cam.frame()
                dots_img = np.maximum(dots_img, on)  # every real dot ends up in one picture of the board
                px = find_dot(on, off)
                row = dict(corner=i, p_cam=p.tolist(), dac=d.tolist(), corner_px=u.tolist(),
                           covered=bool(covered(p, c)[0]), dot_px=None)
                marks.append((u, True))
                if px is not None:
                    row.update(dot_px=px.tolist(), err_mm=board_err(px, K, dist, R, t, p).tolist(),
                               err_px=float(np.linalg.norm(px - u)))
                    marks.append((px, False))
                rows.append(row)
                if show(on, f"corner {i + 1}/{len(P)}", marks) == ord("q"):
                    raise KeyboardInterrupt
            gal.stop()
            hit = [r for r in rows if r["dot_px"]]
            if not hit:
                print("no dots seen - laser on? board in reach?"); continue
            E = np.array([r["err_mm"] for r in hit]); m = np.linalg.norm(E, axis=1)
            ctr = P.mean(0)
            bias = (E @ R[:, :2].T).mean(0)  # camera frame (x right, y down): the detector may flip the board's axes
            s = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), session=sess.name, label=a.label, pass_no=n,
                     dist_m=float(np.linalg.norm(ctr)), board_x_m=float(ctr[0]), board_y_m=float(ctr[1]), board_z_m=float(ctr[2]),
                     tilt_deg=float(np.degrees(np.arccos(min(abs(R[2, 2]), 1.0)))), corners=len(rows), seen=len(hit),
                     outside_calibrated=sum(not r["covered"] for r in rows),
                     median_mm=float(np.median(m)), p95_mm=float(np.percentile(m, 95)), max_mm=float(m.max()),
                     bias_right_mm=float(bias[0]), bias_down_mm=float(bias[1]),
                     spread_mm=float(np.median(np.linalg.norm(E - E.mean(0), axis=1))),  # error left after removing the bias
                     median_px=float(np.median([r["err_px"] for r in hit])))
            vis = annotate(dots_img, hit, s)
            stem = sess / f"pass_{n:02d}"
            cv2.imwrite(f"{stem}_board.png", off); cv2.imwrite(f"{stem}_dots.png", dots_img); cv2.imwrite(f"{stem}_annotated.png", vis)
            Path(f"{stem}.json").write_text(json.dumps(dict(summary=s, R=R.tolist(), t=t.tolist(), corners=rows), indent=1))
            csv_path = VOUT / "summary.csv"
            new = not csv_path.exists()
            with open(csv_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(s))
                if new:
                    w.writeheader()
                w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
            cv2.imshow("last pass", cv2.resize(vis, None, fx=0.75, fy=0.75))
            print(f"{sess.name}/pass_{n:02d} @ {s['dist_m']:.2f} m, tilt {s['tilt_deg']:.0f} deg: {s['seen']}/{s['corners']} dots | "
                  f"error median {s['median_mm']:.1f} mm, 95% {s['p95_mm']:.1f}, max {s['max_mm']:.1f} | "
                  f"bias right {s['bias_right_mm']:+.1f} down {s['bias_down_mm']:+.1f} mm, spread {s['spread_mm']:.1f} mm"
                  + (f" | {s['outside_calibrated']} corners outside calibrated area" if s["outside_calibrated"] else ""))
            n += 1
    except KeyboardInterrupt:
        print("pass aborted")
    finally:
        gal.close(); cam.close(); cv2.destroyAllWindows()


def selftest():
    """board_err: a dot that landed 3 mm along board x and -4 mm along board y from the corner reads (3, -4)."""
    K, dist = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]]), np.zeros(5)
    R, t = cv2.Rodrigues(np.radians([10.0, -20, 5]))[0], np.array([0.1, 0, 2.0])
    corner, landed = R @ [0.19, 0.095, 0] + t, R @ [0.193, 0.091, 0] + t
    px = cv2.projectPoints(landed.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)[0].ravel()
    assert np.allclose(board_err(px, K, dist, R, t, corner), [3, -4], atol=1e-6)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--speed", type=int, default=4000)
    ap.add_argument("--settle", type=float, default=0.25, help="s to wait after moving the dot")
    ap.add_argument("--dry-run", action="store_true", help="no serial output (laser stays off)")
    ap.add_argument("--label", default="", help="name for this spot, e.g. door_near (folder name + summary.csv)")
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run(a)
