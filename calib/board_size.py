"""How big does the L2 think the A1 checkerboard is?  Histograms over every capture vs the real print.

    python calib/board_size.py              # captures/ and captures/old/  ->  captures/calibration/board_size.png

Per capture (calib_target detection, cached by calibrate.py):
  square size  - fitted checker square (real: 95 mm -> 8x6 pattern 760 x 570 mm)
  sheet size   - extent of the connected flat region the board sits in (real A1 sheet: 841 x 594 mm)
Each raw and range-corrected (captures/calibration/extrinsics.json: lateral size scales by r_true / r_meas).
Note the correction was fitted from these square sizes, so "corrected square size ~95 mm" is by construction;
the sheet size is a separate measurement of the same board.

Used for: checking how accurately the L2 LiDAR measures size/distance, and whether the range
          correction from calibrate.py actually fixes it (output: a 4-panel histogram PNG).
Works with: calibrate.py (reuses its board detector and its range model in extrinsics.json),
          captures made by capture_gui.py. Needs numpy, matplotlib, and the SMART-MARK calib_target package.
"""
import argparse, json
from pathlib import Path

import numpy as np
from calibrate import CALIB_TARGET, lidar_board

SQUARE, SHEET = 95.0, (841.0, 594.0)  # mm


def collect(roots, calib_target):
    """Go through every capture folder under `roots`, find the checkerboard in its LiDAR scan, and
    return one row per capture: (distance to board m, square size mm, sheet long side mm, sheet short side mm).
    Captures where the board isn't found are printed and skipped."""
    rows = []
    for root in roots:
        for d in sorted(p for p in Path(root).iterdir() if (p / "lidar.npz").exists()):
            print(f"{d.parent.name}/{d.name}: ", end="", flush=True)
            res = lidar_board(d, calib_target)
            if isinstance(res, str):
                print(f"no board ({res})"); continue
            z = np.load(d / "lidar_board.npz")
            r, sq, sheet = float(np.linalg.norm(z["centre"])), float(z["scale"]) * SQUARE, np.array(z["sheet"]) * 1000
            print(f"{r:.2f} m, square {sq:.1f} mm, sheet {sheet[0]:.0f} x {sheet[1]:.0f} mm")
            rows.append((r, sq, *sheet))
    return np.array(rows)


def lateral_factor(r_meas, kappa, delta):
    """A lateral length at measured range r shrinks by r_true / r_meas once the range error is removed.
    Returns the factor to multiply a LiDAR-measured size by to get the corrected size.
    kappa, delta: the range model from extrinsics.json (r_meas = kappa * r_true + delta)."""
    return (r_meas - delta) / kappa / r_meas


def main():
    """Collect board sizes from all captures, print raw vs range-corrected stats against the real
    print size, and save a histogram/scatter figure to captures/calibration/board_size.png."""
    root =Path(__file__).resolve().parent.parent / "captures"
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="+", default=[str(root), str(root / "old")])
    ap.add_argument("--calib", default=str(root / "calibration" / "extrinsics.json"))
    ap.add_argument("--calib-target", default=CALIB_TARGET)
    ap.add_argument("--out", default=str(root / "calibration" / "board_size.png"))
    a = ap.parse_args()

    rows = collect([c for c in a.captures if Path(c).is_dir()], a.calib_target)
    assert len(rows), "no board detections"
    r, sq, sl, ss = rows.T
    rm = json.loads(Path(a.calib).read_text())["range_model"]
    f = lateral_factor(r, rm["kappa"], rm["delta_m"])

    def stats(name, raw, true):
        """Print mean/std and % error of raw and corrected sizes vs the true size; return corrected sizes."""
        cor = raw * f
        print(f"{name:13s} real {true:5.0f} mm | raw {raw.mean():6.1f} +- {raw.std():4.1f} ({(raw.mean() / true - 1) * 100:+.1f}%) "
              f"| corrected {cor.mean():6.1f} +- {cor.std():4.1f} ({(cor.mean() / true - 1) * 100:+.1f}%)")
        return cor
    print(f"\n{len(rows)} boards, {r.min():.2f}-{r.max():.2f} m, range model kappa {rm['kappa']:.3f} delta {rm['delta_m']:.3f} m")
    panels = [("Square size", sq, SQUARE), ("Sheet long side", sl, SHEET[0]), ("Sheet short side", ss, SHEET[1])]
    cors = [stats(n, raw, t) for n, raw, t in panels]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for axi, (name, raw, true), cor in zip(ax.flat, panels, cors):
        bins = np.linspace(min(raw.min(), cor.min(), true) * 0.97, max(raw.max(), cor.max(), true) * 1.03, 30)
        axi.hist(raw, bins, alpha=0.6, color="#d9534f", label=f"LiDAR raw  (mean {raw.mean():.0f})")
        axi.hist(cor, bins, alpha=0.6, color="#2c7fb8", label=f"range-corrected  (mean {cor.mean():.0f})")
        axi.axvline(true, color="k", ls="--", lw=2, label=f"real A1 print  {true:.0f}")
        axi.set_title(f"{name} (mm)"); axi.set_xlabel("mm"); axi.set_ylabel("captures"); axi.legend(fontsize=9)
    axi = ax[1, 1]
    axi.scatter(r, sq, c="#d9534f", label="raw square size")
    axi.scatter(r, sq * f, c="#2c7fb8", label="range-corrected")
    axi.axhline(SQUARE, color="k", ls="--", lw=2, label="real 95 mm")
    axi.set_title("Square size vs board distance"); axi.set_xlabel("LiDAR distance to board (m)"); axi.set_ylabel("mm")
    axi.legend(fontsize=9)
    fig.suptitle(f"Unitree L2: A1 checkerboard size over {len(rows)} captures")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
