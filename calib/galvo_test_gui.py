"""GUI: draw basic shapes with the galvo laser anywhere in its full mirror range (0..65535 both axes). No camera, no calibration.

    python calib/draw_rect.py                # GUI, laser on COM5
    python calib/draw_rect.py --dry-run      # GUI, prints the serial command instead of sending (laser stays off)
    python calib/draw_rect.py --selftest     # no-hardware check of the shape maths

The square canvas is the whole mirror range. Pick a shape, then:
    rectangle / circle / triangle / star / line : press-drag-release (box or end points), sent on release
    polygon : left-click corners, right-click (or double-click) to close and send
    dot     : click
Esc or "Stop" turns the laser off. Flip X / Y if the canvas moves the opposite way to the laser.

Works with: galvo_calib.py (Galvo class, shape_message). Hardware: galvo laser on COM5. Needs pyserial, numpy, tkinter.
"""
import argparse
import tkinter as tk
from tkinter import ttk

import numpy as np

from galvo_calib import Galvo, shape_message

FULL = 65535
SIZE = 640  # canvas px = whole mirror range
SHAPES = ["rectangle", "circle", "triangle", "star", "line", "polygon", "dot"]


def shape_points(kind, p0, p1, n=36):
    """Mirror points of a shape spanning the box/segment p0 -> p1, in drawing order (the galvo closes the loop)."""
    (x0, y0), (x1, y1) = p0, p1
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    if kind == "rectangle":
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if kind == "line":
        return [(x0, y0), (x1, y1)]
    if kind == "triangle":
        return [(cx, min(y0, y1)), (max(x0, x1), max(y0, y1)), (min(x0, x1), max(y0, y1))]
    if kind == "circle":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    elif kind == "star":
        a = np.linspace(0, 2 * np.pi, 10, endpoint=False) - np.pi / 2
        rx, ry = np.where(np.arange(10) % 2, 0.4, 1.0) * rx, np.where(np.arange(10) % 2, 0.4, 1.0) * ry
    else:
        raise ValueError(kind)
    return list(zip(cx + rx * np.cos(a), cy + ry * np.sin(a)))


class App:
    def __init__(self, root, galvo, dry):
        self.g, self.dry, self.poly, self.p0 = galvo, dry, [], None
        self.kind, self.speed, self.wait = tk.StringVar(value="rectangle"), tk.IntVar(value=7000), tk.IntVar(value=1000)
        self.n, self.fx, self.fy = tk.IntVar(value=36), tk.BooleanVar(), tk.BooleanVar()
        self.status = tk.StringVar(value="pick a shape, drag on the canvas")

        bar = ttk.Frame(root, padding=4)
        bar.pack(fill="x")
        for k in SHAPES:
            ttk.Radiobutton(bar, text=k, value=k, variable=self.kind, command=self.reset).pack(side="left")
        bar2 = ttk.Frame(root, padding=4)
        bar2.pack(fill="x")
        for label, var in [("speed", self.speed), ("wait", self.wait), ("circle pts", self.n)]:
            ttk.Label(bar2, text=label).pack(side="left")
            ttk.Spinbox(bar2, textvariable=var, from_=1, to=100000, width=7).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(bar2, text="flip X", variable=self.fx).pack(side="left")
        ttk.Checkbutton(bar2, text="flip Y", variable=self.fy).pack(side="left")
        ttk.Button(bar2, text="Full-range box", command=self.full_box).pack(side="left", padx=8)
        ttk.Button(bar2, text="Stop (Esc)", command=self.stop).pack(side="left")

        self.c = tk.Canvas(root, width=SIZE, height=SIZE, bg="black", highlightthickness=0, cursor="crosshair")
        self.c.pack()
        for i in range(1, 8):
            q = i * SIZE / 8
            col = "#555" if i == 4 else "#222"
            self.c.create_line(q, 0, q, SIZE, fill=col)
            self.c.create_line(0, q, SIZE, q, fill=col)
        ttk.Label(root, textvariable=self.status, padding=4).pack(fill="x")

        self.c.bind("<ButtonPress-1>", self.press)
        self.c.bind("<B1-Motion>", self.drag)
        self.c.bind("<ButtonRelease-1>", self.release)
        self.c.bind("<Button-3>", self.close_poly)
        self.c.bind("<Double-Button-1>", self.close_poly)
        self.c.bind("<Motion>", lambda e: self.status.set("x %d  y %d" % self.mirror(e.x, e.y)) if not self.poly else None)
        root.bind("<Escape>", lambda e: self.stop())

    # canvas px <-> mirror units
    def mirror(self, px, py):
        x, y = np.clip(px, 0, SIZE) / SIZE * FULL, np.clip(py, 0, SIZE) / SIZE * FULL
        return (FULL - x if self.fx.get() else x), (FULL - y if self.fy.get() else y)

    def canvas(self, x, y):
        x, y = (FULL - x if self.fx.get() else x), (FULL - y if self.fy.get() else y)
        return x / FULL * SIZE, y / FULL * SIZE

    def reset(self):
        self.poly, self.p0 = [], None
        self.c.delete("shape")

    def press(self, e):
        if self.kind.get() == "polygon":
            self.poly.append(self.mirror(e.x, e.y))
            self.preview(self.poly, closed=False)
            self.status.set(f"polygon: {len(self.poly)} corners, right-click to close")
        else:
            self.reset()
            self.p0 = self.mirror(e.x, e.y)

    def drag(self, e):
        if self.p0 and self.kind.get() not in ("polygon", "dot"):
            self.preview(shape_points(self.kind.get(), self.p0, self.mirror(e.x, e.y), self.n.get()))

    def release(self, e):
        k, p0 = self.kind.get(), self.p0
        if k == "polygon" or not p0:
            return
        self.p0 = None
        if k == "dot":
            self.preview([p0, p0])
            self.send([p0])
        else:
            self.send(shape_points(k, p0, self.mirror(e.x, e.y), self.n.get()))

    def close_poly(self, e):
        if self.kind.get() == "polygon" and len(self.poly) >= 2:
            pts, self.poly = self.poly, []
            self.send(pts)

    def full_box(self):
        self.reset()
        self.send(shape_points("rectangle", (0, 0), (FULL, FULL)))

    def preview(self, pts, closed=True):
        self.c.delete("shape")
        xy = [v for p in pts for v in self.canvas(*p)]
        if closed and len(pts) > 2:
            self.c.create_polygon(xy, outline="#0f0", fill="", width=2, tags="shape")
        elif len(pts) >= 2:
            self.c.create_line(xy, fill="#0f0", width=2, tags="shape")
        for x, y in zip(xy[::2], xy[1::2]):
            self.c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#0f0", outline="", tags="shape")

    def send(self, pts):
        self.preview(pts)
        self.g.speed = self.speed.get()
        if len(pts) == 1:
            self.g.dot(*pts[0])
        else:
            self.g.shape(pts, self.wait.get())
        if self.dry:
            print(repr(shape_message(pts, self.g.speed, self.wait.get())))
        self.status.set(f"{self.kind.get()}: {len(pts)} points sent" + (" (dry run)" if self.dry else ""))

    def stop(self):
        self.g.stop()
        self.reset()
        self.status.set("stopped (laser off)")


def selftest():
    r = shape_points("rectangle", (100, 200), (300, 600))
    assert r == [(100, 200), (300, 200), (300, 600), (100, 600)]
    c = np.array(shape_points("circle", (0, 0), (2000, 1000), 24))
    assert len(c) == 24 and np.allclose(c.min(0), [0, 0], atol=1e-6) and np.allclose(c.max(0), [2000, 1000], atol=5)
    s = np.array(shape_points("star", (0, 0), (100, 100)))
    d = np.hypot(*(s - 50).T)
    assert len(s) == 10 and np.allclose(d[::2], 50) and np.allclose(d[1::2], 20)
    t = shape_points("triangle", (10, 10), (30, 50))
    assert t == [(20, 10), (30, 50), (10, 50)]
    assert shape_points("line", (1, 2), (3, 4)) == [(1, 2), (3, 4)]
    assert "X0Y0" in shape_message(shape_points("rectangle", (-50, -50), (70000, 70000)))  # clamped to the mirror grid
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        g = Galvo(a.port, dry=a.dry_run)
        root = tk.Tk()
        root.title("Galvo draw" + (" (dry run)" if a.dry_run else f" ({a.port})"))
        App(root, g, a.dry_run)
        try:
            root.mainloop()
        finally:
            g.close()
