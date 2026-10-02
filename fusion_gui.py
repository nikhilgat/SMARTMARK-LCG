"""Live LiDAR-camera fusion: L2 points drawn on the D435i image, and the L2 cloud coloured by the camera.

    .\\bridge\\start_lidar.ps1
    python calib/fusion_gui.py                     # uses captures/calibration/extrinsics.json
    python calib/fusion_gui.py --selftest

Left: camera image + projected LiDAR points (colour = distance). Right: 3D cloud, points inside the camera
view take the pixel colour, the rest are dark blue (tick "camera view only" to hide them).

Used for: visually checking the LiDAR-camera calibration live (if it's right, LiDAR points line up with
          edges in the image) and as a demo of fused colour point clouds.
Works with: calibrate.py (needs its extrinsics.json, reuses correct_range), capture_gui.py (reuses its Rig
          for the sensor threads), bridge/start_lidar.ps1 (LiDAR data). Needs pyrealsense2, open3d, opencv.
"""
import argparse, json, threading, time
from pathlib import Path

import capture_gui as cg  # first: disables OpenCV OpenCL before cv2 loads (iGPU crash)
import cv2
import numpy as np
from calibrate import correct_range


def load_calib(path):
    """Read extrinsics.json into a dict: R, t (LiDAR -> camera), K, dist (camera lens) and the
    range model kappa, delta (defaults to 'no correction' for older files)."""
    e =json.loads(Path(path).read_text())
    T = np.array(e["T_cam_lidar"])
    rm = e.get("range_model", {"kappa": 1.0, "delta_m": 0.0})
    return dict(R=T[:3, :3], t=T[:3, 3], K=np.array(e["K"]), dist=np.array(e["dist"]),
                kappa=rm["kappa"], delta=rm["delta_m"])


def project(xyz, c, w, h):
    """Work out where each LiDAR point lands in the camera image.
    LiDAR points -> (range-corrected lidar xyz, pixel uv (N,2) int, inside mask, camera depth).
    `inside` is True for points in front of the camera and within the w x h image."""
    xyz = correct_range(xyz.astype(np.float64), c["kappa"], c["delta"])
    p = xyz @ c["R"].T + c["t"]
    uv = np.zeros((len(p), 2), int)
    K = c["K"]
    front = p[:, 2] > 0.1
    # pinhole pre-check with margin: the L2 sees ~360 deg, and the distortion polynomial folds far off-axis
    # points back into the image, so only points near the view go through projectPoints
    with np.errstate(divide="ignore", invalid="ignore"):
        pu = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]; pv = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    near = front & (pu > -0.2 * w) & (pu < 1.2 * w) & (pv > -0.2 * h) & (pv < 1.2 * h)
    if near.any():
        uv[near] = np.round(cv2.projectPoints(p[near], np.zeros(3), np.zeros(3), K, c["dist"])[0].reshape(-1, 2)).astype(int)
    inside = near & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return xyz, uv, inside, p[:, 2]


def overlay(img, uv, inside, depth, max_depth=6.0):
    """Return a copy of the camera image with the projected LiDAR points drawn on it.
    Draw 2x2 px points coloured by distance, far first so near points stay on top."""
    view = img.copy()
    h, w = img.shape[:2]
    idx = np.where(inside)[0]
    idx = idx[np.argsort(-depth[idx])]
    col = cv2.applyColorMap(np.clip(depth[idx] / max_depth * 255, 0, 255).astype(np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_TURBO)[:, 0]
    u, v = uv[idx, 0], uv[idx, 1]
    for du in (0, 1):
        for dv in (0, 1):
            view[np.minimum(v + dv, h - 1), np.minimum(u + du, w - 1)] = col
    return view


def run(rig, calib, a):
    """Open the fusion window (left: camera + LiDAR overlay and controls, right: 3D coloured cloud),
    start the sensor threads and redraw ~15 times a second until the window is closed."""
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    win = app.create_window("L2 + D435i fusion", 1880, 900)
    em = win.theme.font_size

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    scene.scene.set_background([0.06, 0.06, 0.08, 1.0])
    scene.scene.show_axes(False)
    mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.5

    pw = a.preview_width
    panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
    image = gui.ImageWidget(o3d.geometry.Image(np.zeros((pw * 9 // 16, pw, 3), np.uint8)))
    status = gui.Label("starting...")
    fov_only = gui.Checkbox("3D: camera view only"); fov_only.checked = False
    decay = gui.Slider(gui.Slider.DOUBLE); decay.set_limits(0.2, 3.0); decay.double_value = rig.decay
    decay.set_on_value_changed(lambda v: setattr(rig, "decay", v))
    panel.add_child(image); panel.add_child(status)
    row = gui.Horiz(0.5 * em); row.add_child(gui.Label("LiDAR history (s)")); row.add_child(decay)
    panel.add_child(row); panel.add_child(fov_only)
    panel.add_child(gui.Label(f"range model: r_true = (r - {calib['delta']:.3f}) / {calib['kappa']:.3f}"))
    win.add_child(scene); win.add_child(panel)

    def on_layout(ctx):
        """Fixed-width panel on the left, 3D view fills the rest."""
        r = win.content_rect
        w0 = pw + 2 * int(em)
        panel.frame = gui.Rect(r.x, r.y, w0, r.height)
        scene.frame = gui.Rect(r.x + w0, r.y, r.width - w0, r.height)
    win.set_on_layout(on_layout)

    first, busy = [True], threading.Event()

    def refresh():
        """One redraw: project the latest cloud into the latest image, update the 2D overlay, colour each
        3D point with its pixel's colour, and on the first frame put the 3D viewpoint at the camera."""
        try:
            img, cloud = rig.cam_img, rig.live_cloud()
            if cloud[0] is None:
                status.text = f"camera: {rig.cam_status} | lidar: {rig.lidar_status}"
                return
            h, w = img.shape[:2] if img is not None else (a.height, a.width)
            xyz, uv, inside, depth = project(cloud[0], calib, w, h)
            rgb = np.tile([0.12, 0.16, 0.30], (len(xyz), 1))  # outside the camera view: dark blue
            if img is None:  # no camera yet: grey cloud only
                inside[:] = False
                status.text = f"{len(xyz)} LiDAR pts | camera: {rig.cam_status}"
            else:
                status.text = f"{len(xyz)} LiDAR pts, {inside.sum()} in camera view"
                view = cv2.resize(overlay(img, uv, inside, depth), (pw, pw * h // w))
                image.update_image(o3d.geometry.Image(np.ascontiguousarray(view[:, :, ::-1])))
                rgb[inside] = img[uv[inside, 1], uv[inside, 0], ::-1] / 255.0
            keep = inside if fov_only.checked else slice(None)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[keep]))
            pcd.colors = o3d.utility.Vector3dVector(rgb[keep])
            scene.scene.clear_geometry()
            scene.scene.add_geometry("cloud", pcd, mat)
            if first[0] and len(pcd.points) > 1000:  # start the 3D view at the RGB camera, looking where it looks
                scene.setup_camera(60.0, pcd.get_axis_aligned_bounding_box(), [0, 0, 0])
                eye = -calib["R"].T @ calib["t"]                 # camera centre in the lidar frame
                fwd, down = calib["R"][2], calib["R"][1]         # camera z / y axes in the lidar frame
                scene.look_at(eye + 3.0 * fwd, eye - 0.5 * fwd, -down)
                first[0] = False
        finally:
            busy.clear()

    def ticker():
        """Background thread asking the GUI thread to redraw ~15x a second."""
        while rig.running:
            if not busy.is_set():  # never queue a redraw behind an unfinished one
                busy.set()
                app.post_to_main_thread(win, refresh)
            time.sleep(1 / 15)

    def on_close():
        """Window closed: stop the sensor threads."""
        rig.running = False
        return True
    win.set_on_close(on_close)

    threading.Thread(target=rig.lidar_loop, daemon=True).start()
    threading.Thread(target=rig.camera_loop, args=(a.width, a.height), daemon=True).start()
    threading.Thread(target=ticker, daemon=True).start()
    app.run()


def selftest():
    """No-hardware check (--selftest): project()/overlay() put known points in the right place, reject points
    behind or far off to the side of the camera, and apply the range model."""
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    c = dict(R=np.eye(3), t=np.zeros(3), K=K, dist=np.array([0.11, -0.26, 0, 0, 0.05]), kappa=1.0, delta=0.0)
    pts = np.array([[0, 0, 2.0],      # straight ahead -> image centre
                    [0, 0, -2.0],     # behind the camera
                    [10.0, 0, 1.0],   # far off-axis: distortion would fold it back into the image
                    [0.5, 0.2, 2.0]])
    _, uv, inside, depth = project(pts, c, 1280, 720)
    assert inside.tolist() == [True, False, False, True], inside
    assert tuple(uv[0]) == (640, 360)
    img = np.zeros((720, 1280, 3), np.uint8)
    assert overlay(img, uv, inside, depth)[360, 640].any() and not img.any()
    c2 = dict(c, kappa=1.0, delta=0.5)  # range model: 2.5 m measured -> 2.0 m true
    assert np.isclose(project(np.array([[0, 0, 2.5]]), c2, 1280, 720)[3][0], 2.0)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(Path(__file__).resolve().parent.parent / "captures" / "calibration" / "extrinsics.json"))
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9899)
    ap.add_argument("--decay", type=float, default=1.0, help="seconds of LiDAR frames overlaid")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--preview-width", type=int, default=1100)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        import open3d.visualization.gui  # noqa: F401  main-thread imports first: concurrent pybind11 imports deadlock
        import pyrealsense2  # noqa: F401
        run(cg.Rig(Path(a.calib).parent, a.host, a.port, a.decay), load_calib(a.calib), a)
