# LiDAR–Camera Calibration Roadmap
Unitree L2 (4D LiDAR) ↔ Intel RealSense D435i (color camera)

Goal: `T_cam_lidar`, the 4×4 transform that maps a LiDAR point into the camera's
**color optical frame** (x right, y down, z forward).

Method: target-based plane matching. A checkerboard is seen by both sensors.
The camera gets the board plane from PnP. The LiDAR gets it from a RANSAC plane fit.
We then solve for the R,t that puts the LiDAR board points onto the camera board
planes, using a closed-form start followed by robust least squares.
This needs ≥3 non-parallel board poses; 12–20 is recommended.

```
L2 --UDP--> udp_relay.py --> lidar_bridge (WSL, Unitree SDK) --TCP :9899--+
D435i --USB3-------------------------------------------------------------+--> calib/capture_gui.py
                                           captures/<time>_<tag>/ image.png, lidar.npz (xyz+intensity), lidar.ply, meta.json
```

## Phase 0: Hardware (do once)
- [ ] Mount both sensors rigidly. Nothing should flex or rotate after calibration.
- [ ] Check that the fields of view overlap in front of the rig.
- [x] Target: A1 board with 6×8 squares of **95 mm**, which gives **7×5 inner corners** (the script defaults).
      Check with a ruler that the printed squares really are 95 mm; if not, pass `--square`.
- [ ] Keep the board flat and rigid, with a white border around the pattern.
- [ ] Put the board on a stand: the camera frame and the LiDAR window must see the same, still scene.

## Phase 1: Data capture (one GUI)
1. Start the native LiDAR bridge: `.\bridge\start_lidar.ps1` (stop: `.\bridge\start_lidar.ps1 -Stop`).
2. Run `python calib/capture_gui.py`. It shows the live camera (with board corners drawn when found)
   and the live LiDAR, coloured by intensity so the checker pattern is visible.
3. Set the LiDAR duration (1–2 s) and a location tag, then press **CAPTURE**. The camera frame
   is taken inside the LiDAR window. The rig must be still.
4. Repeat from as many places and board poses as you need. Each capture gets its own folder under `captures/`.
5. LiDAR board detection: pass `lidar.npz` to `calib_target.detect.extract_target(xyz, intensity, ...)` (Unitree4DLidarL2).

## Phase 2: Camera intrinsics
- [ ] `calibrate.py` runs Zhang calibration on every pose and cam image, starting from the factory values.
      The reprojection RMS should be **< 0.5 px**, and fx/fy/cx/cy should be within a few px of factory.
- [ ] If that fails, use `--intrinsics factory`.

## Phase 3: Solve extrinsics
```
python calib/calibrate.py
```
- It segments the board by subtracting the background and fitting a RANSAC plane,
  then initialises with Kabsch on the plane normals and refines with point-to-plane Huber least squares.
- It writes `data/extrinsics.json`, which contains T_cam_lidar, the K and dist it used, xyz and rpy,
  and a ready `static_transform_publisher` line.

## Phase 4: Validate
- [ ] Per-pose point-to-plane RMS should be **< 2 cm**. The L2 range noise is about 2 cm.
      Delete the files for outlier poses and run the solve again.
- [ ] Look at `data/overlays/pose_XX.png`. LiDAR points (coloured by range) should line up with
      board edges, door frames and table edges.
- [ ] Check the translation against a tape measure (within ~1–2 cm).
- [ ] Repeat the capture on another day. The two results should agree within about 1 cm and 0.3°.

## Phase 5: Use
- Publish the static TF (the command is in `extrinsics.json`).
- Colourise point clouds, or project LiDAR depth into the image for fusion.

## Later (only if needed)
- Time sync or motion compensation for a **moving** rig. The static calibration above does not need it.
  Both sensors have IMUs, which you can use for this.
- Board-edge constraints, if the in-plane translation stays poorly constrained.
- Targetless refinement (ICP between D435i depth and the LiDAR).
