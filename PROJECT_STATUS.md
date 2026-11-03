# SMARTMARK-LCG: Project Status

*As of 3 Nov 2026 (previous version 25 Oct). Covers everything in `UNITREE-INTEL/`: hardware, software, calibrations, experiments, results, decisions and open items.*

**New since 25 Oct:**
- **Wall/whiteboard range test against a laser meter** (28 Oct, 1.06–5.95 m, `Lidar-Test/`). The fixed 19.4 cm correction **does not hold beyond the 1–2 m it was fitted on**: after correction the error drifts 9.3 cm over the range (verdict FAIL), see [§7.7](#77-wall-test-against-a-laser-meter-28-oct-lidar-test). This changes the "±2 cm after correction" claim.
- **Onsite data-collection rig** (`Onsite-Data-Collection/capture.py`, 2 Nov): one-button capture of the D435i (colour, aligned depth, coloured cloud, IMU tilt), the **Insta360 X4** 360° panorama and an L2 recording, into one named folder per capture, see [§12](#12-onsite-data-collection-rig-tripod-payload). Verified on hardware; 4 captures so far.
- **Galvo shape-drawing GUI** (`calib/galvo_test_gui.py`): draws shapes anywhere in the full mirror range, no calibration needed.
- **`lidar_bridge` fix** (in WSL, not in the repo): one client that stops reading no longer freezes the stream for every other client.
- `calibrate.py` `fit_plane` fix (it hung on ~50k-point wall planes); ICP rerun for 203 on 28 Oct (same result as 26 Aug).

---

## Contents
1. [What the project is](#1-what-the-project-is)
2. [Hardware and the rig](#2-hardware-and-the-rig)
3. [Data path and software stack](#3-data-path-and-software-stack)
4. [Repository map (every file)](#4-repository-map-every-file)
5. [Calibration chain: how it works and what we got](#5-calibration-chain-how-it-works-and-what-we-got)
6. [Laser object tracking (end application)](#6-laser-object-tracking-end-application)
7. [L2 LiDAR accuracy investigation](#7-l2-lidar-accuracy-investigation)
8. [Sensor decisions (LiDAR and RealSense)](#8-sensor-decisions-lidar-and-realsense)
9. [Scan ↔ BIM alignment (ICP)](#9-scan--bim-alignment-icp)
10. [LOCALIZATION-360: panorama room layout](#10-localization-360-panorama-room-layout)
11. [YOLO experiments](#11-yolo-experiments)
12. [Onsite data-collection rig (tripod payload)](#12-onsite-data-collection-rig-tripod-payload)
13. [Current state at a glance](#13-current-state-at-a-glance)
14. [Known issues and open items](#14-known-issues-and-open-items)
15. [Command cheat sheet](#15-command-cheat-sheet)
16. [Timeline](#16-timeline)

---

## 1. What the project is

A **Unitree L2 LiDAR**, an **Intel RealSense D435i** camera and a **green galvo laser** are mounted on one rigid rig and calibrated against each other. The calibrated rig then:

- **Fuses** LiDAR points with the camera image (LiDAR points drawn on the image, and a 3D cloud coloured by the camera).
- **Detects objects** with YOLO and **outlines the chosen object with the laser** (one basic shape per object).

A third use of the same sensors is a **tripod payload for construction sites** (`Onsite-Data-Collection/`): RealSense, Insta360 X4 and L2 captured together with one button, raw data only (§12).

Two neighbouring research tracks live in the same repo:
- **Scan ↔ BIM alignment (ICP):** registers LiDAR room scans to clouds generated from the building's IFC model.
- **LOCALIZATION-360:** estimates room layout from a 360° panorama, as a candidate route for registering to the BIM.

Repo name: `SMARTMARK-LCG` (LiDAR–Camera–Galvo). Git: `main` branch, 4 commits (latest `ee13e8c` "Updated fit test", 18 Oct 2026). Everything since 18 Oct is uncommitted (see §14 item 19).

---

## 2. Hardware and the rig

| Device | Details | Connection |
|---|---|---|
| **Unitree L2** 4D LiDAR | IP `192.168.1.62`. Streams UDP to `192.168.1.2:6201` (packets come from port 6101). ~5,380 points per frame, ~12 frames/s (18 frames in a 1.5 s capture), ~64k points/s. 360° coverage. | Ethernet; the Windows adapter is set to `192.168.1.2` |
| **Intel RealSense D435i** | Colour 1280×720 @ 30 fps (BGR8), depth 1280×720 z16 aligned to colour. Factory colour K: fx 909.03, fy 908.82, cx 651.64, cy 377.13, distortion zero. IMU accel used at 100 Hz (firmware 5.17.0.10 offers 100/200/400 Hz only). | USB3 |
| **Insta360 X4** | 360° camera, used in **USB webcam (UVC) mode**: in-camera stitched equirect, 2880×1440 MJPEG. Found by name through ffmpeg's DirectShow device list. The 72 MP stills / Insta360 SDK route is deliberately **not** used. | USB |
| **Galvo laser** | Green laser, two-mirror galvo, mirror values 0–65535 per axis. Serial at 1,000,000 baud. Can only draw **basic shapes** (point lists, no blanking), so **one simple shape per object**. | Serial `COM5` |
| **Checkerboard** | A1 sheet (841×594 mm), 8×6 squares of 95 mm, so **7×5 inner corners**. The pattern spans −1…7 × −1…5 squares in the inner-corner frame. | n/a |
| **PC** | Windows 11 + WSL Ubuntu, Python 3.10, NVIDIA RTX Pro Blackwell GPU (noted in the localization360 notes). | n/a |

**Rig geometry (from calibration, not measured by hand):**
- Camera colour lens ↔ L2 optical centre: **~7.2–7.4 cm** apart.
  - `extrinsics.json`: (−5.3, +4.2, +2.5) cm in the LiDAR frame.
  - `extrinsics_joint.json`: (−4.6, +5.7, −0.3) cm.
- Camera rotated about **−115°** about its viewing axis relative to the LiDAR frame; roll and pitch ≈ 1°.
- Galvo ↔ camera: galvo sits **~14 cm below and 5.5 cm right of** the camera lens (`galvo_in_cam_m` = (0.055, 0.139, −0.005) m, camera frame x right / y down / z forward).
- Not yet confirmed with a tape measure (ROADMAP phase 4 item still open).
- **The sensors will move to a new mount** (stated 2 Nov). Every calibration in §5 belongs to the current rig and will need redoing on the new one if fusion or laser targeting is needed there. The onsite rig deliberately does no extrinsic calibration.

**Laser safety and constraints (standing rule):** basic shapes only (box or circle), one object at a time, green laser on COM5. Nobody in front of the rig during sweeps.

---

## 3. Data path and software stack

```
Unitree L2 ──UDP──> 192.168.1.2:6201 ──> bridge/udp_relay.py (Windows) ──> WSL IP:6201
                                                                    │
                                                    lidar_bridge (WSL Ubuntu, Unitree SDK)
                                                                    │
                                                         TCP 127.0.0.1:9899  ("PCLD" frames)
                                                                    │
RealSense D435i ──USB3───────────────────────────────────────> calib/*.py, Onsite-Data-Collection/capture.py
Insta360 X4   ──USB (webcam mode, DirectShow)─────────────────> Onsite-Data-Collection/capture.py
Galvo laser   <──serial COM5────────────────────────────────── calib/galvo_calib.py, laser_track.py, galvo_test_gui.py
```

- **Why the relay:** WSL2's NAT network doesn't own `192.168.1.2`, so sensor UDP never reaches WSL directly. `udp_relay.py` forwards each packet unchanged to the WSL IP, which it auto-detects with `hostname -I` because the IP changes on restart. Outbound traffic from WSL to the sensor works through normal NAT.
- **`lidar_bridge`** is *not in this repo*. It's a native program at `/root/lidar_bridge/lidar_bridge` in WSL that parses L2 packets with the Unitree SDK. Wire format: header `<4sdI` = magic `"PCLD"`, sensor timestamp (double), point count N, then N × (x, y, z, intensity) float32.
  - **Patched 2 Nov:** it used to `send` to all clients with a blocking call, so one client that stopped reading (a hung GUI) froze the stream for everyone. Now it sends with `MSG_DONTWAIT` and a 4 MB send buffer and drops a client that can't keep up (logs `Client fd=N dropped`). Tested: 11.9 frames/s with a stuck client attached, longest gap 95 ms. Backups: `lidar_bridge.bak`, `lidar_bridge.cpp.bak`. Rebuild: `g++ -O2 -std=c++17 lidar_bridge.cpp -Iinclude -Llib/x86_64 -lunilidar_sdk2 -lpthread -o lidar_bridge`.
- **`start_lidar.ps1`** starts the bridge inside a hidden, persistent `wsl.exe`. A `nohup`ed process got killed when WSL shut the idle distro down, which froze the stream. It then starts the relay and checks both. `-Stop` kills both. Logs: `/root/bridge.log` (WSL), `bridge/relay.log`, `bridge/relay_err.log`.
- **`calib_target`** is *not in this repo*. It's the LiDAR checkerboard detector from `C:\Users\nikhi\Documents\SMART-MARK\Unitree4DLidarL2`, with the path hard-coded in `calibrate.py` (`CALIB_TARGET`). It finds the board plane from saturated (white) returns and applies an **intensity-dependent range-walk correction**: dark returns get flattened onto the white plane. It reports board points, apparent `scale`, sheet dimensions, NCC and a transform.
- **Alternative driver:** `docker/` holds a ROS 2 Foxy Dockerfile (`unilidar_sdk2` + `unitree_lidar_ros2`, RViz2, rosbag2 MCAP). It isn't used by the current pipeline. Note that its README has a typo: `ping 192.169.1.62` should be `.168.`.
- **OpenCV/OpenCL workaround:** the Intel iGPU's OpenCL (shared with the Open3D renderer) crashed with `CL_OUT_OF_HOST_MEMORY`. Every entry point sets `OPENCV_OPENCL_DEVICE=disabled` *before* importing cv2. `capture_gui` and `galvo_calib` are imported first for that reason.
- **Camera driver hang on exit:** a Python process that exits or is killed while the RealSense/X4 are streaming can hang in the driver and keep holding the cameras (zombie process; only a USB replug or reboot frees them). `Onsite-Data-Collection/capture.py` calls `os._exit(0)` after the window closes. Close the GUI with its X button rather than killing it.

---

## 4. Repository map (every file)

### Root
| File | What it is |
|---|---|
| `README.md` | Setup and workflow (bridge → capture → calibrate → galvo → laser track), ICP, localization360, selftests, repo layout |
| `ROADMAP.md` | LiDAR–camera calibration method and validation checklist (phases 0–5). Some paths are outdated (`data/extrinsics.json` is now `captures/calibration/`). |
| `architecture.html` / `architecture.json` | Architecture diagram (L2 → relay → bridge → calibrate → calibration files; D435i → capture_gui; galvo_calib / laser / laser_track / fusion_gui) |
| `.gitignore` | Ignores caches, `*.pt`, raw captures (except `captures/calibration/` minus overlays, and `galvo_calib.json`), logs, `icp/data/`, `icp/results/`, vendored localization360 repos and runs |
| `yolo11n.pt`, `yolov8s-worldv2.pt` | Weights used by `laser_track.py` (git-ignored) |
| `PROJECT_STATUS.md` | This file (untracked) |
| `.vscode/settings.json` | Adds `./calib` to Pylance's import path (untracked) |

### `bridge/`
| File | Purpose |
|---|---|
| `start_lidar.ps1` | Start/stop `lidar_bridge` (WSL) and `udp_relay.py` |
| `udp_relay.py` | UDP `192.168.1.2:6201` → WSL IP `:6201` |
| `relay.log`, `relay_err.log` | Relay logs (last run shows packets from `192.168.1.62:6101`) |

### `calib/` (the core)
| Script | Purpose | Main output |
|---|---|---|
| `capture_gui.py` | Open3D GUI: live camera (board corners drawn) and live L2 cloud (intensity). **CAPTURE** grabs 1–2 s of LiDAR plus one colour frame taken inside that window. Its `Rig` class is reused by `fusion_gui`. | `captures/<YYYYmmdd-HHMMSS>_<tag>/` with `image.png`, `lidar.npz` (xyz, intensity, frame_sizes, stamps, host times), `lidar.ply`, `meta.json` |
| `calibrate.py` | Camera intrinsics (Zhang, seeded with factory K) plus LiDAR→camera extrinsics (plane matching), plus an L2 range model fitted from the apparent board size. Writes overlays. | `captures/calibration/extrinsics.json`, `intrinsics_calibrated.json`, `overlays/*.png`; caches `lidar_board.npz` per capture |
| `calibrate_joint.py` | **New (test).** Fits T_cam_lidar and the range error **together** on board interiors (edges eroded by 10 cm), after dropping detections where the board merged with its surroundings. Offset and linear models, with leave-one-out and split-half reproducibility. | `captures/calibration/extrinsics_joint.json` |
| `fusion_gui.py` | Live view: LiDAR points projected on the camera image (colour = distance) plus a 3D cloud coloured by the camera. `--calib` chooses the extrinsics file. | n/a |
| `board_size.py` | Histograms of the LiDAR-measured square size and sheet size vs the real print, raw and range-corrected | `captures/calibration/board_size.png` (currently deleted, regenerate) |
| `galvo_calib.py` | Galvo ↔ camera: `capture` (sweeps a laser dot grid per board pose), `fit`, `aim` (click to fire), `selftest` | `captures/galvo/view_NN/{off.png, dots.json}`, `captures/galvo/galvo_calib.json` |
| `galvo_verify.py` | Aims at all 35 inner corners per board pose and measures the miss in mm on the board plane (bias and spread) | `captures/galvo/verify/<run>/…`, `verify/summary.csv` (**never run yet**) |
| `laser_track.py` | End application: YOLO detect/track → D435i depth → 3D box/circle → galvo mirror values → laser | n/a |
| `verify_lidar.py` | Lets you check the L2 claims yourself (see [§7.6](#76-how-to-verify-everything-yourself-verify_lidarpy)): `boards`, `points`, `scans`, `wall`, `wall-fit`, `selftest`. Its `wall_distance` is reused by `Lidar-Test/wall_capture.py`. | `captures/verify/points.csv`, `scans_<room>.png`, `wall.csv` |
| `galvo_test_gui.py` | **New.** Tk GUI for drawing with the galvo in raw mirror values, **no camera, no calibration**. The canvas is the full 0–65535 range. Shapes: rectangle, circle/ellipse, triangle, star, line, polygon (click corners) and dot. Drag to draw, sent on release. Speed, dwell, circle points, flip X/Y, full-range box, Stop/Esc. `--dry-run` prints the serial commands instead. It replaced the earlier `draw_rect.py`; its docstring still says `draw_rect.py`. | n/a |

`calibrate.py` has one uncommitted change: `fit_plane` uses `svd(..., full_matrices=False)`, because the full U matrix (N×N) hung on a wall's ~50k points. `calibrate_joint.py` is committed; the working copy differs only in line endings (CRLF). `verify_lidar.py` and `galvo_test_gui.py` are **untracked (not committed yet)**.

### `Lidar-Test/` (new, untracked): L2 range against a laser meter
| File | Purpose | Output |
|---|---|---|
| `wall_capture.py` | Live preview, capture and type the laser-meter distance. LiDAR distance = perpendicular distance to the plane in front of the camera, raw and range-corrected (`--calib`, default `extrinsics_joint.json`). RealSense distance from a plane fit on the central depth patch. `--gate` keeps only points near the typed distance, for a board in front of a wall. `--fit` prints the verdict. | `Lidar-Test/<time>_<meter>m/{lidar.npz, lidar.ply}`, a row in `readings.csv` |
| `board_size.py` | Board width × height as the L2 sees it, from the saved clouds (RANSAC plane, the connected patch the camera axis hits, minimum-area rectangle; full and 1–99 % trimmed). `--real W H` prints the excess. | console |
| `readings.csv` | 18 readings, 28 Oct (whiteboard, 6 distances × 3) | n/a |
| `20261028-*_<m>m/` | 18 raw capture folders | n/a |

### `Onsite-Data-Collection/` (new, untracked): tripod payload capture
| File | Purpose |
|---|---|
| `capture.py` | **One self-contained script** (the Insta360 code is inline; the former `insta360.py` is merged in). Open3D GUI with live previews of all sensors and one CAPTURE button; reuses `calib/capture_gui.py`'s `Rig` and LiDAR loop. Details in §12. |
| `captures/<name>_<YYYYmmdd-HHMMSS>/` | One folder per capture (4 so far, 2 Nov) |
| `captures/index.csv` | One row per capture: folder, name, timestamp, seconds, LiDAR points, tilt, missing sensors, notes |

The hardware-free test `test_capture.py` written with it no longer exists in the folder.

### `captures/`
| Path | Contents |
|---|---|
| `captures/20261014-*_capture/` (54 folders) | The 14 Oct calibration session, 1.5 s LiDAR each. 58 originally; **4 suspect ones deleted** (`123110`, `123334`, `123924`, `135515`, where the LiDAR board merged with its surroundings). 29 have a usable LiDAR board; 25 are camera-only close-ups for intrinsics (L2 found no board). |
| `captures/calibration/` | `extrinsics.json`, `extrinsics_joint.json`, `intrinsics_calibrated.json`, `overlays/` (33 PNGs, **stale**: includes the 4 deleted captures) |
| `captures/galvo/` | `view_00` … `view_25` (26 views) and `galvo_calib.json` |
| `captures/verify/` | `points.csv` (per-board accuracy/precision), `scans_203.png`, `scans_207.png`, `scans_214.png` |

### `icp/` (scan ↔ BIM)
| File | Purpose |
|---|---|
| `generate.py` | IFC → per-room BIM reference cloud: wall + floor lattice (5 cm), door openings carved, ceiling dropped |
| `align.py` | Scan → BIM: level, outlier removal, wall band 0.6–1.9 m, footprint check, coarse (yaw sweep), ICP distance ladder, scores, PNGs |
| `webui.py` | Flask dashboard at `127.0.0.1:5000`: run both stages, browse results, 3D overlay |
| `config.yaml` | All settings |
| `data/ifc/H_DA_D21_Final.ifc` | Building model (18 MB) |
| `data/bim/{203,207,214}.ply/.json` | Generated BIM clouds. Room 203 = "HMI-Labor", floor z 7.58, clear height 3.04 m, 6.11 × 9.22 m. |
| `data/scans/{203,207,214}.ply` + `*_2seconds.ply` | L2 room scans: one short and one 2 s scan per room (203 "awa", 207 "kitchen", 214 "lab"). Already levelled (floor at z = 0), sensor origin not recorded. |
| `data/misc/` | Older/cleaned 203 scans (`203_cleaned`, `203_scan_cleaned`, `203room`, `D21_203_pcl(_old)`) |
| `results/` | 7 runs (26 Aug, two rounds × 3 rooms; 28 Oct, 203 yaw only), plus `203_wall_offset_investigation.png`, `203/207_scan_overview.png`, `footprint_debug_203.png`, latest-run files |

### `localization360/`
`app.py` (Flask at `127.0.0.1:5001`) wraps vendored **HorizonNet** and **LGT-Net**, each with a `NOTES.md` and small compatibility patches. `runs/<id>/` holds web-UI runs. Weights: `HorizonNet/ckpt/resnet50_rnn__panos2d3d.pth`, `LGT-Net/checkpoints/SWG_Transformer_LGT_Net/mp3d/best.pkl`. All git-ignored.

### `YOLO/`
`detections.ipynb` (YOLO26 tests on `room.jpg`), weights `yolo26s.pt`, `yolo26x.pt`, `yolo26x-seg.pt`.

### `docker/`
`Dockerfile` and `README.txt`: ROS 2 Foxy L2 driver with RViz (alternative, unused).

---

## 5. Calibration chain: how it works and what we got

Order matters: **camera intrinsics → LiDAR↔camera → galvo↔camera**. The galvo calibration uses `intrinsics_calibrated.json`. If the camera moves, all three must be redone; if only the LiDAR or galvo moves, redo that step.

### 5.1 Capture session (14 Oct 2026)
- Tool: `capture_gui.py`, 1.5 s per capture. Two batches: 12:30–12:52 and 13:54–13:58.
- **Extrinsic captures:** board on a stand, 1–2 m away, tilted in yaw and pitch, with free space behind it.
- **Intrinsic-only captures:** board close (0.6–1 m), filling the frame and pushed into image corners and edges. The L2 doesn't need to see the board for these.
- Board-capture guidance used: hold still, even light, no glare, full board in frame, both tilt axes ±30–45°, varied distances and image positions.

### 5.2 Camera intrinsics (`calibrate.py`)
- **Method:** OpenCV `calibrateCamera` (Zhang) on every image where the 7×5 corners are found. Detection uses `findChessboardCornersSB`, falling back to classic detection plus `cornerSubPix`; SB misses about half of these images on its own. Seeded with the factory K (`CALIB_USE_INTRINSIC_GUESS`).
- **Result** (`intrinsics_calibrated.json`): **58 images, reprojection 0.16 px**, which is excellent.
  - fx 897.12, fy 894.91, cx 651.85, cy 381.07 (factory: 909.03 / 908.82 / 651.64 / 377.13)
  - dist k1 0.0374, k2 0.3507, p1 0.0017, p2 0.0008, **k3 −1.363**
- **Known weakness:** the large k3 makes the radial model **fold beyond ~600 px from the centre** (undistorted r ≈ 0.742). The leftmost ~50 px, rightmost ~30 px and all four corners map to wrong rays, and LiDAR points just outside the view can show up as ghosts in those strips. It doesn't affect the galvo, whose whole reach is inside the valid region. **Fix proposed, not applied:** add `cv2.CALIB_FIX_K3` in `calibrate_intrinsics`.
- `intrinsics_calibrated.json` still carries `"source": "realsense factory"` copied from meta. The label is wrong but harmless.

### 5.3 LiDAR ↔ camera extrinsics

#### a) `calibrate.py` (main method, writes `extrinsics.json`)
1. **Camera:** PnP on the board corners gives the board plane (n, d) in the camera frame.
2. **LiDAR:** `calib_target.extract_target` gives the board points, cached in `lidar_board.npz`. It tries a coarse configuration for sparse or far scans first, then the default one.
3. **Range model:** the L2 sees the board `scale` times too large. Fitting `scale = κ + δ·scale/r` gives `r_meas = κ·r_true + δ`, and `correct_range` moves each point along its ray to its true range.
4. **Solve:** closed-form start (Kabsch on plane normals, linear t), then robust point-to-plane least squares (Huber, 2 cm), ≤ 1000 points per pose.
5. **Checks:** tilt spread = smallest/largest singular value of the stacked normals (need > 0.15, good > 0.3); per-pose plane RMS with outlier flag; overlays.
6. **Outputs:** `T_cam_lidar`, `T_lidar_cam`, K, dist, range model, camera-in-LiDAR xyz/rpy, a ROS 2 `static_transform_publisher` command.

**Run history:**
| Run | Boards | Tilt spread | Range model | Camera in LiDAR | Notes |
|---|---|---|---|---|---|
| 1st (12:xx captures) | 22 | 0.25 (borderline) | κ 1.0413, δ 10.8 cm (1.5 m reads 1.670) | (−6.7, 4.5, 3.0) cm | Plane RMS 1.41 cm without the range model, 1.16 cm with it; **translation moved 17 cm** depending on the range model |
| 2nd (current `extrinsics.json`) | 33 | **0.338** | **κ 1.0291, δ 12.45 cm** (1.5 m reads 1.668) | **(−5.29, 4.15, 2.47) cm = 7.2 cm**, rpy (1.18°, −0.46°, −114.53°) | Plane RMS 0.9–2.1 cm, no outliers flagged. Stable vs run 1 (Δt ≈ 1.5 cm, Δrot < 1°). |

⚠️ `extrinsics.json` was **not re-run after the 4 suspect captures were deleted**, so it still includes them.

#### b) `calibrate_joint.py` (new, writes `extrinsics_joint.json`)
**Why:** the board-size range model mixes two effects. The L2's beam footprint makes every board edge about 10 cm too wide ("edge bloom"), and the board-size fit attributes that to range error.

**Method** (Mirzaei et al. 2012, planar-target LiDAR intrinsic + extrinsic calibration):
1. Drop detections whose sheet short side is > 0.95 m (board merged with its stand or background).
2. Erode each board 10 cm from its outline, keeping interior points only.
3. Least squares over (R, t, δ[, κ]) so the range-corrected LiDAR points lie on the camera planes. A range offset moves points along rays that fan out, so it can be separated from t.

**Result** (29 boards, LiDAR distance 1.12–2.08 m, tilt spread 0.32):
| Model | Fit | Leave-one-out | First half → second half | Plane RMS |
|---|---|---|---|---|
| **offset** (written) | **δ = 19.4 cm**, κ = 1 | δ std 0.95 cm | δ 19.3 vs 20.0 cm (**reproducible**) | 1.33 cm (held-out 1.68) |
| linear | κ 1.0336, δ 19.2 cm | κ std 0.0015 | κ **1.023 vs 1.061** (**not reproducible**); 4 m reads 4.26 vs 4.43 | 1.15 cm (held-out 1.48) |
| current `extrinsics.json` | κ 1.0291, δ 12.5 cm | n/a | n/a | 1.19 cm on the same interior points |

- Camera in LiDAR (offset model): (−4.6, 5.7, −0.3) cm = **7.4 cm**, rpy (1.25°, 0.21°, −115.13°).
- Takes about 10 minutes to run because of the ~60 refits.

### 5.4 Galvo ↔ camera (`galvo_calib.py`)
**Model:** the galvo is treated as an **inverse camera**. Its mirror values (0–65535) play the role of pixels, so OpenCV camera calibration gives its intrinsics, distortion (the keystone/pincushion of the two-mirror optics) and its pose relative to the camera.

**Capture, per board pose (SPACE):**
1. Take a laser-off frame; the board pose comes from PnP.
2. Sweep a coarse 5×5 dot grid over mirror range 8000–57535.
3. Use a rough mirror→pixel homography to aim a dense 7×7 grid at the board (85% of the board).
4. Find each dot as the green-channel difference against the laser-off frame (a saturated dot turns white, so plain green-excess doesn't work). Sub-pixel centroid; patches > 800 px rejected as exposure changes.
5. Turn each dot into 3D by intersecting its camera ray with the board plane. Keep only dots on the printed sheet.

**Fit:**
1. Drop false dots per view with a RANSAC homography (8 px).
2. Zhang initialisation per view, then **one joint calibration over all dots in the camera frame**, where the single pose *is* camera→galvo.
3. Second pass without dots whose error is > max(4×median, 100 units).
4. Reports error in mm (small-angle conversion) and the covered range of camera-ray directions.

**Fix committed 18 Oct:** scale `S` changed from 1024/65536 to **1.0** (K in mirror units), image size set to 65536² so OpenCV's start guess sits among the dots. `load_calib` rescales older files.

**Result** (`galvo_calib.json`):
| Metric | Value |
|---|---|
| Views | **26**, board at **1.46–2.54 m**, tilts 3–53° |
| Dots | 1304 used, 68 dropped (5%) |
| Reprojection | 121 mirror units RMS |
| **On-target error** | **median 1.92 mm, p95 5.42 mm** (in-sample) |
| Galvo K | fx 81,229, fy 148,987 mirror units/rad (axes have different gain: ~46° × 25° field) |
| Coverage (camera-ray x/z, y/z) | x −0.44 … 0.28, y −0.28 … 0.11, which is basically the laser's full physical reach (lower and right parts of the image are out of reach) |
| Galvo position in camera frame | (5.5, 13.9, −0.5) cm |

**Validation status:** `aim` mode exists (click → depth → fire → pixel error). **`galvo_verify.py` has not been run yet** (no `captures/galvo/verify/`), so the 1.9 mm figure is in-sample only. Depth range is narrow (1.5–2.5 m); 1 m and 3.5 m+ are extrapolated.

---

## 6. Laser object tracking (end application)

`calib/laser_track.py`, a Tk GUI. Per camera frame:
1. **Detect + track:** COCO class names use **YOLO11n** (`model.track`, persistent IDs). Any other name (door, whiteboard, …) switches to open-vocabulary **YOLO-World v2-s** (`set_classes`). Defaults: class `chair`, ignore `person`, confidence 0.35.
2. **Pick one target** (the laser can only do one): `lock` keeps the current track ID; `cycle` hops left to right every N s; otherwise the highest-confidence detection.
3. **Depth:** median D435i depth over the central 50% of the box (aligned depth).
4. **Shape:** box or circle around the 2D box, placed in 3D on a camera-facing plane at that depth, plus a 5 cm margin (circle = 24 points).
5. **Mirror values** via `to_dac` (galvo calibration), exponential smoothing (0.5). Re-sent only if moved > 250 mirror units (avoids flicker).
6. **Safety and blocking:** no fire outside the calibrated coverage (unless "Draw outside" is ticked) or outside mirror range 0–65535. The laser stops if the target is lost > 0.7 s, if blocked, or if toggled off. SPACE toggles the laser.
7. **Falls back to no laser** if COM5 can't be opened. `--dry-run` does everything except firing.

**Uses D435i depth, not the LiDAR.** Depth error barely moves the hit point: about `0.15 m × depth error / distance`, so about 3 mm at 2 m and about 5 mm at 4 m. YOLO box jitter matters more.

---

## 7. L2 LiDAR accuracy investigation

### 7.1 Why
We needed to decide whether the L2 is accurate enough or a new LiDAR must be bought. The 1st calibration showed the range model shifting the translation by 17 cm, so the size and repeatability of the L2's range error had to be established.

### 7.2 Findings, chronologically
1. **ICP room scans were larger than the BIM** in every room (footprint deltas 0.4–0.8 m), consistent with a range bias. This was first evidence, later confirmed properly in 7.5.
2. **All board detections** (`repro.py` in scratch, now `verify_lidar.py boards`):
   - The L2 reads long on 28/29 boards (direct comparison: LiDAR board-centre range vs camera PnP range, no board-size model involved). Mean +16.1 cm, single-board scatter 8.3 cm.
   - **Depends on tilt:** boards within 0–40° read +18–19 cm; boards > 40° only +8.6 cm with more scatter. This is why the two halves differ (11.6 vs 20.3 cm): the first half had steeper boards.
   - **Edge bloom:** the 841×594 mm sheet reads ≈ 1026×760 mm (+185/+166 mm, about 9 cm per edge). It's an additive smear, not a scale error.
   - 4 detections had a sheet of ~1150×1030 mm (merged with surroundings) and were deleted.
3. **Joint calibration** (§5.3b): the offset of **~19.4 cm is reproducible to about ±1 cm** within 1–2 m. The slope (κ) is **not determinable** from 1.1–2.1 m data.
4. **Raw-point accuracy and precision** (§7.3) using the camera as reference, out-of-sample.
5. **Room scans vs BIM** (§7.5): the error **does not grow with distance** out to room size (5–10 m).

### 7.3 Accuracy and precision of the raw L2 points on the board (`verify_lidar.py points`)
Raw points from `lidar.npz` (not the detector's walk-flattened ones), board interior only, 29 boards at 0.6–1.8 m. For each board, T and the range offset were fitted on **all other boards** (leave-one-out).

**Precision (random):**
| | 1σ |
|---|---|
| Single point, same-colour surface | **~1 cm** (white 10.2 mm, black 9.0 mm) |
| Single point, mixed black/white board | 2.9 cm (inflated by range walk) |
| Board distance from one frame (~5k pts) | **~5 mm** frame to frame |
| Face-on (<25°) vs steep (>40°) | 0.4–0.9 cm vs 1.2–2.5 cm per point |

**Accuracy (systematic):**
| | Value |
|---|---|
| Raw range | **+17.8 cm too long** (per board +15.1 … +23.5 cm) |
| After offset correction (out-of-sample) | **−1.5 cm mean, 1.6 cm std, worst 3.6 cm** |
| White squares after correction | **≈ 0 ± 2 cm** (at 0.6–1.8 m only; see §7.7 for longer range) |
| **Reflectivity (range walk)** | **black squares read 7.3 cm closer than white**, on every board (5.3–9.8 cm) |
| Incidence 0–25° / 25–40° / >40° | −0.9 / −1.8 / −2.3 cm corrected |
| Distance < 1.5 m / > 1.5 m | −2.2 / +1.1 cm corrected (hint of a small distance effect) |
| **Session drift** | the 13:5x session reads ~2 cm shorter than the 12:xx session at the same distances. **Possibly warm-up:** let the L2 run 15–20 min first. |

### 7.4 Error models compared
| Model | Source | Status |
|---|---|---|
| `r_meas = r_true + 19.4 cm` (offset) | `calibrate_joint.py` | Best supported at 1–2 m: reproducible on boards, consistent with room widths. **Fails the 28 Oct whiteboard test over 1–6 m** (§7.7). |
| `r_meas = 1.0336 r + 19.2 cm` (linear) | `calibrate_joint.py --model linear` | Rejected: halves disagree, room scans contradict it |
| `r_meas = 1.0291 r + 12.5 cm` (board size) | `calibrate.py` | Biased by edge bloom; overshoots on rooms |

### 7.5 Room scans vs BIM (`verify_lidar.py scans`)
Each scan was placed on its BIM room with the existing ICP transform. For every scan wall point, the script measures the offset outside the nearest BIM wall, then adds opposite walls together so the ICP translation cancels. That gives the room-width excess per axis: `≈ 2δ + (κ−1)·width`, independent of where the sensor stood.

| Room | Short axis reads wider | Long axis reads wider | 2nd scan of the same room |
|---|---|---|---|
| 203 (6.11 × 9.22 m) | 34.5 cm | 28.6 cm | 34.3 / 28.1 cm |
| 207 (4.84 × 8.23 m) | 31.1 cm | 27.8 cm | 32.2 / 28.8 cm |
| 214 (7.21 × 9.98 m) | 47.3 cm ⚠ | 30.5 cm | 51.4 / 32.7 cm |

- Every room reads wider, typically **28–34 cm, about 14–17 cm per wall**. Repeat scans agree to 1 cm (203, 207) and 2–4 cm (214).
- **No growth with room size** (4.8 → 10 m), so κ ≈ 1 and the linear model is ruled out. Per-wall bias is a bit smaller than on boards because walls are mostly hit obliquely (consistent with the tilt and incidence finding).
- Wall "thickness" (noise plus real unevenness) is about **2.3–2.6 cm** at room distances.
- Caveats: the 214 −x wall has two surfaces (~5 cm and ~47 cm; furniture or an as-built difference). Peak picking uses "tallest peak, unless an outer peak is ≥ 80% as tall". The BIM is the reference, so as-built deviations count as error. Plots: `captures/verify/scans_<room>.png`.

### 7.6 How to verify everything yourself (`verify_lidar.py`)
| Claim | Command | Look at |
|---|---|---|
| L2 reads long; tilt dependence; edge bloom | `python calib/verify_lidar.py boards` | Per-capture table, CLAIM lines |
| Precision, accuracy, reflectivity, session drift | `python calib/verify_lidar.py points` (~5 min) | Summary lines, `captures/verify/points.csv` |
| Bias at room distances, no growth with distance | `python calib/verify_lidar.py scans` | Table plus `captures/verify/scans_<room>.png` |
| Offset reproducible, slope not | `python calib/calibrate_joint.py` (~10 min) | Leave-one-out and split-half lines |
| Fusion consistency | `calibrate.py` output, `fusion_gui.py` | Plane RMS; points on image edges |
| Far range with a reference (needs hardware) | `verify_lidar.py wall` then `wall-fit` | Slope; corrected error drift < 3 cm = PASS |
| Laser accuracy (needs hardware) | `python calib/galvo_verify.py --label <spot>` | Median/p95 mm, bias vs spread |
| Range vs a laser meter, 1–6 m (done 28 Oct) | `python Lidar-Test/wall_capture.py --fit` | Slope, corrected − meter per spot, drift verdict |

### 7.7 Wall test against a laser meter (28 Oct, `Lidar-Test/`)
**Setup:** a whiteboard (`--gate 0.4`, label `whiteboard`) at 6 distances measured with a laser meter from a fixed mark on the rig, 3 captures each, 1.06–5.95 m. The LiDAR distance is the perpendicular distance to the fitted board plane, so it doesn't depend on how square the rig stands. Correction: `extrinsics_joint.json` (offset model, −19.4 cm).

**Result** (`wall_capture.py --fit`):
| Meter (m) | Raw L2 − meter | Corrected − meter | Repeat std | Points per capture |
|---|---|---|---|---|
| 1.06 | +5.4 cm | −13.8 cm | 0.1 cm | ~55k |
| 1.83 | +7.7 cm | −11.5 cm | 0.2 cm | ~55k |
| 2.90 | +9.9 cm | −9.4 cm | 0.2 cm | ~52k |
| 3.89 | +8.1 cm | −11.2 cm | 0.3 cm | ~37k |
| 4.98 | +4.7 cm | −14.6 cm | 0.1 cm | ~29k |
| 5.95 | +0.7 cm | −18.7 cm | 0.0 cm | ~24k |

- **Linear fit:** raw L2 = **0.9897 × meter + 9.6 cm**. The offset includes the unknown distance from the meter mark to the L2's optical centre, so **only the shape of the error is meaningful, not its absolute level**.
- **Verdict: FAIL.** After correction the error still drifts **9.3 cm** over 1–6 m (pass limit 3 cm). It isn't a straight line either: it rises by ~4.5 cm from 1 to 3 m, then falls ~9 cm by 6 m.
- **Very repeatable:** the 3 captures per spot agree to ≤ 0.3 cm; plane scatter is 0.5–0.7 cm. So this is a real systematic effect, not noise.
- **Conflicts with the room scans (§7.5),** where the per-wall bias looked constant out to 5–10 m. Likely difference: the whiteboard is a bright, partly glossy, face-on target, while room walls are matte and mostly hit obliquely. **Hypothesis, not tested:** an intensity-dependent range walk (the same mechanism that makes black squares read 7.3 cm closer than white), with return intensity falling with distance. Checking the board-point intensity per distance in the saved `lidar.npz` files would test it.
- **What it means:** the 19.4 cm offset is valid on the checkerboard at 1–2 m. It should not be quoted as "±2 cm after correction" at longer range or on other surfaces until a matte-wall run is done.
- **RealSense on the same board:** RealSense = 0.72 × meter + 47.9 cm, residual std 89 cm. Fine up to ~3 m (0.95 / 1.76 / 2.86 m at 1.06 / 1.84 / 2.91 m), useless beyond (plane tilts of 40–65° at 4–6 m show the fit is landing on noise).
- **Board size** (`board_size.py`): trimmed rectangle ≈ **1.48 × 1.25 m at 4–6 m, 1.53 × 1.28 m at 3 m, 1.37–1.55 × 1.26–1.31 m at 1–2 m**. Full rectangles are larger and noisy (stand or wall merging). The real board size wasn't recorded, so the excess can't be computed yet (`--real W H`).
- Not yet tested: the L2 warm-up before this session isn't recorded; no matte-wall run; 8 m not reached.

---

## 8. Sensor decisions (LiDAR and RealSense)

### Keep the Unitree L2 (no new LiDAR)
- **Noise ~1 cm**, the same class as Livox Mid-360, Ouster and Hesai (1–3 cm). A replacement costing 2–5× more would be a sideways move.
- **Main error is a repeatable offset** (~19 cm at 1–2 m, ~15 cm per wall in rooms), corrected in software to about ±2 cm on bright surfaces.
- **Laser targeting doesn't use the LiDAR.**
- A real accuracy step up would be a **survey-grade static scanner** (Leica BLK360/RTC360, Faro Focus): a different category, not a live sensor.
- Switching costs weeks: bridge, detector, range model, recalibration.
- **Caveat added 3 Nov:** the 28 Oct whiteboard test (§7.7) shows the single-offset correction drifting 9 cm over 1–6 m. The decision to keep the L2 still stands (noise ~0.5–1 cm and excellent repeatability), but the correction model needs more work beyond 2 m.
- **Limits to state honestly:** dark surfaces read ~7 cm short; steep incidence reduces the bias and adds scatter; edges bloom ~9 cm. Board data covers ≤ 2 m; rooms show the error stays constant to ~5–10 m.
- **If the project needs ≤ 5 mm** (construction layout marking from BIM), no robotics LiDAR does that.

### Keep the D435i (no new RealSense)
- Colour calibration is 0.16 px reprojection.
- Depth error barely affects laser aim (~3 mm at 2 m, ~5 mm at 4 m).
- Only consider a D455 (wider stereo baseline) if targets are regularly at 4 m+ *and* LiDAR depth can't be used. A new camera means redoing all three calibrations.
- **Free upgrade (not implemented):** for far targets, use the range-corrected L2 depth via the existing extrinsics instead of D435i depth.

### Email text (updated with all tests so far)
> Based on the calibration testing so far, we recommend keeping the Unitree L2 rather than buying a new LiDAR. Across 29 checkerboard captures at 1–2 m, the L2 reads distances about 18–19 cm too long, but this error is repeatable: the fitted offset varies by only about ±1 cm between test runs and data subsets, so we correct it in software. After correction its points are accurate to about ±2 cm (worst 3.6 cm) on bright surfaces, with about 1 cm random noise per point. Dark surfaces read about 7 cm short, a reflectivity effect common to this class of sensor. Checked against the BIM on six room scans (rooms 4.8–10 m across), the error stays a constant ~15 cm per wall and does not grow with distance, and repeat scans of the same room agree to 1–4 cm. Other robotic LiDARs we would consider (Livox, Ouster, Hesai) are in the same 1–3 cm class, so switching would cost 2–5× more without a meaningful accuracy gain, and would mean rebuilding the driver, the board detector and every calibration. A real step up in accuracy would require a survey-grade static scanner, which is a different category of tool. The LiDAR–camera fusion is also consistent: across independent calibration runs, the camera position relative to the LiDAR stayed at about 7.2–7.4 cm and the rotation within about 1°, with the fused points matching the camera's board planes to about 1.2 cm. The laser targeting does not depend on the LiDAR at all; it uses the camera and galvo calibration, which reaches about 2 mm median error in calibration.

The first draft (written 18 Oct, before the point and scan tests) said "repeatable to ±1 cm" and "beyond 2 m not yet verified". The version above replaces both.

⚠️ **Before sending:** the sentence "after correction its points are accurate to about ±2 cm" holds only for the checkerboard at 1–2 m. The 28 Oct whiteboard test (§7.7) shows the corrected error drifting ~9 cm over 1–6 m. Either qualify the sentence with "at 1–2 m" or rerun on a matte wall first.

---

## 9. Scan ↔ BIM alignment (ICP)

**Goal:** register an L2 room scan to the BIM so the rig is localised in building coordinates.

**Stage 1, `generate.py`:**
1. Reads Revit IfcSpace solids (inner wall faces = what a LiDAR in the room sees).
2. Classifies triangles by normal (wall/floor/ceiling; ceiling dropped).
3. Samples a world-anchored lattice (5 cm xy and z).
4. Clips walls at the detected ceiling (lowest IfcCovering/IfcSlab).
5. Carves door openings (IfcDoor bbox + 6 cm margin); windows are left solid on purpose.
6. Stays in IFC world coordinates and merges Revit phase duplicates.

**Stage 2, `align.py`:**
1. Levels the scan (seeded RANSAC; most horizontal plane → +Z, floor → z = 0).
2. Outlier removal (20 neighbours, 1.5 σ).
3. Keeps the wall band 0.6–1.9 m.
4. Voxel 3 cm + normals.
5. **Footprint check** (scan vs BIM rectangle; > 100 mm difference floors the achievable residual).
6. Coarse alignment: **yaw sweep** (2°, 0.30 m) is the reliable method. FPFH is weak on lattice clouds; PCA fails on near-square rooms; manual picking is the fallback.
7. ICP ladder 0.30 → 0.15 → 0.08 → 0.05 m (point-to-point; point-to-plane is rank-deficient on a wall-only slice).
8. Scores: fitness, inlier RMSE, median/p90 over all points ("the honest number").

**Results (26 Aug):**
| Run | Room | Footprint delta (mm) | Fitness | Inlier RMSE | Median | p90 | Yaw |
|---|---|---|---|---|---|---|---|
| 15:29 (no outlier removal) | 203 | 446 / 1552 | 0.324 | 27.8 | 197 | 388 | 74.1° |
| | 207 | 2188 / 677 | 0.324 | 29.1 | 122 | 358 | −109.9° |
| | 214 | 2093 / 1022 | 0.429 | 29.8 | 89 | 606 | −63.3° |
| **16:32 (with outlier removal)** | 203 | 427 / 1432 | 0.326 | 26.9 | 195 | 384 | 74.2° |
| | 207 | 672 / 529 | 0.333 | 29.0 | 113 | 340 | −110.0° |
| | 214 | 773 / 642 | 0.436 | 29.6 | 80 | 495 | −63.2° |

- **Rerun 28 Oct** (`20261028-113841_203_yaw`, the current `latest.txt`): room 203 with yaw only, giving footprint delta 424 / 1430 mm, fitness 0.323, inlier RMSE 27.6 mm, median 205 mm, p90 474 mm, yaw 74.1°. Essentially the same as 26 Aug; nothing in the method changed.
- Outlier removal fixed the footprint inflation from stray multipath returns (207: 2.19 → 0.67 m).
- Room 203 investigation (`203_wall_offset_investigation.png`): the scan sees through the door and windows; there is a **~550 mm inward offset east of the door** (likely an unmodelled counter/cabinet run with no IfcFurniture there) and two unexplained wall features (nothing in the IFC). The room has a real non-Manhattan **recess**.
- **What limits ICP now:** the uncorrected L2 range bias (rooms ~30 cm too wide), see-through at doors and windows, and unmodelled furniture. Sensor noise isn't the limit. The range correction **can't be applied to these scans** because they are levelled and the sensor origin isn't stored. Future scans should keep the raw sensor frame, then be corrected, then aligned.

---

## 10. LOCALIZATION-360: panorama room layout

**Idea:** estimate room shape from one 360° panorama, then register that shape to the BIM floor plan. Scale would come from the BIM floor-to-ceiling height, followed by 2D polygon matching.

| Model | Tried on | Result |
|---|---|---|
| **HorizonNet** (CVPR 2019, ResNet-50 + BiRNN, `resnet50_rnn__panos2d3d.pth`) | Room 203 Insta360 panorama (5888×2944) on 25 Aug; demo; ZInD sample panoramas; web-UI run `203_1.jpg` | Runs cleanly. Boundary tracks ceiling and floor well, but predicts a **plain 4-wall box** and misses 203's recess. |
| **LGT-Net** (CVPR 2022, SWG-Transformer, MP3D weights, Manhattan post-processing) | 203, demo, lab panoramas (`lab1-2`, `lab2-1`, `lab2-2`, `lab_1-1`) | Runs cleanly and also draws a top-down floor plan. Also a **plain rectangle** for 203. |

- Web UI: `python localization360/app.py` → `127.0.0.1:5001`. Choose a model, upload an equirectangular image or use the demo; output goes to `runs/<id>/`.
- Patches for current numpy/OpenCV/PyTorch: `torch.load weights_only`, `np.float` aliases, `findContours` tuple, JSON floats. LGT-Net is forced to the `Agg` backend (its `plt.show()` blocked headless runs).
- Datasets (ZInD, Structured3D) are access-gated; bundled weights and demos avoided that.
- **Status:** works technically. Both models agree on a too-simple footprint for 203. Not yet usable for BIM registration without recess handling (other post-processing options, `atalanta`/`original`, are untried). Scale resolution and 2D registration are **not implemented**.

---

## 11. YOLO experiments

`YOLO/detections.ipynb`:
- **YOLO26x** detection on `room.jpg` found 2 chairs, 1 couch, 6 potted plants, 1 dining table, 2 books and 1 vase in **65.7 ms** at 448×640.
- **YOLO26x-seg** segmentation was also run.
- Weights present: `yolo26s.pt`, `yolo26x.pt`, `yolo26x-seg.pt`.
- These are **exploration only**; `laser_track.py` still uses `yolo11n.pt` and `yolov8s-worldv2.pt`. Swapping in YOLO26 would be a one-argument change (`--model`) for COCO classes.

---

## 12. Onsite data-collection rig (tripod payload)

**Goal:** take the sensors to a construction site on a tripod and, with one button, record what all of them see at the same moment, for processing later. **Raw data only:** no extrinsic calibration is applied or stored, because the sensors will move to a new mount (decision 2 Nov).

**Script:** `Onsite-Data-Collection/capture.py`, a single file. It needs the LiDAR bridge running (`.\bridge\start_lidar.ps1`), the D435i, and the X4 in webcam mode.

**GUI (Open3D):**
- Left: RealSense colour and depth side by side, the 360° preview, a status line per sensor, live IMU tilt.
- Right: the live L2 cloud. It starts levelled with +z into the screen and zoomed out to the whole cloud. "Reset LiDAR view" restores that.
- Name and Notes fields, a LiDAR length slider (0.5–10 s), CAPTURE.

**Per capture** → `captures/<name>_<YYYYmmdd-HHMMSS>/`, every file `<name>_<sensor>`:
| File | Contents |
|---|---|
| `_realsense_color.png` | 1280×720 colour |
| `_realsense_depth.png` | uint16 depth **aligned to colour**, × `depth_scale` (0.001) = metres |
| `_realsense.ply` | Coloured point cloud from that depth (pinhole, ignores the near-zero colour distortion) |
| `_insta360.jpg` | 2880×1440 stitched equirect |
| `_lidar.npz` / `_lidar.ply` | All L2 frames in the window: xyz, intensity, frame sizes, sensor stamps, host times. **Raw, not range-corrected.** |
| `_meta.json` | Name, notes, per-sensor timestamps, RealSense intrinsics and depth_scale, IMU gravity (mean, std, `tilt_deg`), missing sensors, file list |

Plus one row in `captures/index.csv`. A sensor that delivers nothing in the window is listed under `missing` and the rest is still saved.

**Sync:** the RealSense and X4 frames are the first ones that arrive after the LiDAR window opens.

**Verified on hardware (2 Nov):**
- **Capture 203** (10 s, notes "on the bench"): nothing missing, 119 LiDAR frames in 9.9 s (11.9 frames/s, longest gap 90 ms), 631,325 points all finite. RealSense +15 ms after the window opened, X4 +28 ms. Depth valid on 99 % of pixels. The RealSense cloud has one point per valid depth pixel (912,360). Tilt 0.2°.
- In general, the RealSense lands 15–29 ms and the X4 9–28 ms after the window opens.
- **Captures so far:** `capture_20261102-105641`, `203_20261102-110157`, `capture_20261102-153803` (tilt 2.6°) and `capture_20261102-153932`. All four are complete.
- Quick sanity check: LiDAR vs RealSense depth on the same surface reads +13.5 cm raw and −3.0 cm range-corrected (85 % within 10 cm), which matches the known L2 long bias.

**Performance fixes made while building it:**
- The depth colour preview is downsampled before colouring.
- Depth→colour alignment runs only on the captured frame.
- Colour and depth share one image widget (two same-size widgets showed the same texture).

The GPU wasn't needed; Open3D already renders on it.

**Known limits:**
- The live view's start orientation uses the **old** `extrinsics.json` to find "up". On the new mount it may start tilted. Only the preview is affected, not the data.
- If the window is killed instead of closed, the cameras can stay locked until a USB replug (§3).
- The hardware-free test (`test_capture.py`) is gone and would have to be rewritten.

---

## 13. Current state at a glance

| Component | State | Evidence |
|---|---|---|
| LiDAR streaming (relay + bridge) | ✅ Working; bridge patched against stuck clients (2 Nov) | Relay log shows packets; 11.9 frames/s in 10 s onsite capture |
| Synchronised capture GUI | ✅ Working | 54 capture folders, selftest ok |
| Camera intrinsics | ✅ Excellent fit (0.16 px), ⚠️ edge-region fold (k3) | `intrinsics_calibrated.json` |
| LiDAR ↔ camera extrinsics | ✅ Working, stable (7.2–7.4 cm, rotation within ~1°) | Plane RMS ~1.2 cm; overlays; fusion GUI |
| L2 range correction | ✅ Offset model validated at 1–2 m on the checkerboard (±2 cm) ❌ **fails 1–6 m whiteboard test** (9.3 cm drift) ⚠️ colour/incidence effects remain | §7.3, §7.7 |
| Live fusion view | ✅ Working | `fusion_gui.py`, selftest ok |
| Galvo ↔ camera | ✅ Fitted, 1.9 mm median in-sample ⚠️ not independently verified | `galvo_calib.json`; `galvo_verify` never run |
| Laser object tracking | ✅ Implemented (YOLO11/World, box/circle, safety blocking) | selftest ok; uses D435i depth |
| Galvo free drawing GUI | ✅ Working (dry run); not yet tried on the laser | `galvo_test_gui.py --selftest` ok |
| Onsite capture rig (D435i + X4 + L2) | ✅ Working on hardware, 4 complete captures | §12 |
| Scan ↔ BIM ICP | 🟡 Runs; accuracy capped by uncorrected range bias + unmodelled geometry | median 80–205 mm (203 rerun 28 Oct) |
| Panorama layout | 🟡 Inference works; footprint too simple; no BIM registration yet | §10 |
| Selftests | ✅ 8 pass (25 Oct); the 3 new ones pass (3 Nov) | `calibrate`, `capture_gui`, `fusion_gui`, `galvo_calib`, `galvo_verify`, `laser_track`, `calibrate_joint`, `verify_lidar`; new: `galvo_test_gui`, `wall_capture`, `board_size` |

---

## 14. Known issues and open items

**Data and calibration hygiene**
1. **Re-run `calibrate.py`** now that the 4 suspect captures are deleted. `extrinsics.json` still uses them, and `overlays/` is stale (33 files).
2. **Apply `CALIB_FIX_K3`** in `calibrate_intrinsics` to remove the edge fold. Afterwards, ideally refit the galvo (its dots store 3D points computed with the current K; the change is small inside its coverage).
3. Decide which extrinsics file is the default: `extrinsics.json` (board-size model) or `extrinsics_joint.json` (offset model, better supported). `fusion_gui` and `board_size` default to `extrinsics.json`.
4. Regenerate `board_size.png` (deleted).

**Validation still missing (needs hardware)**
5. **`galvo_verify.py`** at about 1 m, 2 m and 3.5 m, at poses not used in calibration. Pass: median < 3 mm, p95 < 8 mm, small bias.
6. ~~Wall test with a laser meter~~ **done 28 Oct on a whiteboard, 1–6 m: FAIL** (§7.7). Follow-ups:
   - a. Repeat on a **matte painted wall** (no board), 1–8 m, after 15–20 min warm-up, to separate a surface effect from a distance effect.
   - b. Check board-point **intensity vs distance** in the saved `Lidar-Test/*/lidar.npz` (the range-walk hypothesis).
   - c. Record the meter mark's position relative to the L2 so the absolute bias can be read, not just its shape.
   - d. Tape-measure the whiteboard and run `board_size.py --real W H`.
   - e. If the drift is confirmed, fit a range- (or intensity-) dependent correction instead of a single offset.
7. More checkerboard captures at **3–5 m**, then `calibrate_joint.py --model linear`, and check the halves agree (κ within 0.005).
8. Tape-measure check of the camera↔LiDAR (~7 cm) and galvo↔camera (~15 cm) offsets. Only worth doing if the current mount stays; the sensors are moving to a new mount.
9. L2 **warm-up** test (the ~2 cm session drift).
10. Try `galvo_test_gui.py` on the real laser and set flip X/Y once.

**Known sensor limits to design around**
11. Dark surfaces read ~7 cm short (range walk); steep incidence lowers the bias and raises scatter; edges bloom ~9 cm. Object sizes measured by the L2 come out ~20 cm too large.
12. Galvo depth range 1.5–2.5 m; reach limited to the upper-left part of the camera image.
13. The D435i depth is only usable to ~3 m on the whiteboard (§7.7).
14. **New mount:** all of §5 (intrinsics are fine if the camera itself is untouched; extrinsics and galvo are not) must be redone on it before fusion or laser targeting.

**Features not built**
15. LiDAR depth for far targets in `laser_track.py`.
16. ICP on range-corrected scans. Future scans must keep the sensor frame/origin; the onsite rig's `_lidar.npz` does (raw sensor frame).
17. Panorama layout → scale → BIM registration. The onsite rig now records an X4 pano together with an L2 scan, which is the data this needs.
18. Onsite rig: the live-view levelling depends on the old `extrinsics.json`. Switch it to the IMU alone, or plain +z, once on the new mount. Rewrite `test_capture.py` if wanted.

**Repo housekeeping**
19. Commit the untracked work: `calib/verify_lidar.py`, `calib/galvo_test_gui.py`, `Lidar-Test/` (scripts and `readings.csv`; the raw clouds are probably for `.gitignore`), `Onsite-Data-Collection/capture.py` (captures probably git-ignored), the `calibrate.py` `fit_plane` fix, `PROJECT_STATUS.md`.
20. Fix the `galvo_test_gui.py` docstring (still says `draw_rect.py`).
21. Update `README.md` (mention `calibrate_joint.py`, `galvo_verify.py`, `verify_lidar.py`, `galvo_test_gui.py`, `Lidar-Test/`, `Onsite-Data-Collection/`) and `ROADMAP.md` (outdated `data/` paths; checklist ticks).
22. Fix the `docker/README.txt` IP typo (`192.169` → `192.168`).
23. `CALIB_TARGET` path is hard-coded to `C:\Users\nikhi\Documents\SMART-MARK\Unitree4DLidarL2`.
24. The `lidar_bridge` patch lives only in WSL (`/root/lidar_bridge`). Consider copying `lidar_bridge.cpp` into `bridge/` so it's versioned.

---

## 15. Command cheat sheet

```powershell
# LiDAR stream
.\bridge\start_lidar.ps1                  # start (-Stop to stop)

# 1+2. Capture and LiDAR-camera / intrinsics
python calib/capture_gui.py
python calib/calibrate.py                 # -> extrinsics.json, intrinsics_calibrated.json, overlays/
python calib/calibrate_joint.py           # -> extrinsics_joint.json (offset model), reproducibility report
python calib/fusion_gui.py [--calib captures/calibration/extrinsics_joint.json]
python calib/board_size.py

# 3. Galvo
python calib/galvo_calib.py capture | fit | aim
python calib/galvo_verify.py --label near_1m

python calib/galvo_test_gui.py [--dry-run]  # draw shapes anywhere in the mirror range, no calibration

# Application
python calib/laser_track.py [--dry-run]

# Onsite tripod capture (bridge running, D435i + X4 in webcam mode plugged in; close with the window X)
python Onsite-Data-Collection/capture.py [--seconds 2] [--out <dir>]

# L2 range vs laser meter
python Lidar-Test/wall_capture.py --label wall [--gate 0.4]   # v preview, Enter capture, q quit
python Lidar-Test/wall_capture.py --fit
python Lidar-Test/board_size.py --real <W> <H>

# Verification (no hardware)
python calib/verify_lidar.py boards | points | scans
# Verification (hardware)
python calib/verify_lidar.py wall ; python calib/verify_lidar.py wall-fit

# ICP / BIM
python icp\webui.py                       # http://127.0.0.1:5000
python icp\generate.py --rooms 203
python icp\align.py --pair 203

# Panorama layout
python localization360\app.py             # http://127.0.0.1:5001

# Selftests (no hardware)
cd calib; python calibrate.py --selftest; python capture_gui.py --selftest; python fusion_gui.py --selftest
python galvo_calib.py selftest; python galvo_verify.py selftest; python laser_track.py --selftest
python calibrate_joint.py --selftest; python verify_lidar.py selftest; python galvo_test_gui.py --selftest
cd ..; python Lidar-Test/wall_capture.py --selftest; python Lidar-Test/board_size.py --selftest
```

---

## 16. Timeline

| Date | What happened |
|---|---|
| 10 Aug 2026 | Early room 203 scans/clouds (`icp/data/misc`) |
| 11 Aug | IFC `H_DA_D21_Final.ifc` obtained |
| 24 Aug | Room 203 wall-offset investigation (unmodelled counter, recess, see-through) |
| 25 Aug | HorizonNet and LGT-Net run on the room 203 Insta360 panorama: both predict a plain box |
| 26 Aug | Room scans 203/207/214 (short and 2 s); BIM clouds generated; ICP runs (15:29 without, 16:32 with outlier removal) |
| 28–30 Sep | ROADMAP and architecture diagram; YOLO weights |
| 2 Oct | First commits (calibration scripts, bridge, docker, README) |
| 6 Oct | README/gitignore updates; ICP and localization360 moved into the repo; docker folder renamed |
| 13–14 Oct | YOLO26 notebook tests |
| **14 Oct** | **Full recalibration:** 58 captures (12:30–12:52, 13:54–13:58) → intrinsics 0.16 px; extrinsics run 1 (spread 0.25) and run 2 (spread 0.34, 33 boards); 26 galvo views (14:08–14:35) → 1.9 mm median |
| 18 Oct | Sensor decision analysis (keep L2 and D435i); repro analysis; 4 suspect captures deleted; `calibrate_joint.py` (offset 19.4 cm reproducible, slope not); email paragraph; commit `ee13e8c` "Updated fit test" (includes the galvo `S = 1.0` fix) |
| 19 Oct | `verify_lidar.py`: `boards` (tilt dependence found), `scans` (no growth with distance), `points` (precision ~1 cm, corrected accuracy ±2 cm, black squares 7.3 cm closer) |
| 25 Oct | All 8 selftests pass; first version of this status document |
| 28 Oct | ICP rerun on 203 (same result). `calibrate.py` `fit_plane` hang fixed. **Whiteboard range test vs laser meter, 1.06–5.95 m** (`Lidar-Test/`, 18 captures): repeatable to ≤ 0.3 cm, but the offset correction drifts 9.3 cm, FAIL |
| 2 Nov | **Onsite capture rig** built and verified (D435i colour, depth, cloud, IMU; X4 360° webcam mode; L2), 4 captures. `lidar_bridge` patched against stuck clients. Galvo free-drawing GUI. Decision: no extrinsics for the onsite rig; sensors moving to a new mount |
| 3 Nov | `insta360.py` merged into `capture.py` (one script). This status update |
