# SMARTMARK-LCG

LiDAR–camera–galvo calibration and laser object tracking.

A Unitree L2 LiDAR, an Intel RealSense D435i camera and a galvo laser are calibrated against each other. Once calibrated, the rig detects objects with YOLO and outlines the chosen object with the laser.

```
Unitree L2 ──UDP──> udp_relay.py ──> lidar_bridge (WSL) ──TCP :9899──┐
RealSense D435i ──USB3───────────────────────────────────────────────┼──> calib/*.py
Galvo laser <──serial (COM5)─────────────────────────────────────────┘
```

## Hardware

- Unitree L2 4D LiDAR, streaming to a Windows network adapter set to `192.168.1.2` (see [unilidar_sdk2](https://github.com/unitreerobotics/unilidar_sdk2))
- Intel RealSense D435i on USB3
- Galvo laser controller on a serial port (default `COM5`)
- A1 checkerboard: 8×6 squares of 95 mm, which gives 7×5 inner corners. Measure your print; if the squares aren't 95 mm, pass `--square`.

All sensors must be mounted rigidly. If anything moves after calibration, calibrate again.

## Setup

Windows with WSL (Ubuntu) and Python 3.10:

```powershell
pip install numpy scipy opencv-python matplotlib open3d pyrealsense2 pyserial ultralytics torch pillow
```

Two pieces are not in this repo:

- **`lidar_bridge`**: a native program that parses the L2 packets with the Unitree SDK and serves point clouds on `127.0.0.1:9899`. `start_lidar.ps1` expects it at `/root/lidar_bridge/lidar_bridge` inside WSL Ubuntu.
- **`calib_target`**: the LiDAR checkerboard detector from the SMART-MARK `Unitree4DLidarL2` repo. Set its path with `--calib-target`, or change `CALIB_TARGET` in [calib/calibrate.py](calib/calibrate.py).

YOLO weights (`yolo11n.pt`, `yolov8s-worldv2.pt`) download automatically the first time they're used.

## Workflow

Run everything from the repo root.

### 1. Start the LiDAR stream

```powershell
.\bridge\start_lidar.ps1          # starts lidar_bridge in WSL and udp_relay.py
.\bridge\start_lidar.ps1 -Stop    # stops both
```

### 2. Capture checkerboard poses

```powershell
python calib/capture_gui.py
```

This shows the live camera and the live LiDAR cloud. Put the board 1–2.5 m away and press **CAPTURE** while the rig is still. Take 12–20 poses, tilting the board ±30–45° left/right and forward/back. Each capture is saved to `captures/<time>_<tag>/`.

### 3. Calibrate LiDAR ↔ camera

```powershell
python calib/calibrate.py
```

This writes `captures/calibration/extrinsics.json` (the LiDAR-to-camera transform, the camera lens parameters and the LiDAR range correction), `intrinsics_calibrated.json`, and overlay images for checking the result by eye. Aim for a plane RMS under 2 cm per pose. If a pose is flagged as an outlier, delete that capture and run the script again.

To check the result:

```powershell
python calib/fusion_gui.py        # live LiDAR points drawn on the camera image, plus a coloured 3D cloud
python calib/board_size.py        # LiDAR-measured board size vs the real print -> board_size.png
```

### 4. Calibrate the galvo laser

```powershell
python calib/galvo_calib.py capture   # press SPACE once per board pose; the laser sweeps a dot grid automatically
python calib/galvo_calib.py fit       # -> captures/galvo/galvo_calib.json
python calib/galvo_calib.py aim       # click in the image and the laser should hit that spot
```

Take 5 or more views at different distances and tilts. Keep everyone out of the laser's path while it sweeps.

### 5. Track objects with the laser

```powershell
python calib/laser_track.py             # GUI; SPACE toggles the laser
python calib/laser_track.py --dry-run   # everything except the laser
```

Type class names (for example `chair`, `bottle` or `door`). The 80 COCO classes run on YOLO11; any other name switches to open-vocabulary YOLO-World. The laser draws one box or circle around one object at a time. It won't fire at targets outside the calibrated area unless you tick the option that allows it.

## Self-tests

These run without any hardware:

```powershell
cd calib
python calibrate.py --selftest
python capture_gui.py --selftest
python fusion_gui.py --selftest
python galvo_calib.py selftest
python laser_track.py --selftest
```

## Repo layout

| Path | Contents |
|---|---|
| `bridge/` | `start_lidar.ps1` and `udp_relay.py`, which forward LiDAR UDP from Windows into WSL |
| `calib/` | Capture, calibration, fusion viewer, galvo calibration and laser tracking scripts; each file's header says what it's for and what it works with |
| `captures/calibration/` | LiDAR–camera calibration results (`extrinsics.json`, `intrinsics_calibrated.json`) |
| `captures/galvo/` | Galvo calibration result (`galvo_calib.json`) |
| `Dockerfile_Driver (1)/` | An alternative ROS 2 Foxy Docker setup for the L2 driver with RViz |
| `ROADMAP.md` | Calibration method and validation checklist |
| `architecture.html` | Architecture diagram |

Raw captures and model weights aren't in the repo (see `.gitignore`).
