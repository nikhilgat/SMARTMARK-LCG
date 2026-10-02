"""
Relays UDP packets arriving on the real Windows adapter (192.168.1.2:6201,
the address the LiDAR is configured to stream to) into WSL, where
lidar_bridge actually listens. Needed because WSL2's default NAT network
doesn't own 192.168.1.2, so inbound sensor data would otherwise never
reach it. Outbound (WSL -> sensor) already works fine via normal NAT and
needs no relay.

WSL's NAT IP changes across restarts, so it's auto-detected at startup
rather than hardcoded.

Used for: getting live Unitree L2 LiDAR packets from Windows into the WSL
          lidar_bridge, which then serves parsed point clouds on 127.0.0.1:9899.
Works with: bridge/start_lidar.ps1 (starts this relay and the WSL bridge),
          and everything in calib/ that reads LiDAR (capture_gui.py, fusion_gui.py).
Needs: Windows, WSL with an "Ubuntu" distro, the LiDAR streaming to 192.168.1.2:6201.
"""

import socket
import subprocess
import sys

LISTEN_ADDR = ("192.168.1.2", 6201)


def get_wsl_ip():
    """Ask WSL for its current IP address (runs `hostname -I` inside Ubuntu) and return the first one."""
    out = subprocess.check_output(
        ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", "hostname", "-I"],
        text=True,
    )
    ip = out.strip().split()[0]
    return ip


try:
    wsl_ip = get_wsl_ip()
except Exception as e:
    print(f"Could not determine WSL IP ({e}). Is WSL/Ubuntu running?")
    sys.exit(1)

FORWARD_ADDR = (wsl_ip, 6201)

# Main loop: receive every UDP packet from the LiDAR and send it on unchanged to WSL.
# Prints the first 5 packets and then every 200th so you can see data is flowing.

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(LISTEN_ADDR)
print(f"Relaying UDP {LISTEN_ADDR} -> {FORWARD_ADDR}")

count = 0
while True:
    data, addr = sock.recvfrom(65535)
    count += 1
    if count <= 5 or count % 200 == 0:
        print(f"[{count}] {len(data)} bytes from {addr}")
    sock.sendto(data, FORWARD_ADDR)
