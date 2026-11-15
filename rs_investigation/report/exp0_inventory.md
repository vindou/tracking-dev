# Experiment 0 — Inventory (PARTIAL: librealsense not installed)

Raw logs: logs/exp0_inventory_raw.txt, logs/exp0_usb_detail.txt
Collected: Jetson clock 2023-11-21 16:32 local (clock is WRONG, not NTP-synced; real date 2026-11-14)

| Item | Value | Source |
|---|---|---|
| Board | Jetson Orin Nano Engineering Reference Developer Kit Super | /proc/device-tree/model |
| L4T | R36.4.3 (nvidia-l4t-core 36.4.3-20250107) -> JetPack 6.2 (inferred; nvidia-jetpack meta-pkg not installed) | /etc/nv_tegra_release |
| Kernel | 5.15.148-tegra, SMP PREEMPT (not RT), OOT variant | uname |
| Power mode | 25W (mode 1) | nvpmodel -q |
| CPU governor | schedutil on all 6 cores, max 1.344 GHz (idle at 729 MHz) | sysfs |
| RAM | 7.4 GiB, 3.7 GiB swap | free |
| librealsense | NOT INSTALLED (no package, lib, pkg-config, pyrealsense2) | dpkg/ls |
| Backend | n/a. Stock uvcvideo, unpatched (dmesg: "Unknown video format 00000050-...") | dmesg |
| Network | Jetson has no internet (DNS fails); Mac does | curl |
| sudo | needs password | sudo -n |
| Build deps | cmake 3.22.1, g++ 11.4; libudev-dev yes; libusb-1.0-0-dev NO (runtime .so present); libssl-dev NO | dpkg |
| /dev/bus/usb nodes | root:root 0664 -> user cannot claim interfaces (RSUSB needs udev rule or root) | ls -l |
| Thermals idle | tj 41 C | tegrastats |

## Cameras (provisional names by USB port; final A/B must be assigned by serial once SDK works)

| Prov. name | USB path | bcdDevice | Negotiated speed | Notes |
|---|---|---|---|---|
| P4 | 1-2.4 (Bus1 Dev4) | 51.13 | **480M (USB 2.0)** | enumerated clean 16:21:43 |
| P3 | 1-2.3 (Bus1 Dev8) | 51.00 | **480M (USB 2.0)** | error -71 on descriptor read, reset, 3 disconnect/re-enumerate cycles 16:22:08-16:22:15 |

Serial numbers: not in USB descriptor (iSerial=0). Firmware version: needs SDK. bcdDevice differs between cameras (51.00 vs 51.13), so the firmware may differ too. Unconfirmed.

## USB topology
Bus 02 (USB3, 10 Gbps root) -> Realtek 4-port USB3 hub 2-1: **nothing attached on any port**
Bus 01 (USB2, 480M root)    -> Realtek 4-port USB2 hub 1-2: camera P3 on port 3, camera P4 on port 4
Both cameras share ONE USB 2.0 hub -> one 480 Mbit/s link to the controller.

## Critical finding
Both D435i enumerated at High-Speed (USB 2.0). The SuperSpeed side of the onboard hub has no devices.
848x480 Z16 @90 fps = 73.3 MB/s = 586 Mbit/s per camera of payload alone. This is above the whole 480 Mbit/s bus
before overhead. With the current cabling the target mode cannot run even on ONE camera.
The SDK limits USB2-connected D400 profiles. At USB2, D435 depth is typically limited to <= 848x480@10 or 640x480@30 (to be confirmed by exp 2).
Likely causes: USB2-only cables or adapters, partially seated USB-C connectors (the -71 errors and re-enumeration on P3 also point to this),
or the cable is plugged in the wrong way for a USB-C connector that is not fully reversible.
**Physical action needed:** reconnect each camera with a known-good USB 3.x cable (short, preferably the original), fully seated, directly to the
Jetson's USB-A ports. Afterwards, `lsusb -t` must show the cameras under Bus 02 at 5000M.

## Update after SDK install (librealsense 2.59.1, RSUSB backend) — logs/exp0_exp2_sdk_inventory.txt, logs/exp0_inventory_repeatability.txt

| Name | Serial | ASIC serial | Firmware | SDK usb_type_descriptor | Kernel speed | SDK physical port |
|---|---|---|---|---|---|---|
| **A** | 234322070408 | 229123060143 | 5.17.3.10 | 2.1 | 480M | 1-2.4-4 |
| B | unknown: cannot enumerate | | | | | USB port 1-2.1, reconnecting continuously |

Camera B (port 1-2.1) has been in a continuous reconnect loop with -71 errors since 16:55:33. That time matches the udevadm trigger.
Earlier at 16:35:40 the kernel gave up on it ("Cannot enable. Maybe the USB cable is bad?").
The first SDK inventory at 16:55 saw 2 devices and failed with "cannot access depth sensor": it caught B during a brief connection.
5 later inventory runs: 1 device, identical output (md5 6dd1b964).
