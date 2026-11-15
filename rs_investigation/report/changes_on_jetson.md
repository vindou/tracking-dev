# Changes made on the Jetson

| When (Jetson clock) | Change | Where | Undo |
|---|---|---|---|
| 16:37 | Copied source tarballs librealsense v2.59.1, libusb 1.0.30, nlohmann/json 3.12.0 (downloaded on the Mac; sha256 in logs) | ~/rs_investigation/deps/ | rm -rf ~/rs_investigation/deps |
| 16:39 | Built and installed libusb 1.0.30 (headers + .so) | ~/.local/{include,lib} | remove libusb-1.0* from ~/.local |
| 16:39 | Installed nlohmann/json 3.12.0 headers + CMake config | ~/.local/{include/nlohmann,share/cmake/nlohmann_json} | rm those dirs |
| 16:40-16:48 | Built and installed librealsense 2.59.1, RSUSB backend, ENABLE_STATS=OFF, CHECK_FOR_UPDATES=OFF, BUILD_ROSBAG2=OFF, no examples/viewer/python/CUDA | ~/.local/{lib/librealsense2*,include/librealsense2,bin/rs-*,lib/cmake/realsense2} | rm those files |
| pending | udev rule 99-realsense-libusb.rules (user runs with sudo) | /etc/udev/rules.d/ | sudo rm /etc/udev/rules.d/99-realsense-libusb.rules && sudo udevadm control --reload-rules |

Not changed: system clock (wrong, 2023; left as is), kernel, firmware, nvpmodel, governor, apt packages.
Build commands: ~/rs_investigation/deps/build_deps.sh (uses tar -m because the clock is wrong).
