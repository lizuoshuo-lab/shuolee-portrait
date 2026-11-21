# FAST-LIVO2 Linux bootstrap

This directory prepares the unchanged official FAST-LIVO2 revision `0d2c0346107b75b59934975adec9a6eeeb913c64` on Ubuntu 20.04 / ROS Noetic. Preparing scripts and probing repositories are separate from actually installing, compiling, or generating a fusion trajectory. There is no CUDA requirement.

## Execution after WSL is available

Use the actual installed Ubuntu 20.04 distribution name in place of `DISTRO_NAME`. The parent Windows setup chooses the distro's storage location; this script does not import, move, unregister, shut down, or restart WSL.

```powershell
wsl.exe -d DISTRO_NAME -u root -- bash /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/fast_livo2_linux/bootstrap.sh --jobs 4
```

The Linux source and compilation directories are under `/opt/gs-slam-lab/fast-livo2`, inside the distro's Linux filesystem. Windows data and audit receipts stay under `/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab`. The source is copied; compiling directly under `/mnt/c` is rejected.

`--stage packages` installs Ubuntu/ROS dependencies. `--stage build` verifies/copies pinned source and dependencies and builds/accepts the environment. `--stage verify` inspects an existing built environment without an apt installation. Repeating a successful stage is supported; changed pinned source files or conflicting dedicated apt configuration fail rather than being overwritten. Runtime writes to upstream `Log/` may make full source verification fail after later experiments: archive results and review their provenance before preparing another clean source copy.

Each invocation creates a distinct UTC/PID run directory under `outputs/fast-livo2-linux-bootstrap-001/linux-executions/`, with the full log and a status record. A failed command leaves its step and nonzero exit code there. The script never reboots, runs `roscore`, plays bags, or starts fusion nodes.

## Immutable dependency selection

The official FAST-LIVO2 README asks for the double-only Sophus revision `a621ff`. Its full SHA is `a621ff2e56c56c839a6c40418d42c3c254424b5c`. The short six-character reference did not resolve through the current GitHub commit API; the full SHA was independently found in the upstream history before the templated transition. Sophus builds unchanged with the C++98 standard used by its original implementation, avoiding the `std::complex::real() =` incompatibility under newer standards. FAST-LIVO2 itself uses its upstream C++17 build. Actual compiler compatibility remains subject to the Linux build, not this static preparation.

Vikit uses the FAST-LIVO2 author's fork `xuankuzcr/rpg_vikit`, pinned to `6c886c8e5d83997806e00294826d528cea3581dd`; do not substitute the older FAST-LIVO1 Vikit fork. Both dependencies use fixed official codeload ZIPs with downloaded archive SHA256 and CRC verification. This avoids relying on a mutable default branch or Git-clone network behavior.

The original FAST-LIVO2 directory is verified against all 75 Git blob hashes and SHA256 values in its existing acquisition receipt. Dependency files are verified directly against the pinned archives before every build, and compiled outputs are separate from upstream files. The catkin invocation explicitly supplies `-fopenmp` to executable/shared-library linking: upstream already supplies OpenMP compile options and calls `omp_get_wtime()`, so its runtime library must be linked as well. This is a toolchain argument, not a source modification. Each Linux invocation snapshots these bootstrap scripts and uses the copied helpers.

## ROS repository and lifecycle

ROS Noetic maintenance ended on 2025-05-31. This work reproduces the official ROS1 code in a local environment; it does not silently migrate to ROS2 or a different Ubuntu release. [Official EOL notice](https://discourse.ros.org/t/ros-noetic-end-of-life-may-31-2025/43160).

The current official main repository's focal InRelease and amd64 Packages.gz were downloaded over HTTP. Its InRelease signature was successfully verified with `gpgv` using the current official key fetched over authenticated GitHub HTTPS; the Packages.gz SHA256 matches the signed metadata. HTTPS connections to packages.ros.org and snapshots.ros.org failed certificate verification on this host. The install uses the standard official HTTP apt source with a dedicated `signed-by` keyring and apt's mandatory signature checks; no certificate or package-signature check is disabled. The final snapshot uses a different key and is not an automatic fallback.

The pinned current ROS key has fingerprint `C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654`, SHA256 `4a91c49af0d6f0016108b93698782b596c27ccd836937e18e0e36c3347dc602f`. Its downloaded bytes are binary OpenPGP packets, not ASCII armor; direct copying to the `.gpg` keyring was separately tested with `gpgv` and succeeded. The read-only probe cache's `.asc` suffix does not describe the actual binary content. A future key change requires explicit revalidation of the lock file rather than an unverified replacement.

## Acceptance evidence

- Ubuntu is exactly 20.04/focal, amd64; ROS reports Noetic.
- Installed versions and apt package origins are archived.
- Original FAST-LIVO2 and dependencies pass source integrity checks.
- Sophus builds, runs its included CTest suite, and installs its library.
- Catkin builds with at most four jobs and produces `devel/lib/fast_livo/fastlivo_mapping`.
- `ldd` reports no missing libraries; the binary SHA256 is archived.
- The official Avia launch parses parameters and node names with `rviz:=false`.
- The acceptance receipt says `fusion_started: false` and `local_trajectory_measured: false`. Compiling alone is not completion of fusion or drift evaluation.

RViz and mimalloc are optional. CUDA, NVIDIA drivers, and a Livox hardware SDK are not installed by this bootstrap. FAST-LIVO2 already supplies the Livox custom ROS message headers for offline bag use. The Avia launch already repackages `/left_camera/image/compressed` into raw `/left_camera/image` via `image_transport`; the required compressed transport plugin is installed.

For the next actual experiment, use a dedicated run-output archive because upstream logging can overwrite prior results, set ROS simulated time, verify bag topics and point timestamps, play a complete bag, and preserve trajectories before another run. Bright_Screen_Wall has no known ground truth; it can establish fusion runtime, not an ATE accuracy measurement. NTU EEE03 plus its Leica position truth is needed for measured local position drift.
