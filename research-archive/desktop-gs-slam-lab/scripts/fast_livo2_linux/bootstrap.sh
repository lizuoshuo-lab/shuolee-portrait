#!/usr/bin/env bash
# FAST-LIVO2 official ROS1/focal bootstrap. This script never starts a fusion run.
set -Eeuo pipefail

lab_input=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
build_root=/opt/gs-slam-lab/fast-livo2
jobs=4
stage=all
while (($#)); do
  case "$1" in
    --windows-lab) lab_input="$2"; shift 2 ;;
    --root) build_root="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --help) printf '%s\n' 'Usage: bash bootstrap.sh [--windows-lab /mnt/c/.../GS-SLAM-Lab] [--root /opt/gs-slam-lab/fast-livo2] [--jobs 4] [--stage all|packages|build|verify]'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] && ((jobs <= 4)) || { printf '%s\n' 'Use 1 to 4 build jobs.' >&2; exit 2; }
[[ "$stage" =~ ^(all|packages|build|verify)$ ]] || exit 2
[[ "$build_root" == /* && "$build_root" != / && "$build_root" != /opt && "$build_root" != /home && "$build_root" != /root && "$build_root" != /mnt* ]] || { printf '%s\n' 'Build root must be a dedicated directory on the Linux filesystem, outside /mnt.' >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="$lab_input/outputs/fast-livo2-linux-bootstrap-001/linux-executions/$run_id"
mkdir -p -- "$output"
mkdir -p -- "$output/script-snapshot"
cp -- "$script_dir"/*.sh "$script_dir"/*.py "$script_dir"/*.json "$script_dir"/*.md "$output/script-snapshot/"
script_dir="$output/script-snapshot"
sha256sum "$script_dir"/* > "$output/script-snapshot.sha256"
exec > >(tee -a "$output/bootstrap.log") 2>&1
current_step=preflight
trap 'code=$?; printf "FAILED step=%s exit=%s\n" "$current_step" "$code"; printf "failed\t%s\t%s\n" "$current_step" "$code" > "$output/status.tsv"; exit "$code"' ERR
printf 'Started UTC=%s mode=%s build_root=%s jobs=%s\n' "$(date -u --iso-8601=seconds)" "$stage" "$build_root" "$jobs"
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 20.04 && "$VERSION_CODENAME" == focal ]] || { printf '%s\n' 'This source revision requires Ubuntu 20.04/focal; refusing a silent OS/ROS migration.'; false; }
[[ "$(dpkg --print-architecture)" == amd64 ]] || { printf '%s\n' 'This checked bootstrap targets amd64.'; false; }
[[ -d "$lab_input/repos/FAST-LIVO2" && -f "$lab_input/outputs/fast-livo2-source-audit-001/repository-acquisition.json" ]] || { printf '%s\n' 'Pinned Windows source or receipt is missing.'; false; }
[[ "$stage" == verify || "$(id -u)" == 0 ]] || { printf '%s\n' 'Run all/packages/build as root; verify may run without root.'; false; }
printf '%s\n' 'ROS Noetic ended maintenance on 2025-05-31. This is an isolated reproduction of the original focal/ROS1 code, without changing distribution.'

install_packages() {
  current_step=ubuntu_base_packages
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o Acquire::Retries=3 -o DPkg::Lock::Timeout=60 update
  apt-get -o Acquire::Retries=3 -o DPkg::Lock::Timeout=60 install -y --no-install-recommends ca-certificates curl gnupg python3 python3-yaml python3-numpy software-properties-common build-essential cmake git unzip pkg-config locales
  add-apt-repository -y -n universe
  current_step=ros_signing_key
  local key_temp="$output/ros.key" keyring=/usr/share/keyrings/gs-lab-ros-archive-keyring.gpg
  local source_line='deb [arch=amd64 signed-by=/usr/share/keyrings/gs-lab-ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main'
  local apt_source=/etc/apt/sources.list.d/gs-lab-ros-noetic.list
  local cached="$lab_input/outputs/fast-livo2-linux-bootstrap-001/official-sources/ros_key.asc"
  if [[ -f "$cached" ]]; then cp -- "$cached" "$key_temp"; else curl --fail --location --retry 3 --connect-timeout 20 --max-time 120 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' --output "$key_temp"; fi
  printf '%s  %s\n' '4a91c49af0d6f0016108b93698782b596c27ccd836937e18e0e36c3347dc602f' "$key_temp" | sha256sum --check -
  # A fresh WSL root account may not yet have GnuPG's required home directory.
  install -d -m 0700 /root/.gnupg
  gpg --batch --no-options --with-colons --show-keys "$key_temp" > "$output/ros-key-inspection.txt"
  grep -q '^fpr:::::::::C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654:' "$output/ros-key-inspection.txt"
  if [[ -f "$keyring" ]] && ! cmp -s -- "$keyring" "$key_temp"; then printf '%s\n' 'Existing dedicated ROS keyring differs; preserve it and review before retry.'; false; fi
  install -m 0644 -- "$key_temp" "$keyring"
  if [[ -f "$apt_source" ]] && [[ "$(cat -- "$apt_source")" != "$source_line" ]]; then printf '%s\n' 'Existing dedicated ROS source differs; preserve it and review before retry.'; false; fi
  printf '%s\n' "$source_line" > "$apt_source"
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 120 'http://packages.ros.org/ros/ubuntu/dists/focal/InRelease' --output "$output/ros-InRelease"
  gpgv --keyring "$keyring" "$output/ros-InRelease"
  current_step=ros_noetic_packages
  apt-get -o Acquire::Retries=3 -o DPkg::Lock::Timeout=60 update
  apt-get -o Acquire::Retries=3 -o DPkg::Lock::Timeout=60 install -y --no-install-recommends \
    ros-noetic-ros-base ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-compressed-image-transport ros-noetic-tf ros-noetic-eigen-conversions \
    ros-noetic-cmake-modules ros-noetic-visualization-msgs ros-noetic-rosbag \
    ros-noetic-roslaunch ros-noetic-message-generation ros-noetic-message-runtime \
    libeigen3-dev libpcl-dev libopencv-dev libboost-thread-dev
  # RViz, mimalloc, Livox hardware SDK, CUDA and NVIDIA drivers are not needed for headless offline fusion.
  dpkg-query -W > "$output/dpkg-packages.tsv"
  apt-cache policy ros-noetic-ros-base ros-noetic-cv-bridge libopencv-dev libpcl-dev libeigen3-dev > "$output/package-policy.txt"
  printf '%s\n' 'packages_ready' > "$output/packages-stage.txt"
}

materialize_and_build() {
  current_step=pinned_source_materialization
  mkdir -p -- "$build_root"
  python3 "$script_dir/verify_sources.py" --windows-lab "$lab_input" --root "$build_root" --output "$output/source-integrity.json"
  current_step=sophus_configure
  cmake -S "$build_root/deps/Sophus" -B "$build_root/deps/Sophus/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=98 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local
  current_step=sophus_build
  cmake --build "$build_root/deps/Sophus/build" --parallel "$jobs"
  current_step=sophus_tests
  (cd -- "$build_root/deps/Sophus/build" && ctest --output-on-failure)
  current_step=sophus_install
  cmake --install "$build_root/deps/Sophus/build"
  ldconfig
  current_step=catkin_build
  # Upstream Sophus exports its build-directory config rather than installing that config file.
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
  (cd -- "$build_root/catkin_ws" && catkin_make -j"$jobs" -l"$jobs" \
    -DCMAKE_BUILD_TYPE=Release -DSophus_DIR="$build_root/deps/Sophus/build" \
    -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -DCMAKE_SHARED_LINKER_FLAGS=-fopenmp \
    -DPYTHON_EXECUTABLE=/usr/bin/python3)
}

verify_environment() {
  current_step=environment_acceptance
  [[ -f /opt/ros/noetic/setup.bash && -f "$build_root/catkin_ws/devel/setup.bash" ]]
  set +u
  source /opt/ros/noetic/setup.bash
  source "$build_root/catkin_ws/devel/setup.bash"
  set -u
  [[ "$(rosversion -d)" == noetic ]]
  rospack find fast_livo | tee "$output/rospack-fast-livo.txt"
  local binary="$build_root/catkin_ws/devel/lib/fast_livo/fastlivo_mapping"
  [[ -x "$binary" ]]
  ldd "$binary" | tee "$output/fastlivo-ldd.txt"
  if grep -q 'not found' "$output/fastlivo-ldd.txt"; then false; fi
  sha256sum "$binary" > "$output/fastlivo-binary.sha256"
  # Ubuntu focal's libpcl-dev installs the versioned pkg-config module.
  pkg-config --modversion opencv4 pcl_common-1.10 > "$output/opencv-pcl-versions.txt"
  dpkg-query -W libpcl-dev libopencv-dev > "$output/pcl-opencv-package-versions.txt"
  roslaunch --nodes fast_livo mapping_avia.launch rviz:=false > "$output/avia-launch-nodes.txt"
  roslaunch --dump-params fast_livo mapping_avia.launch rviz:=false > "$output/avia-launch-params.yaml"
  python3 - "$output" "$build_root" "$jobs" <<'PY'
import datetime, json, pathlib, platform, sys
out, root = map(pathlib.Path, sys.argv[1:3])
result = {"status": "compiled_environment_acceptance_passed", "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "system": platform.platform(), "ros": "noetic", "ubuntu": "20.04", "fast_livo2_commit": "0d2c0346107b75b59934975adec9a6eeeb913c64", "build_jobs": int(sys.argv[3]), "binary": str(root / "catkin_ws/devel/lib/fast_livo/fastlivo_mapping"), "fusion_started": False, "local_trajectory_measured": False, "gpu_required": False, "launch_scope": "XML/parameter parse only; no ROS master or fusion nodes started"}
(out / "environment-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
PY
}

if [[ "$stage" == all || "$stage" == packages ]]; then install_packages; fi
if [[ "$stage" == all || "$stage" == build ]]; then materialize_and_build; fi
if [[ "$stage" == all || "$stage" == build || "$stage" == verify ]]; then verify_environment; fi
current_step=complete
printf 'passed\t%s\t0\n' "$stage" > "$output/status.tsv"
printf 'Completed stage=%s output=%s\n' "$stage" "$output"
