#!/usr/bin/env python3
"""Verify a downloaded official ROS apt index, without installing packages."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REQUIRED = ["ros-noetic-ros-base", "ros-noetic-pcl-ros", "ros-noetic-cv-bridge", "ros-noetic-image-transport", "ros-noetic-compressed-image-transport", "ros-noetic-tf", "ros-noetic-eigen-conversions", "ros-noetic-cmake-modules", "ros-noetic-visualization-msgs", "ros-noetic-rosbag", "ros-noetic-roslaunch", "ros-noetic-message-generation", "ros-noetic-message-runtime"]

def gpg_path(path):
    value = str(path.resolve()).replace("\\", "/")
    if os.name == "nt" and re.match(r"^[A-Za-z]:/", value):
        return "/" + value[0].lower() + value[2:]
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpgv", required=True)
    args = parser.parse_args()
    base = args.output_root
    key = base / "official-sources/ros_key.asc"
    metadata = base / "official-sources-http-check/ros_focal_inrelease_http.txt"
    packages = base / "official-sources-http-check/ros_focal_packages_http.txt"
    gpg_home = base / "gpg-inspection"
    gpg_home.mkdir(parents=True, exist_ok=True)
    command = [args.gpgv, "--homedir", gpg_path(gpg_home), "--keyring", gpg_path(key), gpg_path(metadata)]
    process = subprocess.run(command, text=True, capture_output=True)
    (base / "ros-main-inrelease-gpgv.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError("ROS InRelease signature verification failed")
    key_sha = hashlib.sha256(key.read_bytes()).hexdigest()
    if key_sha != "4a91c49af0d6f0016108b93698782b596c27ccd836937e18e0e36c3347dc602f":
        raise ValueError("unexpected official ROS key bytes")
    signed_metadata = metadata.read_text(encoding="utf-8")
    sha_section = signed_metadata.split("SHA256:\n", 1)[1].split("-----BEGIN", 1)[0]
    rows = [line.split() for line in sha_section.splitlines() if line.strip()]
    expected = next(row for row in rows if row[2] == "main/binary-amd64/Packages.gz")
    raw = packages.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected[0] or len(raw) != int(expected[1]):
        raise ValueError("package index does not match signed metadata")
    entries = {}
    for stanza in gzip.decompress(raw).decode("utf-8").split("\n\n"):
        fields = {line.split(": ", 1)[0]: line.split(": ", 1)[1] for line in stanza.splitlines() if ": " in line and not line.startswith(" ")}
        if fields.get("Package") in REQUIRED:
            entries[fields["Package"]] = {key: fields.get(key) for key in ["Version", "Architecture", "Filename", "SHA256"]}
    if set(entries) != set(REQUIRED):
        raise ValueError("required ROS packages missing from official index: " + str(set(REQUIRED) - set(entries)))
    receipt = {"status": "official_ros_focal_index_signature_and_package_hash_verified", "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "repository": "http://packages.ros.org/ros/ubuntu", "codename": "focal", "distribution": "noetic", "inrelease_gpgv_exit_code": process.returncode, "ros_key_sha256": key_sha, "ros_key_fingerprint": "C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654", "inrelease_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(), "package_index_sha256": actual, "package_index_bytes": len(raw), "required_packages_in_signed_index": entries, "package_installation_executed": False, "linux_build_executed": False, "snapshot_fallback": "not used; final snapshot uses a different signing key"}
    (base / "ros-apt-verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["status"])

if __name__ == "__main__":
    main()
