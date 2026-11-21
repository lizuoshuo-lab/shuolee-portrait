#!/usr/bin/env python3
"""Read-only official dependency/apt endpoint probe; no package installation."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCES = {
    "ros_focal_inrelease": "https://packages.ros.org/ros/ubuntu/dists/focal/InRelease",
    "ros_focal_packages": "https://packages.ros.org/ros/ubuntu/dists/focal/main/binary-amd64/Packages.gz",
    "ros_final_snapshot_inrelease": "https://snapshots.ros.org/noetic/final/ubuntu/dists/focal/InRelease",
    "ros_focal_inrelease_http": "http://packages.ros.org/ros/ubuntu/dists/focal/InRelease",
    "ros_focal_packages_http": "http://packages.ros.org/ros/ubuntu/dists/focal/main/binary-amd64/Packages.gz",
    "ros_final_snapshot_inrelease_http": "http://snapshots.ros.org/noetic/final/ubuntu/dists/focal/InRelease",
    "ros_key": "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key",
    "ubuntu_focal_inrelease": "https://archive.ubuntu.com/ubuntu/dists/focal/InRelease",
    "sophus_archive": "https://codeload.github.com/strasdat/Sophus/zip/a621ff2e56c56c839a6c40418d42c3c254424b5c",
    "vikit_archive": "https://codeload.github.com/xuankuzcr/rpg_vikit/zip/6c886c8e5d83997806e00294826d528cea3581dd",
}

def fetch(item, out):
    name, url = item
    suffix = ".zip" if name.endswith("archive") else ".gz" if name.endswith("packages") else ".asc" if name == "ros_key" else ".txt"
    target = out / (name + suffix)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GS-SLAM-Lab dependency audit/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = response.read(20 * 1024 * 1024 + 1)
            if len(raw) > 20 * 1024 * 1024:
                raise ValueError("endpoint exceeds 20 MiB probe limit")
            status, resolved = response.status, response.url
        target.write_bytes(raw)
        result = {"name": name, "url": url, "resolved_url": resolved, "status": status, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "saved_file": target.name}
        if name.endswith("archive"):
            with zipfile.ZipFile(target) as zf:
                invalid = zf.testzip()
                if invalid:
                    raise ValueError("ZIP CRC failed: " + invalid)
                result["zip_crc_passed"] = True
                result["entry_count"] = len(zf.infolist())
        if "inrelease" in name:
            decoded = raw.decode("utf-8", errors="strict")
            result["metadata_fields"] = {k: next((line[len(k)+2:] for line in decoded.splitlines() if line.startswith(k + ": ")), None) for k in ["Origin", "Label", "Suite", "Codename", "Date", "Valid-Until"]}
            result["signature_verified"] = False
        return result
    except Exception as exc:
        return {"name": name, "url": url, "status": "failed", "error": str(exc), "package_installation_attempted": False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda item: fetch(item, args.output), SOURCES.items()))
    receipt = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "scope": "HTTP reachability and downloaded-byte integrity only; apt GPG verification and Linux compilation not executed", "ros_noetic_eol": "2025-05-31", "sources": results}
    (args.output / "official-endpoint-probe.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == "__main__":
    main()
