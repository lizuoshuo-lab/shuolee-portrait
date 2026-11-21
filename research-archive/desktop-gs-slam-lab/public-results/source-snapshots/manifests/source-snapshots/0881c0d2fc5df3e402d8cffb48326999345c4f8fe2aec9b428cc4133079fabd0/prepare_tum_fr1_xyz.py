"""Acquire and validate one official TUM fr1_xyz archive on CPU only.

The 500 MB ceiling is an internal preparation guard. Never changes the frozen
loader or training scripts. Publisher hashes are not available here: SHA256 is
computed locally for provenance, not presented as publisher authentication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import urllib.request

import numpy as np
from PIL import Image
import tum_rgbd

ROOT = Path(__file__).resolve().parents[1]
NAME = "rgbd_dataset_freiburg1_xyz"
URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"+NAME+".tgz"
FINAL_URL = "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/"+NAME+".tgz"
EXPECTED_BYTES = 448204271
MAX_BYTES = 500000000
MAX_UNCOMPRESSED_BYTES = 2000000000
MANIFEST = ROOT/"manifests/tum-fr1-xyz-preparation.json"
ASSOCIATIONS = ROOT/"manifests/tum-fr1-xyz-association.json"
ARCHIVE = ROOT/"data/downloads"/(NAME+".tgz")
STAGING = ROOT/"data/tum/.fr1_xyz_preparation_tmp"
DESTINATION = ROOT/"data/tum"/NAME
LOCKED = {"tum_rgbd.py": "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11",
          "train_tum_online_gs_v4.py": "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(manifest):
    manifest["last_update_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def request(method):
    response = urllib.request.urlopen(urllib.request.Request(URL, method=method, headers={
        "User-Agent": "GS-SLAM-Lab-data-audit/1.0", "Accept-Encoding": "identity"}), timeout=45)
    if response.status != 200 or response.url not in (URL, FINAL_URL):
        response.close()
        raise ValueError("Unexpected archive response status/source")
    size = int(response.headers["Content-Length"])
    if size != EXPECTED_BYTES or size > MAX_BYTES:
        response.close()
        raise ValueError("Official archive length changed or exceeded preflight ceiling")
    return response


def download(manifest):
    with request("HEAD") as response:
        manifest["download_preflight"] = {"requested_url": URL, "final_url": response.url,
            "content_length_bytes": int(response.headers["Content-Length"]), "authorized_max_bytes": MAX_BYTES,
            "ceiling_type": "internal agent preflight guard, not a user budget constraint",
            "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag")}
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != EXPECTED_BYTES:
            raise ValueError("Existing archive has unexpected size; refusing to overwrite it")
        manifest["download"]["reused_existing_archive"] = True
    else:
        part = ARCHIVE.with_suffix(".tgz.part")
        if part.exists():
            raise ValueError("A previous partial file exists; manual inspection required before resuming")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        total, next_log = 0, 32*1024*1024
        with request("GET") as response, part.open("xb") as stream:
            manifest["download"].update({"archive_GET_performed": True, "final_url": response.url})
            save(manifest)
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES or total > EXPECTED_BYTES:
                    raise ValueError("Downloaded bytes exceed declared size/cap")
                stream.write(chunk)
                if total >= next_log:
                    print(f"downloaded {total}/{EXPECTED_BYTES} bytes", flush=True)
                    next_log += 32*1024*1024
        if total != EXPECTED_BYTES:
            raise ValueError("Truncated official archive")
        part.rename(ARCHIVE)
    manifest["download"].update({"archive_bytes": ARCHIVE.stat().st_size,
        "sha256_local_computed": sha(ARCHIVE), "publisher_checksum_available": False,
        "archive_relative_path": ARCHIVE.relative_to(ROOT).as_posix(),
        "hash_scope": "local exact-file identifier; not a published checksum or independent authenticity proof"})
    manifest["status"] = "archive_downloaded_validating"
    save(manifest)


def safe_member(member):
    name = member.name
    if "\\" in name or "\x00" in name:
        raise ValueError("Backslash/NUL in tar path")
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or parts[0] != NAME:
        raise ValueError("Unexpected tar root or absolute path")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(p in (".", "..") or ":" in p or p.rstrip(" .") != p or p.split(".")[0].upper() in reserved for p in parts):
        raise ValueError("Unsafe Windows tar component")
    if not (member.isfile() or member.isdir()) or member.issym() or member.islnk():
        raise ValueError("Only regular files and directories permitted; no links/devices")
    target = STAGING.joinpath(*parts).resolve()
    if not target.is_relative_to(STAGING.resolve()) or not target.is_relative_to((ROOT/"data/tum").resolve()):
        raise ValueError("Tar target escapes dataset staging directory")
    return target


def extract(manifest):
    if DESTINATION.exists() or STAGING.exists():
        raise ValueError("Dataset/staging directory already exists; refusing to merge extraction")
    uncompressed = 0
    with gzip.open(ARCHIVE, "rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            uncompressed += len(chunk)
            if uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Expanded gzip exceeds safety ceiling")
    # Reading to EOF above checks the gzip CRC/ISIZE trailer independently of
    # tar's stop-at-end-of-archive behavior.
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        targets = [(member, safe_member(member)) for member in members]
        normalized = [p.relative_to(STAGING).as_posix().casefold() for _, p in targets]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Duplicate/case-alias tar names")
        total_bytes = sum(m.size for m in members if m.isfile())
        if total_bytes > MAX_UNCOMPRESSED_BYTES or shutil.disk_usage(ROOT).free < total_bytes+MAX_BYTES:
            raise ValueError("Expanded size limit or insufficient free space")
        STAGING.mkdir()
        for member, target in targets:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as stream:
                shutil.copyfileobj(source, stream, 1024*1024)
            if target.stat().st_size != member.size:
                raise ValueError("Extracted member size mismatch")
    manifest["archive_validation"] = {"gzip_integrity": "passed_read_to_EOF_CRC_and_size_trailer",
        "gzip_uncompressed_bytes": uncompressed, "tar_member_count": len(members),
        "regular_file_count": sum(m.isfile() for m in members), "expanded_regular_file_bytes": total_bytes,
        "tar_safety": "all paths under exact xyz root; no symlink/hardlink/device/absolute/dotdot/Windows alias; no duplicate normalized names",
        "unsafe_members": 0, "extraction": "manual regular-file copy after complete member preflight"}
    save(manifest)
    return STAGING/NAME


def validate_images(dataset, rows, kind):
    seen, modes, positive_fractions = set(), set(), []
    for _, relative in rows:
        path = tum_rgbd.safe_data_path(dataset, relative)
        if relative in seen or PurePosixPath(relative).parts[0] != kind or path.suffix != ".png":
            raise ValueError("Duplicate or unexpected indexed PNG path")
        seen.add(relative)
        with path.open("rb") as f:
            header = f.read(33)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError("Invalid PNG signature/IHDR")
        bit_depth, color_type = header[24], header[25]
        if (bit_depth, color_type) != ((8, 2) if kind == "rgb" else (16, 0)):
            raise ValueError("Unexpected RGB/depth PNG storage format")
        with Image.open(path) as image:
            if image.size != (640, 480) or image.format != "PNG":
                raise ValueError("Invalid sensor dimensions/format")
            image.verify()
        with Image.open(path) as image:
            image.load()
            modes.add(image.mode)
            pixels = np.asarray(image)
            if kind == "rgb":
                assert pixels.shape == (480, 640, 3) and pixels.dtype == np.uint8
            else:
                assert pixels.shape == (480, 640) and np.issubdtype(pixels.dtype, np.integer)
                assert pixels.min() >= 0 and pixels.max() <= 65535
                positive_fractions.append(float(np.mean(pixels > 0)))
    actual = {p.relative_to(dataset).as_posix() for p in (dataset/kind).glob("*.png")}
    if actual != seen:
        raise ValueError("PNG files and index references differ")
    result = {"indexed_images": len(rows), "all_fully_decoded_and_crc_verified": True,
        "dimensions": [640, 480], "modes": sorted(modes), "IHDR_bit_depth": bit_depth,
        "IHDR_color_type": color_type, "index_paths_unique_and_safe": True, "unindexed_PNG_count": 0}
    if positive_fractions:
        result["raw_positive_depth_fraction_min_mean_max"] = [min(positive_fractions), float(np.mean(positive_fractions)), max(positive_fractions)]
    return result


def independent_association(rgb, depth, gt):
    times = [np.array([float(row[0]) for row in rows]) for rows in (depth, gt)]
    last, used, matches = -np.inf, set(), []
    for i, row in enumerate(rgb):
        stamp = float(row[0]); indices = []
        for values in times:
            high = int(np.searchsorted(values, stamp))
            candidates = [j for j in (high-1, high) if 0 <= j < len(values)]
            indices.append(min(candidates, key=lambda j: (abs(values[j]-stamp), j)))
        j, k = indices
        if abs(times[0][j]-stamp) >= .02 or abs(times[1][k]-stamp) >= .02 or stamp-last <= 1/32 or j in used:
            continue
        matches.append((i, j, k)); used.add(j); last = stamp
    return matches


def validate_dataset(dataset, manifest):
    rgb = tum_rgbd.read_rows(dataset/"rgb.txt", 2)
    depth = tum_rgbd.read_rows(dataset/"depth.txt", 2)
    gt = tum_rgbd.read_rows(dataset/"groundtruth.txt", 8)
    # Validate every indexed GT pose structurally, not against an estimator.
    for row in gt:
        tum_rgbd.pose_from_tum(row)
    images = {"rgb": validate_images(dataset, rgb, "rgb"), "depth": validate_images(dataset, depth, "depth")}
    frames, association = tum_rgbd.associate_frames(dataset)
    assert independent_association(rgb, depth, gt) == [(f.rgb_index, f.depth_index, f.gt_index) for f in frames]
    sensor_rows = []
    valid_fractions = []
    for start in range(0, len(frames), 32):
        batch = frames[start:start+32]
        _, _, masks = tum_rgbd.load_frames(dataset, batch, 160, 120, .2, 5.)
        valid_fractions.extend(masks.mean(axis=(1, 2)).tolist())
    for frame in frames:
        sensor_rows.append({k: v for k, v in frame.metadata().items() if k not in ("c2w", "w2c")})
    associations = {"dataset": NAME, "status": "complete_frozen_loader_association_checked_by_searchsorted_reference",
        "loader_sha256": LOCKED["tum_rgbd.py"], "statistics": association, "frames": sensor_rows,
        "scope": "data integrity/association only; GT timestamps gate association, GT poses only structurally parsed, no estimator/evaluation/training"}
    ASSOCIATIONS.write_text(json.dumps(associations, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    manifest["dataset_validation"] = {"indices": {name: {"rows": len(rows), "sha256": sha(dataset/name)} for name, rows in (("rgb.txt", rgb), ("depth.txt", depth), ("groundtruth.txt", gt))},
        "all_index_timestamps_finite_and_strictly_increasing": True, "all_GT_poses_finite_proper_rigid": True,
        "images": images, "association": association,
        "independent_nearest_timestamp_reference_exact_match": True,
        "associated_frame_ids": list(range(len(frames))),
        "associated_rgb_time_span_seconds": frames[-1].rgb_timestamp-frames[0].rgb_timestamp,
        "frozen_loader_all_associated_frames_loaded_at_160x120": True,
        "depth_valid_fraction_0p2_to_5m_min_mean_max": [min(valid_fractions), float(np.mean(valid_fractions)), max(valid_fractions)],
        "association_manifest": ASSOCIATIONS.relative_to(ROOT).as_posix(), "association_manifest_sha256": sha(ASSOCIATIONS)}


def main():
    old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if old.get("status") == "completed_cpu_dataset_preparation":
        raise SystemExit("Dataset already prepared; manifest retained without re-downloading")
    manifest = {"dataset": NAME, "status": "preparing", "script_sha256": sha(Path(__file__)),
        "scope": "second-sequence data preparation only; no model, GPU, cloud, GT-based selection or tuning",
        "preflight_history": [{"status": old.get("status"), "checked_utc": old.get("checked_utc"),
            "download_preflight": old.get("download_preflight"), "interpretation": "earlier 300 MB internal guard; superseded by 500 MB internal guard, not a user-imposed budget or user blocker"}],
        "official_source_checks": old.get("official_source_checks", {}), "download": {},
        "locked_sources_before": {name: sha(ROOT/"scripts"/name) for name in LOCKED},
        "limitations": ["fr1_xyz is a second sequence from the same Freiburg1 sensor/domain, not demonstrated cross-scene generalization.",
                        "Data is prepared; no pose estimation or reconstruction result exists for this sequence from this preparation.",
                        "fr1_desk start-200 is an already inspected diagnostic clip, not unseen evaluation."]}
    try:
        assert manifest["locked_sources_before"] == LOCKED
        save(manifest)
        download(manifest)
        print("archive complete; validating gzip/tar and all sensor files", flush=True)
        dataset = extract(manifest)
        validate_dataset(dataset, manifest)
        # Both final targets are fixed children of data/tum and were checked
        # absent above; rename only this newly created verified dataset.
        assert dataset.resolve().is_relative_to((ROOT/"data/tum").resolve())
        assert DESTINATION.resolve().parent == (ROOT/"data/tum").resolve()
        if DESTINATION.exists():
            raise ValueError("Destination appeared during preparation")
        dataset.rename(DESTINATION)
        STAGING.rmdir()  # Empty directory only; no recursive deletion.
        manifest["dataset_relative_path"] = DESTINATION.relative_to(ROOT).as_posix()
        manifest["locked_sources_after"] = {name: sha(ROOT/"scripts"/name) for name in LOCKED}
        assert manifest["locked_sources_after"] == LOCKED
        manifest["status"] = "completed_cpu_dataset_preparation"
        save(manifest)
        print(json.dumps({"status": manifest["status"], "archive_sha256": manifest["download"]["sha256_local_computed"],
                          "association": manifest["dataset_validation"]["association"]}, indent=2), flush=True)
    except BaseException as error:
        manifest["status"] = "failed_preparation_incomplete"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error),
            "partial_archive_retained": ARCHIVE.with_suffix(".tgz.part").exists(), "staging_retained": STAGING.exists()}
        save(manifest)
        raise


if __name__ == "__main__":
    main()
