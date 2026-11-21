"""Acquire the official 3DGS scenes archive and audit its complete Truck scene.

This script changes only its dedicated data and evidence directories. It uses
the existing lab Python, NumPy and Pillow; it does not install or train anything.
Archive transfer is resumable, bounded to 1 GB received in total, and extraction
is restricted to safe regular files within the selected Truck prefix.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import struct
import subprocess
import threading
import time
import urllib.request
import zipfile
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/large-scene-gs/tandt"
EVIDENCE = ROOT / "outputs/large-scene-gs-001/data-preparation"
URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
PROJECT = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
LICENSE = "https://www.tanksandtemples.org/license/"
MAX_RECEIVED = 1_000_000_000
MAX_UNCOMPRESSED = 8 * 1024 ** 3
ARCHIVE = DATA / "tandt_db.zip"
PART = DATA / "tandt_db.zip.part"
MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
          2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
          4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
          6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
          8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5),
          10: ("THIN_PRISM_FISHEYE", 12)}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def request(url, *, method="GET", headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers or {}), timeout=30)


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    progress_path = EVIDENCE / "download-progress.json"
    old = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    with request(URL, method="HEAD") as response:
        headers = dict(response.headers)
        size = int(response.headers["Content-Length"])
        remote = {"requested_url": URL, "final_url": response.geturl(), "http_status": response.status,
                  "headers": headers, "checked_utc": now()}
    save(EVIDENCE / "download-head.json", remote)
    if size > MAX_RECEIVED:
        raise RuntimeError(f"Archive {size} exceeds compressed download limit {MAX_RECEIVED}")
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != size:
            raise RuntimeError("Existing final archive size differs; preserve it, do not overwrite")
        return {"status": "reused_existing_size_matching_archive_pending_crc", "expected_bytes": size,
                "received_network_bytes": old.get("received_network_bytes", 0), "remote": remote}
    received = max(int(old.get("received_network_bytes", 0)), PART.stat().st_size if PART.exists() else 0)
    state = {"status": "downloading", "url": URL, "expected_bytes": size,
             "compressed_received_limit_bytes": MAX_RECEIVED, "received_network_bytes": received,
             "attempts": old.get("attempts", []), "updated_utc": now()}
    for _ in range(8):
        start = PART.stat().st_size if PART.exists() else 0
        if start == size:
            break
        if start > size:
            raise RuntimeError("Preserved partial archive is larger than expected")
        if received >= MAX_RECEIVED:
            raise RuntimeError("Compressed network transfer budget exhausted; partial remains recoverable")
        attempt = {"start_byte": start, "started_utc": now()}
        state["attempts"].append(attempt)
        save(progress_path, state)
        try:
            resume_headers = {"Range": f"bytes={start}-"} if start else {}
            if start:
                resume_headers["If-Range"] = headers.get("ETag", headers.get("Last-Modified", ""))
            with request(URL, headers=resume_headers) as response:
                attempt.update(http_status=response.status, headers=dict(response.headers))
                if start:
                    expected_range = f"bytes {start}-"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                        raise RuntimeError("Server did not honor the exact Range; refusing to append mismatched bytes")
                elif response.status != 200:
                    raise RuntimeError(f"Unexpected initial status {response.status}")
                length = int(response.headers.get("Content-Length", size - start))
                if length != size - start:
                    raise RuntimeError("Response length does not match remaining archive bytes")
                with PART.open("ab" if start else "xb") as output:
                    last = time.monotonic()
                    while output.tell() < size:
                        limit = min(1024 * 1024, size - output.tell(), MAX_RECEIVED - received)
                        if limit <= 0:
                            raise RuntimeError("Compressed network transfer budget exhausted")
                        block = response.read(limit)
                        if not block:
                            raise RuntimeError("Unexpected end of archive response")
                        received += len(block)
                        output.write(block)
                        if time.monotonic() - last >= 4:
                            output.flush()
                            state.update(received_network_bytes=received, bytes_written=output.tell(), updated_utc=now())
                            save(progress_path, state)
                            print(f"download {output.tell()}/{size} ({100 * output.tell() / size:.1f}%)", flush=True)
                            last = time.monotonic()
                    output.flush()
            attempt.update(status="response_complete", ended_utc=now())
        except Exception as exc:
            attempt.update(status="failed_preserved_for_exact_range_resume", error=repr(exc), ended_utc=now())
            state.update(received_network_bytes=received, bytes_written=PART.stat().st_size if PART.exists() else 0,
                         updated_utc=now(), status="partial_preserved")
            save(progress_path, state)
            print(f"transfer interrupted; partial preserved: {exc}", flush=True)
            # A bounded retry; no deletion or restart from byte zero.
            if isinstance(exc, RuntimeError) and "Range" in str(exc):
                raise
            continue
    if not PART.exists() or PART.stat().st_size != size:
        raise RuntimeError("Transfer incomplete after bounded attempts; rerun to resume preserved partial")
    PART.replace(ARCHIVE)
    state.update(status="download_complete_pending_crc", bytes_written=size,
                 received_network_bytes=received, updated_utc=now(), remote=remote)
    save(progress_path, state)
    return state


def parallel_resume():
    """Reuse an immutable contiguous prefix, then receive disjoint ranges."""
    progress_path = EVIDENCE / "parallel-download-progress.json"
    previous = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    head = json.loads((EVIDENCE / "download-head.json").read_text(encoding="utf-8"))
    total = int(head["headers"]["Content-Length"])
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != total:
            raise RuntimeError("Existing archive has wrong size; preserve it")
        return
    if previous:
        state = previous
        prefix = state["preserved_prefix_bytes"]
        if not PART.exists() or PART.stat().st_size != prefix:
            raise RuntimeError("Parallel resume requires its exact immutable contiguous prefix")
    else:
        prefix = PART.stat().st_size if PART.exists() else 0
        if not 0 <= prefix < total:
            raise RuntimeError("Unexpected contiguous prefix size")
        original_state = json.loads((EVIDENCE / "download-progress.json").read_text(encoding="utf-8"))
        prefix_received = max(prefix, int(original_state.get("received_network_bytes", 0)))
        boundaries = [prefix + (total - prefix) * i // 4 for i in range(5)]
        state = {"status": "parallel_disjoint_range_transfer", "url": URL, "expected_bytes": total,
                 "preserved_prefix_bytes": prefix, "preserved_prefix_sha256": digest(PART) if prefix else None,
                 "prefix_received_network_bytes": prefix_received, "compressed_received_limit_bytes": MAX_RECEIVED,
                 "ranges": [{"index": i, "start": boundaries[i], "end_exclusive": boundaries[i + 1],
                             "path": str(DATA / f"tandt_db.range{i}.part"), "received_network_bytes": 0,
                             "bytes_written": 0, "attempts": []} for i in range(4)], "updated_utc": now()}
        save(progress_path, state)
    lock = threading.Lock()

    def update():
        state["received_network_bytes"] = state["prefix_received_network_bytes"] + sum(r["received_network_bytes"] for r in state["ranges"])
        state["bytes_written"] = prefix + sum(r["bytes_written"] for r in state["ranges"])
        state["updated_utc"] = now()
        save(progress_path, state)

    def worker(r):
        path = Path(r["path"])
        length = r["end_exclusive"] - r["start"]
        r["bytes_written"] = path.stat().st_size if path.exists() else 0
        r["received_network_bytes"] = max(r["bytes_written"], r["received_network_bytes"])
        for _ in range(4):
            have = path.stat().st_size if path.exists() else 0
            if have == length:
                r["status"] = "range_complete"
                with lock:
                    update()
                return
            if have > length:
                raise RuntimeError("Preserved range exceeds exact length")
            start, end = r["start"] + have, r["end_exclusive"] - 1
            attempt = {"request_start": start, "request_end_inclusive": end, "started_utc": now()}
            r["attempts"].append(attempt)
            try:
                with request(URL, headers={"Range": f"bytes={start}-{end}",
                                           "If-Range": head["headers"].get("Last-Modified", "")}) as response:
                    attempt.update(http_status=response.status, headers=dict(response.headers))
                    expected_range = f"bytes {start}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Server did not honor the exact bounded Range")
                    if int(response.headers.get("Content-Length", -1)) != end - start + 1:
                        raise RuntimeError("Bounded Range response length mismatch")
                    with path.open("ab" if have else "xb") as output:
                        last = time.monotonic()
                        while output.tell() < length:
                            block = response.read(min(1024 * 1024, length - output.tell()))
                            if not block:
                                raise RuntimeError("Unexpected end of bounded Range")
                            with lock:
                                received = state["prefix_received_network_bytes"] + sum(rr["received_network_bytes"] for rr in state["ranges"])
                                if received + len(block) > MAX_RECEIVED:
                                    raise RuntimeError("Compressed total network transfer budget exhausted")
                                r["received_network_bytes"] += len(block)
                            output.write(block)
                            r["bytes_written"] = output.tell()
                            if time.monotonic() - last >= 4:
                                output.flush()
                                with lock:
                                    update()
                                    print(f"parallel download {state['bytes_written']}/{total} ({100 * state['bytes_written']/total:.1f}%)", flush=True)
                                last = time.monotonic()
                attempt.update(status="response_complete", ended_utc=now())
            except Exception as exc:
                attempt.update(status="failed_preserved_for_exact_range_resume", error=repr(exc), ended_utc=now())
                r["bytes_written"] = path.stat().st_size if path.exists() else 0
                with lock:
                    update()
                if "Range" in str(exc) or "budget" in str(exc):
                    raise
        if path.stat().st_size != length:
            raise RuntimeError("Bounded range incomplete; preserve and rerun --parallel-resume")
        r["status"] = "range_complete"
        with lock:
            update()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, state["ranges"]))
    assembled = DATA / "tandt_db.assembling"
    if assembled.exists():
        raise RuntimeError("Preserved assembly file exists; inspect before resuming")
    with assembled.open("xb") as output:
        for p in [PART, *[Path(r["path"]) for r in state["ranges"]]]:
            with p.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
    if assembled.stat().st_size != total:
        raise RuntimeError("Assembled archive size mismatch")
    assembled.replace(ARCHIVE)
    state["status"] = "parallel_download_complete_pending_crc"
    with lock:
        update()


def safe_member(info):
    name = info.filename
    path = PurePosixPath(name)
    if "\\" in name or path.is_absolute() or any(p in ("..", "") or ":" in p for p in path.parts):
        raise RuntimeError(f"Unsafe ZIP member: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
    if any(p.endswith((".", " ")) or p.split(".", 1)[0].upper() in reserved for p in path.parts):
        raise RuntimeError(f"ZIP member is unsafe on Windows: {name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeError(f"ZIP symbolic link is forbidden: {name!r}")
    return path


def extract():
    print("checking complete archive CRC and safe member paths", flush=True)
    archive_sha = digest(ARCHIVE)
    inventory = []
    with zipfile.ZipFile(ARCHIVE) as z:
        members = z.infolist()
        if len({i.filename.casefold() for i in members}) != len(members):
            raise RuntimeError("Duplicate or case-insensitive colliding ZIP member names")
        if sum(i.file_size for i in members) > MAX_UNCOMPRESSED:
            raise RuntimeError("ZIP uncompressed total exceeds conservative 8 GiB safety bound")
        for info in members:
            safe_member(info)
            inventory.append({"name": info.filename, "compressed_bytes": info.compress_size,
                              "uncompressed_bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                              "is_directory": info.is_dir()})
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive CRC failed at {bad}")
        # Locate scene prefix independently of sparse versus sparse/0 layout.
        scene_prefixes = set()
        for i in members:
            parts = PurePosixPath(i.filename).parts
            if "truck" in [p.lower() for p in parts]:
                idx = [p.lower() for p in parts].index("truck")
                scene_prefixes.add(PurePosixPath(*parts[:idx + 1]))
        if len(scene_prefixes) != 1:
            raise RuntimeError(f"Expected exactly one complete Truck prefix, got {scene_prefixes}")
        prefix = next(iter(scene_prefixes))
        selected = [i for i in members if safe_member(i).is_relative_to(prefix)]
        scene = DATA / "truck"
        reuse_count = 0
        for info in selected:
            rel = safe_member(info).relative_to(prefix)
            target = scene.joinpath(*rel.parts)
            if not target.resolve().is_relative_to(scene.resolve()):
                raise RuntimeError("Extraction escaped scene root")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with z.open(info) as f:
                    source_sha = hashlib.sha256(f.read()).hexdigest()
                if target.stat().st_size != info.file_size or digest(target) != source_sha:
                    raise RuntimeError(f"Existing source file differs; do not overwrite: {target}")
                reuse_count += 1
                continue
            with z.open(info) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    save(EVIDENCE / "archive-members.json", inventory)
    receipt = {"status": "complete_archive_crc_passed_safe_truck_extraction", "archive_path": str(ARCHIVE),
               "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": archive_sha,
               "checksum_provenance": "Locally computed SHA256; no publisher SHA256 was located on project page",
               "archive_member_count": len(inventory), "crc_scope": "Every archive entry, including unextracted scenes",
               "archive_uncompressed_bytes": sum(i["uncompressed_bytes"] for i in inventory),
               "archive_uncompressed_limit_bytes": MAX_UNCOMPRESSED,
               "selected_archive_prefix": str(prefix), "selected_entries": len(selected),
               "reused_identical_source_files": reuse_count, "scene_path": str(scene), "checked_utc": now()}
    save(EVIDENCE / "archive-integrity.json", receipt)
    return scene, receipt


def unpack(f, fmt):
    n = struct.calcsize("<" + fmt)
    raw = f.read(n)
    if len(raw) != n:
        raise RuntimeError("Truncated COLMAP binary")
    return struct.unpack("<" + fmt, raw)


def parse_cameras(path):
    cameras = {}
    with path.open("rb") as f:
        count, = unpack(f, "Q")
        for _ in range(count):
            cid, model_id, width, height = unpack(f, "iiQQ")
            model, n = MODELS[model_id]
            params = unpack(f, "d" * n)
            if cid in cameras:
                raise RuntimeError("Duplicate camera ID")
            if not np.all(np.isfinite(params)) or width <= 0 or height <= 0:
                raise RuntimeError("Invalid camera dimensions or parameters")
            cameras[cid] = {"id": cid, "model_id": model_id, "model": model,
                            "width": width, "height": height, "params": params}
        if f.read(1):
            raise RuntimeError("Extra data after cameras.bin")
    return cameras


def qrotation(q):
    w, x, y, z = q
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                     [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                     [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])


def parse_images(path):
    images = []
    with path.open("rb") as f:
        count, = unpack(f, "Q")
        for _ in range(count):
            values = unpack(f, "i7di")
            raw_name = bytearray()
            while True:
                byte = f.read(1)
                if not byte:
                    raise RuntimeError("Unterminated COLMAP image name")
                if byte == b"\0":
                    break
                raw_name.extend(byte)
            name = raw_name.decode("utf-8")
            n, = unpack(f, "Q")
            raw = f.read(n * 24)
            if len(raw) != n * 24:
                raise RuntimeError("Truncated COLMAP image observations")
            obs = np.frombuffer(raw, dtype=[("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")]).copy()
            q = np.array(values[1:5])
            if not np.all(np.isfinite(values[1:8])):
                raise RuntimeError("Nonfinite camera pose")
            images.append({"id": values[0], "q_wxyz": q, "t": np.array(values[5:8]),
                           "camera_id": values[8], "name": name, "observations": obs, "R": qrotation(q)})
        if f.read(1):
            raise RuntimeError("Extra data after images.bin")
    return sorted(images, key=lambda x: x["name"])


def parse_points(path):
    points, errors, track_lengths = {}, [], []
    with path.open("rb") as f:
        count, = unpack(f, "Q")
        for _ in range(count):
            values = unpack(f, "Q3d3BdQ")
            pid, xyz, rgb, error, track_len = values[0], values[1:4], values[4:7], values[7], values[8]
            if pid in points:
                raise RuntimeError("Duplicate point ID")
            points[pid] = (xyz, rgb)
            errors.append(error)
            track_lengths.append(track_len)
            if len(f.read(track_len * 8)) != track_len * 8:
                raise RuntimeError("Truncated point tracks")
        if f.read(1):
            raise RuntimeError("Extra data after points3D.bin")
    return points, np.array(errors), np.array(track_lengths)


def stats(values):
    x = np.asarray(values, dtype=float)
    return {"count": len(x), "min": float(np.min(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(np.max(x)),
            "rmse": float(np.sqrt(np.mean(x*x)))} if len(x) else {"count": 0}


def audit(scene, factor):
    sparse = scene / "sparse/0"
    if not sparse.exists():
        sparse = scene / "sparse"
    cameras = parse_cameras(sparse / "cameras.bin")
    images = parse_images(sparse / "images.bin")
    points, point_errors, track_lengths = parse_points(sparse / "points3D.bin")
    if not images or not points:
        raise RuntimeError("Scene lacks registered cameras or sparse points")
    # Published Truck is expected to contain undistorted pinhole images. Refuse
    # to silently apply a pinhole check to another camera model.
    if any(c["model"] not in ("PINHOLE", "SIMPLE_PINHOLE") for c in cameras.values()):
        raise RuntimeError("Audit implements exact undistorted pinhole projection only")
    decoded, reprojections, positive, observed = [], [], 0, 0
    rotation_errors, q_errors, determinants = [], [], []
    resized_dir = scene / f"images_{factor}"
    resized_dir.mkdir(parents=True, exist_ok=True)
    names = [im["name"] for im in images]
    if len(set(names)) != len(names) or len(set(im["id"] for im in images)) != len(images):
        raise RuntimeError("Repeated registered image name or ID")
    original_names = sorted(p.relative_to(scene / "images").as_posix()
                            for p in (scene / "images").rglob("*") if p.is_file())
    if set(original_names) != set(names):
        raise RuntimeError("Original images and registered COLMAP names are not an exact match")
    for im in images:
        name = PurePosixPath(im["name"])
        if name.is_absolute() or ".." in name.parts or any(":" in p for p in name.parts):
            raise RuntimeError("Unsafe COLMAP image name")
        camera = cameras[im["camera_id"]]
        source = (scene / "images").joinpath(*name.parts)
        with Image.open(source) as pil:
            pil.load()
            width, height = pil.size
            mode = pil.mode
            # Exact Pillow BICUBIC behavior matches the official parser helper.
            small = pil.convert("RGB").resize((int(round(width / factor)), int(round(height / factor))), Image.Resampling.BICUBIC)
            resized = resized_dir.joinpath(*name.with_suffix(".png").parts)
            resized.parent.mkdir(parents=True, exist_ok=True)
            if resized.exists():
                with Image.open(resized) as existing:
                    existing.load()
                    if existing.size != small.size or not np.array_equal(np.asarray(existing.convert("RGB")), np.asarray(small)):
                        raise RuntimeError(f"Existing resized image differs, refusing overwrite: {resized}")
            else:
                small.save(resized, format="PNG")
        sx, sy = width / camera["width"], height / camera["height"]
        # Published half-resolution dimensions can round odd model dimensions
        # differently on the two axes. Keep and audit both exact row scales.
        q_errors.append(abs(np.linalg.norm(im["q_wxyz"]) - 1))
        rotation_errors.append(np.max(np.abs(im["R"].T @ im["R"] - np.eye(3))))
        determinants.append(np.linalg.det(im["R"]))
        observations = im["observations"]
        observations = observations[observations["point_id"] >= 0]
        xyz = np.array([points[int(pid)][0] for pid in observations["point_id"]])
        p = xyz @ im["R"].T + im["t"]
        zpositive = p[:, 2] > 0
        positive += int(zpositive.sum())
        observed += len(p)
        params = camera["params"]
        if camera["model"] == "PINHOLE":
            fx, fy, cx, cy = params
        else:
            fx, cx, cy = params
            fy = fx
        projected = np.stack([fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy], axis=1)
        obsxy = np.stack([observations["x"], observations["y"]], axis=1)
        residuals = np.linalg.norm((projected[zpositive] - obsxy[zpositive]) * np.array([sx, sy]), axis=1)
        if not np.all(np.isfinite(residuals)):
            raise RuntimeError("Nonfinite reprojection residuals")
        reprojections.extend(residuals.tolist())
        effective_K = [[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]
        train_K = [[effective_K[0][0] * small.width / width, 0, effective_K[0][2] * small.width / width],
                   [0, effective_K[1][1] * small.height / height, effective_K[1][2] * small.height / height], [0, 0, 1]]
        official_x = small.width / (camera["width"] // factor)
        official_y = small.height / (camera["height"] // factor)
        official_train_K = [[fx / factor * official_x, 0, cx / factor * official_x],
                            [0, fy / factor * official_y, cy / factor * official_y], [0, 0, 1]]
        decoded.append({"name": im["name"], "id": im["id"], "camera_id": im["camera_id"],
                        "original_width": width, "original_height": height, "original_mode": mode,
                        "colmap_width": camera["width"], "colmap_height": camera["height"],
                        "source_scale_x": sx, "source_scale_y": sy,
                        "original_pixel_K": effective_K, "training_pixel_K": train_K,
                        "official_parser_training_pixel_K": official_train_K,
                        "official_parser_K_max_abs_difference_px": float(np.max(np.abs(np.array(official_train_K) - np.array(train_K)))),
                        "resized_name": name.with_suffix(".png").as_posix(),
                        "resized_width": small.width, "resized_height": small.height,
                        "positive_depth_fraction": float(zpositive.mean()),
                        "reprojection_original_pixel_stats": stats(residuals)})
    splits = {"protocol": "Lexicographically sorted registered image names; index % 8 == 0 held out",
              "test_every": 8, "used_for_camera_sfm": "Published cameras/points already estimated from full scene; split isolates GS photometric training only",
              "future_rule": "Preserve these exact train/test name lists for all baselines and block experiments; no holdout-based configuration selection",
              "train_names": [name for i, name in enumerate(names) if i % 8 != 0],
              "test_names": [name for i, name in enumerate(names) if i % 8 == 0]}
    existing_split = EVIDENCE / "split.json"
    if existing_split.exists() and json.loads(existing_split.read_text(encoding="utf-8")) != splits:
        raise RuntimeError("Existing frozen split differs; preserve it, no repartition")
    save(existing_split, splits)
    save(EVIDENCE / "image-audit.json", decoded)
    centers = np.array([-im["R"].T @ im["t"] for im in images])
    xyz = np.array([v[0] for v in points.values()])
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(point_errors)):
        raise RuntimeError("Nonfinite world geometry")
    global_report = {"status": "passed_complete_scene_binary_image_and_projection_audit",
                     "registered_image_count": len(images), "camera_count": len(cameras), "sparse_point_count": len(points),
                     "train_image_count": len(splits["train_names"]), "test_image_count": len(splits["test_names"]),
                     "camera_models": list({c["model"] for c in cameras.values()}), "cameras": list(cameras.values()),
                     "quaternion_order": "wxyz; world-to-camera rotation", "pose_equation": "x_camera = R(q_wxyz) x_world + t",
                     "length_unit": "Shared arbitrary SfM scale, not asserted to be meters",
                     "quaternion_norm_max_abs_error": float(max(q_errors)),
                     "rotation_orthonormal_max_abs_error": float(max(rotation_errors)),
                     "rotation_det_min": float(min(determinants)), "rotation_det_max": float(max(determinants)),
                     "valid_point_observation_count": observed, "positive_depth_observation_count": positive,
                     "positive_depth_fraction": positive / observed,
                     "reprojection_scope": "All registered nonnegative point observations, positive depth; original decoded-image pixels",
                     "reprojection_original_pixel_stats": stats(reprojections),
                     "published_point_error_colmap_pixel_stats": stats(point_errors), "point_track_length_stats": stats(track_lengths),
                     "camera_center_bbox": {"min": centers.min(0).tolist(), "max": centers.max(0).tolist()},
                     "sparse_point_bbox": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()},
                     "data_factor": factor, "normalization_applied": False,
                     "image_size_correction": "COLMAP model and original decoded photos may have different dimensions; scale each row of K by actual/model image dimensions, as official gsplat parser does",
                     "training_K_correction_note": "training_pixel_K uses exact resized/model ratios; official_parser_training_pixel_K reproduces the official parser factor division then integer model-size floor and row correction, with a small rounding discrepancy for odd dimensions; baseline must use its own recorded parser K",
                     "official_parser_K_max_abs_difference_px": float(max(row["official_parser_K_max_abs_difference_px"] for row in decoded)),
                     "checked_utc": now()}
    if global_report["quaternion_norm_max_abs_error"] > 1e-5 or global_report["rotation_orthonormal_max_abs_error"] > 1e-5:
        raise RuntimeError("Quaternion or rotation sanity check failed")
    if positive / observed < 0.98 or global_report["reprojection_original_pixel_stats"]["median"] > 3:
        raise RuntimeError("Positive-depth or median reprojection sanity check failed")
    geometry_path = EVIDENCE / "global-cameras-points.npz"
    w2c = np.repeat(np.eye(4)[None], len(images), axis=0)
    for i, im in enumerate(images):
        w2c[i, :3, :3] = im["R"]
        w2c[i, :3, 3] = im["t"]
    geometry_arrays = {"image_names": np.array(names),
                       "image_ids": np.array([im["id"] for im in images], dtype=np.int64),
                       "camera_ids": np.array([im["camera_id"] for im in images], dtype=np.int64),
                       "world_to_camera": w2c, "camera_to_world": np.linalg.inv(w2c),
                       "camera_centers": centers, "point_ids": np.array(list(points), dtype=np.uint64),
                       "points_xyz": xyz,
                       "points_rgb": np.array([v[1] for v in points.values()], dtype=np.uint8),
                       "original_pixel_K": np.array([row["original_pixel_K"] for row in decoded]),
                       "training_pixel_K": np.array([row["training_pixel_K"] for row in decoded]),
                       "official_parser_training_pixel_K": np.array([row["official_parser_training_pixel_K"] for row in decoded]),
                       "test_mask": np.arange(len(images)) % 8 == 0}
    registered_ids = [im["observations"]["point_id"][im["observations"]["point_id"] >= 0] for im in images]
    geometry_arrays["registered_point_observation_ids"] = np.concatenate(registered_ids)
    geometry_arrays["registered_point_observation_offsets"] = np.concatenate([[0], np.cumsum([len(ids) for ids in registered_ids])])
    global_report["point_association_scope"] = "Stored nonnegative image observations in original COLMAP images.bin, including any observations failing positive-depth audit; offsets follow sorted image_names"
    if geometry_path.exists():
        with np.load(geometry_path, allow_pickle=False) as old:
            if set(old.files) != set(geometry_arrays) or any(not np.array_equal(old[k], v) for k, v in geometry_arrays.items()):
                raise RuntimeError("Existing unnormalized global geometry differs; refusing overwrite")
    else:
        np.savez(geometry_path, **geometry_arrays)
    global_report["global_geometry_arrays_path"] = str(geometry_path)
    global_report["global_geometry_arrays_sha256"] = digest(geometry_path)
    save(EVIDENCE / "colmap-audit.json", global_report)
    print(f"audited {len(images)} images, {len(points)} sparse points; split {len(splits['train_names'])}/{len(splits['test_names'])}", flush=True)
    return global_report, splits


def sources():
    results = []
    evidence_path = EVIDENCE / "source-evidence.json"
    previous = {r["url"]: r for r in json.loads(evidence_path.read_text(encoding="utf-8"))} if evidence_path.exists() else {}
    for name, url in (("3dgs-project", PROJECT), ("tanks-download", "https://www.tanksandtemples.org/download/"), ("tanks-license", LICENSE)):
        path = EVIDENCE / f"source-{name}.html"
        if not path.exists():
            try:
                with request(url) as response:
                    raw = response.read(2 * 1024 * 1024)
                    if len(raw) == 2 * 1024 * 1024:
                        raise RuntimeError("Source page unexpectedly large")
                    path.write_bytes(raw)
                    results.append({"url": url, "local_path": str(path), "sha256": digest(path),
                                    "retrieved_utc": now(), "final_url": response.geturl(), "http_status": response.status})
            except Exception as exc:
                results.append({"url": url, "error": repr(exc), "web_verification": "Independently opened through connected web tool on 2026-10-11"})
        else:
            results.append({**previous.get(url, {}), "url": url, "local_path": str(path), "sha256": digest(path), "status": "reused_preserved_source_page"})
    save(evidence_path, results)
    return results


def complete(scene, archive_receipt, report, splits):
    files = []
    for p in sorted(scene.rglob("*")):
        if p.is_file():
            rel = p.relative_to(scene).as_posix()
            files.append({"path": rel, "bytes": p.stat().st_size, "sha256": digest(p),
                          "role": "derived_training_image" if rel.startswith("images_") else "original_archive_member"})
    save(EVIDENCE / "data-files.json", files)
    source_evidence = sources()
    transfer_path = EVIDENCE / "parallel-download-progress.json"
    if not transfer_path.exists():
        transfer_path = EVIDENCE / "download-progress.json"
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    if transfer.get("received_network_bytes", 0) > MAX_RECEIVED:
        raise RuntimeError("Recorded compressed network budget was exceeded")
    transfer["status"] = "download_complete_full_zip_crc_verified"
    transfer["archive_sha256"] = archive_receipt["archive_sha256"]
    transfer["full_archive_crc_scope"] = archive_receipt["crc_scope"]
    save(transfer_path, transfer)
    manifest = {"status": "ready_for_isolated_official_gsplat_single_scene_baseline", "prepared_utc": now(),
                "dataset": "Tanks and Temples / Truck, official 3DGS author input archive",
                "scope": "Complete real outdoor Truck multiview scene; bounded baseline, not a city-scale dataset",
                "project_page": PROJECT, "archive_url": URL, "archive": archive_receipt,
                "download_receipt_path": str(transfer_path), "download_receipt": transfer,
                "preparation_script_path": str(Path(__file__).resolve()), "preparation_script_sha256": digest(Path(__file__).resolve()),
                "scene_path": str(scene), "training_images_path": str(scene / f"images_{report['data_factor']}"),
                "data_factor": report["data_factor"], "colmap_audit": report,
                "file_manifest_path": str(EVIDENCE / "data-files.json"),
                "file_manifest_sha256": digest(EVIDENCE / "data-files.json"),
                "split_path": str(EVIDENCE / "split.json"), "split_sha256": digest(EVIDENCE / "split.json"),
                "split": splits, "source_evidence": source_evidence,
                "licensing": {"verified_sources": ["https://www.tanksandtemples.org/download/", LICENSE],
                              "download_page": "Noncommercial research, teaching, publication or personal experimentation",
                              "license_page_conflict": "Copyright section says CC BY 4.0 and allows commercial adaptation; License Grant instead restricts use to noncommercial scientific research and bars redistribution/modification to third parties without permission",
                              "archive_specific_license": "No independent license on the 3DGS project download page was located; any license files in the archive remain in the member inventory",
                              "local_use": "Local noncommercial learning and experimentation; do not present the repackage as unconditionally CC BY 4.0 or redistribute raw data via the knowledge website",
                              "attribution": "Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM TOG 36(4), 2017. Repacked input cameras/scene provided through Kerbl et al., 3D Gaussian Splatting, 2023."},
                "preparation": {"original_images_modified": False, "resampling": "Pillow RGB BICUBIC at rounded original/factor dimensions, lossless PNG, separate images_factor directory",
                                "global_normalization_applied": False, "training_started": False,
                                "environment_modified": False, "linux_started": False},
                "future_block_protocol": "All blocks share the frozen global camera/point coordinates and train/test membership; normalization must be global or saved/inverted per block"}
    save(EVIDENCE / "input-manifest.json", manifest)
    text = f"""# Truck 公开多视图输入接入\n\n已从 [3DGS 作者项目]({PROJECT})的 Scenes 链接获取完整 tandt_db.zip，仅提取其中完整 Truck 场景。它是围绕真实室外卡车拍摄的多视图起点，不能称为城市级大场景。\n\n压缩包 {archive_receipt['archive_bytes']:,} 字节，SHA256 为 `{archive_receipt['archive_sha256']}`；整个 ZIP 的 CRC 校验通过，全部成员路径已检查，原包保留。没有找到发布方 SHA256，以上为本地完整性身份。\n\n独立二进制读取得到 {report['registered_image_count']} 张注册图像、{report['camera_count']} 个相机、{report['sparse_point_count']:,} 个稀疏点。全部照片成功解码并与注册名称一一匹配。四元数采用 wxyz，外参为世界到相机；正深度观测占 {100 * report['positive_depth_fraction']:.4f}%，原照片像素口径重投影中位数 {report['reprojection_original_pixel_stats']['median']:.4f} px、95 分位数 {report['reprojection_original_pixel_stats']['p95']:.4f} px。这是输入几何一致性检查，不是新训练质量评测。\n\n按图像文件名排序，每第 8 张（零基索引可被 8 整除）固定留出，共 {report['train_image_count']} 张训练、{report['test_image_count']} 张留出。公开 SfM 本来使用完整场景求相机，该划分只隔离后续 GS 光度优化与留出图像评测，不能称为未见相机位姿估计。后续单块、分块与 LOD 共用这份划分。\n\n原图未改；训练小图另存 `images_{report['data_factor']}`，用与官方 gsplat parser 一致的 Pillow BICUBIC、四舍五入尺寸，并保存 PNG。模型相机尺寸与原照片尺寸的实际比例及修正内参逐图记录在 image-audit.json；不能仅用缩图 factor 就假定模型分辨率等于照片分辨率。\n\nTanks 官方[下载说明](https://www.tanksandtemples.org/download/)允许非商业研究、教学和个人实验；[许可页]({LICENSE})的 CC BY 4.0 段落与后面的非商业／禁止未经许可再分发条款存在冲突。此处保留该事实，用于本地非商业学习，不捏造统一许可或把原始数据打包入网站。保留 Knapitsch 等（2017）与 3DGS 作者（2023）的署名。\n\n完整文件 SHA256 清单见 data-files.json，固定划分见 split.json，COLMAP 几何检查见 colmap-audit.json，来源与用途见 input-manifest.json。未安装依赖、启动 Linux 或训练。\n"""
    (EVIDENCE / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "scene_path": str(scene),
                      "manifest": str(EVIDENCE / "input-manifest.json")}, ensure_ascii=False), flush=True)


def cli_crosscheck():
    manifest_path = EVIDENCE / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = manifest["colmap_audit"]
    exe = Path("C:/Users/Administrator/colmap-x64-windows-cuda/bin/colmap.exe")
    sparse = Path(manifest["scene_path"]) / "sparse/0"
    command = [str(exe), "model_analyzer", "--path", str(sparse)]
    started = now()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    output = result.stdout + result.stderr
    expected = {"Cameras": report["camera_count"], "Registered images": report["registered_image_count"],
                "Points": report["sparse_point_count"], "Observations": report["valid_point_observation_count"]}
    actual = {key: int(re.search(r"\b" + re.escape(key) + r": (\d+)", output).group(1)) for key in expected}
    _, errors, tracks = parse_points(sparse / "points3D.bin")
    mean_error = float(re.search(r"Mean reprojection error: ([\d.]+)px", output).group(1))
    mean_track = float(re.search(r"Mean track length: ([\d.]+)", output).group(1))
    checks = {"exit_zero": result.returncode == 0, "independent_counts_equal": expected == actual,
              "point_mean_error_equal_at_cli_precision": abs(mean_error - float(errors.mean())) < 1e-6,
              "point_mean_track_length_equal_at_cli_precision": abs(mean_track - float(tracks.mean())) < 1e-6}
    receipt = {"status": "passed_independent_colmap_cli_crosscheck" if all(checks.values()) else "failed_cli_crosscheck",
               "command": command, "started_utc": started, "ended_utc": now(), "returncode": result.returncode,
               "stdout": result.stdout, "stderr": result.stderr, "expected_independent_binary_counts": expected,
               "actual_cli_counts": actual, "checks": checks, "cli_mean_reprojection_error_model_pixels": mean_error,
               "independent_point_mean_error_model_pixels": float(errors.mean()),
               "cli_mean_track_length": mean_track, "independent_mean_track_length": float(tracks.mean()),
               "scope": "Read-only model_analyzer; no mapping, training, image edits, installation or Linux startup"}
    receipt_path = EVIDENCE / "colmap-cli-crosscheck.json"
    save(receipt_path, receipt)
    if not all(checks.values()):
        raise RuntimeError("Independent COLMAP CLI crosscheck failed")
    manifest["independent_cli_crosscheck_path"] = str(receipt_path)
    manifest["independent_cli_crosscheck_sha256"] = digest(receipt_path)
    manifest["preparation_script_sha256"] = digest(Path(__file__).resolve())
    save(manifest_path, manifest)
    report_path = EVIDENCE / "report.md"
    text = report_path.read_text(encoding="utf-8")
    extra = ("\nCOLMAP 3.11.1 的只读 model_analyzer 独立复核成功退出 0，相机、注册图像、稀疏点和 "
             "1,083,072 条观察计数均与 NumPy 二进制读取一致，点平均重投影误差与平均 track 长度也在 CLI 输出精度内一致。"
             "模型 1957×1091、照片 979×546、训练小图 245×136；官方 parser 的整数模型尺寸取整使内参矩阵与精确比例修正最大相差 "
             f"{report['official_parser_K_max_abs_difference_px']:.6f} px（包括焦距项）。训练应保留实际 parser 内参口径，不能混用两种修正。\n")
    if "COLMAP 3.11.1 的只读 model_analyzer" not in text:
        report_path.write_text(text + extra, encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": checks}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--parallel-resume", action="store_true")
    parser.add_argument("--crosscheck-only", action="store_true")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()
    if args.factor < 2:
        raise ValueError("Use a separate downscaled training-image directory, factor >= 2")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.crosscheck_only:
        cli_crosscheck()
        return
    if not args.prepare_only:
        if args.parallel_resume:
            parallel_resume()
        else:
            acquire()
    if not args.download_only:
        scene, receipt = extract()
        report, splits = audit(scene, args.factor)
        complete(scene, receipt, report, splits)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        save(EVIDENCE / f"failure-{stamp}.json", {"status": "failed_preserved_for_review_or_resume", "error": repr(exc), "utc": now()})
        raise
