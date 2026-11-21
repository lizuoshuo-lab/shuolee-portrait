#!/usr/bin/env python3
"""Verify pinned sources and materialize them in a Linux filesystem without edits."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

FAST_COMMIT = "0d2c0346107b75b59934975adec9a6eeeb913c64"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe source path: " + name)
    return path

def verify_fast(source, receipt):
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record.get("commit") != FAST_COMMIT or record.get("status") != "passed_immutable_official_archive_all_blobs_match_tree":
        raise ValueError("FAST-LIVO2 acquisition receipt is not the expected verified commit")
    checks = []
    for row in record["files"]:
        rel = safe_relative(row["path"])
        file = source.joinpath(*rel.parts)
        raw = file.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if actual_sha != row["sha256"] or actual_blob != row["expected_git_sha"] or len(raw) != row["bytes"]:
            raise ValueError("FAST-LIVO2 source mismatch: " + row["path"])
        checks.append({"path": row["path"], "sha256": actual_sha, "git_blob_sha1": actual_blob, "bytes": len(raw)})
    return checks

def materialize_fast(source, receipt, target):
    checks = verify_fast(source, receipt)
    if target.exists():
        verify_fast(target, receipt)
    else:
        target.mkdir(parents=True)
        for row in checks:
            rel = safe_relative(row["path"])
            dest = target.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.joinpath(*rel.parts), dest)
        verify_fast(target, receipt)
    return {"commit": FAST_COMMIT, "files_verified": len(checks), "source": str(source), "target": str(target), "source_edits": False, "files": checks}

def materialize_dependency(dep, cache, target, local_cache):
    archive = cache / (dep["name"] + ".zip")
    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        cached = local_cache / dep["cached_file"]
        if cached.exists() and digest(cached) == dep["sha256"]:
            shutil.copyfile(cached, archive)
        else:
            request = urllib.request.Request(dep["url"], headers={"User-Agent": "GS-SLAM-Lab pinned dependency bootstrap/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                archive.write_bytes(response.read())
    if digest(archive) != dep["sha256"]:
        raise ValueError("pinned dependency archive SHA256 mismatch: " + dep["name"])
    files = []
    with zipfile.ZipFile(archive) as zf:
        invalid = zf.testzip()
        if invalid:
            raise ValueError("dependency ZIP CRC failed: " + invalid)
        for info in zf.infolist():
            rel = safe_relative(info.filename)
            if rel.parts[0] != dep["archive_prefix"]:
                raise ValueError("unexpected dependency archive root")
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("dependency symlink entry is not supported")
            local = PurePosixPath(*rel.parts[1:])
            dest = target.joinpath(*local.parts)
            raw = zf.read(info)
            expected = hashlib.sha256(raw).hexdigest()
            if dest.exists():
                if digest(dest) != expected:
                    raise ValueError("existing dependency source edited: " + str(dest))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
            files.append({"path": str(local), "bytes": len(raw), "sha256": expected})
    return {"name": dep["name"], "repository": dep["repository"], "commit": dep["commit"], "archive_sha256": dep["sha256"], "files_verified": len(files), "source_edits": False, "files": files}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-lab", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lab = args.windows_lab
    receipt = lab / "outputs/fast-livo2-source-audit-001/repository-acquisition.json"
    source = lab / "repos/FAST-LIVO2"
    results = {"status": "source_verification_only", "fast_livo2": {"commit": FAST_COMMIT, "files": verify_fast(source, receipt)}, "compilation_executed": False}
    if not args.verify_only:
        if not args.root:
            raise ValueError("--root is required for materialization")
        lock = json.loads((Path(__file__).parent / "dependency-lock.json").read_text(encoding="utf-8"))
        results["fast_livo2"] = materialize_fast(source, receipt, args.root / "catkin_ws/src/FAST-LIVO2")
        results["dependencies"] = []
        for dep in lock["dependencies"]:
            target = args.root / dep["destination"]
            results["dependencies"].append(materialize_dependency(dep, args.root / "download-cache", target, lab / "outputs/fast-livo2-linux-bootstrap-001/official-sources"))
        results["status"] = "pinned_source_materialization_passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(results["status"])

if __name__ == "__main__":
    main()
