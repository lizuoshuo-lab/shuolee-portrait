"""Inventory declared local evidence; never parse metrics or scientific arrays."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


LAB = Path(__file__).resolve().parents[1]
PROGRESS = "manifests/progress.json"
OUTPUT = "manifests/local-evidence-index.json"
SOURCE = "scripts/build_local_evidence_index.py"


def resolve_local(declared):
    if not isinstance(declared, str) or not declared:
        raise ValueError("Artifact paths must be nonempty relative strings")
    path = Path(declared)
    if path.is_absolute() or path.drive or "://" in declared:
        raise ValueError("Only relative local artifact paths are allowed")
    resolved = (LAB / path).resolve()
    try:
        relative = resolved.relative_to(LAB).as_posix()
    except ValueError:
        raise ValueError("Artifact resolves outside the lab") from None
    return resolved, relative


def file_record(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise RuntimeError("A declared file changed during inventory")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def main():
    output_path, _ = resolve_local(OUTPUT)
    if output_path.exists():
        raise FileExistsError("Inventory already exists; no overwrite allowed")
    progress_path, _ = resolve_local(PROGRESS)
    progress_bytes = progress_path.read_bytes()
    progress_sha = hashlib.sha256(progress_bytes).hexdigest()
    manifest = json.loads(progress_bytes.decode("utf-8-sig"))
    entries = manifest["results"]
    if not isinstance(entries, list):
        raise ValueError("Progress results must be a list")
    results, files, missing, references = [], {}, [], 0
    for entry in entries:
        result = {key: entry[key] for key in ("id", "title", "status", "scope")}
        declared_paths = entry["artifacts"]
        if not isinstance(declared_paths, list):
            raise ValueError("Declared artifacts must be a list")
        result["artifacts"] = declared_paths.copy()
        results.append(result)
        for declared in declared_paths:
            references += 1
            path, relative = resolve_local(declared)
            if relative == OUTPUT:
                raise ValueError("Cannot bind the inventory to its own output")
            if relative not in files:
                exists = path.is_file()
                record = {"declared_paths": [], "is_file": exists}
                if exists:
                    record.update(file_record(path))
                else:
                    record["reason"] = "missing_or_not_a_file"
                    missing.append(relative)
                files[relative] = record
            if declared not in files[relative]["declared_paths"]:
                files[relative]["declared_paths"].append(declared)
    if progress_path.read_bytes() != progress_bytes:
        raise RuntimeError("Progress manifest changed during inventory")
    source_path, _ = resolve_local(SOURCE)
    index = {
        "schema_version": 1,
        "scope": "Local file inventory only; no metrics recomputation or scientific validation.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": {"path": PROGRESS, "bytes": len(progress_bytes), "sha256": progress_sha},
        "builder_source": {"path": SOURCE, **file_record(source_path)},
        "counts": {"results": len(results), "declared_artifact_references": references,
                   "unique_resolved_artifacts": len(files), "missing_or_not_files": len(missing)},
        "results": results,
        "files": files,
        "missing_or_not_files": missing,
    }
    payload = (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with output_path.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"output": OUTPUT, "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "source_sha256": index["builder_source"]["sha256"],
                      "progress_sha256": progress_sha, **index["counts"],
                      "missing_or_not_files": missing}, ensure_ascii=True))


if __name__ == "__main__":
    main()
