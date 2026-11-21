"""Export a small, portable documentation bundle. No network calls or publishing.

Run from any working directory with the lab Python environment:
    python scripts/export_site.py
    python scripts/export_site.py --check

Only top-level Markdown in knowledge/, curriculum/, experiments/ and the single
manifests/progress.json file are read. Dataset/model directories are never walked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import yaml


LAB = Path(__file__).resolve().parent.parent
EXPORT = LAB / "site-export"
MANIFEST = LAB / "manifests" / "site-export.json"
GROUPS = ("curriculum", "knowledge", "experiments")
REQUIRED = ("id", "slug", "title", "order", "status", "updated", "sources")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FENCE = re.compile(r"(?ms)^\s*(```|~~~).*?^\s*\1[^\n]*(?:\n|$)")
LINK = re.compile(
    r'!?\[[^\]\n]*\]\(\s*(<[^>\n]*>|(?:\\.|[^()\s]|\([^()]*\))+)(?:\s+"[^"]*")?\s*\)'
)
REF_LINK = re.compile(r"(?m)^\s{0,3}\[[^\]]+\]:\s*(<[^>\n]+>|\S+)")
ABS_LOCAL = re.compile(r"(?i)(?:(?<![\w])[a-z]:[\\/]|\\\\[^\\\s]+\\|file://|/(?:Users|home)/)")
BLOCKED_PARTS = {
    "data", "datasets", "raw", "raw-data", "models", "checkpoints", "repos",
    ".venv", ".git", "credentials", "secrets", ".ssh", ".aws",
}
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy", ".ply",
    ".splat", ".bin", ".bag", ".zip", ".tar", ".gz", ".7z", ".pem", ".key",
}
ALLOWED_ARTIFACT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".json", ".txt", ".csv", ".md", ".html"}


class ExportError(ValueError):
    pass


def relative(path: Path) -> str:
    return path.relative_to(LAB).as_posix()


def require_inside(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ExportError("A path or link escapes its permitted directory")
    return resolved


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def portable_text(value: str, *, strict: bool = True) -> str:
    # Keep local source files unchanged. Only exported text becomes portable.
    for spelling in (str(LAB), LAB.as_posix()):
        value = re.sub(re.escape(spelling), ".", value, flags=re.IGNORECASE)
    if ABS_LOCAL.search(value):
        if strict:
            raise ExportError("An exported text field contains an absolute local path outside the lab")
        return "[omitted-local-path]"
    return value


def source_url(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ExportError(f"{label}: sources must be a list of URL strings")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ExportError(f"{label}: invalid or credential-bearing source URL")
    if parsed.query and re.search(r"(?i)(?:token|secret|password|signature|credential)=", parsed.query):
        raise ExportError(f"{label}: signed/private URL is not an exportable source")
    return value


def parse_document(path: Path) -> tuple[dict, str]:
    label = relative(path)
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", text, flags=re.DOTALL)
    if not match:
        raise ExportError(f"{label}: YAML frontmatter is required")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ExportError(f"{label}: invalid YAML frontmatter") from exc
    if not isinstance(metadata, dict):
        raise ExportError(f"{label}: frontmatter must be an object")
    missing = [key for key in REQUIRED if key not in metadata]
    if missing:
        raise ExportError(f"{label}: missing fields {', '.join(missing)}")
    # Export only the public schema, never unknown frontmatter fields.
    metadata = {key: metadata[key] for key in REQUIRED}
    for key in ("id", "slug", "title", "status"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ExportError(f"{label}: {key} must be a nonempty string")
        metadata[key] = portable_text(metadata[key])
    if not SLUG.fullmatch(metadata["slug"]):
        raise ExportError(f"{label}: slug must use lowercase words separated by hyphens")
    if not isinstance(metadata["order"], int) or isinstance(metadata["order"], bool):
        raise ExportError(f"{label}: order must be an integer")
    date = metadata["updated"]
    if isinstance(date, dt.date) and not isinstance(date, dt.datetime):
        date = date.isoformat()
    try:
        if not isinstance(date, str) or dt.date.fromisoformat(date).isoformat() != date:
            raise ValueError
    except ValueError as exc:
        raise ExportError(f"{label}: updated must be YYYY-MM-DD") from exc
    metadata["updated"] = date
    if not isinstance(metadata["sources"], list) or not metadata["sources"]:
        raise ExportError(f"{label}: sources must be a nonempty list of public URLs")
    metadata["sources"] = [source_url(item, label) for item in metadata["sources"]]
    return metadata, text[match.end():].lstrip("\n")


def document_links(body: str, path: Path, known_docs: dict[Path, dict]) -> tuple[str, list[dict]]:
    # Ignore examples inside fenced code, while preserving offsets for rewriting.
    visible = FENCE.sub(lambda item: " " * len(item.group()), body)
    occurrences = list(LINK.finditer(visible)) + list(REF_LINK.finditer(visible))
    occurrences.sort(key=lambda item: item.start(1))
    links, replacements = [], []
    for match in occurrences:
        target = match.group(1).strip("<>")
        if target.startswith("#"):
            continue
        parsed = urlsplit(target)
        is_drive = bool(re.match(r"^[a-zA-Z]:[\\/]", target))
        if parsed.scheme and not is_drive:
            if parsed.scheme not in {"http", "https", "mailto"}:
                raise ExportError(f"{relative(path)}: unsupported link scheme")
            if parsed.scheme != "mailto":
                source_url(target, relative(path))
            continue
        if parsed.netloc and not is_drive:
            raise ExportError(f"{relative(path)}: protocol-relative or network paths are not supported")
        raw_path = unquote(target.split("#", 1)[0].split("?", 1)[0])
        target_path = Path(raw_path)
        if not target_path.is_absolute():
            target_path = path.parent / target_path
        target_path = require_inside(target_path, LAB)
        if not target_path.exists():
            raise ExportError(f"{relative(path)}: missing local link {relative(target_path)}")
        entry = {"target": relative(target_path), "bundled": target_path in known_docs}
        if target_path in known_docs:
            entry["slug"] = known_docs[target_path]["slug"]
            entry["bundle_path"] = "markdown/" + relative(target_path)
        links.append(entry)
        if is_drive or Path(raw_path).is_absolute():
            local_target = Path(os.path.relpath(target_path, path.parent)).as_posix()
            if "#" in target:
                local_target += "#" + target.split("#", 1)[1]
            replacements.append((match.start(1), match.end(1), "<" + local_target + ">"))
    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return portable_text(body), links


def safe_progress_value(value: object) -> object:
    if isinstance(value, str):
        return portable_text(value, strict=False)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [safe_progress_value(item) for item in value]
    if isinstance(value, dict):
        return {
            portable_text(str(key), strict=False): safe_progress_value(item)
            for key, item in value.items()
            if not re.search(r"(?i)token|password|secret|credential|api.?key|private.?key", str(key))
        }
    raise ExportError("progress.json contains an unsupported value")


def safe_artifact(value: object) -> str | None:
    if not isinstance(value, str) or ABS_LOCAL.search(value) or urlsplit(value).scheme:
        return None
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        return None
    if any(part.lower() in BLOCKED_PARTS or part.startswith(".") for part in candidate.parts):
        return None
    if candidate.suffix.lower() in BLOCKED_SUFFIXES or candidate.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES:
        return None
    require_inside(LAB / Path(*candidate.parts), LAB)
    return candidate.as_posix()


def read_progress() -> tuple[dict, list[str]]:
    path = LAB / "manifests" / "progress.json"
    if not path.exists():
        return {"results": [], "next_steps": []}, []
    require_inside(path, LAB)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ExportError("manifests/progress.json exceeds the 2 MiB metadata limit")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ExportError("manifests/progress.json is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("results", []), list):
        raise ExportError("progress.json must be an object with a results array")
    progress = {key: safe_progress_value(raw[key]) for key in ("updated", "current_stage", "next_steps") if key in raw}
    progress.setdefault("next_steps", [])
    if not isinstance(progress["next_steps"], list):
        raise ExportError("progress.next_steps must be an array")
    progress["results"] = []
    warnings = []
    for result in raw.get("results", []):
        if not isinstance(result, dict):
            raise ExportError("Every progress result must be an object")
        clean = {key: safe_progress_value(result[key]) for key in ("id", "title", "status", "metrics", "scope") if key in result}
        artifacts = result.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ExportError("Every progress artifacts field must be an array")
        clean["artifacts"] = []
        for artifact in artifacts:
            safe = safe_artifact(artifact)
            if safe is None:
                warnings.append("Omitted an absolute, unsupported, dataset, model, or credential artifact reference")
            else:
                clean["artifacts"].append(safe)
        progress["results"].append(clean)
    return progress, warnings


def build_bundle() -> tuple[dict[Path, bytes], dict]:
    records: dict[Path, dict] = {}
    bodies: dict[Path, str] = {}
    seen_ids, seen_slugs = set(), set()
    for group in GROUPS:
        folder = require_inside(LAB / group, LAB)
        if not folder.exists():
            continue
        # Deliberately nonrecursive: never enumerate raw datasets or repositories.
        for path in sorted(folder.glob("*.md")):
            path = require_inside(path, folder)
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ExportError(f"{relative(path)} exceeds the 2 MiB document limit")
            metadata, body = parse_document(path)
            if metadata["id"] in seen_ids or metadata["slug"] in seen_slugs:
                raise ExportError(f"{relative(path)}: duplicate document id or slug")
            seen_ids.add(metadata["id"])
            seen_slugs.add(metadata["slug"])
            records[path] = {**metadata, "kind": group, "source_path": relative(path)}
            bodies[path] = body
    if not records:
        raise ExportError("No Markdown documents were found")
    outputs, documents = {}, []
    ordered = sorted(records, key=lambda item: (GROUPS.index(records[item]["kind"]), records[item]["order"], records[item]["slug"]))
    for path in ordered:
        record = dict(records[path])
        body, links = document_links(bodies[path], path, records)
        portable_metadata = {key: record[key] for key in REQUIRED}
        md = "---\n" + yaml.safe_dump(portable_metadata, allow_unicode=True, sort_keys=False).rstrip() + "\n---\n\n" + body.rstrip() + "\n"
        payload = md.encode("utf-8")
        bundle_path = "markdown/" + relative(path)
        outputs[EXPORT / bundle_path] = payload
        record.update({"body_markdown": body, "markdown_path": bundle_path, "local_links": links, "sha256": digest(payload)})
        documents.append(record)
    progress, warnings = read_progress()
    content = {
        "schema_version": 1,
        "exporter_version": "1.0",
        "entry_slug": "day01-local-first-three-hours",
        "documents": documents,
        "progress": progress,
        "results": progress["results"],
        "warnings": sorted(set(warnings)),
    }
    outputs[EXPORT / "content.json"] = json_bytes(content)
    readme = (
        "# Local documentation export\n\n"
        "Generated by scripts/export_site.py. This folder is not a published site.\n\n"
        "- content.json contains ordered documents, public sources, relative links and filtered progress.\n"
        "- markdown/ preserves document bodies with portable lab paths and normalized frontmatter.\n"
        "- Source Markdown remains unchanged. Stable ids/slugs are taken from its frontmatter.\n"
        "- Local links are checked for existence; links outside the Markdown bundle are labeled bundled=false.\n"
        "- Artifact paths are references only. Images, logs, datasets, models and credentials are not copied.\n"
        "- Only listed files belong to the current export; stale unlisted files are not deleted automatically.\n"
        "- Re-run the exporter after changing reports or manifests/progress.json; --check validates without writing.\n"
    )
    outputs[EXPORT / "README.md"] = readme.encode("utf-8")
    file_records = [{"path": relative(path), "bytes": len(data), "sha256": digest(data)} for path, data in sorted(outputs.items())]
    manifest = {
        "schema_version": 1,
        "exporter": "scripts/export_site.py",
        "destination": "site-export",
        "content_path": "site-export/content.json",
        "source_documents": [record["source_path"] for record in documents],
        "progress_source": "manifests/progress.json" if (LAB / "manifests" / "progress.json").exists() else None,
        "document_count": len(documents),
        "result_count": len(progress["results"]),
        "files": file_records,
        "published": False,
        "warnings": content["warnings"],
    }
    outputs[MANIFEST] = json_bytes(manifest)
    return outputs, manifest


def write_owned(path: Path, data: bytes) -> bool:
    # A resolved symlink/junction may not redirect generated files outside EXPORT.
    if path == MANIFEST:
        require_inside(path, LAB / "manifests")
        if path.resolve() != path:
            raise ExportError("The export manifest may not redirect through a symbolic link")
    else:
        require_inside(path, EXPORT)
    require_inside(EXPORT, LAB)
    require_inside(LAB / "manifests", LAB)
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate current inputs and compare with the last export, without writing")
    args = parser.parse_args()
    try:
        outputs, manifest = build_bundle()
        changed = 0
        stale = []
        for path, data in outputs.items():
            if args.check:
                if not path.exists() or path.read_bytes() != data:
                    stale.append(relative(path))
            else:
                changed += write_owned(path, data)
        print(json.dumps({"documents": manifest["document_count"], "results": manifest["result_count"], "mode": "check" if args.check else "export", "files_written": changed, "stale_files": stale, "warnings": manifest["warnings"], "published": False}, ensure_ascii=False, indent=2))
        return 1 if stale else 0
    except (ExportError, OSError, UnicodeError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
