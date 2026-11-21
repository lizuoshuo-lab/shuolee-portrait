"""Preserve local experiment source bytes by SHA256 before future edits."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "manifests" / "source-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted((ROOT / "scripts").glob("*.py")):
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        archived = directory / digest / source.name
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            if archived.read_bytes() != raw:
                raise ValueError(f"Hash archive conflict: {archived}")
        else:
            with archived.open("xb") as stream:
                stream.write(raw)
        records.append({"source": source.relative_to(ROOT).as_posix(), "sha256": digest,
                        "archive": archived.relative_to(ROOT).as_posix(), "bytes": len(raw)})
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "top-level local Python experiment/helper/export sources; versioned archives append only",
                "restore_note": "Restore a desired source version into the scripts directory; archived ROOT paths are not meant to run in place.",
                "files": records}
    (ROOT / "manifests" / "source-snapshots.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"archived_sources": len(records), "manifest": "manifests/source-snapshots.json"}))


if __name__ == "__main__":
    main()
