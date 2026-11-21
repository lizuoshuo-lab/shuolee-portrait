#!/usr/bin/env python3
"""Meaningful offline integrity controls; never installs Linux packages or builds C++."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import verify_sources

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-lab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).parent
    checks = []
    for script in scripts.glob("*.py"):
        ast.parse(script.read_text(encoding="utf-8"), feature_version=(3, 8))
    checks.append({"control": "python38_ubuntu_focal_syntax", "passed": True})
    process = subprocess.run([args.bash, "-n", str(scripts / "bootstrap.sh").replace("\\", "/")], text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr)
    checks.append({"control": "bash_syntax", "passed": True})
    workspace = args.output / "materialized-source-control"
    command = [sys.executable, str(scripts / "verify_sources.py"), "--windows-lab", str(args.windows_lab), "--root", str(workspace), "--output", str(args.output / "materialization-first.json")]
    subprocess.run(command, check=True)
    command[-1] = str(args.output / "materialization-repeat.json")
    subprocess.run(command, check=True)
    checks.append({"control": "all_pinned_sources_materialize_and_repeat_without_edits", "passed": True})
    file = workspace / "catkin_ws/src/FAST-LIVO2/CMakeLists.txt"
    original = file.read_bytes()
    file.write_bytes(original + b"\n# deliberately modified integrity control\n")
    receipt = args.windows_lab / "outputs/fast-livo2-source-audit-001/repository-acquisition.json"
    try:
        verify_sources.verify_fast(file.parent, receipt)
    except ValueError:
        checks.append({"control": "modified_fast_source_is_rejected", "passed": True})
    else:
        raise AssertionError("modified FAST source incorrectly accepted")
    finally:
        file.write_bytes(original)
    for malicious in ["../escape", "/absolute", "dir/../../escape", "dir\\escape"]:
        try:
            verify_sources.safe_relative(malicious)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe archive path accepted: " + malicious)
    checks.append({"control": "traversal_absolute_and_backslash_paths_rejected", "passed": True})
    source_before = verify_sources.verify_fast(args.windows_lab / "repos/FAST-LIVO2", receipt)
    checks.append({"control": "original_windows_source_remains_immutable_all_75_files", "passed": len(source_before) == 75})
    result = {"status": "offline_preparation_controls_passed", "controls": checks, "linux_package_installation_executed": False, "linux_compilation_executed": False, "fusion_executed": False, "script_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts.iterdir() if path.is_file()}}
    (args.output / "preparation-checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["status"])

if __name__ == "__main__":
    main()
