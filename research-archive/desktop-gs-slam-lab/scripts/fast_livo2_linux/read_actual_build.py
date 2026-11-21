#!/usr/bin/env python3
"""Independent archived-log readback; no Linux command or installation is invoked."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-lab", type=Path, required=True)
    args = parser.parse_args()
    out = args.windows_lab / "outputs/fast-livo2-linux-bootstrap-001"
    first = out / "linux-executions/20261010T034800Z-408"
    built = out / "linux-executions/20261010T035029Z-372"
    passed = out / "linux-executions/20261010T035957Z-309"
    first_log = (first / "bootstrap.log").read_text(encoding="utf-8")
    build_log = (built / "bootstrap.log").read_text(encoding="utf-8")
    assert "gpg: Fatal: /root/.gnupg: directory does not exist!" in first_log
    assert (first / "status.tsv").read_text().strip() == "failed\tros_signing_key\t2"
    assert "100% tests passed, 0 tests failed out of 6" in build_log
    assert "[100%] Built target fastlivo_mapping" in build_log
    assert (built / "status.tsv").read_text().strip() == "failed\tenvironment_acceptance\t1"
    assert (passed / "status.tsv").read_text().strip() == "passed\tverify\t0"
    acceptance = json.loads((passed / "environment-acceptance.json").read_text())
    assert acceptance["status"] == "compiled_environment_acceptance_passed"
    assert acceptance["ubuntu"] == "20.04" and acceptance["ros"] == "noetic"
    assert not acceptance["fusion_started"] and not acceptance["local_trajectory_measured"]
    ldd = (passed / "fastlivo-ldd.txt").read_text()
    assert "not found" not in ldd
    first_binary = (built / "fastlivo-binary.sha256").read_text().split()[0]
    final_binary = (passed / "fastlivo-binary.sha256").read_text().split()[0]
    assert first_binary == final_binary
    versions = (passed / "opencv-pcl-versions.txt").read_text().splitlines()
    assert versions == ["4.2.0", "1.10.0"]
    nodes = (passed / "avia-launch-nodes.txt").read_text()
    assert "laserMapping" in nodes and "republish" in nodes
    artifacts = [first / "bootstrap.log", first / "status.tsv", built / "bootstrap.log", built / "status.tsv", passed / "bootstrap.log", passed / "status.tsv", passed / "environment-acceptance.json", passed / "fastlivo-binary.sha256", passed / "fastlivo-ldd.txt", passed / "opencv-pcl-versions.txt", passed / "pcl-opencv-package-versions.txt", passed / "avia-launch-nodes.txt", passed / "avia-launch-params.yaml"]
    receipt = {"status": "actual_linux_build_and_environment_acceptance_independent_readback_passed", "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "scope": "archived real Linux execution logs, receipts and binary hashes; no WSL invocation by this readback", "successful_compile_run": built.name, "successful_acceptance_run": passed.name, "first_failures_preserved": [{"run": first.name, "step": "ros_signing_key", "exit_code": 2, "cause": "missing GnuPG home directory"}, {"run": built.name, "step": "environment_acceptance", "exit_code": 1, "cause": "focal pkg-config module is pcl_common-1.10 rather than pcl_common"}], "acceptance": acceptance, "sophus_ctest": {"passed": 6, "failed": 0}, "opencv_version": versions[0], "pcl_version": versions[1], "binary_sha256": final_binary, "binary_unchanged_between_compile_and_verify": True, "missing_dynamic_libraries": [], "fusion_runtime_measured": False, "artifacts": [{"path": str(file.relative_to(out)).replace("\\", "/"), "bytes": file.stat().st_size, "sha256": sha(file)} for file in artifacts]}
    target = out / "actual-build-readback-001.json"
    if target.exists():
        raise FileExistsError("Preserve prior readback rather than overwrite it")
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["status"])

if __name__ == "__main__":
    main()
