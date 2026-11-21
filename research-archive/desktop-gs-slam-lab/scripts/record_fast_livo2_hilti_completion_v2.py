"""Append one hash-bound, actually accepted Hilti result to local progress.

--validate-only never writes. --execute is for root AFTER the real evaluator
and summarizer have completed. Frozen sources/candidate/manifest are read only.
Existing result ids are preserved; an identical existing candidate is a no-op,
and a differing record with the same id is rejected. A unique original-byte
backup precedes any progress replacement. Exclusive cooperative lock plus hash
checks before each atomic replace reject observed concurrent modifications.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import types
import uuid


LAB_DEFAULT = Path(__file__).resolve().parents[1]
GROUP = "outputs/fast-livo2-hilti-gt-001"
ID = "fast-livo2-hilti-gt-001"
STATUS = "completed_local_two_mode_dense_reference_comparison"
SUMMARIZER_SHA = "3ad7b3c2dc3e618523e7709a9690ea7091e74d38628d2eb754ee2cf598879b5d"
FIXED_GT_FILE = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\fast-livo2-data\hilti2022-recovery-003-exp14\ground_truth\exp14_basement_2_imu.txt")
FIXED_GT_SHA = "dec7b20cc11002482ccb416a6e1a1c410d385b843504d3aedf7b5956746af1b6"
BASE_IDS = frozenset("""
splatam-common-gt-view-map-evaluation-001
splatam-official-local-xyz-full-001
splatam-official-local-desk-640-full-001
splatam-official-local-full-001
splatam-official-local-smoke-001
splatam-source-and-data-001
fast-livo2-local-bright-replay-002
fast-livo2-official-source-001
fast-livo2-actual-stereo-calibration-001
fast-livo2-real-bright-timing-002
fast-livo2-position-evaluator-author-saved-001
official-image-fit-001
tiny-nerf-gsplat-001
tum-geometry-001
gaussian-count-study-001
tum-rgbd-gsplat-001
tum-rgbd-rgbonly-001
tiny-fixed-map-pose-002
tum-fixed-map-pose-001
tum-online-study-001
tum-online-v2-transfer-study-001
tum-icp-transfer100-001
renderer-feature-semantics-002
tum-online-v3-icpprefix-controls
tum-icp-full549-001
tum-pose-gradient-diagnostic-001
tum-online-v4-prior-controls
tum-pose-prior-context-replay-001
tum-xyz-v4-fixed-configuration-transfer
tum-v4-three-seed-study
tum-v4-known-camera-final-map-quality
full-icp-two-sequence-controls
rgbd-photogeometric-fixed-real-sequence-controls
rgbd-last-accepted-reference-recovery-controls
affine-brightness-eight-dimensional-synthetic-controls
full-sequence-online-v4-original-three-plus-recovery
affine-brightness-fixed-real-CPU-controls
full-sequence-known-camera-seen-map-diagnostics
schur-pose-brightness-fixed-linear-observability
same-map-saved-camera-seen-view-consistency
rgbd-affine-full-lastaccepted-controls-001
rgbd-affine-adjacent-icp-fallback-full-controls-001
gaussian-map-pose-correction-cpu-controls-001
gaussian-global-gauge-renderer-controls-001
splatam-ablation-repeat-001
""".split())
NEXT_STEP = "FAST-LIVO2 的物理标定、同步与更多场景验证仍需独立完成；exp14 共同 dense IMU 开发参考对照已有真实验收。"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode(raw):
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON object key: " + key)
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8-sig"), parse_constant=constant, object_pairs_hook=pairs)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def local(lab, value):
    path = Path(str(value).replace("\\", "/"))
    result = (path if path.is_absolute() else lab / path).resolve()
    require(result.is_relative_to(lab), "Evidence path outside LAB: " + str(value))
    return result


def verified_completion(lab):
    paths = {"candidate": local(lab, GROUP + "/progress-record-candidate.json"),
             "manifest": local(lab, "manifests/fast-livo2-hilti-gt.json"),
             "execution": local(lab, GROUP + "/evaluation/execution.json"),
             "seal": local(lab, GROUP + "/evaluation/preregistration.json"),
             "metrics": local(lab, GROUP + "/evaluation/metrics.json"),
             "plan": local(lab, GROUP + "/evaluation-plan.json"),
             "summarizer": local(lab, "scripts/summarize_fast_livo2_hilti_v2.py")}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    before = {key: digest(value) for key, value in raw.items()}
    data = {key: decode(value) for key, value in raw.items() if key != "summarizer"}
    candidate, manifest = data["candidate"], data["manifest"]
    require(before["summarizer"] == SUMMARIZER_SHA == manifest["summarizer_source_sha256"],
            "Frozen strict summarizer source/manifest SHA mismatch")
    # Execute only the reviewed module's definitions; main is not invoked and
    # bytecode cache writing is disabled. summarize() performs read-only gates.
    module = types.ModuleType("_frozen_hilti_read_only_completion_validation")
    module.__file__ = str(paths["summarizer"])
    old_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        exec(compile(raw["summarizer"].decode("utf-8"), str(paths["summarizer"]), "exec"), module.__dict__)
        metric, modes, evidence, execution, plan_sha, seal_sha = module.summarize(lab, paths["execution"], paths["seal"])
    finally:
        sys.dont_write_bytecode = old_bytecode_setting
    require(execution == data["execution"] and metric == data["metrics"], "Actual evaluator inputs changed during validation")
    shared = {key: metric[key] for key in ("dense_GT_sha256", "dense_reference_rows", "common_reference_rows",
                                          "common_reference_fraction", "common_reference_indices",
                                          "common_start_timestamp_ns", "common_end_timestamp_ns")}
    require(manifest["id"] == candidate["id"] == ID and manifest["status"] == candidate["status"] == STATUS,
            "Candidate/manifest are not the actually accepted Hilti result")
    require(manifest["modes"] == modes and manifest["common_reference"] == shared and manifest["evidence"] == evidence,
            "Manifest raw run, trajectory, parameter or source evidence differs from current strict acceptance")
    require(manifest["source_commit"] == module.COMMIT and manifest["dataset"] == "Hilti 2022 exp14_basement_2"
            and manifest["local_only"] is True and manifest["progress_json_edited"] is False,
            "Manifest scope/source changed")
    require(manifest["updated_utc"] == execution["finished_utc"]
            and manifest["actual_evaluator_child_exit_code"] == execution["child_exit_code"] == 0
            and manifest["evaluation_metrics_sha256"] == before["metrics"]
            and manifest["evaluation_plan_sha256"] == plan_sha == before["plan"]
            and manifest["preregistration_sha256"] == seal_sha == before["seal"]
            and manifest["evaluator_source_sha256"] == metric["evaluator_source_sha256"]
            and manifest["protocol"] == metric["protocol"] and manifest["limitations"] == metric["limits"],
            "Manifest actual exit/protocol/hash binding failed")
    require(module.GT_FILE == FIXED_GT_FILE and module.GT_SHA == FIXED_GT_SHA,
            "Reviewed summarizer's exact external GT identity changed")
    require(manifest["dense_reference_sha256"] == module.GT_SHA
            and manifest["dense_reference_file"] == str(module.GT_FILE), "Dense development reference binding changed")
    expected = {"id": ID, "title": "FAST-LIVO2 本机 Hilti exp14 双模式与共同 IMU 开发真值对照", "status": STATUS,
                "metrics": {"actual_runs": modes, "common_reference": shared, "actual_evaluator_child_exit_code": 0,
                            "evaluation_metrics_sha256": before["metrics"], "evaluation_plan_sha256": plan_sha,
                            "preregistration_sha256": seal_sha, "evaluator_source_sha256": metric["evaluator_source_sha256"]},
                "artifacts": ["manifests/fast-livo2-hilti-gt.json", GROUP + "/evaluation/metrics.json",
                              GROUP + "/evaluation/execution.json", GROUP + "/evaluation/preregistration.json",
                              GROUP + "/evaluation/paired-errors.csv", GROUP + "/evaluation/paired-reference-errors.png",
                              GROUP + "/evaluation-plan.json"],
                "scope": "完整官方 exp14 输入，串行 0.5× LIO/LIVO 本机回放；共同原始 dense IMU 开发参考时刻，"
                         "各模式事后 SE3 尺度1对齐。主轨迹为 LiDAR 后、当前 VIO 前；LIVO 含既往视觉历史。"
                         "无杆臂或时间偏移拟合，无稀疏测量成绩，不证明物理标定、同步、实时性或跨场景优越性。"}
    require(candidate == expected, "Candidate differs from the frozen summarizer's exact hash-bound record")
    bindings = {path.relative_to(lab).as_posix(): before[key] for key, path in paths.items()}
    for relative, binding in evidence.items():
        path = local(lab, relative)
        name = path.relative_to(lab).as_posix()
        expected_hash = binding["sha256"]
        require(re.fullmatch("[0-9a-f]{64}", expected_hash), "Malformed accepted evidence SHA: " + name)
        require(name not in bindings or bindings[name] == expected_hash, "Conflicting evidence binding: " + name)
        bindings[name] = expected_hash
    for path, expected_hash in bindings.items():
        require(sha(local(lab, path)) == expected_hash, "Completion input changed before merge: " + path)
    require(sha(FIXED_GT_FILE) == FIXED_GT_SHA, "Exact external dense reference changed before merge")
    return candidate, bindings


def propose(progress, candidate):
    require(isinstance(progress, dict) and isinstance(progress.get("results"), list), "Current progress results schema mismatch")
    require(isinstance(progress.get("updated"), str) and isinstance(progress.get("current_stage"), str)
            and isinstance(progress.get("next_steps"), list) and all(isinstance(v, str) for v in progress["next_steps"]),
            "Current updated/current_stage/next_steps schema mismatch")
    dt.date.fromisoformat(progress["updated"])
    require(all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in progress["results"]), "Malformed existing result")
    ids = [item["id"] for item in progress["results"]]
    require(len(ids) == len(set(ids)) and BASE_IDS.issubset(ids), "Existing 45 result identities missing or duplicated")
    existing = [item for item in progress["results"] if item["id"] == ID]
    if existing:
        require(existing[0] == candidate, "Same id has a differing existing record; refusing to overwrite")
        return None
    after = copy.deepcopy(progress)
    after["results"].append(copy.deepcopy(candidate))
    after["updated"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    after["current_stage"] = "FAST_LIVO2_Hilti_exp14_dense_reference_pair_completed"
    # Only this now-completed exp14 comparison is removed. Other plans and
    # slam_status are preserved; completion of other projects is not inferred.
    after["next_steps"] = [step for step in progress["next_steps"]
                           if not ("FAST-LIVO2" in step and "Hilti exp14" in step
                                   and any(word in step for word in ("两模式对照", "双模式对照", "两模式回放与对照", "paired comparison")))]
    if NEXT_STEP not in after["next_steps"]:
        after["next_steps"].append(NEXT_STEP)
    require(after["results"][:-1] == progress["results"], "Existing results were changed")
    allowed = {"updated", "current_stage", "next_steps", "results"}
    require(all(after[key] == value for key, value in progress.items() if key not in allowed), "Unrelated progress fields changed")
    return after


def save_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=LAB_DEFAULT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-progress-sha256", help="Optional root-observed SHA; mismatch rejects before any backup/write")
    args = parser.parse_args()
    lock, lock_stream, attempt, info = None, None, None, None
    try:
        lab = args.lab.resolve()
        candidate, bindings = verified_completion(lab)
        target = local(lab, "manifests/progress.json")
        original = target.read_bytes()
        before_sha = digest(original)
        require(sha(target) == before_sha, "Progress changed while reading")
        if args.expected_progress_sha256 is not None:
            require(re.fullmatch("[0-9a-f]{64}", args.expected_progress_sha256)
                    and before_sha == args.expected_progress_sha256, "Root-observed progress SHA mismatch")
        progress = decode(original)
        proposed = propose(progress, candidate)
        if proposed is None:
            print(json.dumps({"status": "already_recorded_identically_no_write", "id": ID, "progress_sha256": before_sha}))
            return 0
        if args.validate_only:
            print(json.dumps({"status": "verified_ready_no_files_written", "id": ID, "existing_results": len(progress["results"]),
                              "proposed_results": len(proposed["results"]), "progress_sha256": before_sha,
                              "completion_bindings_sha256": bindings, "changed_fields": ["updated", "current_stage", "next_steps", "results"]}))
            return 0
        # Cooperative writer lock is exclusive and owned by this invocation.
        # The content hash is checked again immediately before every replace.
        lock = local(lab, "manifests/.fast-livo2-hilti-completion.lock")
        lock_stream = lock.open("x", encoding="utf-8")
        lock_stream.write(json.dumps({"pid": os.getpid(), "utc": utc(), "progress_sha256": before_sha}) + "\n")
        lock_stream.flush()
        require(sha(target) == before_sha, "Concurrent progress change before backup; rejected")
        token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        attempt = local(lab, GROUP + "/progress-merges/" + token)
        attempt.mkdir(parents=True, exist_ok=False)
        backup = attempt / "progress-before.json"
        save_new(backup, original)
        require(sha(backup) == before_sha, "Independent original-byte backup failed")
        info = {"status": "prepared_merge_backup_verified", "created_utc": utc(), "progress_before_sha256": before_sha,
                "original_results_count": len(progress["results"]), "preserved_result_ids": [row["id"] for row in progress["results"]],
                "backup": backup.relative_to(lab).as_posix(), "backup_sha256": sha(backup),
                "completion_bindings_sha256": bindings, "merger_source_sha256": sha(Path(__file__)),
                "fixed_dense_reference": {"path": str(FIXED_GT_FILE), "sha256": FIXED_GT_SHA},
                "candidate_id": ID, "changed_fields": ["updated", "current_stage", "next_steps", "results"]}
        save_new(attempt / "started.json", encoded(info))
        proposed_raw = encoded(proposed)
        temp = target.with_name(target.name + "." + token + ".tmp")
        save_new(temp, proposed_raw)
        for retry in range(40):
            require(sha(target) == before_sha, "Concurrent progress change before atomic replace; rejected")
            for path, expected in bindings.items():
                require(sha(local(lab, path)) == expected, "Completion evidence changed during merge: " + path)
            require(sha(FIXED_GT_FILE) == FIXED_GT_SHA, "Exact external dense reference changed during merge")
            try:
                temp.replace(target)
                break
            except PermissionError:
                if retry == 39:
                    raise
                time.sleep(.25)
        committed_sha = digest(proposed_raw)
        require(sha(target) == committed_sha and decode(target.read_bytes()) == proposed, "Atomic commit readback mismatch")
        info.update(status="actual_progress_result_appended", finished_utc=utc(), progress_after_sha256=committed_sha,
                    final_results_count=len(proposed["results"]), all_existing_results_and_other_fields_preserved=True)
        save_new(attempt / "receipt.json", encoded(info))
        print(json.dumps({"status": info["status"], "progress_sha256": committed_sha, "backup": str(backup),
                          "actual_merge_receipt": str(attempt / "receipt.json"), "results": len(proposed["results"])}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        if attempt is not None:
            failure = dict(info or {}, status="progress_merge_rejected_or_failed", error=str(error), observed_utc=utc())
            if not (attempt / "receipt.json").exists():
                save_new(attempt / "receipt.json", encoded(failure))
        print("Hilti progress merge rejected: " + str(error), file=sys.stderr)
        return 1
    finally:
        if lock_stream is not None:
            lock_stream.close()
            # Remove only this invocation's newly created cooperative lock.
            if lock is not None:
                lock.unlink()


if __name__ == "__main__":
    sys.exit(main())
