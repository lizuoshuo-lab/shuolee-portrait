"""Run native video intake, original SMPLify-X and static Gaussian delivery."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

from update_leafwoman129_state import ROOT, atomic, read


HERE = Path(__file__).resolve().parent
DISTRO = "Ubuntu-20.04"
LINUX_PYTHON = "/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python"
PYTHON_FLAGS = ["-X", "utf8", "-B", "-u"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linux_path(path):
    path = Path(path).absolute()
    value = path.as_posix()
    for host in ("wsl.localhost", "wsl$"):
        prefix = f"//{host}/{DISTRO}/"
        if value.lower().startswith(prefix.lower()):
            return value[len(prefix) - 1:]
    if re.fullmatch(r"[A-Za-z]:", path.drive):
        return f"/mnt/{path.drive[0].lower()}/{value[3:]}"
    raise ValueError(f"Expected a local drive or {DISTRO} path: {path}")


def linux_command(script, *args):
    return [str(Path(os.environ["SystemRoot"]) / "System32/wsl.exe"), "-d", DISTRO,
            "--exec", LINUX_PYTHON, *PYTHON_FLAGS, linux_path(HERE / script), *map(str, args)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_receipt(path, video_sha):
    from review_full_sequence123 import relocate

    receipt = read(path)
    require(receipt.get("video_sha256") == video_sha and receipt.get("observations_ready") is True,
            f"Native receipt does not belong to this video or observations are incomplete: {path}")
    data = relocate(receipt["data_root"]) / "data"
    require(sha256(data / "inputs.json") == receipt.get("observation_manifest_sha256")
            and sha256(data / "observations.npz") == receipt.get("observations_sha256"),
            f"Native observation files differ from their receipt: {path}")
    return sha256(path)


def fit_receipt(folder, video_sha, native_sha, arc):
    plan_path, result_path = folder / "plan.json", folder / "result.json"
    plan, result = read(plan_path), read(result_path)
    require(plan.get("source_video_sha256") == video_sha,
            f"Fitting archive does not belong to this video: {folder}")
    require(native_sha is None or plan.get("native_receipt_sha256") == native_sha,
            f"Fitting archive does not belong to the supplied native receipt: {folder}")
    require(result.get("fitting_complete") is True and result.get("plan_sha256") == sha256(plan_path),
            f"Fitting archive is incomplete or its plan changed: {folder}")
    require(arc == "half" or plan["view_targets"]["rear"] > 0,
            f"The existing fit has no registered rear arc: {folder}")
    return sha256(result_path)


def delivery_receipt(folder):
    from review_full_sequence123 import relocate

    publication = folder / "delivery/publication.json"
    result = read(publication)
    require(result.get("state") == "delivered_pending_quality_review" and result.get("format_passed") is True
            and result.get("run") == folder.name, f"Missing format-verified delivery: {publication}")
    for artifact, name in (("ply", "point_cloud.ply"), ("camera", "cameras.json"), ("package", "scene.zip")):
        expected = folder / "delivery/step-018000" / name
        actual = relocate(result[f"{artifact}_path"])
        require(actual.resolve() == expected.resolve() and sha256(actual) == result[f"{artifact}_sha256"],
                f"Delivered {artifact} does not match its receipt: {actual}")
    return result


def run(args):
    require(os.name == "nt", "Run this entry point from Windows Python.")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run), "Use a single alphanumeric run name with optional - or _.")
    capture_arcs = args.capture_arcs
    declares_layout = capture_arcs is not None or args.front_last_source_index is not None
    require(not declares_layout or args.fit_dir is None, "Omit capture layout options when reusing a fit.")
    require(args.arc != "full" or args.fit_dir is not None or declares_layout,
            "Fitting a full capture requires --capture-arcs or --front-last-source-index.")
    video = args.video.resolve(strict=True)
    require(video.is_file(), f"Expected an input video file: {video}")
    started = time.monotonic()
    control = ROOT / f"{args.run}-pipeline"
    output = ROOT / args.run
    require(not output.exists(), f"Gaussian output already exists. Use a fresh run name: {output}")
    control.mkdir(exist_ok=False)
    state_path = control / "pipeline.json"
    state = dict(run=args.run, arc=args.arc, source_video=str(video), output=str(output),
                 state="running", current_stage="validate_source", stages={}, quality_accepted=False,
                 cold_start=args.native_receipt is None and args.fit_dir is None)
    native = args.native_receipt or control / "native/result.json"
    fit = (args.fit_dir or control / "fit").resolve()
    arc = ["--arc", args.arc]

    def save():
        state.update(updated=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        atomic(state_path, state)

    def stage(name, command):
        log_path = control / f"{name}.log"
        row = dict(command=command, log=str(log_path), state="running")
        state["stages"][name] = row
        state["current_stage"] = name
        save()
        print(f"{name}: {log_path}", flush=True)
        before = time.monotonic()
        try:
            with log_path.open("xb") as log:
                completed = subprocess.run(command, cwd=HERE, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            row["exit_code"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"{name} exited with code {completed.returncode}. See {log_path}")
            row["state"] = "command_completed"
        finally:
            row["seconds"] = time.monotonic() - before
            save()

    save()
    try:
        video_sha = sha256(video)
        state["source_video_sha256"] = video_sha
        if capture_arcs is not None:
            capture_arcs = capture_arcs.resolve(strict=True)
            layout = read(capture_arcs)
            digest = layout.get("video_sha256") if isinstance(layout, dict) else None
            require(isinstance(digest, str) and digest.lower() == video_sha,
                    "Capture layout must identify this video SHA256.")
            state["capture_arcs"] = str(capture_arcs)
            state["capture_arcs_sha256"] = sha256(capture_arcs)
        native_sha = None
        if args.fit_dir is None and args.native_receipt is None:
            data = f"/opt/gs-slam-lab/shuolee-cross-subject-124/{args.run}-pipeline"
            state["native_data_root"] = data
            stage("intake", linux_command("run_fixed_smplify175.py", "intake", linux_path(video),
                                          linux_path(native.parent), data))
        if args.fit_dir is None or args.native_receipt is not None:
            native = native.resolve(strict=True)
            state["current_stage"] = "verify_native_receipt"
            native_sha = native_receipt(native, video_sha)
            state["native_receipt"] = str(native)
            state["native_receipt_sha256"] = native_sha
            if args.native_receipt is not None:
                state["stages"]["intake"] = dict(state="reused", seconds=0)

        if args.fit_dir is None:
            command = [sys.executable, *PYTHON_FLAGS, str(HERE / "run_fixed_smplify175.py"),
                       "fit", str(native), str(fit), "--automatic", *arc]
            if capture_arcs is not None:
                command.extend(["--capture-arcs", str(capture_arcs)])
            elif args.front_last_source_index is not None:
                command.extend(["--front-last-source-index", str(args.front_last_source_index)])
            stage("fit", command)
        else:
            state["stages"]["fit"] = dict(state="reused", seconds=0)
            if args.native_receipt is None:
                state["stages"]["intake"] = dict(state="included_in_reused_fit", seconds=0)
        state["current_stage"] = "verify_fit_receipt"
        state["fit_result_sha256"] = fit_receipt(fit, video_sha, native_sha, args.arc)
        state["fit_dir"] = str(fit)

        stage("gaussian_and_delivery", linux_command("prepare_reviewed_sh0.py", "run", linux_path(fit), args.run, *arc))
        state["current_stage"] = "verify_delivery"
        state["delivery"] = delivery_receipt(output)
        state.update(state="delivered_pending_quality_review", current_stage=None)
        save()
        print(f"DELIVERED_PENDING_QUALITY_REVIEW: {output / 'delivery/step-018000/scene.zip'}", flush=True)
        print(f"Pipeline receipt: {state_path}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        failure = output / "failure.json"
        row = state["stages"].get(state["current_stage"])
        if row is not None:
            row["state"] = "failed"
        recovery = dict(use_fresh_run_name=True)
        if state.get("fit_dir"):
            recovery["reuse_arguments"] = ["--fit-dir", str(fit)]
            if (output / "process-exit.json").is_file():
                recovery["delivery_retry_command"] = linux_command(
                    "prepare_reviewed_sh0.py", "deliver", linux_path(fit), args.run, *arc)
        elif state.get("native_receipt"):
            recovery["reuse_arguments"] = ["--native-receipt", str(native)]
            if capture_arcs is not None:
                recovery["reuse_arguments"].extend(["--capture-arcs", str(capture_arcs)])
            elif args.front_last_source_index is not None:
                recovery["reuse_arguments"].extend(["--front-last-source-index", str(args.front_last_source_index)])
        state.update(state="failed", error=repr(error), recovery=recovery,
                     child_failure_record=str(failure) if failure.is_file() else None)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Original local MP4")
    parser.add_argument("run", help="New local run name")
    parser.add_argument("--arc", required=True, choices=("half", "full"))
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--front-last-source-index", type=int, help="Last original frame index in a contiguous front arc")
    layout.add_argument("--capture-arcs", type=Path, help="Video-bound JSON with disjoint front/rear source intervals")
    parser.add_argument("--native-receipt", type=Path, help="Reuse a completed native observations result.json for this MP4")
    parser.add_argument("--fit-dir", type=Path, help="Reuse an already completed original SMPLify-X fit directory for this MP4")
    args = parser.parse_args()
    if args.front_last_source_index is not None and args.front_last_source_index < 0:
        parser.error("--front-last-source-index must be nonnegative")
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
