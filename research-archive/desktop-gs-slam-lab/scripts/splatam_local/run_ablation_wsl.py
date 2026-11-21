"""Windows entry point for isolated official SplaTAM runs in Ubuntu WSL2.

Full data, environment, checkpoints and model stay on this computer. After the
actual run, mirror its final model/config/logs for local independent evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

LAB = Path(__file__).resolve().parents[2]
ROOT = "/opt/gs-slam-lab/splatam"
UNC = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\splatam")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=20, help="-1 for the complete associated sequence")
    p.add_argument("--sequence", choices=("freiburg1_desk", "freiburg1_xyz", "freiburg1_desk2", "freiburg1_room"), default="freiburg1_desk")
    p.add_argument("--width", type=int, choices=(320, 640), default=320)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracking-loss", choices=("rgbd", "rgb_only", "depth_only"), default="rgbd")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.frames != -1 and args.frames < 2:
        p.error("frames must be >=2, or -1")
    if args.seed < 0:
        p.error("seed must be nonnegative")
    python_ready = subprocess.run(["wsl.exe", "-d", "Ubuntu-20.04", "-u", "root", "--exec", "test", "-x", ROOT + "/.venv/bin/python"], capture_output=True)
    if not (UNC / "env.sh").is_file() or python_ready.returncode != 0:
        raise SystemExit("Isolated WSL environment is not ready; inspect the bootstrap receipt first.")
    native_script = UNC / "scripts/run_ablation_official.py"
    native_script.parent.mkdir(exist_ok=True)
    shutil.copy2(LAB / "scripts/splatam_local/run_ablation_official.py", native_script)
    wrapper_source = Path(__file__).read_bytes()
    command = (f"source {ROOT}/env.sh && {ROOT}/.venv/bin/python {ROOT}/scripts/run_ablation_official.py "
               f"--root {ROOT} --frames {args.frames} --sequence {args.sequence} --width {args.width} --seed {args.seed} --tracking-loss {args.tracking_loss}"
               + (" --check-only" if args.check_only else ""))
    argv = ["wsl.exe", "-d", "Ubuntu-20.04", "-u", "root", "--exec", "bash", "-lc", command]
    result = subprocess.run(argv, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="replace")
    print(stdout, end="", flush=True)
    states = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict) and "run_dir" in item:
                states.append(item)
        except ValueError:
            pass
    if not states:
        print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        raise SystemExit(f"WSL launcher produced no run receipt, exit {result.returncode}")
    run_name = Path(states[0]["run_dir"]).name
    native = UNC / "outputs/runs" / run_name
    mirrored = LAB / "outputs/splatam-official-runs-001" / run_name
    mirrored.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in ("execution.json", "effective-config.json", "input_config.py", "config.py", "sensor-config.yaml", "environment-acceptance.json", "launcher-source.py", "params.npz", "stdout.log"):
        source = native / name
        if source.is_file():
            target = mirrored / name
            shutil.copy2(source, target)
            copied.append({"name": name, "bytes": target.stat().st_size,
                           "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    # Preserve qualitative render outputs and original metric text separately.
    if (native / "eval").is_dir():
        shutil.copytree(native / "eval", mirrored / "upstream-eval")
    receipt = {"native_run_name": run_name, "native_path": states[0]["run_dir"],
               "windows_wrapper_exit": result.returncode, "wsl_stderr_bytes": len(result.stderr),
               "copied": copied, "all_local": True, "check_only": args.check_only,
               "windows_wrapper_sha256": hashlib.sha256(wrapper_source).hexdigest()}
    (mirrored / "windows-wrapper-source.py").write_bytes(wrapper_source)
    (mirrored / "mirror-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (mirrored / "wsl-wrapper-stderr.bin").write_bytes(result.stderr)
    print(json.dumps({"mirrored_run": str(mirrored), "child_launcher_exit": result.returncode}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
