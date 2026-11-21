"""Three actual candidate Engine face-composite inputs; no GS optimization.

--check-only is CPU-only. Only explicit --execute-sr initializes CUDA/SwinIR.
K4 is checked algebraically; this is not a native-renderer preflight.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time


VIDEO_SHA = "965befa346fdb0b5e5140745ab5651ef6ed7f8a87ae32777e1d88017bc45e547"
INPUT_SHA = "fc7ed4330ffa1a008469412313ecc524962f21691dc2ee93105281998c988fc6"
MIN_FREE_BYTES = 25 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name != "nt":
        for prefix in ("//wsl.localhost/Ubuntu-20.04/", "//wsl$/Ubuntu-20.04/"):
            if value.lower().startswith(prefix.lower()):
                return Path("/" + value[len(prefix) :])
        if len(value) > 2 and value[1:3] == ":/":
            return Path("/mnt/" + value[0].lower() + "/" + value[3:])
    return Path(value)


def candidate_files(source):
    paths = sorted((source / "msv2").rglob("*.py"))
    return {p.relative_to(source).as_posix(): sha(p) for p in paths}


def prepare(args):
    import numpy as np
    from PIL import Image

    source, record = Path(args.candidate_source).resolve(), Path(args.record).resolve()
    template_path = Path(args.template_job).resolve()
    template = read(template_path)
    assert template["input_sha256"] == INPUT_SHA
    prep = Path(template["run_dir"]) / "prepared"
    lineage = read(prep / "pixel-lineage.json")
    assert lineage["passed"] and len(lineage["records"]) == 65
    actual_lineage = read(args.native_lineage)
    assert actual_lineage["video_sha256"] == VIDEO_SHA
    actual_by_name = {r["image"]: r for r in actual_lineage["records"]}
    numbers = args.frames
    assert (
        numbers
        and len(set(numbers)) == len(numbers)
        and all(1 <= n <= 65 for n in numbers)
    )
    rows = [copy.deepcopy(lineage["records"][n - 1]) for n in numbers]
    cameras = {r["name"]: r for r in read(prep / "camera-inputs.json")["records"]}
    protocol = read(record / "实验协议.json")
    protected = dict(protocol["protected_old_SHA256"])
    for path in (
        template_path,
        prep / "camera-inputs.json",
        prep / "pixel-lineage.json",
    ):
        protected.setdefault(str(path), sha(path))
    sources, algebra = [], []
    for number, row in zip(numbers, rows):
        path = prep / "dataset/images" / row["image"]
        assert sha(path) == row["sha256"]
        protected[str(path)] = row["sha256"]
        with Image.open(path) as image:
            assert image.mode == "RGB" and list(image.size) == row["wh"] == [1520, 2704]
            rgb = np.array(image)
        rgb_sha = hashlib.sha256(memoryview(rgb)).hexdigest()
        assert rgb_sha == row["rgb_sha256"]
        original = actual_by_name[row["image"]]
        assert original["rgb_sha256"] == rgb_sha
        original_path = (
            Path(args.native_lineage).parent / "dataset/images" / row["image"]
        )
        assert sha(original_path) == original["sha256"]
        protected[str(original_path)] = original["sha256"]
        with Image.open(original_path) as image:
            assert np.array_equal(np.array(image.convert("RGB")), rgb)
        camera = cameras[row["image"]]
        assert camera["source_sha256"] == row["sha256"] and camera["wh"] == row["wh"]
        k = np.asarray(camera["K"], dtype=np.float64)
        k4 = np.diag([4.0, 4.0, 1.0]) @ k
        xyz = np.array([[0.1, 0.2, 2.0], [-0.4, -0.15, 3.0], [0.0, 0.0, 1.0]]).T
        uv, uv4 = k @ xyz, k4 @ xyz
        uv, uv4 = uv[:2] / uv[2:], uv4[:2] / uv4[2:]
        error = float(np.max(np.abs(uv4 - 4 * uv)))
        assert error < 1e-10
        cvk4 = k4.copy()
        cvk4[:2, 2] -= 0.5
        algebra.append(
            dict(
                name=row["image"],
                native_K=k.tolist(),
                full4_K=k4.tolist(),
                full4_K_integer_centers=cvk4.tolist(),
                scaled_projection_error_px=error,
                w2c_unchanged=camera["w2c"],
                crop_translation_applied=False,
                source_split="test"
                if (number - 1) % template["options"]["holdout"] == 0
                else "train",
            )
        )
        sources.append(
            dict(
                name=row["image"],
                source_index=original["source_index"],
                time_seconds=original["time_seconds"],
                png_sha256=row["sha256"],
                decoded_rgb_sha256=rgb_sha,
                source_png=str(path),
                native_wh=row["wh"],
            )
        )
    before = {}
    for name, expected in protected.items():
        path = host_path(name)
        current = sha(path)
        assert current == expected, "Protected old file differs before probe: " + str(
            path
        )
        before[str(path)] = current
    files = candidate_files(source)
    assert "msv2/engine.py" in files and "msv2/mask_sr.py" in files
    if args.expected_engine_sha:
        assert files["msv2/engine.py"] == args.expected_engine_sha
    free = shutil.disk_usage(host_path("D:/")).free
    assert free >= MIN_FREE_BYTES, (
        "Physical D free space below25 GiB; generation/training forbidden"
    )
    probe = record / "input_probe"
    root = probe / "run"
    assert not root.exists(), "Probe run exists; preserve it and use a different record"
    config = read(record / "训练配置.json")
    assert config["sr_region"] == "face" and config["scales"] == [4]
    assert (
        config["iterations"] == 18000
        and config["seed"] == 0
        and config["importance_metric"] == "outdoor"
    )
    job = copy.deepcopy(template)
    job.update(
        run_dir=str(root),
        options=config,
        created="three-frame-input-probe",
        implementation_sha256=hashlib.sha256(canonical(files).encode()).hexdigest(),
        probe_only=True,
        probe_kind="three-full4-face-composite-inputs-no-training",
        template_job_sha256=sha(template_path),
        candidate_package=str(source),
    )
    job["task_sha256"] = hashlib.sha256(
        canonical(
            dict(
                input_sha256=INPUT_SHA,
                options=config,
                selected_frames=numbers,
                probe_only=True,
            )
        ).encode()
    ).hexdigest()
    job_path = probe / "job.json"
    if job_path.exists():
        assert read(job_path) == job, (
            "Existing prepared probe job differs; do not overwrite"
        )
    else:
        save(job_path, job)
    readiness = dict(
        schema="msv2-face-fullframe-input-probe-readiness-v1",
        checked=now(),
        passed=True,
        phase="CPU input and camera algebra only",
        selected_frames=numbers,
        source_video_sha256=VIDEO_SHA,
        source_input_sha256=INPUT_SHA,
        sources=sources,
        camera_algebra=algebra,
        prepared_readonly=str(prep),
        candidate_source=str(source),
        candidate_files=files,
        old_files_before=before,
        job=str(job_path),
        job_sha256=sha(job_path),
        physical_D_free_bytes=free,
        minimum_free_bytes=MIN_FREE_BYTES,
        cuda_imported=False,
        cuda_initialized=False,
        gpu_SR_executed=False,
        training_executed=False,
        native_renderer_preflight_executed=False,
        stage_cameras_called=False,
        quality_accepted=False,
    )
    save(probe / "readiness.json", readiness)
    return source, root, job_path, prep, rows, readiness


def verify_pixels(root, rows, readiness):
    import numpy as np
    from PIL import Image, ImageDraw

    variant = root / "swinir4x"
    sr, pixels = read(variant / "superresolution.json"), read(variant / "pixels.json")
    assert sr["state"] == "completed" and sr["source_frames"] == sr[
        "view_count"
    ] == len(rows)
    assert sr["mask_scope"] == "face" and sr["view_layout"] == "full4x_mask_composite"
    assert sr["full_frame_4x_generated"] and not sr["additional_training_views"]
    assert (
        pixels["count"] == len(rows)
        and pixels["one_complete_training_image_per_source"]
    )
    assert not pixels["additional_training_views"] and not pixels["training_mask_used"]
    assert sr["parser_detector_assets"]["head_labels"] == list(range(1, 14))
    wanted = {r["image"] for r in rows}
    assert {p.name for p in (variant / "dataset/images").iterdir()} == wanted
    by_name = {item["name"]: item for item in sr["records"]}
    assert set(by_name) == wanted
    samples = root.parent / "samples"
    samples.mkdir()
    results = []
    sources = {r["name"]: r for r in readiness["sources"]}
    for row in rows:
        item = by_name[row["image"]]
        source = Path(sources[row["image"]]["source_png"])
        assert sha(source) == row["sha256"] == item["native_sha256"]
        with Image.open(source) as image:
            native = np.array(image)
            base = np.array(image.resize((6080, 10816), Image.Resampling.LANCZOS))
        output_path = variant / "dataset/images" / row["image"]
        assert sha(output_path) == item["output_sha256"]
        with Image.open(output_path) as image:
            assert image.mode == "RGB" and image.size == (6080, 10816)
            output = np.array(image)
        union = np.zeros(native.shape[:2], dtype=bool)
        assert len(item["faces"]) == item["face_count"]
        for face in item["faces"]:
            mask_path = variant / "face-masks" / face["mask_name"]
            assert sha(mask_path) == face["mask_sha256"]
            with Image.open(mask_path) as image:
                assert image.mode == "L" and image.size == (1520, 2704)
                assert image.getbbox() == tuple(face["bbox_native"])
                mask = np.array(image)
            assert set(np.unique(mask)).issubset({0, 255})
            mask = mask != 0
            assert (
                hashlib.sha256(memoryview(mask)).hexdigest()
                == face["native_mask_array_sha256"]
            )
            assert int(mask.sum()) == face["mask_pixels"]
            union |= mask
        composition = item["composition"]
        assert (
            hashlib.sha256(memoryview(native)).hexdigest()
            == composition["source_rgb_sha256"]
        )
        assert (
            hashlib.sha256(memoryview(base)).hexdigest()
            == composition["base_full4_rgb_sha256"]
        )
        assert (
            hashlib.sha256(memoryview(output)).hexdigest()
            == composition["output_rgb_sha256"]
        )
        assert (
            hashlib.sha256(memoryview(union)).hexdigest()
            == composition["union_mask_native_sha256"]
        )
        assert int(union.sum()) == composition["union_mask_native_pixels"]
        assert (
            composition["one_complete_training_image"]
            and composition["added_views"] == 0
        )
        assert (
            composition["outside_mask_equal_reference"]
            and composition["feather_native"] == 2
        )
        changed_inside, outside_mismatch = 0, 0
        for y in range(native.shape[0]):
            support = np.repeat(union[y], 4)
            different = np.any(
                output[4 * y : 4 * y + 4] != base[4 * y : 4 * y + 4], axis=2
            )
            outside_mismatch += int(different[:, ~support].sum())
            changed_inside += int(different[:, support].sum())
        assert outside_mismatch == 0
        contexts = []
        for region in composition["regions"]:
            if not region["sr_inference_called"]:
                continue
            assert region["feather_native"] == 2
            assert region["sr_receipt"]["semantic_mask_used"] is False
            x0, y0, x1, y1 = region["context_bbox_native"]
            raw_context = np.ascontiguousarray(native[y0:y1, x0:x1])
            assert (
                hashlib.sha256(memoryview(raw_context)).hexdigest()
                == region["sr_receipt"]["context_rgb_sha256"]
            )
            contexts.append(
                dict(
                    bbox_native=region["bbox_native"],
                    context_bbox_native=region["context_bbox_native"],
                    raw_context_rgb_sha256=region["sr_receipt"]["context_rgb_sha256"],
                )
            )
        assert all(c["crop_input"] and c["raw_pixels"] for c in item["swinir_calls"])
        if union.any():
            yy, xx = np.nonzero(union)
            box = (
                max(0, int(xx.min()) - 16),
                max(0, int(yy.min()) - 16),
                min(1520, int(xx.max()) + 17),
                min(2704, int(yy.max()) + 17),
            )
        else:
            box = (0, 0, 256, 256)
        x0, y0, x1, y1 = box
        left = Image.fromarray(base[4 * y0 : 4 * y1, 4 * x0 : 4 * x1])
        right = Image.fromarray(output[4 * y0 : 4 * y1, 4 * x0 : 4 * x1])
        stem = Path(row["image"]).stem
        left.save(samples / (stem + "_whole_lanczos4_crop.png"))
        right.save(samples / (stem + "_face_composite4_crop.png"))
        overlay = native[y0:y1, x0:x1].copy()
        m = union[y0:y1, x0:x1]
        overlay[m] = np.rint(overlay[m] * 0.65 + np.array([0, 255, 128]) * 0.35).astype(
            np.uint8
        )
        Image.fromarray(overlay).save(samples / (stem + "_native_mask_overlay.png"))
        canvas = Image.new("RGB", (left.width * 2, left.height + 40), "white")
        canvas.paste(left, (0, 40))
        canvas.paste(right, (left.width, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), row["image"] + " | whole-original Lanczos4", fill="black")
        draw.text((left.width + 8, 10), "face SR in same full4", fill="black")
        canvas.save(samples / (stem + "_compare.png"))
        results.append(
            dict(
                name=row["image"],
                output_wh=[6080, 10816],
                output_png_sha256=item["output_sha256"],
                face_count=item["face_count"],
                native_union_pixels=int(union.sum()),
                mask_exterior_mismatched_pixels=outside_mismatch,
                changed_inside_mask_4x_pixels=changed_inside,
                native_raw_contexts=contexts,
                display_crop_native_xyxy=list(box),
                comparison=str(samples / (stem + "_compare.png")),
            )
        )
        del native, base, output, union
    assert not any(root.rglob("point_cloud.ply"))
    assert not any(root.rglob("chkpnt*"))
    return results, sr


def execute(prepared):
    source, root, job_path, prep, rows, readiness = prepared
    result_path = root.parent / "receipt.json"
    result = dict(
        schema="msv2-face-fullframe-input-probe-v1",
        started=now(),
        passed=False,
        phase="real face SR input probe; zero GS optimization",
        quality_accepted=False,
        native_renderer_preflight_executed=False,
        training_executed=False,
        zero_optimizer_updates=True,
        stage_cameras_called=False,
        note="K4 algebra only; not native renderer or training quality acceptance",
    )
    attempts, forbidden = [], []
    original_connect, original_create = socket.socket.connect, socket.create_connection

    def deny_network(*args, **kwargs):
        attempts.append(repr(args[:2]))
        raise PermissionError("Network disabled for local-asset input probe")

    def no_training(*args, **kwargs):
        forbidden.append("optimizer/backward/training")
        raise RuntimeError("GS optimization is forbidden in input probe")

    started = time.monotonic()
    try:
        assert shutil.disk_usage(host_path("D:/")).free >= MIN_FREE_BYTES
        socket.socket.connect, socket.create_connection = deny_network, deny_network
        sys.path.insert(0, str(source))
        from msv2.engine import Engine
        import torch

        tensor_backward, autograd_backward = (
            torch.Tensor.backward,
            torch.autograd.backward,
        )
        optimizer_init = torch.optim.Optimizer.__init__
        torch.Tensor.backward = torch.autograd.backward = no_training
        torch.optim.Optimizer.__init__ = no_training
        try:
            root.mkdir(parents=True)
            engine = Engine(job_path)
            assert engine.sr_region == "face"
            engine.prep = prep
            engine.native_rows = lambda: copy.deepcopy(rows)
            engine.stage_cameras = no_training
            engine.train = no_training
            engine.superresolve()
        finally:
            torch.Tensor.backward, torch.autograd.backward = (
                tensor_backward,
                autograd_backward,
            )
            torch.optim.Optimizer.__init__ = optimizer_init
        results, sr = verify_pixels(root, rows, readiness)
        assert not forbidden and not attempts
        assert candidate_files(source) == readiness["candidate_files"]
        result.update(
            passed=True,
            gpu_SR_executed=True,
            frames=results,
            source_frames=len(rows),
            complete_training_images=len(rows),
            added_training_views=0,
            sr_peak_reserved_gib=sr.get("peak_reserved_gib"),
            parser_detector_assets=sr["parser_detector_assets"],
            candidate_sources_unchanged=True,
            source_pngs_unchanged=True,
            no_backward_or_optimizer_calls=True,
            network_attempts=attempts,
        )
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            forbidden_calls=forbidden,
            network_attempts=attempts,
        )
        raise
    finally:
        socket.socket.connect, socket.create_connection = (
            original_connect,
            original_create,
        )
        changes = {}
        for path, expected in readiness["old_files_before"].items():
            actual = sha(path) if Path(path).is_file() else None
            if actual != expected:
                changes[path] = dict(before=expected, after=actual)
        result.update(
            finished=now(),
            seconds=time.monotonic() - started,
            old_protected_file_count=len(readiness["old_files_before"]),
            old_protected_files_unchanged=not changes,
            changed_protected_files=changes,
        )
        if changes:
            result["passed"] = False
        save(result_path, result)
        if changes:
            raise RuntimeError("Protected file changed; inspect receipt.json")
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                receipt=str(result_path),
                comparisons=[r["comparison"] for r in result["frames"]],
            ),
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="CPU only; freeze job, never import torch",
    )
    mode.add_argument(
        "--execute-sr",
        action="store_true",
        help="Explicit GPU SwinIR only; never GS train",
    )
    parser.add_argument(
        "--candidate-source", default="/mnt/d/Projects/msv2-mask-compose"
    )
    parser.add_argument(
        "--template-job",
        default="/opt/gs-slam-lab/msv2-video15-20261105/runs/video15-head4x/job.json",
    )
    parser.add_argument(
        "--record", default="/mnt/d/00/记录/2026-11-06_MSV2_video15_面部mask整图合成"
    )
    parser.add_argument(
        "--native-lineage",
        default="/mnt/d/GS-SLAM-Data/msv2-video15-20261105/pixel-lineage.json",
    )
    parser.add_argument("--frames", nargs="+", type=int, default=[1, 33, 65])
    parser.add_argument("--expected-engine-sha")
    args = parser.parse_args()
    prepared = prepare(args)
    if args.check_only:
        assert "torch" not in sys.modules
        print(
            json.dumps(
                dict(
                    passed=True,
                    CPU_only=True,
                    job=str(prepared[2]),
                    readiness=str(prepared[1].parent / "readiness.json"),
                ),
                ensure_ascii=False,
            )
        )
        return
    execute(prepared)


if __name__ == "__main__":
    main()
