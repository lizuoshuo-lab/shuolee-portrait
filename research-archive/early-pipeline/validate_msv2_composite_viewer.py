"""Independent synthetic CPU contract checks, never training or publication."""

import argparse
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).parent
HELPER = WORKSPACE / "publish_msv2_pipeline_viewer.py"
SPEC = importlib.util.spec_from_file_location("local_publisher", HELPER)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
sys.path.insert(0, "D:/Projects/msv2-mask-compose")
from msv2.export import camera_input_mapping, camera_records
from msv2.mask_sr import compose_face
from msv2.report import route_metadata


TRANSFORM = np.diag([1.0, -1.0, -1.0, 1.0])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(p.dump(value), encoding="utf-8")


def synthetic_case(root, mode, count):
    run = root / mode
    output, runtime = run / "export/swinir4x", run / "swinir4x"
    native_dir = run / "prepared/images"
    for directory in (
        output,
        native_dir,
        runtime / "dataset/images",
        runtime / "face-masks",
        run / "pipeline-source/msv2",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (run / "pipeline-source/msv2/frozen.py").write_text(
        "# Synthetic contract fixture; never executed as training.\n"
    )
    options = dict(
        scales=[4], iterations=18000, importance_metric="outdoor", sr_region="face"
    )
    job = dict(
        schema="msv2-job-v1",
        run_dir=str(run),
        input="synthetic-video.mp4",
        options=options,
        created="2026-11-06T00:00:00+08:00",
        source_dir="synthetic source, not executed",
        implementation_sha256=p.implementation_sha(run / "pipeline-source/msv2"),
    )
    save(run / "job.json", job)
    save(
        run / "state.json",
        dict(
            state="completed",
            training_completed=True,
            failures={},
            completed_scales=[4],
            frames=count,
        ),
    )
    save(
        run / "plan.json",
        dict(
            state="completed",
            training_completed=True,
            fresh_gaussians=True,
            fresh_optimizer=True,
            old_fitted_state_loaded=False,
            frame_count=count,
            options=options,
        ),
    )
    native_rows, trained, sr_items = [], [], []
    for index in range(count):
        frame, name = f"frame_{index + 1:06d}", f"frame_{index + 1:06d}.png"
        rgb = np.random.default_rng(index).integers(0, 256, (40, 36, 3), np.uint8)
        original = native_dir / name
        Image.fromarray(rgb).save(original)
        digest = p.sha(original)
        w2c = np.eye(4)
        w2c[0, 3] = index * 0.01
        native = dict(
            frame=frame,
            name=name,
            wh=[36, 40],
            K=[[70.0, 0.0, 18.5], [0.0, 71.0, 20.5], [0.0, 0.0, 1.0]],
            w2c=w2c.tolist(),
            split="test" if index % 8 == 0 else "train",
            native_sha256=digest,
            source_sha256=digest,
            native_image=str(original),
            image=str(original),
        )
        native_rows.append(native)
        regions, faces, calls = [], [], []
        present = mode != "no_face" and (
            mode != "heldout_only" or native["split"] == "test"
        )
        boxes = [[7, 8, 24, 31]] if present else []
        if present and mode == "overlap_generic7":
            boxes += [[14, 15, 32, 36]]
        for number, bbox in enumerate(boxes, 1):
            mask = np.zeros(rgb.shape[:2], bool)
            x0, y0, x1, y1 = bbox
            mask[y0:y1, x0:x1] = True
            # An interior hole tests alpha/mask support rather than bbox-only masking.
            mask[y0 + 3 : y0 + 5, x0 + 3 : x0 + 5] = False
            regions.append(dict(bbox_native=bbox, mask_native=mask, mask_scope="face"))
            mask_name = f"{frame}_face_{number:03d}.png"
            mask_path = runtime / "face-masks" / mask_name
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            faces.append(
                dict(
                    bbox_native=bbox,
                    mask_name=mask_name,
                    mask_sha256=p.sha(mask_path),
                    mask_pixels=int(mask.sum()),
                    native_mask_array_sha256=__import__("hashlib")
                    .sha256(mask.tobytes())
                    .hexdigest(),
                )
            )

        def synthetic_sr(crop):
            height, width = crop.shape[:2]
            calls.append(
                dict(
                    source_wh=[width, height],
                    output_wh=[4 * width, 4 * height],
                    tile=None,
                    window_padding=[(-width) % 8, (-height) % 8],
                    crop_input=True,
                    raw_pixels=True,
                )
            )
            return np.clip(
                crop.repeat(4, 0).repeat(4, 1).astype(np.int16) + 25, 0, 255
            ).astype(np.uint8)

        pixels, composition = compose_face(rgb, regions, synthetic_sr)
        target = runtime / "dataset/images" / name
        Image.fromarray(pixels).save(target)
        row = deepcopy(native)
        row.update(
            image=str(target),
            wh=[144, 160],
            K=(np.diag([4, 4, 1]) @ np.asarray(native["K"])).tolist(),
            source_sha256=p.sha(target),
        )
        trained.append(row)
        sr_items.append(
            dict(
                name=name,
                native_sha256=digest,
                output_sha256=p.sha(target),
                sr_region="face",
                mask_scope="face",
                face_count=len(faces),
                faces=faces,
                composition=composition,
                swinir_calls=calls,
                face_extraction=dict(
                    mask_scope="face",
                    geometry_model_used=False,
                    source_rgb_sha256=composition["source_rgb_sha256"],
                ),
            )
        )
        previews = run / "export/frames" / frame
        previews.mkdir(parents=True)
        for stem in ("input", "swinir4x"):
            shutil.copy2(original, previews / (stem + ".png"))
    used = any(item["face_count"] for item in sr_items)
    sr = dict(
        state="completed",
        sr_region="face",
        mask_scope="face",
        mask_source=p.HEAD_MASK_SOURCE,
        view_layout=p.COMPOSITE_LAYOUT,
        training_mask_used=False,
        additional_training_views=False,
        full_frame_4x_generated=True,
        source_frames=count,
        view_count=count,
        input_composite_mask=used,
        input_face_mask_used=used,
        records=sr_items,
        face_frames=sum(bool(item["face_count"]) for item in sr_items),
        no_face_frames=sum(not item["face_count"] for item in sr_items),
        face_region_count=sum(item["face_count"] for item in sr_items),
        parser_detector_assets=dict(mask_scope="face", head_labels=list(range(1, 14))),
    )
    save(runtime / "superresolution.json", sr)
    save(output / "superresolution.json", sr)
    save(run / "original/training-inputs.json", dict(records=native_rows))
    training = dict(
        records=trained,
        mask_used=False,
        training_mask_used=False,
        additional_training_views=False,
        crop_used=False,
        full_frame_4x_generated=True,
        mask_scope="face",
        view_layout=p.COMPOSITE_LAYOUT,
        source_frames=count,
        view_count=count,
        input_composite_mask=used,
        input_face_mask_used=used,
    )
    save(runtime / "training-inputs.json", training)
    save(
        output / "camera-input-mapping.json",
        camera_input_mapping(trained, "face", sr=sr),
    )
    transformed_rows = deepcopy(trained)
    for row in transformed_rows:
        row["w2c"] = (np.asarray(row["w2c"]) @ np.linalg.inv(TRANSFORM)).tolist()
    save(output / "cameras.json", camera_records(transformed_rows))
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        + "".join("property float " + name + "\n" for name in names)
        + "end_header\n"
    )
    values = np.zeros((2, len(names)), dtype="<f4")
    values[:, 0] = [1, 2]
    values[:, -4] = 1
    world = runtime / "model/point_cloud/iteration_18000/point_cloud.ply"
    world.parent.mkdir(parents=True)
    world.write_bytes(header.encode() + values.tobytes())
    for name in ("point_cloud.ply", "point_cloud_world.ply"):
        shutil.copy2(world, output / name)
    execution = dict(
        state="training_completed",
        iterations=18000,
        recipe=options,
        native_sources_unchanged=True,
        points=2,
        train_frames=sum(row["split"] == "train" for row in trained),
        test_frames=sum(row["split"] == "test" for row in trained),
        ply_sha256=p.sha(world),
    )
    save(runtime / "execution.json", execution)
    save(output / "execution.json", execution)
    save(
        output / "native-ply-parity.json",
        dict(
            passed=True, sha256=p.sha(world), checks=[dict(max_abs=0.0, mean_abs=0.0)]
        ),
    )
    save(
        output / "upright-parity.json",
        dict(
            passed=True,
            transform=TRANSFORM.tolist(),
            covariance_relative=0.0,
            SH_error=0.0,
            inverse_position_error=0.0,
            position_tolerance=1e-6,
            render_checks=[dict(max_abs=0.0, mean_abs=0.0)],
        ),
    )
    # Placeholder parity values exercise metadata guards, never certify rendering.
    bundle = output / "scene.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in ("point_cloud.ply", "cameras.json"):
            archive.write(output / name, name)
    save(
        output / "delivery.json",
        dict(
            parity_passed=True,
            points=2,
            cameras=count,
            bundle=bundle.name,
            sha256=p.sha(bundle),
        ),
    )
    provenance = route_metadata(SimpleNamespace(root=run, config=options), 4, trained)
    save(
        output / "metrics.json",
        dict(
            scale=4,
            execution=execution,
            metrics=[
                dict(frame=r["frame"], split=r["split"], wh=r["wh"])
                for r in native_rows
            ],
            means={s: dict(psnr=20.0, ssim=0.8) for s in ("train", "test")},
            mask_used=False,
            training_mask_used=False,
            input_composite_mask=used,
            input_face_mask_used=used,
            view_layout=p.COMPOSITE_LAYOUT,
            input_provenance=provenance,
        ),
    )
    return run


def checked(root, run):
    args = SimpleNamespace(
        job=str(run / "job.json"),
        record=str(run / "record"),
        viewer=str(root / "viewer"),
        scales=[4],
        name="synthetic-face",
        date=None,
        wsl_distribution="Ubuntu-20.04",
    )
    context = p.context_from_args(args)
    item = p.verify_variant(context, 4)
    return context, item, p.metadata(context, item)


def rejection(root, run, files, mutate):
    saved = {path: path.read_bytes() for path in files}
    try:
        mutate()
        try:
            checked(root, run)
        except (ValueError, KeyError, FileNotFoundError):
            return
        raise AssertionError("Tampered synthetic evidence was accepted")
    finally:
        for path, value in saved.items():
            path.write_bytes(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    root = Path(ap.parse_args().output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    viewer = root / "viewer"
    viewer.mkdir()
    (viewer / "viewer.js").write_text('const BUILTINS = ["unchanged"];\n')
    (viewer / "index.html").write_text("Synthetic viewer, not opened")
    before = {str(path): p.sha(path) for path in viewer.iterdir()}
    cases = {}
    for mode, count in [
        ("all_faces", 65),
        ("heldout_only", 65),
        ("no_face", 65),
        ("overlap_generic7", 7),
    ]:
        run = synthetic_case(root, mode, count)
        context, item, meta = checked(root, run)
        assert len(item["cameras"]) == len(item["source_rows"]) == count
        assert all(r["wh"] == [36, 40] for r in item["source_rows"])
        assert all(r["wh"] == [144, 160] for r in item["rows"])
        assert (
            meta["qualityAccepted"] is False
            and meta["additionalTrainingViews"] is False
        )
        assert meta["inputRoute"] == p.COMPOSITE_LAYOUT and "整图" in meta["label"]
        if mode == "heldout_only":
            assert meta["faceTrainFrames"] == 0 and "仅留出" in meta["label"]
        if mode == "no_face":
            assert not meta["inputFaceMaskUsed"] and "无面部" in meta["label"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            p.main(
                [
                    "--job",
                    str(run / "job.json"),
                    "--viewer",
                    str(viewer),
                    "--record",
                    str(run / "record"),
                    "--scales",
                    "4",
                    "--check-only",
                ]
            )
        assert (
            not (run / "record").exists()
            and not json.loads(captured.getvalue())["viewer_modified"]
        )
        cases[mode] = dict(
            source_frames=count,
            cameras=count,
            train_views=meta["trainViews"],
            test_views=meta["testViews"],
            face_train_frames=meta["faceTrainFrames"],
            face_test_frames=meta["faceTestFrames"],
            label=meta["label"],
        )
    run = root / "all_faces"
    output = run / "export/swinir4x"
    runtime = run / "swinir4x"
    metric = output / "metrics.json"

    def wrong_metrics():
        d = p.read(metric)
        d["metrics"][0]["wh"] = [144, 160]
        save(metric, d)

    rejection(root, run, [metric], wrong_metrics)
    paths = [
        output / "camera-input-mapping.json",
        output / "cameras.json",
        runtime / "training-inputs.json",
    ]

    def coherent_K():
        m, c, t = map(p.read, paths)
        m["records"][0]["K"][0][2] += 1
        t["records"][0]["K"] = deepcopy(m["records"][0]["K"])
        c[0]["K"] = deepcopy(m["records"][0]["K"])
        c[0]["cx"] += 1
        for path, d in zip(paths, [m, c, t]):
            save(path, d)

    rejection(root, run, paths, coherent_K)
    source = run / "prepared/images/frame_000001.png"
    rejection(
        root, run, [source], lambda: Image.new("RGB", (36, 40), (1, 2, 3)).save(source)
    )
    mask = runtime / "face-masks/frame_000001_face_001.png"
    rejection(root, run, [mask], lambda: Image.new("L", (36, 40), 0).save(mask))
    target = runtime / "dataset/images/frame_000001.png"
    sr_runtime = runtime / "superresolution.json"
    sr_export = output / "superresolution.json"

    def outside_RGB_with_coherent_hashes():
        with Image.open(target) as im:
            a = np.array(im)
        a[0, 0, 0] = (int(a[0, 0, 0]) + 1) % 256
        Image.fromarray(a).save(target)
        sr = p.read(sr_runtime)
        item = sr["records"][0]
        item["output_sha256"] = p.sha(target)
        item["composition"]["output_rgb_sha256"] = item["composition"][
            "full_rgb_sha256"
        ] = __import__("hashlib").sha256(a.tobytes()).hexdigest()
        save(sr_runtime, sr)
        save(sr_export, sr)
        for path in [
            runtime / "training-inputs.json",
            output / "camera-input-mapping.json",
        ]:
            d = p.read(path)
            d["records"][0]["source_sha256"] = item["output_sha256"]
            save(path, d)

    rejection(
        root,
        run,
        [
            target,
            sr_runtime,
            sr_export,
            runtime / "training-inputs.json",
            output / "camera-input-mapping.json",
        ],
        outside_RGB_with_coherent_hashes,
    )
    checked(root, run)
    # Re-run previous mixed-head scenarios in a NEW fixture tree; old receipts stay intact.
    old = Path("D:/GS-SLAM-Data/msv2-publisher-mixed-cpu-fixture-20261106/validate.py")
    spec = importlib.util.spec_from_file_location("legacy_mixed_fixture", old)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.ROOT = root / "legacy-mixed"
    legacy.ROOT.mkdir()
    legacy.main()
    assert (
        before == {str(path): p.sha(path) for path in viewer.iterdir()}
        and "torch" not in sys.modules
    )
    result = dict(
        state="passed",
        synthetic_only=True,
        cpu_only=True,
        GPU_used=False,
        actual_training_run_verified=False,
        actual_SR_inference_run=False,
        native_parity_measured=False,
        actual_publication=False,
        publisher_sha256=p.sha(HELPER),
        cases=cases,
        rejected=[
            "metrics_wrong4x_domain",
            "coherent_crop_K",
            "native_source_PNG",
            "mask_PNG",
            "outside_RGB_even_with_coherent_receipt_hashes",
        ],
        legacy_mixed_four_scenarios_passed=True,
        check_only_zero_viewer_and_record_writes=True,
        torch_imported=False,
    )
    save(root / "result.json", result)
    print(p.dump(result), end="")


if __name__ == "__main__":
    main()
