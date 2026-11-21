"""Read-only CPU audit of all full-frame face-composite training inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from probe_msv2_face_fullframe import VIDEO_SHA, INPUT_SHA, host_path, read, save, sha


INITIAL_POINTS_SHA = "7e049a3b338912f28e87c181fbea9efc6ae1ffbfd0a522d6be7f874719f4f787"
IMPLEMENTATION_SHA = "686342cb669fbe6813c54aac950944163caa9ac4baf698ecbe6839842ce78b8d"


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def audit(args):
    started = time.monotonic()
    run, record = Path(args.run), Path(args.record)
    prep, variant = run / "prepared", run / "swinir4x"
    job = read(run / "job.json")
    assert (
        job["input_sha256"] == INPUT_SHA
        and job["implementation_sha256"] == IMPLEMENTATION_SHA
    )
    config = job["options"]
    assert config["sr_region"] == "face" and config["scales"] == [4]
    assert config["iterations"] == 18000 and config["seed"] == 0
    assert config["importance_metric"] == "outdoor" and config["holdout"] == 8
    sr = read(variant / "superresolution.json")
    assert sr["state"] == "completed", "All65 SR images must finish before this audit"
    training, pixels = (
        read(variant / "training-inputs.json"),
        read(variant / "pixels.json"),
    )
    lineage, cameras = (
        read(prep / "pixel-lineage.json"),
        read(prep / "camera-inputs.json"),
    )
    original = read(args.native_lineage)
    assert original["passed"] and original["video_sha256"] == VIDEO_SHA
    assert (
        lineage["passed"] and len(original["records"]) == len(lineage["records"]) == 65
    )
    names = [r["image"] for r in lineage["records"]]
    assert len(set(names)) == 65 and names == [r["image"] for r in original["records"]]
    assert (
        names
        == [r["name"] for r in cameras["records"]]
        == [r["name"] for r in training["records"]]
    )
    assert len(sr["records"]) == len(pixels["items"]) == 65
    assert {p.name for p in (variant / "dataset/images").iterdir()} == set(names)
    assert not any("_head_" in n or "_face_" in n for n in names)
    assert (
        training["source_frames"] == training["view_count"] == 65
        and not training["crop_used"]
    )
    for data in (sr, pixels, training):
        assert data["mask_scope"] == data["sr_region"] == "face"
        assert data["view_layout"] == "full4x_mask_composite"
        assert not data["additional_training_views"] and not data["training_mask_used"]
        assert data["full_frame_4x_generated"]
    assert not training["mask_used"] and training["background_trained"]
    assets = sr["parser_detector_assets"]
    assert assets["device"] == "cpu" and assets["head_labels"] == list(range(1, 14))
    assert assets["excluded_labels"] == [0, 14, 15, 16, 17, 18]
    assert (
        assets["parser"]["source_commit"] == "8a4729d95118d0e97c44185f9bdef3d6bfeaaf99"
    )
    assert (
        assets["parser"]["weight_sha256"]
        == "18c44e06b77e47da86fe2b48fac78e0ccbd0b5241c829f9487ebf7fd4a6389e6"
    )
    assert (
        assets["detector"]["detector_weight_sha256"]
        == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    )
    assert (
        sr["weight_sha256"]
        == "99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a"
    )
    assert (
        sr["precision"] == "float32" and sr["tile"] == 256 and sr["tile_overlap"] == 32
    )
    baseline = Path(args.baseline)
    baseline_rows = read(baseline / "swinir4x/training-inputs.json")["records"]
    assert names == [r["name"] for r in baseline_rows]
    protected = {}

    def protect(path, expected=None):
        path = host_path(path)
        digest = sha(path)
        assert expected is None or digest == expected, "SHA mismatch: " + str(path)
        if str(path) in protected:
            assert protected[str(path)] == digest
        protected[str(path)] = digest
        return digest

    for path, expected in read(record / "input_probe/readiness.json")[
        "old_files_before"
    ].items():
        protect(path, expected)
    for path in (
        run / "job.json",
        prep / "pixel-lineage.json",
        prep / "camera-inputs.json",
        variant / "superresolution.json",
        variant / "pixels.json",
        variant / "training-inputs.json",
    ):
        protect(path)
    protect(args.native_lineage)
    video_path = host_path(original["source"])
    protect(video_path, VIDEO_SHA)
    for relative, value in assets["parser"]["source_files"].items():
        protect(value["path"], value["raw_sha256"])
    protect(assets["parser"]["weight"], assets["parser"]["weight_sha256"])
    protect(
        assets["detector"]["detector_weight"],
        assets["detector"]["detector_weight_sha256"],
    )
    for filename in ("images.bin", "points3D.bin"):
        expected = protect(prep / "dataset/sparse/0" / filename)
        assert protect(variant / "dataset/sparse/0" / filename) == expected
    assert protected[str(prep / "dataset/sparse/0/points3D.bin")] == INITIAL_POINTS_SHA
    assert training["initial_sparse_sha256"] == INITIAL_POINTS_SHA
    sys.path.insert(0, args.candidate_source)
    from msv2.data import _read_cameras, _read_images

    native_cameras = _read_cameras(prep / "dataset/sparse/0")
    output_cameras = _read_cameras(variant / "dataset/sparse/0")
    input_images, output_images = (
        _read_images(prep / "dataset/sparse/0"),
        _read_images(variant / "dataset/sparse/0"),
    )
    assert len(input_images) == len(output_images) == 65
    assert native_cameras.keys() == output_cameras.keys()
    for cid, camera in native_cameras.items():
        output = output_cameras[cid]
        parameters = np.asarray(camera.params)
        if camera.model == 0:
            parameters = parameters[[0, 0, 1, 2]]
        assert output.model == 1 and (output.width, output.height) == (
            camera.width * 4,
            camera.height * 4,
        )
        assert np.array_equal(np.asarray(output.params), parameters * 4)
    protect(prep / "dataset/sparse/0/cameras.bin")
    protect(variant / "dataset/sparse/0/cameras.bin")
    inputs = {r["name"]: r for r in training["records"]}
    sr_by = {r["name"]: r for r in sr["records"]}
    pixel_by = {r["name"]: r for r in pixels["items"]}
    rows, train_count, test_count = [], 0, 0
    for number, (source_row, raw_row, camera, baseline_row) in enumerate(
        zip(lineage["records"], original["records"], cameras["records"], baseline_rows),
        1,
    ):
        name = source_row["image"]
        out_row, item = inputs[name], sr_by[name]
        assert source_row["wh"] == raw_row["wh"] == camera["wh"] == [1520, 2704]
        expected_split = "test" if (number - 1) % config["holdout"] == 0 else "train"
        assert out_row["split"] == baseline_row["split"] == expected_split
        test_count += expected_split == "test"
        train_count += expected_split == "train"
        k, k4 = np.asarray(camera["K"]), np.asarray(out_row["K"])
        assert np.array_equal(k4, np.diag([4.0, 4.0, 1.0]) @ k)
        assert out_row["w2c"] == camera["w2c"] == baseline_row["w2c"]
        assert out_row["K"] == baseline_row["K"] and out_row["wh"] == [6080, 10816]
        k_cv = k4.copy()
        k_cv[:2, 2] -= 0.5
        assert np.array_equal(np.asarray(out_row["K_integer_centers"]), k_cv)
        assert out_row["colmap_image_id"] == camera["colmap_image_id"]
        assert out_row["colmap_camera_id"] == camera["colmap_camera_id"]
        assert (
            out_row.get("view_type", "full") == "full"
            and out_row.get("native_bbox") is None
        )
        source_path = prep / "dataset/images" / name
        raw_path = Path(args.native_lineage).parent / "dataset/images" / name
        source_sha = protect(source_path, source_row["sha256"])
        protect(raw_path, raw_row["sha256"])
        assert (
            item["native_sha256"]
            == out_row["native_sha256"]
            == camera["source_sha256"]
            == source_sha
        )
        assert Path(out_row["native_image"]) == source_path
        with Image.open(source_path) as image:
            assert image.mode == "RGB" and image.size == (1520, 2704)
            native = np.array(image)
            base = np.array(image.resize((6080, 10816), Image.Resampling.LANCZOS))
        assert array_sha(native) == source_row["rgb_sha256"] == raw_row["rgb_sha256"]
        with Image.open(raw_path) as image:
            assert np.array_equal(np.array(image.convert("RGB")), native)
        target_path = variant / "dataset/images" / name
        output_sha = protect(target_path, item["output_sha256"])
        assert output_sha == out_row["source_sha256"] == pixel_by[name]["image_sha256"]
        assert Path(out_row["image"]) == target_path
        with Image.open(target_path) as image:
            assert image.mode == "RGB" and image.size == (6080, 10816)
            output = np.array(image)
        extraction, composition = item["face_extraction"], item["composition"]
        assert extraction["mask_scope"] == "face" and extraction["head_labels"] == list(
            range(1, 14)
        )
        assert extraction["source_rgb_sha256"] == array_sha(native)
        assert composition["source_rgb_sha256"] == array_sha(native)
        assert composition["base_full4_rgb_sha256"] == array_sha(base)
        assert composition["output_rgb_sha256"] == array_sha(output)
        assert (
            composition["one_complete_training_image"]
            and composition["training_views_per_source"] == 1
        )
        assert composition["added_views"] == 0 and composition["no_added_views"]
        assert (
            composition["no_source_downsampling"]
            and not composition["training_mask_used"]
        )
        assert (
            composition["feather_native"] == 2
            and composition["full_frame_4x_generated"]
        )
        union = np.zeros((2704, 1520), bool)
        mask_rows = []
        assert item["face_count"] == len(item["faces"])
        for face in item["faces"]:
            path = variant / "face-masks" / face["mask_name"]
            protect(path, face["mask_sha256"])
            with Image.open(path) as image:
                assert image.mode == "L" and image.size == (1520, 2704)
                assert image.getbbox() == tuple(face["bbox_native"])
                assert sum(image.histogram()[1:255]) == 0
                mask = np.array(image) != 0
            assert mask.any() and array_sha(mask) == face["native_mask_array_sha256"]
            assert int(mask.sum()) == face["mask_pixels"]
            union |= mask
            mask_rows.append(
                dict(
                    name=face["mask_name"],
                    bbox_native=face["bbox_native"],
                    pixels=int(mask.sum()),
                    sha256=face["mask_sha256"],
                )
            )
        assert array_sha(union) == composition["union_mask_native_sha256"]
        assert int(union.sum()) == composition["union_mask_native_pixels"]
        assert composition["union_mask_4x_pixels"] == int(union.sum()) * 16
        contexts = []
        for region in composition["regions"]:
            if not region["sr_inference_called"]:
                continue
            receipt = region["sr_receipt"]
            assert not receipt["semantic_mask_used"] and not receipt["blending_used"]
            assert receipt["no_source_downsampling"] and receipt["context_native"] == 32
            assert receipt["source_rgb_sha256"] == array_sha(native)
            x0, y0, x1, y1 = region["bbox_native"]
            cx0, cy0, cx1, cy1 = region["context_bbox_native"]
            expected_context = [
                max(0, ((x0 - 32) // 8) * 8),
                max(0, ((y0 - 32) // 8) * 8),
                min(1520, math.ceil((x1 + 32) / 8) * 8),
                min(2704, math.ceil((y1 + 32) / 8) * 8),
            ]
            assert (
                region["context_bbox_native"]
                == expected_context
                == receipt["context_bbox_native"]
            )
            assert receipt["native_head_rgb_sha256"] == array_sha(native[y0:y1, x0:x1])
            assert receipt["context_rgb_sha256"] == array_sha(native[cy0:cy1, cx0:cx1])
            assert receipt["context_native_wh"] == [cx1 - cx0, cy1 - cy0]
            contexts.append(
                dict(
                    bbox_native=region["bbox_native"],
                    context_bbox_native=expected_context,
                    source_rgb_sha256=receipt["source_rgb_sha256"],
                    context_rgb_sha256=receipt["context_rgb_sha256"],
                )
            )
        assert len(contexts) == len(item["swinir_calls"]) == composition["sr_calls"]
        assert all(c["raw_pixels"] and c["crop_input"] for c in item["swinir_calls"])
        outside_mismatch, changed_inside = 0, 0
        for y in range(2704):
            support = np.repeat(union[y], 4)
            different = np.any(
                output[4 * y : 4 * y + 4] != base[4 * y : 4 * y + 4], axis=2
            )
            outside_mismatch += int(different[:, ~support].sum())
            changed_inside += int(different[:, support].sum())
        assert outside_mismatch == 0, "Pixels outside face mask changed: " + name
        rows.append(
            dict(
                frame=source_row["frame"],
                name=name,
                source_index=raw_row["source_index"],
                time_seconds=raw_row["time_seconds"],
                split=expected_split,
                native_wh=[1520, 2704],
                target_wh=[6080, 10816],
                source_png_sha256=source_sha,
                target_png_sha256=output_sha,
                decoded_rgb_sha256=array_sha(native),
                face_count=item["face_count"],
                masks=mask_rows,
                union_native_pixels=int(union.sum()),
                union_sha256=array_sha(union),
                outside_mismatched_4x_pixels=outside_mismatch,
                changed_inside_4x_pixels=changed_inside,
                raw_contexts=contexts,
                K4_exact=True,
                w2c_unchanged=True,
                full_frame_camera=True,
            )
        )
        print(
            "AUDITED",
            number,
            name,
            "outside",
            outside_mismatch,
            "inside_changed",
            changed_inside,
            flush=True,
        )
        del native, base, output, union
    assert train_count == 56 and test_count == 9
    changes = {}
    for path, expected in protected.items():
        actual = sha(path)
        if actual != expected:
            changes[path] = dict(before=expected, after=actual)
    assert not changes, "Protected files changed: " + str(changes)
    assert "torch" not in sys.modules, "CPU audit must never import torch/CUDA"
    result = dict(
        schema="msv2-full65-face-composite-input-audit-v1",
        passed=True,
        created=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic() - started,
        phase="CPU input pixels, calibration and provenance audit only",
        run=str(run),
        implementation_sha256=job["implementation_sha256"],
        source_video_sha256=VIDEO_SHA,
        input_sha256=INPUT_SHA,
        source_frames=65,
        target_images=65,
        registered_image_views=len(output_images),
        intrinsic_camera_models=len(output_cameras),
        target_wh=[6080, 10816],
        train_source_count=train_count,
        test_source_count=test_count,
        added_head_views=0,
        crop_cameras=False,
        all_K4_exact=True,
        all_w2c_unchanged=True,
        all_source_splits_match_baseline=True,
        all_native_RGB_unchanged=True,
        parser_detector_assets=assets,
        all_mask_exterior_pixels_equal_whole_Lanczos4=True,
        total_outside_mismatched_4x_pixels=sum(
            r["outside_mismatched_4x_pixels"] for r in rows
        ),
        total_changed_inside_4x_pixels=sum(r["changed_inside_4x_pixels"] for r in rows),
        face_region_count=sum(r["face_count"] for r in rows),
        empty_face_frames=sum(not r["face_count"] for r in rows),
        initial_sparse_sha256=INITIAL_POINTS_SHA,
        SfM_points_not_old_learned_Gaussians=True,
        old_model_files_hashed_only_not_loaded_for_initialization=True,
        protected_file_count=len(protected),
        all_protected_SHA256_unchanged=True,
        protected_SHA256=protected,
        torch_imported=False,
        GPU_used=False,
        native_renderer_preflight_executed=False,
        training_modified=False,
        quality_accepted=False,
        limitation="Pixel identity and camera checks do not prove semantic segmentation accuracy, reconstruction quality or a single causal explanation; scope and input layout both changed from old head trial.",
        records=rows,
    )
    save(record / "全65帧输入核查.json", result)
    print("ALL65_INPUT_AUDIT_PASSED", json_summary(result), flush=True)


def json_summary(result):
    import json

    return json.dumps(
        {
            k: result[k]
            for k in (
                "passed",
                "seconds",
                "target_images",
                "registered_image_views",
                "train_source_count",
                "test_source_count",
                "total_outside_mismatched_4x_pixels",
                "total_changed_inside_4x_pixels",
                "protected_file_count",
            )
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="/opt/gs-slam-lab/msv2-video15-20261105/runs/video15-face-composite4x",
    )
    parser.add_argument(
        "--record", default="/mnt/d/00/记录/2026-11-06_MSV2_video15_面部mask整图合成"
    )
    parser.add_argument(
        "--native-lineage",
        default="/mnt/d/GS-SLAM-Data/msv2-video15-20261105/pixel-lineage.json",
    )
    parser.add_argument(
        "--baseline", default="/opt/gs-slam-lab/msv2-video15-20261105/runs/video15"
    )
    parser.add_argument(
        "--candidate-source", default="/mnt/d/Projects/msv2-mask-compose"
    )
    args = parser.parse_args()
    try:
        audit(args)
    except Exception as error:
        save(
            Path(args.record) / "全65帧输入核查错误.json",
            dict(
                passed=False,
                error_type=type(error).__name__,
                error=str(error),
                GPU_used=False,
                training_modified=False,
                quality_accepted=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
